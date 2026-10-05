# Defines a class that orchestrates the entire process to execute an experiment

import os
import torch
import random
import numpy as np
from typing import Optional, Dict, Union, Tuple, List
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from sklearn.model_selection import KFold

from src.reconstruction import ReconstructionConfig, run_reconstruction
from src.privacy_model import PrivacyCheckModel

from src.callbacks import StrategyCallback, HeatmapLogger, LogMetricsCallback, DeltaDecay, SaveSelectedFeaturesCallback

import wandb


class Execution:
    def __init__(self,
                 dataset: Dataset,
                 dataset_name: str,
                 dfs_model_module: pl.LightningModule,
                 privacy_model_module: pl.LightningModule = PrivacyCheckModel,
                 dfs_model_module_kwargs: Optional[Dict] = None,
                 privacy_model_module_kwargs: Optional[Dict] = None,
                 privacy_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 privacy_test_dataset: Optional[Dataset] = None,
                 validation_size: Optional[float] = None,
                 test_size: Optional[float] = 0.2,
                 collate_preprocessing_fn: Optional[callable] = None,
                 loaders_workers: int = 10,
                 cl_epochs: int = 100,
                 privacy_epochs: int = 100,
                 log_interval: Optional[int] = None,
                 seed: int = 42,
                 num_experiments: int = 5,
                 batch_size: Union[int, Tuple[int, int, int]] = 512,
                 n_folds: Optional[int] = None,
                 project_name: str = "dfs_xxx",
                 experiment_name: str = "dfs_xxx",
                 save_selected_features: bool = False,
                 save_features_dir: Optional[str] = None,
                 save_features_n_images: int = 16,
                 reconstruction=None):

        self.reconstruction = ReconstructionConfig.from_value(reconstruction)
        self.dataset = dataset
        self.privacy_dataset = privacy_dataset if privacy_dataset is not None else dataset
        self.datasets = {'cl': self.dataset, 'privacy': self.privacy_dataset}
        self.dataset_name = dataset_name
        self.test_dataset = test_dataset
        self.privacy_test_dataset = privacy_test_dataset if privacy_test_dataset is not None else test_dataset
        self.test_datasets = {'cl': self.test_dataset, 'privacy': self.privacy_test_dataset}
        self.collate_fn = collate_preprocessing_fn
        self.loaders_workers = loaders_workers

        example, _ = dataset[0]
        self.input_dim = example.size()

        self.dfs_model_module = dfs_model_module
        self.dfs_model_module_kwargs = dict(dfs_model_module_kwargs or {})
        self.dfs_model_module_kwargs["input_dim"] = self.input_dim
        self.privacy_model_module = privacy_model_module
        self.privacy_model_module_kwargs = dict(privacy_model_module_kwargs or {})
        self.privacy_model_module_kwargs["input_dim"] = self.input_dim

        self.seed = seed
        self._set_seeds(self.seed)  # set seeds in numpy, torch and random

        # Modify the number of epochs according to the training strategy
        self.strategy = self.dfs_model_module_kwargs.get('training_strategy', 0)
        self.cl_epochs = cl_epochs if self.strategy in [0, 1, 4] else cl_epochs * 2
        self.privacy_epochs = privacy_epochs

        self.times_training_dds_over_bias = self.dfs_model_module_kwargs.get('times_training_dds_over_bias', None)

        self.num_experiments = num_experiments

        if isinstance(batch_size, int):
            self.batch_size = (batch_size, batch_size, batch_size)
        elif isinstance(batch_size, (tuple, list)) and len(batch_size) == 3:
            self.batch_size = tuple(batch_size)
        else:
            raise ValueError("batch_size must be a positive integer or three batch sizes")
        if any(type(size) is not int or size < 1 for size in self.batch_size):
            raise ValueError("batch sizes must be positive integers")

        self.log_interval = log_interval
        self.delta = self.dfs_model_module_kwargs.get('delta', 1)  # for the delta decay callback

        self.validation_size = validation_size
        self.test_size = test_size
        self.n_folds = n_folds

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.save_selected_features = save_selected_features
        self.save_features_dir = save_features_dir
        self.save_features_n_images = save_features_n_images

    def _set_seeds(self, seed):
        """
        Set seeds in pytorch, numpy and random to ensure reproducibility. Also sets seeds for
        DataLoader workers (only applicable under worker parallelism).
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)  # pseudo randomness on cpu
        torch.cuda.manual_seed_all(seed)  # pseudo randomness on gpu (all available gpus)
        random.seed(seed)

    def get_loaders(self, model):
        """
        Generates the train, validation and test DataLoaders
        """
        if self.test_datasets[model] is None:
            train_dataset, test_dataset = random_split(self.datasets[model], [1 - self.test_size, self.test_size])
        else:
            train_dataset = self.datasets[model]
            test_dataset = self.test_datasets[model]

        if self.validation_size:
            train_dataset, val_dataset = random_split(train_dataset, [1 - self.validation_size, self.validation_size])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size[1], shuffle=False,
                                    num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)
        else:
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size[0], shuffle=True,
                                  num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size[2], shuffle=False,
                                 num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)

        return train_loader, val_loader, test_loader

    def get_loaders_from_indices(self, model: str, train_indices: np.ndarray, val_indices: np.ndarray):
        """
        Generates DataLoaders for K-Fold Cross-Validation using the specified indices.

        Args:
            model: 'cl' for classification or 'privacy' for privacy model
            train_indices: indices of the training set for this fold
            val_indices: indices of the validation set for this fold

        Returns:
            train_loader, val_loader, test_loader
        """
        # Creates Subsets from the original datasets using the indices
        train_subset = Subset(self.datasets[model], train_indices)
        val_subset = Subset(self.datasets[model], val_indices)

        # The test set remains separate (if provided) or is None
        if self.test_datasets[model] is not None:
            test_dataset = self.test_datasets[model]
        else:
            # If there is no separate test set, use the validation fold as test
            test_dataset = val_subset

        train_loader = DataLoader(train_subset, batch_size=self.batch_size[0], shuffle=True,
                                  num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size[1], shuffle=False,
                                num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size[2], shuffle=False,
                                 num_workers=self.loaders_workers, collate_fn=self.collate_fn, pin_memory=True, persistent_workers=self.loaders_workers > 0)

        return train_loader, val_loader, test_loader

    def run_experiments(self, **trainer_kwargs):
        """
        Orchestrates the execution of experiments.
        If n_folds is specified, executes K-Fold Cross-Validation.
        Otherwise, executes repeated experiments with the same split.
        """
        exp_seed = np.random.randint(0, 1000)
        self._set_seeds(exp_seed)

        num_gpus = torch.cuda.device_count()
        print(f'\n\n- Number of available GPUs: {num_gpus}')
        print(f'- Number of EPOCHS: {self.cl_epochs} for classification and {self.privacy_epochs} for privacyCheck')
        print(f'- Training strategy: {self.strategy}')

        if self.times_training_dds_over_bias:
            print(f'- DDS trainings over 1 of Bias: {self.times_training_dds_over_bias}\n')
        else:
            print("\n")

        # If n_folds is specified, use K-Fold Cross-Validation
        if self.n_folds is not None and self.n_folds > 1:
            results = self.run_cross_validation(**trainer_kwargs)
        else:
            # Original behavior: repeated experiments with same split
            results = self._run_repeated_experiments(**trainer_kwargs)
        return results

    def _run_repeated_experiments(self, **trainer_kwargs):
        """
        Executes repeated experiments with the same train/val/test split (original behavior).
        """
        exp_number = 1

        cl_train_loader, cl_val_loader, cl_test_loader = self.get_loaders('cl')
        if not self.reconstruction.enabled:
            privacy_train_loader, privacy_val_loader, privacy_test_loader = None, None, None
        elif self.privacy_dataset is self.dataset and self.privacy_test_dataset is self.test_dataset:
            # Reuse the split so reconstruction never trains on classifier test samples.
            privacy_train_loader, privacy_val_loader, privacy_test_loader = cl_train_loader, cl_val_loader, cl_test_loader
        else:
            privacy_train_loader, privacy_val_loader, privacy_test_loader = self.get_loaders('privacy')
        train_loader, val_loader, test_loader = [cl_train_loader, privacy_train_loader], [cl_val_loader,
                                                                                          privacy_val_loader], [
            cl_test_loader, privacy_test_loader]

        results = []
        for _ in range(self.num_experiments):
            result = self.run_one_experiment(exp_number, train_loader, val_loader, test_loader, **trainer_kwargs)
            results.append(result)
            exp_number += 1
        return results

    def run_cross_validation(self, **trainer_kwargs):
        """
        Executes K-Fold Cross-Validation on the dataset.

        For each fold:
        - The dataset is divided into K parts
        - K-1 parts are used for training
        - 1 part is used for validation
        - The separate test set (if provided) is used for final evaluation

        Results are aggregated and logged at the end.
        """
        print(f'\n### Running {self.n_folds}-Fold Cross-Validation ###\n')

        if self.reconstruction.enabled and len(self.privacy_dataset) != len(self.dataset):
            raise ValueError("Cross-validation requires aligned classification and privacy datasets of equal length")

        # Creates the KFold object
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # Prepares the dataset indices
        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)

        # List to save results for each fold
        fold_results = []
        # Iterates over folds
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            fold_number = fold_idx + 1
            print(f'\n### Fold {fold_number}/{self.n_folds} ###')
            print(f'- Training samples: {len(train_indices)}')
            print(f'- Validation samples: {len(val_indices)}\n')
            print(
                f'- Test samples: {len(self.test_datasets["cl"]) if self.test_datasets["cl"] is not None else "N/A"} (separate)')

            # Gets the DataLoaders for this fold
            cl_train_loader, cl_val_loader, cl_test_loader = self.get_loaders_from_indices('cl', train_indices,
                                                                                           val_indices)
            privacy_train_loader, privacy_val_loader, privacy_test_loader = (
                self.get_loaders_from_indices('privacy', train_indices, val_indices)
                if self.reconstruction.enabled else (None, None, None)
            )

            train_loader = [cl_train_loader, privacy_train_loader]
            val_loader = [cl_val_loader, privacy_val_loader]
            test_loader = [cl_test_loader, privacy_test_loader]

            # Executes the experiment for this fold
            fold_result = self.run_one_experiment(
                exp_number=fold_number,
                train_set=train_loader,
                val_set=val_loader,
                test_set=test_loader,
                fold_number=fold_number,
                **trainer_kwargs
            )

            if fold_result is not None:
                fold_results.append(fold_result)

            # Cleanup memory after each fold
            self._cleanup_memory_after_fold(fold_number)

        # Print aggregated results
        if fold_results:
            self._print_cv_summary(fold_results)
        return fold_results

    def _cleanup_memory_after_fold(self, fold_number: int):
        """
        Cleans up memory after each fold of the cross-validation.
        """
        import gc

        print(f"🧹 Cleaning memory after fold {fold_number}...")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear MPS cache if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Force garbage collection
        gc.collect()

        print(f"✅ Memory cleaned after fold {fold_number}")

    def _print_cv_summary(self, fold_results: List[Dict]):
        """
        Prints a summary of the cross-validation results.
        """
        print(f'\n{"=" * 50}')
        print(f'### {self.n_folds}-Fold Cross-Validation Summary ###')
        print(f'{"=" * 50}\n')

        # Calculates statistics if results contain metrics
        fold_results = [result[0] if isinstance(result, list) and result else result for result in fold_results]
        if fold_results and isinstance(fold_results[0], dict):
            for metric_name in fold_results[0].keys():
                values = [r[metric_name] for r in fold_results if metric_name in r]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f'{metric_name}: {mean_val:.4f} ± {std_val:.4f}')

    def run_one_experiment(self, exp_number: int, train_set, val_set, test_set, fold_number: Optional[int] = None,
                           **trainer_kwargs):
        """
        Handles the execution of a single experiment.

        Args:
            exp_number: number of the experiment
            train_set: list of DataLoaders for training [cl, privacy]
            val_set: list of DataLoaders for validation [cl, privacy]
            test_set: list of DataLoaders for test [cl, privacy]
            fold_number: number of the fold (if in cross-validation mode)

        Returns:
            dict with the results of the experiment (for aggregation in CV)
        """
        if fold_number is not None:
            print(f'\n\n### Running Fold {fold_number} ###\n\n')
            exp_name = f"{self.experiment_name}_fold{fold_number}"
        else:
            print(f'\n\n### Running experiment {exp_number} ###\n\n')
            exp_name = self.experiment_name

        # Defines the logger
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            wandb_logger = CSVLogger("lightning_logs", name=exp_name)
        else:
            wandb_logger = WandbLogger(project=self.project_name, name=exp_name)
        try:
            wandb_logger.log_hyperparams({'dataset': self.dataset_name,
                                          'batch_size': self.batch_size,
                                          'cl_epochs': self.cl_epochs,
                                          'privacy_epochs': self.privacy_epochs,
                                          'fold_number': fold_number,
                                          'reconstruction': self.reconstruction.to_dict()})

            ### DFS CLASSIFIER MODEL ###
            # First, add the necessary arguments and define the model

            self.dfs_model_module_kwargs['experiment_number'] = exp_number

            dfs_model = self.dfs_model_module(**self.dfs_model_module_kwargs)

            # Defines the loggers and trainer
            strategyCallback = StrategyCallback(self.cl_epochs)
            heatmapLogger = HeatmapLogger()
            logMetricsCallback_cl = LogMetricsCallback(fase="classification")

            # To regulate the delta value
            delta_decay = DeltaDecay(self.delta, self.cl_epochs, self.strategy)

            callbacks_list = [strategyCallback, heatmapLogger, logMetricsCallback_cl, delta_decay]

            # Se richiesto, aggiunge il callback per salvare le feature selezionate
            if self.save_selected_features:
                if self.save_features_dir:
                    save_dir = os.path.join(self.save_features_dir, f"{self.experiment_name}")
                else:
                    save_dir = os.path.join("selected_features_images", f"{self.experiment_name}")
                save_features_cb = SaveSelectedFeaturesCallback(
                    test_dataset=test_set[0].dataset,
                    save_dir=save_dir,
                    dataset_name=self.dataset_name,
                    n_images=self.save_features_n_images
                )
                callbacks_list.append(save_features_cb)

            options = dict(trainer_kwargs)
            options.update(reload_dataloaders_every_n_epochs=0,
                           devices=options.get("devices", "auto"),
                           accelerator=options.get("accelerator", "auto"),
                           max_epochs=self.cl_epochs, logger=wandb_logger,
                           callbacks=callbacks_list + list(options.get("callbacks") or []))
            trainer_dfs = pl.Trainer(**options)

            # Launch the training
            trainer_dfs.fit(dfs_model, train_set[0], val_set[0])

            # Finally, evaluate the model
            result = trainer_dfs.test(dfs_model, test_set[0])

            print("\n\n")

            if self.reconstruction.enabled:
                reconstruction_result = run_reconstruction(
                    config=self.reconstruction,
                    dfs_model=dfs_model,
                    model_class=self.privacy_model_module,
                    model_kwargs=self.privacy_model_module_kwargs,
                    input_dim=self.input_dim,
                    exp_number=exp_number,
                    train_loader=train_set[1],
                    val_loader=val_set[1],
                    test_loader=test_set[1],
                    logger=wandb_logger,
                    dataset_name=self.dataset_name,
                    default_epochs=self.privacy_epochs,
                    log_interval=self.log_interval,
                    trainer_kwargs=trainer_kwargs,
                )
                # Preserve the existing list-of-dictionaries result contract.
                for metrics, reconstruction_metrics in zip(result, reconstruction_result):
                    metrics.update(reconstruction_metrics)

        finally:
            wandb.finish()

        # Cleanup memory after this experiment/fold
        if fold_number is not None:
            # Memory cleanup will be handled by _cleanup_memory_after_fold
            pass
        else:
            # For non-CV experiments, cleanup here
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        # Return test metrics for repeated experiments and CV aggregation.
        return result
