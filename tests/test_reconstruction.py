"""CPU regression tests; no downloads or online tracking required."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from src.dfs import DDS_Classifier
from src.main_class import Execution
from src.privacy_model import PrivacyCheckModel
from src.reconstruction import ReconstructionConfig
from src.simple_autoencoder import SimpleAutoencoder


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.dataset = TensorDataset(torch.randn(8, 1, 8, 8), torch.arange(8) % 2)

    def classifier(self, **kwargs):
        return DDS_Classifier(input_dim=(1, 8, 8), num_classes=2,
                              autoencoder=SimpleAutoencoder, n_features_to_select=3, **kwargs)

    def test_configuration_validation(self):
        self.assertFalse(ReconstructionConfig.from_value(None).enabled)
        self.assertTrue(ReconstructionConfig.from_value(True).enabled)
        for value in ({'epochs': 0}, {'epochs': True}, {'enabled': 'false'}, {'autoencoder': 'bad'}):
            with self.assertRaises(ValueError):
                ReconstructionConfig.from_value(value)

    def test_forward_contract_and_selected_input(self):
        x, _ = self.dataset[:4]
        for kwargs in ({}, {'baseline': True}, {'method': 'l2x'}, {'training_strategy': 1}):
            with self.subTest(kwargs=kwargs):
                model = self.classifier(**kwargs).eval()
                selected, _, _ = model.selected_input(x)
                output = model(x)
                self.assertEqual(len(output), 3)
                torch.testing.assert_close(output[0], model.classifier(selected.flatten(1)))

    def test_reconstruction_freezes_selector_and_batchnorm(self):
        selector = self.classifier()
        model = PrivacyCheckModel(selector, SimpleAutoencoder, (1, 8, 8)).train()
        state = {key: value.clone() for key, value in selector.state_dict().items()}
        loss = model.training_step(self.dataset[:4], 0)
        loss.backward()
        self.assertFalse(selector.training)
        self.assertTrue(all(p.grad is None for p in selector.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.privacy_autoencoder.parameters()))
        for key, value in selector.state_dict().items():
            torch.testing.assert_close(state[key], value)

    def test_execution_on_cpu_with_and_without_reconstruction(self):
        for enabled, validation_size in ((False, 0.5), (True, 0.5), (True, None)):
            with self.subTest(enabled=enabled, validation_size=validation_size):
                options = {'num_classes': 2, 'autoencoder': SimpleAutoencoder, 'n_features_to_select': 3}
                execution = Execution(
                    dataset=self.dataset, test_dataset=self.dataset, dataset_name='MNIST',
                    dfs_model_module=DDS_Classifier, dfs_model_module_kwargs=options,
                    cl_epochs=1, num_experiments=1, batch_size=4, loaders_workers=0,
                    validation_size=validation_size,
                    reconstruction={'enabled': enabled, 'autoencoder': 'simple', 'epochs': 1, 'log_images': True},
                )
                self.assertNotIn('input_dim', options)
                with patch('src.main_class.WandbLogger') as logger, patch('src.main_class.CSVLogger') as csv_logger:
                    logger.return_value = False
                    # Logger=False has no log_hyperparams, so use a local CSV logger.
                    from pytorch_lightning.loggers import CSVLogger
                    import tempfile
                    with tempfile.TemporaryDirectory() as directory:
                        logger.return_value = csv_logger.return_value = CSVLogger(directory)
                        result = execution.run_experiments(accelerator='cpu', devices=1,
                            enable_checkpointing=False, enable_progress_bar=False,
                            enable_model_summary=False, num_sanity_val_steps=0)
                self.assertIn('classification_test_accuracy', result[0][0])
                self.assertEqual('privacyCheck_test_loss' in result[0][0], enabled)

    def test_all_decoder_architectures_and_rgb(self):
        from src.conv_autoencoder import UNet
        from src.residual_autoencoder import ResidualAutoencoder
        for channels in (1, 3):
            for architecture in (UNet, SimpleAutoencoder, ResidualAutoencoder):
                with self.subTest(channels=channels, architecture=architecture.__name__):
                    selector = DDS_Classifier(input_dim=(channels, 8, 8), num_classes=2,
                                              n_features_to_select=3, autoencoder=SimpleAutoencoder)
                    model = PrivacyCheckModel(selector, architecture, (channels, 8, 8),
                        enc_channels={'input_channels': channels, 'output_channels': channels, 'base_channels': 4})
                    x = torch.randn(2, channels, 8, 8)
                    self.assertEqual(model(x).shape, x.shape)

    def test_augmented_selector_evaluation_and_reconstruction(self):
        from src.dfs_augmentation import DDS_Classifier as AugmentedClassifier
        selector = AugmentedClassifier(input_dim=(1, 8, 8), num_classes=2,
                                       n_features_to_select=3, autoencoder=SimpleAutoencoder).eval()
        x, _ = self.dataset[:4]
        first = selector(x)
        second = selector(x)
        torch.testing.assert_close(first[0], second[0])
        model = PrivacyCheckModel(selector, SimpleAutoencoder, (1, 8, 8))
        self.assertEqual(model(x).shape, x.shape)

    def test_disabled_stage_does_not_build_privacy_loaders(self):
        execution = Execution(dataset=self.dataset, dataset_name='MNIST',
                              dfs_model_module=DDS_Classifier, num_experiments=1, loaders_workers=0)
        with patch.object(execution, 'get_loaders', return_value=(None, None, None)) as loaders, \
             patch.object(execution, 'run_one_experiment', return_value=[]):
            execution._run_repeated_experiments()
        loaders.assert_called_once_with('cl')


if __name__ == '__main__':
    unittest.main()
