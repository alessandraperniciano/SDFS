import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union, Sequence, Optional
from src.conv_autoencoder import UNet
from src.dds_model import DDS
from src.l2x import Sample_Concrete

class DDS_Classifier(pl.LightningModule):
    def __init__(self,  
                 input_dim: Union[torch.Size, Sequence[int], int],
                 n_features_to_select: int = 1, 
                 autoencoder: nn.Module = UNet,
                 method: Optional[str] = 'dds',
                 baseline: Optional[bool] = False,
                 training_strategy: Optional[int] = 0,
                 times_training_dds_over_bias: Optional[int] = 3,
                 penalty_loss: Optional[bool] = True,
                 hidden_dims = {'hidden_dim1': 200, 'hidden_dim2': 100, 'hidden_dim3': 50, 'hidden_dim4': 20}, 
                 enc_channels = {'input_channels':1, 'output_channels': 1, 'base_channels':16}, 
                 num_classes = 10, 
                 alpha = 0.1, beta = 2/3, delta_min = .2, delta_decay = .975, landa = 0.3,
                 learning_rate = 1e-3,
                 step_size = 400, gamma = 0.2,
                 experiment_number = None):
        
        super(DDS_Classifier, self).__init__()
        self.comparison_call_count = 0  # Contatore per il numero di chiamate
        self.max_comparison_calls = 10
        self.baseline = baseline
                
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_classifiers = 5
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha
        self.beta = beta
        self.delta_min = delta_min
        self.delta_decay = delta_decay
        self.delta = 1.
        self.landa = landa
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.experiment_number = experiment_number
        self.training_strategy = None if training_strategy == 0 else training_strategy
        self.times_training_dds_over_bias = times_training_dds_over_bias
        self.penalty_loss = penalty_loss

        self.accuracy = nn.ModuleDict({
            'train_acc': torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes),
            'val_acc': torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes),
            'test_acc': torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        })

        self.loss_function = nn.CrossEntropyLoss()

        #to save the loss
        self.step_outputs = {'train': [], 'val': [], 'test': []}

        for stage in ("train", "val", "test"):
            self.register_buffer(f"{stage}_times_selected", torch.zeros(self.input_dim), persistent=False)

        # Se definen los modelos autoencoder, dds, l2x, bias y clasificador
        if autoencoder == UNet:
            autoencoder_kwargs = enc_channels
        else:
            autoencoder_kwargs = {'input_dim': np.prod(self.input_dim)}
            autoencoder_kwargs.update(hidden_dims)

        self.dds_autoencoder = autoencoder(**autoencoder_kwargs)
        self.bias = autoencoder(**autoencoder_kwargs)

        self.method = method
        self.dds = DDS(n_features_to_select = self.n_features_to_select, alpha = self.alpha, beta = self.beta, delta = self.delta)

        self.l2x = Sample_Concrete(k = self.n_features_to_select)

        self.classifier = nn.Sequential(
            nn.Linear(np.prod(self.input_dim), self.num_classes)
        )

        self.save_hyperparameters()
        self.phase = 'classification'
        self.schedulers = [] #para poder actualizar el scheduler en medio del entrenamiento
        

    @property
    def times_selected(self):
        return {stage: getattr(self, f"{stage}_times_selected") for stage in ("train", "val", "test")}

    def selected_input(self, x):
        """Return the classifier input, selection penalty and feature mask."""
        # Primero se aplica el dds autoencoder
        if not self.baseline:
            encoded = self.dds_autoencoder(x)

        if self.method == 'dds' and not self.baseline:
            # Aplicación del modelo DDS
            # s probabilities, m the feature mask

            s, m = self.dds(encoded)

             # Se calcula la penalización del loss
            penalty = self.n_features_to_select*s

            # Se entrena el bias
            if self.training_strategy in [1, 2, 3]:
                b = self.bias(x)
                masked_data = s * m*(x + b)
            else:
                masked_data = s * m * x
        else:
            if not self.baseline:
                m = self.l2x(encoded)
                masked_data = m*x
            else:
                masked_data = x
                m = torch.ones(x.size(), device=x.device)
            
            # Se definen las siguientes dos variables de forma que no afecten nada
            penalty = torch.zeros(x.size(), device=x.device)
            s = torch.ones(x.size(), device=x.device)

        
        return masked_data, penalty, m

    def forward(self, x):
        masked_data, penalty, m = self.selected_input(x)
        # Apply the classifier to the selected input.
        masked_data = masked_data.view(masked_data.size(0), -1)
        output = self.classifier(masked_data)

        return output, penalty, m
        

    def shared_step(self, batch, stage):
        x, y = batch

        # forward pass calling the model
        # y_hat = output of the model
        y_hat, penalty, mask = self(x)

        loss = self.loss_function(y_hat, y)

        self.step_outputs[stage].append(loss.detach())

        # Cálculo de la precisión
        probabilities = torch.softmax(y_hat, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        self.accuracy[f'{stage}_acc'].update(predictions, y)

        # Se actualizan los contadores de selección
        self.times_selected[stage] += mask.detach().sum(dim=0)

        return loss

    def shared_epoch_end(self, stage):
        acc = self.accuracy[f'{stage}_acc'].compute()
        print(f'{stage}_acc = {acc:.4f}')
        self.accuracy[f'{stage}_acc'].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.delta = max(self.delta_min, self.delta_decay * self.delta)
        self.dds.delta = self.delta


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]


