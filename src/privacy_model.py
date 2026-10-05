
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union, Sequence

from src.conv_autoencoder import UNet
    

class PrivacyCheckModel(pl.LightningModule):
    def __init__(self,
                 dfs_model: pl.LightningModule,
                 autoencoder: nn.Module, 
                 input_dim: Union[torch.Size, Sequence[int], int],
                 hidden_dims = {'hidden_dim1': 200, 'hidden_dim2': 100, 'hidden_dim3': 50, 'hidden_dim4': 20}, 
                 enc_channels = {'input_channels':1, 'output_channels': 1, 'base_channels':16}, 
                 learning_rate = 1e-2, 
                 step_size = 40, gamma = 0.2,
                 experiment_number = None):
        
        super(PrivacyCheckModel, self).__init__()
                
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.experiment_number = experiment_number

        self.loss_function = nn.MSELoss()

        self.step_outputs = {'train':[], 'val':[], 'test':[]} #para almacenar el loss

        # Se definen el modelo dfs y el autoencoder
        if not callable(getattr(dfs_model, "selected_input", None)):
            raise TypeError("Reconstruction requires a classifier implementing selected_input(x).")
        self.dfs_model = dfs_model
        self.dfs_model.requires_grad_(False)
        self.dfs_model.eval()

        if autoencoder == UNet:
            autoencoder_kwargs = enc_channels
        else:
            autoencoder_kwargs = {'input_dim': np.prod(self.input_dim)}
            autoencoder_kwargs.update(hidden_dims)

        self.privacy_autoencoder = autoencoder(**autoencoder_kwargs)

        

    def train(self, mode=True):
        """Keep the trained feature selector frozen, including BatchNorm statistics."""
        super().train(mode)
        self.dfs_model.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            selected, _, _ = self.dfs_model.selected_input(x)
        return self.privacy_autoencoder(selected)

    def shared_step(self, batch, stage):
        x, _ = batch
        x_hat = self(x)

        loss = self.loss_function(x_hat, x)
        self.step_outputs[stage].append(loss.detach())
        return loss
            

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.privacy_autoencoder.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]
