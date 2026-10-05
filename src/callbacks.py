
# Se definen los callbacks

import os
import wandb
import torch
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pytorch_lightning.callbacks import Callback

from src.utils import denormalize_mnist, denormalize_cifar10, denormalize_tiny_imagenet



### CALLBACK PARA MANEJAR ESTRATEGIAS DE ENTRENAMIENTO

class StrategyCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_train_epoch_start(self, trainer, pl_module):
        '''
        Cuando se emplea la estrategia 2, orquesta el entrenamiento para entrenar primero el dds y luego el bias
        '''
        current_epoch = trainer.current_epoch

        if pl_module.training_strategy == 2:
            if current_epoch < self.epochs // 2: #primera parte del entrenamiento
                for param in pl_module.bias.parameters():
                    param.requires_grad = False
            else: #segunda parte del entrenamiento
                if current_epoch == self.epochs // 2: # reinicia el learning rate y el scheduler
                    for group in trainer.optimizers[0].param_groups:
                        group['lr'] = pl_module.learning_rate
                    new_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 
                                                             step_size=pl_module.step_size, gamma=pl_module.gamma)
                    pl_module.schedulers = [new_scheduler] #se reinicia el scheduler cuando se empieza a entrenar el bias

                for param in pl_module.dds_autoencoder.parameters():
                    param.requires_grad = False
                for param in pl_module.dds.parameters():
                    param.requires_grad = False
                for param in pl_module.bias.parameters():
                    param.requires_grad = True


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        '''
        Cuando se emplea la estrategia 3, orquesta el entrenamiento para entrenar el bias 
        '''
        if pl_module.training_strategy == 3:
            times_training_dds_over_bias = pl_module.times_training_dds_over_bias
            p_dds = 1 / (1 + times_training_dds_over_bias) #se establecen los porcentajes
            p_bias = 1 - p_dds

            freeze_bias = np.random.choice([True, False], p=[p_bias, p_dds])
            if freeze_bias:
                for param in pl_module.bias.parameters():
                    param.requires_grad = False
                for param in pl_module.dds_autoencoder.parameters():
                    param.requires_grad = True
                for param in pl_module.dds.parameters():
                    param.requires_grad = True
            else:
                for param in pl_module.dds_autoencoder.parameters():
                    param.requires_grad = False
                for param in pl_module.dds.parameters():
                    param.requires_grad = False
                for param in pl_module.bias.parameters():
                    param.requires_grad = True



### CALLBACK PARA LOGGEAR MAPAS DE CALOR

class HeatmapLogger(Callback):
    def __init__(self):
        super().__init__()
        self.names = {'train': 'Entrenamiento', 'val': 'Validación', 'test': 'Test'}

    def log_heatmaps(self, pl_module, stage):
        tensor = pl_module.times_selected[stage]

        # Calcula el número de canales
        num_channels = tensor.shape[0]

        # Crea una figura con una fila de subplots
        fig, axs = plt.subplots(1, num_channels, figsize=(num_channels*4, 4))

        # Añade un título general a la figura
        plt.suptitle(self.names[stage], fontsize=16)

        # Para cada canal en el tensor
        for i in range(num_channels):
            # Selecciona el subplot correspondiente
            ax = axs[i] if num_channels > 1 else axs

            channel = tensor[i].cpu()

            # Crea el mapa de calor
            cax = ax.imshow(channel.detach().numpy(), cmap='Blues')
            colorbar = fig.colorbar(cax, ax=ax)

            colorbar.locator = plt.MaxNLocator(integer=True) #se indica que los ticks de la barra de color sean enteros
            colorbar.update_ticks()

            ax.set_title(f'Channel {i}')

        # Ajusta la separación entre los subplots
        plt.subplots_adjust(wspace=0.5)

        # Guarda la figura en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Loggea la imagen en wandb
        if wandb.run is not None:
            wandb.log({f'{stage}_times_selected': wandb.Image(fig)})

        # Cierra la figura para evitar que se muestren gráficos superpuestos
        plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        self.log_heatmaps(pl_module, 'train')

    def on_validation_end(self, trainer, pl_module):
        self.log_heatmaps(pl_module, 'val')

    def on_test_end(self, trainer, pl_module):
        self.log_heatmaps(pl_module, 'test')



### CALLBACK PARA LOGGEAR METRICAS

class LogMetricsCallback(Callback):
    def __init__(self, fase):
        super().__init__()
        self.fase = fase

    def shared_epoch_end(self, stage, trainer, pl_module):
        if not pl_module.step_outputs[stage]:
            return
        metrics = {}

        if self.fase == 'classification':
            # Se obtiene el valor del accuracy
            avg_accuracy = pl_module.accuracy[f'{stage}_acc'].compute()
            metrics[f'{self.fase}_{stage}_accuracy'] = avg_accuracy

            # Se resetea la métrica de precisión
            pl_module.accuracy[f'{stage}_acc'].reset()


        # Se obtiene el valor del loss
        avg_loss = torch.stack(pl_module.step_outputs[stage]).mean()
        metrics[f'{self.fase}_{stage}_loss'] = avg_loss

        if pl_module.step_outputs.get(stage + '_allignment_loss'):
            metrics[f'{self.fase}_{stage}_allignment_loss'] = torch.stack(pl_module.step_outputs[stage + '_allignment_loss']).mean()

        if stage + '_allignment_loss' in pl_module.step_outputs:
            pl_module.step_outputs[stage + '_allignment_loss'] = []
        pl_module.step_outputs[stage] = [] #se reinicia el objeto que almacena el loss
            
        # Se loggean las métricas
        pl_module.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_train_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end("train", trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end("test", trainer, pl_module)



### CALLBACK PARA LOGGEAR IMAGENES DE RECONSTRUCCIÓN DURANTE VALIDACIÓN DE PRIVACYCHECK

class ImageLoggerCallback(Callback):
    def __init__(self, privacy_epochs, log_interval, dataset_name):
        super().__init__()
        self.epochs = privacy_epochs
        self.log_interval = log_interval
        self.dataset_name = dataset_name
        self.last_batch = None

        if self.dataset_name in ("MNIST", "FashionMNIST"):
            self.denormalize_func = denormalize_mnist
            self.cmap = 'gray'
        elif self.dataset_name == "CIFAR10":
            self.denormalize_func = denormalize_cifar10
            self.cmap = None

        else:
            self.denormalize_func = denormalize_tiny_imagenet if "imagenet" in self.dataset_name.lower() else lambda x: x
            self.cmap = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.last_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_interval:
            condition = trainer.current_epoch % self.log_interval == 0
        else:
            condition = trainer.current_epoch == (self.epochs-1)

        if condition and self.last_batch is not None and not trainer.sanity_checking:
            batch_size = self.last_batch[0].size(0)
            indices = np.random.choice(batch_size, size=min(8, batch_size), replace=False)
            images = self.last_batch[0][indices]

            # Se obtienen las reconstrucciones de las imágenes seleccionadas
            with torch.no_grad():
                reconstructions = pl_module(images)

            # Se desnormalizan las imagenes
            images = self.denormalize_func(images)
            reconstructions = self.denormalize_func(reconstructions)

            # Se crea un subplot de 4x4
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
            for i, ax in enumerate(axs.flat):
                if i // 2 >= len(images):
                    ax.axis("off")
                    continue
                # Se muestra la imagen original y su reconstrucción
                if i % 2 == 0:
                    img = images[i // 2].cpu().numpy()
                else:
                    img = reconstructions[(i - 1) // 2].detach().cpu().numpy()

                # Si las imágenes son en escala de grises, se eliminan las dimensiones extra
                if self.cmap == 'gray':
                    img = img.squeeze()
                else:
                    img = img.transpose((1, 2, 0))

                ax.imshow(img, cmap=self.cmap)
                ax.set_title(f'{"Original" if i % 2 == 0 else "Reconstruction"} {i // 2 + 1}')
            plt.suptitle(f'Epoch {trainer.current_epoch}')
            plt.tight_layout()
            # Se loggea el subplot en wandb
            if wandb.run is not None:
                wandb.log({"reconstruction": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
        self.last_batch = None



### CALLBACK PARA delta DECAY

class DeltaDecay(Callback):
    def __init__(self, delta, cl_epochs, strategy):
        super().__init__()
        self.delta = delta
        self.half_epochs = max(1, cl_epochs // 4 if strategy == 2 else cl_epochs // 2)
        self.delta_reduction = self.delta / self.half_epochs  #cantidad a reducir 'delta' en cada epoch
        self.strategy = strategy

    def shared_epoch_start(self, stage, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if self.strategy == 2: #se 
            current_epoch = current_epoch%(self.half_epochs*2)
            if current_epoch == 0:
                pl_module.delta = self.delta #se reinicia para cuando empiece a entrenar el Bias
        
        if current_epoch >= 1 and current_epoch < self.half_epochs and pl_module.delta > 0.1:
            pl_module.delta -= self.delta_reduction
        elif current_epoch == self.half_epochs:
            pl_module.delta = 0.1 #para evitar problemas de redondeo
        
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.shared_epoch_start("train", trainer, pl_module)


### CALLBACK PER SALVARE LE FEATURE SELEZIONATE E LE IMMAGINI CON OVERLAY

class SaveSelectedFeaturesCallback(Callback):
    """
    Callback che, al termine del test, salva:
      - Le feature selezionate (maschera) come tensore .pt e come immagine .png
      - Le immagini originali con le feature selezionate sovraimpresse (overlay)
      - Una griglia riassuntiva con le immagini affiancate (originale | overlay)
    """

    def __init__(self, test_dataset, save_dir, dataset_name="CIFAR10", n_images=16):
        """
        Args:
            test_dataset: il dataset di test da cui prendere le immagini
            save_dir: cartella in cui salvare le immagini
            dataset_name: nome del dataset (CIFAR10, MNIST, FashionMNIST)
            n_images: numero di immagini da salvare
        """
        super().__init__()
        self.test_dataset = test_dataset
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.n_images = n_images

    def on_test_end(self, trainer, pl_module):
        """Viene chiamato alla fine del test. Salva feature e immagini."""
        os.makedirs(self.save_dir, exist_ok=True)
        paired_dir = os.path.join(self.save_dir, "paired")
        os.makedirs(paired_dir, exist_ok=True)

        pl_module.eval()
        device = next(pl_module.parameters()).device

        # Determina cmap in base al dataset
        is_grayscale = self.dataset_name in ["MNIST", "FashionMNIST"]
        cmap = 'gray' if is_grayscale else None

        n = min(self.n_images, len(self.test_dataset))
        indices = list(range(n))

        # Raccoglie immagini originali e maschere
        all_masks = []
        all_images = []
        all_labels = []

        with torch.no_grad():
            for idx in indices:
                img, label = self.test_dataset[idx]
                img_tensor = img.unsqueeze(0).to(device)

                # Forward pass per ottenere la maschera
                mask = pl_module(img_tensor)[2]

                all_images.append(img.cpu())
                all_masks.append(mask.squeeze(0).cpu())
                all_labels.append(label)

        # Salva il tensore delle maschere
        masks_tensor = torch.stack(all_masks)
        torch.save(masks_tensor, os.path.join(self.save_dir, "selected_features_masks.pt"))

        # Salva le immagini singole
        for i in range(n):
            img = all_images[i]
            mask = all_masks[i]
            label = all_labels[i]
            prefix = f"img{i}_label{label}"

            # --- Salva immagine originale ---
            self._save_image(img, os.path.join(self.save_dir, f"{prefix}_original.png"),
                             cmap=cmap, title="Original")

            # --- Salva maschera ---
            self._save_mask(mask, os.path.join(self.save_dir, f"{prefix}_mask.png"))

            # --- Salva immagine con feature selezionate sovraimpresse ---
            self._save_overlay(img, mask, os.path.join(self.save_dir, f"{prefix}_selected.png"),
                               cmap=cmap, is_grayscale=is_grayscale)

            # --- Salva coppia (originale | overlay) ---
            self._save_pair(img, mask, os.path.join(paired_dir, f"{prefix}_pair.png"),
                            cmap=cmap, is_grayscale=is_grayscale)

        # --- Salva griglia riassuntiva ---
        n_pixels = pl_module.n_features_to_select
        self._save_grid(all_images, all_masks, all_labels,
                        os.path.join(self.save_dir, f"grid_highlighted_{n_pixels}px.png"),
                        cmap=cmap, is_grayscale=is_grayscale, n_pixels=n_pixels)

        print(f"\n✅ Immagini con feature selezionate salvate in: {self.save_dir}")

    @staticmethod
    def _to_numpy_img(img_tensor):
        """Converte un tensore immagine (C, H, W) in numpy (H, W, C) o (H, W) per grayscale."""
        img = img_tensor.numpy()
        if img.shape[0] == 1:
            return img.squeeze(0)
        else:
            return np.transpose(img, (1, 2, 0))

    def _save_image(self, img_tensor, path, cmap=None, title=None):
        """Salva un'immagine singola."""
        fig, ax = plt.subplots(figsize=(3, 3))
        img_np = self._to_numpy_img(img_tensor)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np, cmap=cmap)
        if title:
            ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_mask(self, mask_tensor, path):
        """Salva la maschera come heatmap."""
        fig, ax = plt.subplots(figsize=(3, 3))
        # Somma sui canali per ottenere maschera 2D
        if mask_tensor.dim() == 3:
            mask_2d = mask_tensor.sum(dim=0).numpy()
        else:
            mask_2d = mask_tensor.numpy()
        ax.imshow(mask_2d, cmap='Reds', interpolation='nearest')
        ax.set_title("Feature Mask")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_overlay(self, img_tensor, mask_tensor, path, cmap=None, is_grayscale=False):
        """Salva l'immagine originale con le feature selezionate sovraimpresse in rosso."""
        fig, ax = plt.subplots(figsize=(3, 3))
        img_np = self._to_numpy_img(img_tensor)
        img_np = np.clip(img_np, 0, 1)

        # Prepara maschera booleana 2D
        if mask_tensor.dim() == 3:
            mask_bool = (mask_tensor.sum(dim=0) > 0).numpy()
        else:
            mask_bool = (mask_tensor > 0).numpy()

        # Mostra l'immagine originale
        ax.imshow(img_np, cmap=cmap)

        # Overlay rosso semitrasparente sulle feature selezionate
        overlay = np.zeros((*mask_bool.shape, 4))  # RGBA
        overlay[mask_bool, 0] = 1.0  # Rosso
        overlay[mask_bool, 3] = 0.5  # Alpha
        ax.imshow(overlay)

        ax.set_title("Selected Features")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_pair(self, img_tensor, mask_tensor, path, cmap=None, is_grayscale=False):
        """Salva una coppia (originale | overlay) fianco a fianco."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

        img_np = self._to_numpy_img(img_tensor)
        img_np = np.clip(img_np, 0, 1)

        # Maschera booleana 2D
        if mask_tensor.dim() == 3:
            mask_bool = (mask_tensor.sum(dim=0) > 0).numpy()
        else:
            mask_bool = (mask_tensor > 0).numpy()

        # Originale
        ax1.imshow(img_np, cmap=cmap)
        ax1.set_title("Original")
        ax1.axis('off')

        # Overlay
        ax2.imshow(img_np, cmap=cmap)
        overlay = np.zeros((*mask_bool.shape, 4))
        overlay[mask_bool, 0] = 1.0
        overlay[mask_bool, 3] = 0.5
        ax2.imshow(overlay)
        ax2.set_title("Selected Features")
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_grid(self, images, masks, labels, path, cmap=None, is_grayscale=False, n_pixels=None):
        """Salva una griglia con tutte le immagini: originale a sinistra, overlay a destra."""
        n = len(images)
        cols = 2  # originale | overlay
        rows = min(n, 8)  # massimo 8 righe

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axs = axs[np.newaxis, :]

        for i in range(rows):
            img_np = self._to_numpy_img(images[i])
            img_np = np.clip(img_np, 0, 1)

            mask = masks[i]
            if mask.dim() == 3:
                mask_bool = (mask.sum(dim=0) > 0).numpy()
            else:
                mask_bool = (mask > 0).numpy()

            # Originale
            axs[i, 0].imshow(img_np, cmap=cmap)
            axs[i, 0].set_title(f"Label {labels[i]}", fontsize=9)
            axs[i, 0].axis('off')

            # Overlay
            axs[i, 1].imshow(img_np, cmap=cmap)
            overlay = np.zeros((*mask_bool.shape, 4))
            overlay[mask_bool, 0] = 1.0
            overlay[mask_bool, 3] = 0.5
            axs[i, 1].imshow(overlay)
            axs[i, 1].set_title("Selected", fontsize=9)
            axs[i, 1].axis('off')

        title = f"Selected Features"
        if n_pixels is not None:
            title += f" ({n_pixels}px)"
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)













