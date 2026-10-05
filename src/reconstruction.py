"""Optional reconstruction stage and its serializable configuration."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ReconstructionConfig:
    """Train an independent decoder on the frozen classifier's selected input."""

    enabled: bool = False
    autoencoder: str = "unet"
    epochs: Optional[int] = None
    log_images: bool = True

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.log_images) is not bool:
            raise ValueError("enabled and log_images must be booleans")
        if self.autoencoder not in ("unet", "simple", "residual"):
            raise ValueError("autoencoder must be unet, simple or residual")
        if self.epochs is not None and (type(self.epochs) is not int or self.epochs < 1):
            raise ValueError("epochs must be a positive integer or null")

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if type(value) is bool:
            return cls(enabled=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("reconstruction must be a boolean, dictionary or ReconstructionConfig")

    def to_dict(self):
        return asdict(self)


def run_reconstruction(*, config, dfs_model, model_class, model_kwargs, input_dim,
                       exp_number, train_loader, val_loader, test_loader, logger,
                       dataset_name, default_epochs, log_interval, trainer_kwargs):
    """Fit and evaluate a reconstruction model without retraining the selector.

    Explicit legacy model_kwargs override the architecture and inferred channels.
    """
    import pytorch_lightning as pl
    from src.callbacks import ImageLoggerCallback, LogMetricsCallback
    from src.conv_autoencoder import UNet
    from src.simple_autoencoder import SimpleAutoencoder
    from src.residual_autoencoder import ResidualAutoencoder

    architectures = {"unet": UNet, "simple": SimpleAutoencoder, "residual": ResidualAutoencoder}
    kwargs = dict(model_kwargs)
    kwargs.setdefault("autoencoder", architectures[config.autoencoder])
    if kwargs["autoencoder"] is UNet:
        kwargs.setdefault("enc_channels", {
            "input_channels": input_dim[0], "output_channels": input_dim[0], "base_channels": 16,
        })
    kwargs.update(input_dim=input_dim, dfs_model=dfs_model, experiment_number=exp_number)
    epochs = config.epochs if config.epochs is not None else default_epochs
    callbacks = [LogMetricsCallback(fase="privacyCheck")]
    if config.log_images and val_loader is not None:
        callbacks.append(ImageLoggerCallback(epochs, log_interval, dataset_name))
    options = dict(trainer_kwargs)
    options.update(devices=options.get("devices", "auto"),
                   accelerator=options.get("accelerator", "auto"),
                   max_epochs=epochs, logger=logger,
                   callbacks=callbacks + list(options.get("callbacks") or []))
    # Restore the original flags after the optional stage, even if fitting fails.
    flags = [(parameter, parameter.requires_grad) for parameter in dfs_model.parameters()]
    was_training = dfs_model.training
    try:
        model = model_class(**kwargs)
        trainer = pl.Trainer(**options)
        trainer.fit(model, train_loader, val_loader)
        return trainer.test(model, test_loader)
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        dfs_model.train(was_training)
