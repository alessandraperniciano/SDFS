"""Run an experiment from a JSON configuration: python -m src.run --config ..."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.setdefault('WANDB_MODE', config.pop('wandb_mode', 'offline'))

    from torchvision import datasets, transforms
    from src.conv_autoencoder import UNet
    from src.dfs import DDS_Classifier
    from src.main_class import Execution
    from src.utils import normalize_mnist, normalize_cifar10

    name = config.pop('dataset', 'MNIST')
    choices = {'MNIST': datasets.MNIST, 'FashionMNIST': datasets.FashionMNIST,
               'CIFAR10': datasets.CIFAR10}
    if name not in choices:
        raise ValueError(f'dataset must be one of {tuple(choices)}')
    channels = 3 if name == 'CIFAR10' else 1
    normalize = normalize_cifar10 if channels == 3 else normalize_mnist
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalize)])
    data_dir = config.pop('data_dir', 'data')
    download = config.pop('download', True)
    train = choices[name](root=data_dir, train=True, transform=transform, download=download)
    test = choices[name](root=data_dir, train=False, transform=transform, download=download)
    model_options = config.pop('model', {})
    model_options.update(autoencoder=UNet)
    model_options.setdefault('enc_channels', {'input_channels': channels, 'output_channels': channels, 'base_channels': 16})
    trainer_options = config.pop('trainer', {})
    execution = Execution(dataset=train, test_dataset=test, dataset_name=name,
                          dfs_model_module=DDS_Classifier,
                          dfs_model_module_kwargs=model_options, **config)
    print(json.dumps(execution.run_experiments(**trainer_options), indent=2))


if __name__ == '__main__':
    main()
