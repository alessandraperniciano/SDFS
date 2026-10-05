
import os
import sys
import torch

from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
# Se obtiene la ruta absoluta del código fuente y se agrega a sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
execute_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["WANDB_DIR"] = os.path.join(execute_dir, "wandb")

from src.main_class import Execution
from src.dfs import DDS_Classifier
from src.privacy_model import PrivacyCheckModel
from src.conv_autoencoder import UNet

from src.utils import normalize_cifar10

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

aug_transform = transforms.Compose(
            [transforms.ToTensor(),
            # transforms.Lambda(normalize_cifar10),
            transforms.RandomHorizontalFlip()
        ])
transform = transforms.Compose(
            [transforms.ToTensor(),
            #transforms.Lambda(normalize_cifar10)
        ])

train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
test_dataset = CIFAR10(root=dataset_dir, train=False, transform=transform, download=True)

execution = Execution(dataset = train_dataset,
                      dataset_name = "CIFAR10",
                      test_dataset = test_dataset,
                      #validation_size = 0.2,
                      dfs_model_module = DDS_Classifier,
                      privacy_model_module = PrivacyCheckModel,
                      dfs_model_module_kwargs = {
                          "autoencoder": UNet,
                          "n_features_to_select": 25,
                          "training_strategy": 0,
                          "enc_channels": {'input_channels':3, 'output_channels': 3, 'base_channels':16}
                      },

                      cl_epochs = 100,
                      batch_size = 512,
                      num_experiments = 1,
                      n_folds=1,
                      project_name = "dfs_CIFAR10_IMAGES",
                      save_selected_features = True,
                      save_features_dir = os.path.join(execute_dir, "selected_features_images"),
                      save_features_n_images = 20,
                      )



if __name__ == "__main__":
    total_results = {}
    for nPixels in [1, 5, 10]:
        torch.cuda.empty_cache()
        if nPixels not in total_results:
            total_results[nPixels] = []
        execution.dfs_model_module_kwargs["n_features_to_select"] = nPixels
        execution.experiment_name = f"dfs_cifar10_{nPixels}pixels_5fold"
        run_results = execution.run_experiments()
        for result in run_results:
            for key, value in result[0].items():
                if 'acc' in key:
                    total_results[nPixels].append(value)
        for result in total_results:
            print('nPixels', result, 'mean_acc', np.mean(total_results[result]), 'std_acc', np.std(total_results[result]))
        torch.cuda.empty_cache()
