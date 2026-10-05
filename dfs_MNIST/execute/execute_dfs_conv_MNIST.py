
import os
import sys
import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets import MNIST

# Se obtiene la ruta absoluta del código fuente y se agrega a sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
execute_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["WANDB_DIR"] = os.path.join(execute_dir, "wandb")
from src.main_class import Execution

from src.dfs import DDS_Classifier
from src.privacy_model import PrivacyCheckModel
from src.conv_autoencoder import UNet

from src.utils import normalize_mnist



dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize_mnist)
        ])

train_dataset = MNIST(root=dataset_dir, train=True, transform=transform, download=True)
test_dataset = MNIST(root=dataset_dir, train=False, transform=transform, download=True)


execution = Execution(dataset = train_dataset,
                      dataset_name = "MNIST",
                      test_dataset = test_dataset,
                      dfs_model_module = DDS_Classifier,
                      privacy_model_module = PrivacyCheckModel,
                      dfs_model_module_kwargs = {
                          "autoencoder": UNet, 
                          "n_features_to_select": 1
                      },
                      
                      cl_epochs =  20,
                      batch_size = 512,
                      n_folds=1,
                      num_experiments = 1,
                      project_name = "dfs_MNIST_images",
                      save_selected_features = True,
                      save_features_dir = os.path.join(execute_dir, "selected_features_images"),
                      save_features_n_images = 16,
                      )



if __name__ == "__main__":
    total_results = {}
    for nFeatures in [1, 5, 10]:
        torch.cuda.empty_cache()
        if nFeatures not in total_results:
            total_results[nFeatures] = []
        execution.dfs_model_module_kwargs["n_features_to_select"] = nFeatures
        execution.experiment_name = f"dfs_mnist_{nFeatures}features"
        run_results = execution.run_experiments()
        for result in run_results:
            for key, value in result[0].items():
                if 'acc' in key:
                    total_results[nFeatures].append(value)
        for result in total_results:
            print('nFeatures', result, 'mean_acc', np.mean(total_results[result]), 'std_acc', np.std(total_results[result]))
        torch.cuda.empty_cache()
