
import os
import sys

from torchvision import transforms
from torchvision.datasets import MNIST

# Se obtiene la ruta absoluta del código fuente y se agrega a sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(src_dir)

from src.main_class import Execution

from src.dfs import DDS_Classifier
from src.privacy_model import PrivacyCheckModel
from src.simple_autoencoder import SimpleAutoencoder

from src.utils import normalize_mnist


dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize_mnist)
        ])

train_dataset = MNIST(root=dataset_dir, train=True, transform=transform)
test_dataset = MNIST(root=dataset_dir, train=False, transform=transform)


execution = Execution(dataset = train_dataset,
                      dataset_name = "MNIST",
                      test_dataset = test_dataset,
                      validation_size = 0.2,
                      dfs_model_module = DDS_Classifier,
                      privacy_model_module = PrivacyCheckModel,
                      dfs_model_module_kwargs = {
                          "autoencoder": SimpleAutoencoder,
                          "n_features_to_select": 1
                      },
                      privacy_model_module_kwargs = {
                          "autoencoder": SimpleAutoencoder
                      },
                      cl_epochs = 100,
                      batch_size = 512,
                      num_experiments = 1,
                      project_name = "dfs_MNIST",
                      experiment_name = "dfs_simple")



if __name__ == "__main__":
    execution.run_experiments()


