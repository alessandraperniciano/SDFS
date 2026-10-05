import sys
import torch
from torchvision import transforms
from torchvision.datasets import ImageNet
import wandb

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Path of the directory where there is this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path of the project folder
project_root = os.path.abspath(os.path.join(script_dir, '../../')) # Risale due livelli dalla cartella 'execute'

# If it is not present, add project_root to sys_path
if project_root not in sys.path:
    sys.path.append(project_root)

# Add the folder src to syspath
src_dir = os.path.abspath(os.path.join(project_root, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Print sys.path for debugging
print("src_dir:", src_dir)

print("Current Working Directory (CWD):", os.getcwd())
print("Script Path:", os.path.abspath(__file__))
print("Python Path (sys.path):", sys.path)


from src.main_class import Execution
from src.privacy_model import PrivacyCheckModel
from src.dfs import DDS_Classifier
from src.conv_autoencoder import UNet
from src.utils import normalize_imagenet

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

aug_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(normalize_imagenet),
            transforms.RandomHorizontalFlip()
        ])
transform = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(normalize_imagenet)
        ])
train_dataset = ImageNet(root=dataset_dir, split='train', transform=transform)
test_dataset = ImageNet(root=dataset_dir, split='val', transform=transform)

execution = Execution(dataset = train_dataset,
                      dataset_name = "imageNet",
                      test_dataset = test_dataset,
                      validation_size = 0.2,
                      dfs_model_module = DDS_Classifier,
                      dfs_model_module_kwargs = {
                          "autoencoder": UNet,
                          "n_features_to_select": 25,
                          "training_strategy": 0,
                          "enc_channels": {'input_channels':3, 'output_channels': 3, 'base_channels':32},
                          "num_classes": len(train_dataset.classes),
                      },

                      cl_epochs = 200,
                      batch_size = 64,
                      num_experiments = 1,
                      project_name = "dfs_gpuTests")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(mode="offline", project="dfs_gpuTests")
    execution.experiment_name = f"dfs_imageNet_25FeaturesClassic"
    execution.run_experiments()
    wandb.finish()
    torch.cuda.empty_cache()