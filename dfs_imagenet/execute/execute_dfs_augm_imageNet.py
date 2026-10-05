import sys
import torch
from src.privacy_model import PrivacyCheckModel
from torchvision import transforms
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


from src.main_class import Execution
from src.dfs_augmentation import DDS_Classifier
from src.conv_autoencoder import UNet
from src.utils import normalize_tiny_imagenet
from torchvision.datasets import ImageFolder


dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
print(f'dataset_dir: {dataset_dir}')
train_path = os.path.abspath(os.path.join(dataset_dir, 'train'))
val_path = os.path.abspath(os.path.join(dataset_dir, 'val'))
test_path = os.path.abspath(os.path.join(dataset_dir, 'test'))


transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(normalize_tiny_imagenet),
        ])


train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=val_path, transform=transform)
#test_dataset = ImageFolder(root=test_path, transform=transform)
#val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

percent = 1
num_features_select = int((percent *(64*64))/100)

execution = Execution(dataset = train_dataset,
                      dataset_name = "tinyImageNet",
                      test_dataset = test_dataset,
                      validation_size = 0.2,
                      dfs_model_module = DDS_Classifier,
                      privacy_model_module = PrivacyCheckModel,
                      dfs_model_module_kwargs = {
                          "autoencoder": UNet,
                          "n_features_to_select": 10,
                          "training_strategy": 0,
                          "enc_channels": {'input_channels':3, 'output_channels': 3, 'base_channels':32},
                          "num_classes": len(train_dataset.classes),
                      },

                      cl_epochs = 100,
                      batch_size = 512,
                      num_experiments = 1,
                      project_name = "dfs_gpuTests")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    #wandb.init(project="dfs_gpuTests")
    execution.experiment_name = f"dfs_TinyimageNet_512batch10FeaturesAugmentation"
    execution.run_experiments()
    #wandb.finish()
    torch.cuda.empty_cache()

