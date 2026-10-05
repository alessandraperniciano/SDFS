import torch
# Se definen funciones útiles que se emplearán durante los experimentos

# Statistiche standard di ImageNet (R, G, B)
TINY_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TINY_IMAGENET_STD = [0.229, 0.224, 0.225]

### FUNCIONES PARA NORMALIZAR Y DESNORMALIZAR
# Para MNIST
def normalize_mnist(tensor):
    return (tensor - 0.1307) / 0.3081

def denormalize_mnist(tensor):
    return tensor * 0.3081 + 0.1307

# Para CIFAR10
def normalize_cifar10(tensor):
    return (tensor - 0.5) / 0.5

def denormalize_cifar10(tensor):
    return tensor * 0.5 + 0.5


def normalize_tiny_imagenet(tensor):
    """
    Normalizza un tensore (C, H, W) usando le statistiche di ImageNet.
    """
    # Creiamo tensori per broadcasting corretto su (3, H, W)
    mean = torch.tensor(TINY_IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD, device=tensor.device).view(3, 1, 1)

    return (tensor - mean) / std


def denormalize_tiny_imagenet(tensor):
    """
    Denormalizza un tensore per riportarlo nel range [0, 1].
    """
    mean = torch.tensor(TINY_IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD, device=tensor.device).view(3, 1, 1)

    return tensor * std + mean

# ImageNet and Tiny ImageNet use the same channel statistics.
normalize_imagenet = normalize_tiny_imagenet
denormalize_imagenet = denormalize_tiny_imagenet
