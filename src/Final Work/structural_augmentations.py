# import's part
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import random

# CIFAR-10 Default Normalization Constants
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

class SobelTransform:
    """
    Applies Sobel filters (horizontal and vertical) to extract edge structure.
    Converts input to grayscale first, computes gradients, and repeats
    gradients across 3 channels to maintain compatibility with RGB architectures.
    """
    def __call__(self, img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)

        # Define horizontal and vertical Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)

        # Compress channels to compute structural gradients on grayscale
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)

        # Extract directional spatial derivatives via 2D convolution
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)

        # Compute gradient magnitude map and replicate for RGB matching
        edge = torch.sqrt(gx**2 + gy**2).squeeze(0)
        return edge.repeat(3, 1, 1)


class GaussianNoise:
    """
    Adds zero-mean additive Gaussian noise to the image.
    Forces the model to be invariant to high-frequency sensor noise.
    """
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)

        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0, 1)


class DiscreteRotation:
    """
    Applies discrete 90-degree rotations (0, 90, 180, 270) to the image.
    """
    def __call__(self, img):
        # Sample an orthogonal perspective switch randomly
        angles = [0, 90, 180, 270]
        return TF.rotate(img, random.choice(angles))


class SimCLRViewGenerator(object):
    """
    Generates multiple stochastic augmented views of a single image.
    Used for creating positive pairs in SimCLR contrastive training.
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # Generate independent stochastic views to form positive contrastive pairs
        return [self.base_transform(x) for _ in range(self.n_views)]



# Core Structural Augmentation Pipelines
# 1. Gaussian Blur Pipeline
get_blur_augmentation = lambda: T.Compose([
    T.RandomResizedCrop(32, scale=(0.6, 1.0)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# 2. Cutout / Random Erasing Pipeline
get_cutout_augmentation = lambda: T.Compose([
    T.RandomResizedCrop(32, scale=(0.6, 1.0)),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
    T.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

# 3. Sobel Edge Extraction Pipeline
get_sobel_augmentation = lambda: T.Compose([
    T.RandomResizedCrop(32, scale=(0.6, 1.0)),
    SobelTransform(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# 4. Gaussian Noise Pipeline
get_noise_augmentation = lambda: T.Compose([
    T.RandomResizedCrop(32, scale=(0.6, 1.0)),
    GaussianNoise(std=0.1),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])