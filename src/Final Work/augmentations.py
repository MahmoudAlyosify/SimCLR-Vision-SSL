
# IMPORT PART
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import random
import yaml
import os

# Load configs - use absolute path relative to this file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'augmentation.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

CIFAR_MEAN = tuple(config['normalization']['cifar_mean'])
CIFAR_STD = tuple(config['normalization']['cifar_std'])
s = config['color_distortion']['s']

# SIMCLR VIEW GENERATOR CLASS 
class SimCLRViewGenerator(object):
    """
    A custom wrapper to generate multiple augmented views of a single image,
    following the SimCLR framework for contrastive learning.
    """
    def __init__(self, base_transform, n_views=2):
        # The core stochastic augmentation pipeline to be applied to each input.
        self.base_transform = base_transform
        # Number of augmented views (typically 2 for creating positive pairs in SimCLR).
        self.n_views = n_views

    def __call__(self, x):
        """
        Executes the transformation pipeline multiple times to produce different 
        stochastic realizations of the same input image 'x'.
        """
        # Returns a list of 'n_views' variations, facilitating the calculation of contrastive loss.
        return [self.base_transform(x) for i in range(self.n_views)]


color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
# Applying jitter with 0.8 probability as per Chen et al. (2020)
rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)


# EXPERIMENT 1 — Random Resized Crop (Crop + Resize)
transform_exp1 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)), 
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 2 — Crop + Flip
transform_exp2 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 3 — Crop + Color
transform_exp3 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    rnd_color_jitter,
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 4 — Crop + Grayscale
transform_exp4 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 5 — Crop + Flip + Color
transform_exp5 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    rnd_color_jitter,
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 6 — Crop + Flip + Grayscale
transform_exp6 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 7 — Crop + Color + Grayscale
transform_exp7 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    rnd_color_jitter,
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# EXP 8 — Crop + Flip + Color + Grayscale
transform_exp8 = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    rnd_color_jitter,
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# ============================================================================
# NATALIE'S 28 EXPERIMENTS (from final augmentation.py)
# ============================================================================

class SobelTransform:
    def __call__(self, img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(gx**2 + gy**2).squeeze(0)
        return edge.repeat(3, 1, 1)

class GaussianNoise:
    def __init__(self, std=0.1): 
        self.std = std
    def __call__(self, img):
        if not torch.is_tensor(img): 
            img = TF.to_tensor(img)
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0, 1)

class DiscreteRotation:
    def __call__(self, img):
        angles = [0, 90, 180, 270]
        return TF.rotate(img, random.choice(angles))

nat_transform_exp1 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp2 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.3))])
nat_transform_exp3 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), SobelTransform(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp4 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

nat_transform_exp5 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp6 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp7 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), SobelTransform(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp8 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp9 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), rnd_color_jitter, T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp10 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), rnd_color_jitter, T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp11 = T.Compose([T.RandomResizedCrop(32), rnd_color_jitter, T.ToTensor(), SobelTransform(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp12 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), rnd_color_jitter, GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp13 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomGrayscale(p=0.2), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp14 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomGrayscale(p=0.2), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp15 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomGrayscale(p=0.2), SobelTransform(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp16 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomGrayscale(p=0.2), GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

nat_transform_exp17 = T.Compose([T.RandomResizedCrop(32), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), SobelTransform(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp18 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp19 = T.Compose([T.RandomResizedCrop(32), T.ToTensor(), SobelTransform(), GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp20 = T.Compose([T.RandomResizedCrop(32), T.GaussianBlur(3), T.ToTensor(), GaussianNoise(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp21 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), rnd_color_jitter, T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp22 = T.Compose([T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.9, scale=(0.05, 0.4), ratio=(0.3, 3.3))])

nat_transform_exp23 = T.Compose([T.RandomResizedCrop(32, scale=(0.08, 1.0)), T.RandomHorizontalFlip(), rnd_color_jitter, T.RandomGrayscale(p=0.2), T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp24 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), rnd_color_jitter, T.RandomGrayscale(p=0.2), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp25 = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), rnd_color_jitter, T.RandomGrayscale(p=0.2), T.ToTensor(), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), T.Normalize(CIFAR_MEAN, CIFAR_STD), T.RandomErasing(p=0.5)])
nat_transform_exp26 = T.Compose([T.RandomResizedCrop(32), T.RandomGrayscale(p=1.0), rnd_color_jitter, T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

nat_transform_exp27 = T.Compose([T.RandomResizedCrop(32, scale=(0.8, 1.0)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
nat_transform_exp28 = T.Compose([DiscreteRotation(), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
