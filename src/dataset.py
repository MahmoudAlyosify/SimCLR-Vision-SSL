# IMPORT PART
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os

# Importing transforms from augmentations.py
from .augmentations import (
    SimCLRViewGenerator, CIFAR_MEAN, CIFAR_STD,
    transform_exp1, transform_exp2, transform_exp3, transform_exp4,
    transform_exp5, transform_exp6, transform_exp7, transform_exp8,
)


#  Experiment Registry
#  Maps exp_id (1–8) → transform + description

EXP_REGISTRY = {
    1: {
        "transform":   transform_exp1,
        "name":        "Exp1 — Random Crop only",
        "description": "Spatial only: RandomResizedCrop. No color augmentation.",
    },
    2: {
        "transform":   transform_exp2,
        "name":        "Exp2 — Crop + Flip",
        "description": "Spatial only: RandomResizedCrop + RandomHorizontalFlip.",
    },
    3: {
        "transform":   transform_exp3,
        "name":        "Exp3 — Color Jitter only",
        "description": "Photometric only: ColorJitter (no spatial transforms).",
    },
    4: {
        "transform":   transform_exp4,
        "name":        "Exp4 — Grayscale only",
        "description": "Photometric only: RandomGrayscale (no spatial transforms).",
    },
    5: {
        "transform":   transform_exp5,
        "name":        "Exp5 — Crop + Color Jitter",
        "description": "Hybrid: RandomResizedCrop + ColorJitter.",
    },
    6: {
        "transform":   transform_exp6,
        "name":        "Exp6 — Crop + Flip + Color Jitter",
        "description": "Hybrid: RandomResizedCrop + Flip + ColorJitter.",
    },
    7: {
        "transform":   transform_exp7,
        "name":        "Exp7 — Crop + Flip + Grayscale",
        "description": "Hybrid: RandomResizedCrop + Flip + RandomGrayscale.",
    },
    8: {
        "transform":   transform_exp8,
        "name":        "Exp8 — Full SimCLR (Crop + Flip + Color + Grayscale)",
        "description": "Full SimCLR pipeline (paper default for CIFAR-10).",
    },
}



#  Custom SimCLR Dataset Wrapper - Positive Pair Generation for Contrastive Learning

class AugmentedDataset(Dataset):
    """
    Dataset wrapper that returns two stochastic augmented views of each image.
    Used for SimCLR contrastive pre-training.

    Returns:
        (view_1, view_2), label
        Unpacked in training loop as: (x_i, x_j), _ = batch
    """
    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = SimCLRViewGenerator(transform, n_views=2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = self.transform(img)          # [view_1, view_2]
        return (views[0], views[1]), label



#  DataLoader: Training
def get_train_dataloader(
    data_dir,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    exp_id=8,
):
    
    # Validate exp_id
    if exp_id not in EXP_REGISTRY:
        raise ValueError(
            f"Invalid exp_id={exp_id}. "
            f"Choose from {list(EXP_REGISTRY.keys())}."
        )

    # Log which experiment is running
    exp_info = EXP_REGISTRY[exp_id]
    print(f"  [Dataset] Augmentation: {exp_info['name']}")
    print(f"            {exp_info['description']}")

    #Load CIFAR-10
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = CIFAR10(root=data_dir, train=True, download=True)

    # Wrap with selected augmentation
    train_augmented = AugmentedDataset(train_dataset, exp_info["transform"])

    #Build DataLoader
    return DataLoader(
        train_augmented,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # ensures consistent batch size for NT-Xent
    )

def get_eval_dataloader(data_dir, train=False, batch_size=256, num_workers=0):

    os.makedirs(data_dir, exist_ok=True)

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    eval_dataset = CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=eval_transform,
    )

    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
