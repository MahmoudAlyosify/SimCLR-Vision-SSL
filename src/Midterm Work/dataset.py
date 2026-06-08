"""
dataset.py — Modular Dataset & Augmentation Registry
Group 20, CISC 867, Queen's University, Spring 2026
Author (Student A): Natalie Nashed  — augmentation definitions
Author (Student B): Mahmoud Alyosify — registry architecture, SupCon loader

TWO REGISTRIES
──────────────
1. EXP_REGISTRY        — the 8 midterm ablation configs (exp_id 1–8)
2. EXTENDED_REGISTRY   — Natalie's 15–20 new augmentations (names as strings)
                          She fills in the transforms; the loader just calls
                          get_train_dataloader(..., aug_name='randaugment')

SUPCON SUPPORT
──────────────
get_supcon_dataloader() returns a labeled DataLoader for Bonus #4:
    - Stratified subsampling to `label_fraction` of the training set
    - Returns (view1, view2), label  so SupConLoss can use class info
"""

import os
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# ── Import from Natalie's augmentation module ───────────────────────────────
from augmentations import (
    SimCLRViewGenerator, CIFAR_MEAN, CIFAR_STD,
    transform_exp1, transform_exp2, transform_exp3, transform_exp4,
    transform_exp5, transform_exp6, transform_exp7, transform_exp8,
    nat_transform_exp1, nat_transform_exp2, nat_transform_exp3, nat_transform_exp4,
    nat_transform_exp5, nat_transform_exp6, nat_transform_exp7, nat_transform_exp8,
    nat_transform_exp9, nat_transform_exp10, nat_transform_exp11, nat_transform_exp12,
    nat_transform_exp13, nat_transform_exp14, nat_transform_exp15, nat_transform_exp16,
    nat_transform_exp17, nat_transform_exp18, nat_transform_exp19, nat_transform_exp20,
    nat_transform_exp21, nat_transform_exp22, nat_transform_exp23, nat_transform_exp24,
    nat_transform_exp25, nat_transform_exp26, nat_transform_exp27, nat_transform_exp28,
    nat_transform_exp29, nat_transform_exp30, nat_transform_exp31, nat_transform_exp32,
    nat_transform_exp33, nat_transform_exp34
)


# ══════════════════════════════════════════════════════════════════════════
# 1. REGISTRY 1 — Midterm 8-Experiment Ablation Configs
# ══════════════════════════════════════════════════════════════════════════

EXP_REGISTRY = {
    1: {
        "transform":   transform_exp1,
        "name":        "Exp1 — Crop only",
        "description": "Spatial only: RandomResizedCrop.",
    },
    2: {
        "transform":   transform_exp2,
        "name":        "Exp2 — Crop + Flip",
        "description": "Spatial: RandomResizedCrop + HorizontalFlip.",
    },
    3: {
        "transform":   transform_exp3,
        "name":        "Exp3 — Crop + ColorJitter",
        "description": "Hybrid: RandomResizedCrop + ColorJitter.",
    },
    4: {
        "transform":   transform_exp4,
        "name":        "Exp4 — Crop + Grayscale",
        "description": "Hybrid: RandomResizedCrop + RandomGrayscale.",
    },
    5: {
        "transform":   transform_exp5,
        "name":        "Exp5 — Crop + Flip + Color",
        "description": "Hybrid: RandomResizedCrop + Flip + ColorJitter.",
    },
    6: {
        "transform":   transform_exp6,
        "name":        "Exp6 — Crop + Flip + Grayscale",
        "description": "Hybrid: RandomResizedCrop + Flip + Grayscale.",
    },
    7: {
        "transform":   transform_exp7,
        "name":        "Exp7 — Crop + Color + Grayscale",
        "description": "Hybrid: RandomResizedCrop + ColorJitter + Grayscale.",
    },
    8: {
        "transform":   transform_exp8,
        "name":        "Exp8 — Full SimCLR",
        "description": "Full pipeline: Crop + Flip + ColorJitter + Grayscale.",
    },
    # NATALIE'S 28 EXPERIMENTS (Offset by 8)
    9:  {"transform": nat_transform_exp1,  "name": "Exp9(N1) — Crop + Blur", "description": "Single: Crop + Gaussian Blur."},
    10: {"transform": nat_transform_exp2,  "name": "Exp10(N2) — Crop + Cutout", "description": "Single: Crop + Random Erasing."},
    11: {"transform": nat_transform_exp3,  "name": "Exp11(N3) — Crop + Sobel", "description": "Single: Crop + Sobel Edge Extraction."},
    12: {"transform": nat_transform_exp4,  "name": "Exp12(N4) — Crop + Noise", "description": "Single: Crop + Gaussian Noise."},
    13: {"transform": nat_transform_exp5,  "name": "Exp13(N5) — Crop + Flip + Blur", "description": "Dual: Spatial + Blur."},
    14: {"transform": nat_transform_exp6,  "name": "Exp14(N6) — Crop + Flip + Cutout", "description": "Dual: Spatial + Corruption."},
    15: {"transform": nat_transform_exp7,  "name": "Exp15(N7) — Crop + Flip + Sobel", "description": "Dual: Spatial + Structural."},
    16: {"transform": nat_transform_exp8,  "name": "Exp16(N8) — Crop + Flip + Noise", "description": "Dual: Spatial + Noise."},
    17: {"transform": nat_transform_exp9,  "name": "Exp17(N9) — Crop + Color + Blur", "description": "Dual: Photometric + Blur."},
    18: {"transform": nat_transform_exp10, "name": "Exp18(N10) — Crop + Color + Cutout", "description": "Dual: Photometric + Corruption."},
    19: {"transform": nat_transform_exp11, "name": "Exp19(N11) — Crop + Color + Sobel", "description": "Dual: Photometric + Structural."},
    20: {"transform": nat_transform_exp12, "name": "Exp20(N12) — Crop + Color + Noise", "description": "Dual: Photometric + Noise."},
    21: {"transform": nat_transform_exp13, "name": "Exp21(N13) — Crop + Gray + Blur", "description": "Dual: Grayscale + Blur."},
    22: {"transform": nat_transform_exp14, "name": "Exp22(N14) — Crop + Gray + Cutout", "description": "Dual: Grayscale + Corruption."},
    23: {"transform": nat_transform_exp15, "name": "Exp23(N15) — Crop + Gray + Sobel", "description": "Dual: Grayscale + Structural."},
    24: {"transform": nat_transform_exp16, "name": "Exp24(N16) — Crop + Gray + Noise", "description": "Dual: Grayscale + Noise."},
    25: {"transform": nat_transform_exp17, "name": "Exp25(N17) — Crop + Blur + Sobel", "description": "Conflict: Smooth vs Sharp Edge."},
    26: {"transform": nat_transform_exp18, "name": "Exp26(N18) — Crop + Flip + Blur + Cutout", "description": "Conflict: Content Erasure with Blur."},
    27: {"transform": nat_transform_exp19, "name": "Exp27(N19) — Crop + Sobel + Noise", "description": "Conflict: Edge Extraction vs Pixel Noise."},
    28: {"transform": nat_transform_exp20, "name": "Exp28(N20) — Crop + Blur + Noise", "description": "Conflict: High-frequency Distortion (JPEG-like)."},
    29: {"transform": nat_transform_exp21, "name": "Exp29(N21) — Crop + Flip + Color + Erasing", "description": "Conflict: Color Distortion with Patch Erasure."},
    30: {"transform": nat_transform_exp22, "name": "Exp30(N22) — Crop + Heavy Cutout Grid", "description": "Extreme Spatial Corruption Baseline."},
    31: {"transform": nat_transform_exp23, "name": "Exp31(N23) — SimCLR Classic Strong", "description": "Standard SimCLR Pipeline (Chen et al. 2020)."},
    32: {"transform": nat_transform_exp24, "name": "Exp32(N24) — SimCLR + Rotation Task", "description": "Pipeline for E-SSL Joint-Task Pretraining."},
    33: {"transform": nat_transform_exp25, "name": "Exp33(N25) — Full Heatmap Suite", "description": "All Invariance Transforms Active Simultaneously."},
    34: {"transform": nat_transform_exp26, "name": "Exp34(N26) — Color Destruction", "description": "Strict Color Elimination: Grayscale p=1.0 + ColorJitter."},
    35: {"transform": nat_transform_exp27, "name": "Exp35(N27) — Weak Baseline", "description": "Minimal Spatial Invariance Only."},
    36: {"transform": nat_transform_exp28, "name": "Exp36(N28) — Pure Discrete Rotation", "description": "Isolated Structural Rotation Alignment."},
    # ── Exp 29 (N29) — The Ultimate Beast (added 18 May 2026) ──────────────────
    37: {
        "transform":   nat_transform_exp29,
        "name":        "Exp37(N29) — The Ultimate Beast",
        "description": (
            "Full pipeline: Crop + Flip + ColorJitter(p=0.8) + Grayscale(p=0.2) "
            "+ RandomErasing/Cutout(p=0.5, scale=5-20%). "
            "Designed to eliminate ALL shortcut-learning pathways simultaneously."
        ),
    },
    # ── Re-runs with Color Jitter (Added 19 May 2026) ──────────────────────────
    38: {"transform": nat_transform_exp30, "name": "Exp38 — Pure Rotation + Jitter (Base: Exp 36)", "description": "Exp 36 + Color Jitter"},
    39: {"transform": nat_transform_exp31, "name": "Exp39 — Weak Baseline + Jitter (Base: Exp 35)", "description": "Exp 35 + Color Jitter"},
    40: {"transform": nat_transform_exp32, "name": "Exp40 — Crop + Blur + Jitter (Base: Exp 9)", "description": "Exp 9 + Color Jitter"},
    41: {"transform": nat_transform_exp33, "name": "Exp41 — Crop + Flip + Blur + Jitter (Base: Exp 13)", "description": "Exp 13 + Color Jitter"},
    42: {"transform": nat_transform_exp34, "name": "Exp42 — Crop + Cutout + Jitter (Base: Exp 10)", "description": "Exp 10 + Color Jitter"},
}


# ══════════════════════════════════════════════════════════════════════════
# 2. REGISTRY 2 — Extended Augmentation Registry (Natalie fills these in)
#
#    Each entry key is the --aug_name CLI argument.
#    Natalie: add your transforms below following the same pattern.
#    Mahmoud's train_master.py will pick them up automatically.
# ══════════════════════════════════════════════════════════════════════════

def _base_crop():
    """All new augs build on top of the base SimCLR crop."""
    return T.RandomResizedCrop(32, scale=(0.2, 1.0))

# ── Placeholder entries — Natalie fills in the transform pipelines ─────────
# For each entry, replace T.Compose([_base_crop(), T.ToTensor(), ...])
# with your actual pipeline.

EXTENDED_REGISTRY = {

    # ── Natalie: fill these in ────────────────────────────────────────

    "randaugment": {
        "transform": T.Compose([
            _base_crop(),
            T.RandAugment(num_ops=2, magnitude=9),   # NATALIE: tune params
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "RandAugment",
        "description": "AutoML-style random augmentation policy search.",
    },

    "autoaugment": {
        "transform": T.Compose([
            _base_crop(),
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "AutoAugment (CIFAR-10 policy)",
        "description": "Learned CIFAR-10 augmentation policy (Cubuk et al.).",
    },

    "trivialaugment": {
        "transform": T.Compose([
            _base_crop(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "TrivialAugmentWide",
        "description": "Parameter-free augmentation (Müller & Hutter, 2021).",
    },

    "solarize": {
        "transform": T.Compose([
            _base_crop(),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomSolarize(threshold=128, p=0.2),   # NATALIE: tune threshold
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "Full SimCLR + Solarize",
        "description": "Exp8 + RandomSolarize (inspired by SimCLRv2).",
    },

    "equalize": {
        "transform": T.Compose([
            _base_crop(),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomEqualize(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "Full SimCLR + Equalize",
        "description": "Exp8 + RandomEqualize histogram normalization.",
    },

    "sharpness": {
        "transform": T.Compose([
            _base_crop(),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "Full SimCLR + Sharpness",
        "description": "Exp8 + RandomAdjustSharpness.",
    },

    "rotation": {
        "transform": T.Compose([
            _base_crop(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "Full SimCLR + Rotation",
        "description": "Exp8 + RandomRotation ±15°.",
    },

    "perspective": {
        "transform": T.Compose([
            _base_crop(),
            T.RandomHorizontalFlip(),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        "name":        "Full SimCLR + Perspective",
        "description": "Exp8 + RandomPerspective distortion.",
    },

    # NOTE: Mixup and CutMix require batch-level operations — they are
    # implemented in the training loop via torchvision.transforms.v2.
    # Natalie: implement these as collate_fn wrappers in augmentations.py
    # and register them here by referencing your collate_fn.
    # Mahmoud's DataLoader accepts an optional collate_fn parameter.
    "mixup": {
        "transform": transform_exp8,   # NATALIE: replace with Mixup-aware transform
        "name":        "Mixup (placeholder)",
        "description": "Batch-level Mixup — requires custom collate_fn.",
        "collate_fn":  None,           # NATALIE: set to your mixup_collate_fn
    },

    "cutmix": {
        "transform": transform_exp8,   # NATALIE: replace with CutMix-aware transform
        "name":        "CutMix (placeholder)",
        "description": "Batch-level CutMix — requires custom collate_fn.",
        "collate_fn":  None,           # NATALIE: set to your cutmix_collate_fn
    },
}


# ══════════════════════════════════════════════════════════════════════════
# 3. AugmentedDataset — returns (view1, view2), label
# ══════════════════════════════════════════════════════════════════════════

class AugmentedDataset(Dataset):
    """
    Wraps any torchvision dataset to return two augmented views.
    Compatible with both NT-Xent (labels ignored) and SupCon (labels used).
    """
    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = SimCLRViewGenerator(transform, n_views=2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = self.transform(img)      # [view_1, view_2]
        return (views[0], views[1]), label


# ══════════════════════════════════════════════════════════════════════════
# 4. DATALOADER FACTORIES
# ══════════════════════════════════════════════════════════════════════════

def _resolve_transform(exp_id: int, aug_name: str):
    """
    Resolves which transform to use.
    Priority: aug_name (extended registry) > exp_id (midterm registry)
    """
    if aug_name is not None:
        aug_name = aug_name.lower().strip()
        if aug_name not in EXTENDED_REGISTRY:
            available = list(EXTENDED_REGISTRY.keys())
            raise ValueError(
                f"Unknown aug_name='{aug_name}'. "
                f"Available: {available}"
            )
        info = EXTENDED_REGISTRY[aug_name]
        print(f"  [Dataset] Augmentation (extended): {info['name']}")
        print(f"            {info['description']}")
        return info["transform"], info.get("collate_fn", None)

    # Fallback to midterm exp_id
    if exp_id not in EXP_REGISTRY:
        raise ValueError(
            f"Invalid exp_id={exp_id}. Choose from {list(EXP_REGISTRY.keys())}."
        )
    info = EXP_REGISTRY[exp_id]
    print(f"  [Dataset] Augmentation (midterm): {info['name']}")
    print(f"            {info['description']}")
    return info["transform"], None


def get_train_dataloader(
    data_dir: str,
    batch_size: int   = 128,
    num_workers: int  = 4,
    shuffle: bool     = True,
    exp_id: int       = 8,
    aug_name: str     = None,
) -> DataLoader:
    """
    Returns a training DataLoader yielding ((view1, view2), label).

    Args:
        data_dir    : root directory for CIFAR-10 download
        batch_size  : training batch size
        num_workers : DataLoader worker processes
        shuffle     : shuffle training data
        exp_id      : midterm ablation config 1–8 (overridden by aug_name)
        aug_name    : named augmentation from EXTENDED_REGISTRY
    """
    transform, collate_fn = _resolve_transform(exp_id, aug_name)

    os.makedirs(data_dir, exist_ok=True)
    base_dataset = CIFAR10(root=data_dir, train=True, download=True)
    augmented    = AugmentedDataset(base_dataset, transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,      # ensures consistent batch size for NT-Xent
        persistent_workers=(num_workers > 0),
    )
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return DataLoader(augmented, **loader_kwargs)


def get_supcon_dataloader(
    data_dir: str,
    batch_size: int     = 256,
    num_workers: int    = 4,
    label_fraction: float = 0.1,
    exp_id: int         = 8,
    aug_name: str       = None,
    seed: int           = 42,
) -> DataLoader:
    """
    Returns a labeled DataLoader for Bonus #4 (SupCon training).
    Subsamples `label_fraction` of the CIFAR-10 training set, stratified
    per class so class balance is maintained.

    Args:
        label_fraction: 0.1 = 10% = 5,000 images (500 per class)
    """
    transform, collate_fn = _resolve_transform(exp_id, aug_name)
    rng = np.random.RandomState(seed)

    os.makedirs(data_dir, exist_ok=True)
    base_dataset  = CIFAR10(root=data_dir, train=True, download=True)
    labels_array  = np.array(base_dataset.targets)

    # Stratified subsampling
    selected = []
    for cls in range(10):
        cls_idx = np.where(labels_array == cls)[0].tolist()
        n_keep  = max(1, int(len(cls_idx) * label_fraction))
        chosen  = rng.choice(cls_idx, size=n_keep, replace=False).tolist()
        selected.extend(chosen)

    print(f"  [SupCon] {len(selected):,} labeled images "
          f"({label_fraction*100:.0f}% of 50k, stratified per class)")

    subset    = Subset(base_dataset, selected)
    augmented = AugmentedDataset(subset, transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return DataLoader(augmented, **loader_kwargs)


def get_eval_dataloader(
    data_dir: str,
    train: bool      = False,
    batch_size: int  = 256,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns an evaluation DataLoader (no augmentation, no shuffling).
    Used for: t-SNE generation, linear probe caching, FAISS index building.
    """
    os.makedirs(data_dir, exist_ok=True)

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    dataset = CIFAR10(
        root=data_dir, train=train, download=True, transform=eval_transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
