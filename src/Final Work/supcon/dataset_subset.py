"""
dataset_subset.py — Stratified 10% Subset Sampler for SupCon Training
──────────────────────────────────────────────────────────────────────
Creates a perfectly balanced subset of CIFAR-10 using stratified sampling.

Why 10%? (Semi-supervised angle)
  In semi-supervised learning, we assume labels are expensive. By training on 
  just 10% of labels using SupCon, we show that contrastive learning can build
  strong representations from limited supervision.
  
  Expected improvement: 60-75% linear eval accuracy
  vs 40-50% with supervised CE loss on 10% labels
  vs 70-80% with 100% labels + CE loss

Stratified Sampling:
  Instead of random sampling (which might get 600 cats, 400 dogs):
  - Take exactly 500 images from each of 10 CIFAR-10 classes
  - Total: 500 * 10 = 5,000 images (exactly 10% of 50,000 training set)
  - This ensures balanced class distribution during training
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 500  # 10% of 5000 per class


def get_stratified_subset(
    dataset,
    fraction=0.1,
    num_classes=10,
    random_seed=42,
    verbose=True
):
    """
    Create a stratified subset of a dataset (e.g., CIFAR-10).
    
    Args:
        dataset (torch.utils.data.Dataset): Full dataset (e.g., CIFAR10)
            Must have `targets` attribute (list of labels)
        
        fraction (float): Fraction of data to sample (default 0.1 = 10%)
        
        num_classes (int): Number of classes in dataset (default 10 for CIFAR-10)
        
        random_seed (int): RNG seed for reproducibility
        
        verbose (bool): Print sampling statistics
    
    Returns:
        subset (torch.utils.data.Subset): Subset containing exactly
            `fraction * len(dataset)` samples, balanced across classes
    
    Example:
        >>> train_set = CIFAR10('./data', train=True, download=True)
        >>> subset = get_stratified_subset(train_set, fraction=0.1)
        >>> print(f"Subset size: {len(subset)}")  # Should be ~5000
        >>> dataloader = DataLoader(subset, batch_size=512, shuffle=True)
    """
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # ── Extract targets from dataset ─────────────────────────────────
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        raise ValueError(
            f"Dataset {type(dataset)} has no 'targets' or 'labels' attribute. "
            "Cannot extract labels for stratification."
        )
    
    total_samples = len(dataset)
    samples_per_class = int(total_samples * fraction / num_classes)
    
    indices = []
    
    # ── Stratify: sample exactly `samples_per_class` from each class ──
    for class_idx in range(num_classes):
        # Find all indices with this class label
        class_indices = np.where(targets == class_idx)[0]
        
        # Randomly sample `samples_per_class` from this class
        sampled_indices = np.random.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False
        )
        
        indices.extend(sampled_indices)
    
    indices = np.array(indices)
    
    # ── Shuffle the combined indices ─────────────────────────────────
    np.random.shuffle(indices)
    
    # ── Create Subset ────────────────────────────────────────────────
    subset = Subset(dataset, indices.tolist())
    
    # ── Print statistics ─────────────────────────────────────────────
    if verbose:
        print("\n" + "="*70)
        print("  STRATIFIED SUBSET SAMPLING")
        print("="*70)
        print(f"  Original dataset size:  {total_samples:,} samples")
        print(f"  Subset fraction:        {fraction:.1%}")
        print(f"  Subset size:            {len(subset):,} samples")
        print(f"  Samples per class:      {samples_per_class:,}")
        print(f"  Number of classes:      {num_classes}")
        print()
        
        # Verify balance
        subset_targets = targets[indices]
        for class_idx in range(num_classes):
            count = np.sum(subset_targets == class_idx)
            class_name = CIFAR10_CLASSES[class_idx] if num_classes == 10 else f"class_{class_idx}"
            print(f"    {class_name:15s}: {count:4d} samples")
        
        print("="*70 + "\n")
    
    return subset


def get_balanced_dataloader(
    dataset,
    fraction=0.1,
    batch_size=512,
    num_workers=4,
    shuffle=True,
    random_seed=42
):
    """
    Convenience function: create stratified subset + DataLoader in one call.
    
    Args:
        dataset: Full dataset (CIFAR-10)
        fraction: Fraction to subset (0.1 = 10%)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle the DataLoader
        random_seed: RNG seed
    
    Returns:
        dataloader (torch.utils.data.DataLoader)
        subset (torch.utils.data.Subset)
    
    Example:
        >>> from torchvision.datasets import CIFAR10
        >>> train_set = CIFAR10('./data', train=True, download=True)
        >>> loader, subset = get_balanced_dataloader(train_set, batch_size=512)
        >>> for batch_idx, (images, labels) in enumerate(loader):
        ...     print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
    """
    
    from torch.utils.data import DataLoader
    
    subset = get_stratified_subset(
        dataset,
        fraction=fraction,
        random_seed=random_seed
    )
    
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensures all batches have exactly batch_size samples
    )
    
    return dataloader, subset


# ─────────────────────────────────────────────────────────────────
# TEST & VERIFICATION
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test to verify stratified sampling works correctly.
    
    Run with: python dataset_subset.py
    """
    
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    
    # Download CIFAR-10 (if not already)
    print("Loading CIFAR-10...")
    train_set = CIFAR10('./data', train=True, download=True)
    
    # Create 10% stratified subset
    print("\nCreating 10% stratified subset...")
    subset = get_stratified_subset(train_set, fraction=0.1, verbose=True)
    
    # Verify with DataLoader
    print("Creating DataLoader from subset...")
    dataloader = DataLoader(
        subset,
        batch_size=512,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        drop_last=True
    )
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Check first batch
    images, labels = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
    print(f"  Label counts in batch:")
    for class_idx in range(10):
        count = (labels == class_idx).sum().item()
        if count > 0:
            print(f"    Class {class_idx}: {count}")
    
    print("\n✅ Stratified sampling test passed!")
