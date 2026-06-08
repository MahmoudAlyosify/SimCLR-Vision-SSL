
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# -- Local imports ----------------------------------------------------
try:
    from loss_supcon import SupConLoss
    from dataset_subset import get_stratified_subset
    from model import SimCLRResNet50, ProjectionHead
    from augmentations import SimCLRViewGenerator, CIFAR_MEAN, CIFAR_STD
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure you're running from the project folder")
    sys.exit(1)


# -- CIFAR-10 configuration -------------------------------------------
NUM_CLASSES = 10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper to return two augmented views + labels for SupCon training.
    
    Each __getitem__ returns:
        (view_1, view_2, label)
    
    where view_1 and view_2 are independent augmentations of the same image.
    """
    
    def __init__(self, cifar_dataset, augmentation_transform):
        self.dataset = cifar_dataset
        self.augmentation = augmentation_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Two independent augmentations
        view_1 = self.augmentation(image)
        view_2 = self.augmentation(image)
        
        return view_1, view_2, label


def build_augmentation():
    """
    Build the augmentation pipeline for SupCon training.
    
    Uses SimCLR's augmentation strategy:
      • RandomResizedCrop (0.2-1.0 scale)
      • RandomHorizontalFlip
      • RandomApply ColorJitter
      • RandomApply GaussianBlur (if available)
      • RandomGrayscale
      • Normalize with CIFAR-10 statistics
    """
    
    s = 0.5  # Color distortion strength (from SimCLR paper)
    
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.8*s,
                contrast=0.8*s,
                saturation=0.8*s,
                hue=0.2*s
            )
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    
    return augmentation


def create_datasets(data_dir, fraction=0.1, batch_size=512, num_workers=4):
    """
    Create stratified training subset + DataLoader for SupCon.
    
    Args:
        data_dir: Where to download/cache CIFAR-10
        fraction: Fraction of data (0.1 = 10%)
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader: DataLoader
        subset: The Subset object (for statistics)
    """
    
    # Download CIFAR-10 (no transforms here, augmentation happens in AugmentedDataset)
    train_set = CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=None
    )
    
    # Create stratified subset (5000 samples for 10%)
    subset = get_stratified_subset(
        train_set,
        fraction=fraction,
        verbose=True
    )
    
    # Wrap with augmentation
    augmentation = build_augmentation()
    augmented_subset = AugmentedDataset(subset, augmentation)
    
    # Create DataLoader
    train_loader = DataLoader(
        augmented_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, subset


def build_model_and_optimizer(learning_rate, use_lars=True):
    """
    Build ResNet-50 encoder + projection head + optimizer.
    
    Uses SGD + momentum 0.9 (Khosla et al., 2020, Section 4.1):
      • LR = 0.05 for batch_size=512
      • Weight decay = 1e-4
      • Momentum = 0.9
    
    Args:
        learning_rate: Base learning rate (0.05 for batch 512)
        use_lars: Use LARS optimizer (recommended for batch >= 4096)
    
    Returns:
        encoder: ResNet-50 backbone
        optimizer: SGD or LARS
    """
    
    # ResNet-50 encoder
    encoder = SimCLRResNet50(projection_dim=128)
    encoder.eval()  # Will be set to train() in training loop
    
    # Collect parameters
    params = list(encoder.parameters())
    
    if use_lars:
        try:
            from torchlars import LARS
            base_optimizer = optim.SGD(
                params,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
            optimizer = LARS(base_optimizer)
        except ImportError:
            print("⚠️ torchlars not found. Using SGD + momentum (SupCon default).")
            optimizer = optim.SGD(
                params,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
    else:
        # SGD + momentum 0.9 (matches SupCon paper exactly)
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
    
    return encoder, optimizer


def build_scheduler(optimizer, total_epochs, warmup_epochs=10):
    """Build cosine annealing scheduler (warmup handled per-step in train_epoch)."""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=0
    )
    return scheduler


def warmup_learning_rate(warmup_epochs, warmup_from, warmup_to,
                         epoch, batch_id, total_batches, optimizer):
    """Per-step linear warmup (matching official SupContrast repo)."""
    if epoch <= warmup_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warmup_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(
    epoch,
    train_loader,
    encoder,
    criterion,
    optimizer,
    warmup_epochs,
    warmup_from,
    warmup_to,
    device
):
    """
    Single epoch of SupCon training (FP32).
    Per-step warmup during first warmup_epochs (matching official repo).
    """
    
    encoder.train()
    loss_meter = 0.0
    time_start = time.time()
    total_batches = len(train_loader)
    
    for batch_idx, (view1, view2, labels) in enumerate(train_loader):
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)
        
        # Per-step warmup (official SupContrast approach)
        warmup_learning_rate(warmup_epochs, warmup_from, warmup_to,
                             epoch, batch_idx, total_batches, optimizer)
        
        optimizer.zero_grad()
        
        # Forward pass (FP32)
        _, z1 = encoder(view1)
        _, z2 = encoder(view2)
        
        # L2-normalize projections
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        # Stack for SupConLoss: [N, 2, 128]
        features = torch.stack([z1, z2], dim=1)
        
        # Compute loss
        loss = criterion(features, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        loss_meter += loss.item()
    
    time_elapsed = time.time() - time_start
    loss_avg = loss_meter / len(train_loader)
    
    return loss_avg, time_elapsed


def train(args):
    """
    Main training loop for SupCon Stage 1.
    """
    
    # -- Setup --------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # Create output directory
    run_id = f"supcon_resnet50_frac{int(args.fraction*100)}_bs{args.batch_size}_ep{args.epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(f"./outputs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  SUPERVISED CONTRASTIVE LEARNING (SupCon) -- STAGE 1")
    print(f"  Khosla et al. (2020)")
    print(f"{'='*70}")
    print(f"\n  [Config] Run ID: {run_id}")
    print(f"  [Device] {device}")
    if device.type == 'cuda':
        print(f"  [GPU]    {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [VRAM]   {vram:.1f} GB")
    
    # -- Create datasets and loaders ----------------------------------
    print(f"\n  [Data] Loading CIFAR-10 ({args.fraction:.1%} subset)...")
    train_loader, subset = create_datasets(
        args.data_dir,
        fraction=args.fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"         Subset size: {len(subset):,} samples")
    print(f"         Batch size: {args.batch_size}")
    print(f"         Steps/epoch: {len(train_loader)}")
    
    # -- Build model --------------------------------------------------
    print(f"\n  [Model] ResNet-50 + 128-dim projection")
    encoder, optimizer = build_model_and_optimizer(
        args.learning_rate,
        use_lars=args.use_lars
    )
    encoder = encoder.to(device)
    
    print(f"         Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
    
    # -- Loss and scheduler -------------------------------------------
    criterion = SupConLoss(temperature=0.1)  # Paper Section 4.5: "All our results used τ=0.1"
    
    warmup_epochs = 10
    warmup_from = 0.01
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=warmup_epochs)
    
    # Compute warmup target LR (from official SupContrast repo)
    import math
    warmup_to = args.learning_rate * (
        1 + math.cos(math.pi * warmup_epochs / args.epochs)) / 2
    
    print(f"  [Loss]  SupConLoss (tau={criterion.temperature})")
    print(f"  [Optim] {'LARS' if args.use_lars else 'SGD'} (momentum=0.9, wd=1e-4)")
    print(f"  [Sched] Cosine annealing + warmup ({warmup_epochs} epochs, from {warmup_from})")
    
    # -- Training loop ------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"   Epoch      Loss          LR        Time")
    print(f"{'-'*70}")
    
    log_data = []
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, time_taken = train_epoch(
            epoch,
            train_loader,
            encoder,
            criterion,
            optimizer,
            warmup_epochs,
            warmup_from,
            warmup_to,
            device
        )
        
        # Step cosine scheduler after warmup ends
        if epoch > warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        log_data.append({
            'epoch': epoch,
            'loss': loss,
            'learning_rate': current_lr,
            'time': time_taken
        })
        
        # Print
        marker = "*" if loss < best_loss else " "
        print(f"   {epoch:3d}    {loss:.4f}     {current_lr:.6f}   {time_taken:6.1f}s  {marker}")
        
        # Save best checkpoint
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save(
                encoder.state_dict(),
                output_dir / "checkpoints" / "supcon_best.pth"
            )
        
        # Save periodic checkpoints
        if epoch % 50 == 0:
            torch.save(
                encoder.state_dict(),
                output_dir / "checkpoints" / f"supcon_epoch_{epoch:03d}.pth"
            )
    
    print(f"{'-'*70}\n")
    
    # -- Save results -------------------------------------------------
    print(f"  [Checkpoint] Best model > {output_dir}/checkpoints/supcon_best.pth")
    
    # Save encoder for linear evaluation
    torch.save(
        encoder.state_dict(),
        output_dir / "supcon_encoder_final.pth"
    )
    print(f"  [Encoder] Saved for Phase 2 > {output_dir}/supcon_encoder_final.pth")
    
    # Save CSV log
    import csv
    with open(output_dir / "logs" / "training_log.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'learning_rate', 'time'])
        writer.writeheader()
        writer.writerows(log_data)
    print(f"  [Log] CSV saved > {output_dir}/logs/training_log.csv")
    
    # Save config
    config = vars(args)
    config['run_id'] = run_id
    config['best_loss'] = best_loss
    config['best_epoch'] = best_epoch
    with open(output_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ SupCon training complete!")
    print(f"  Best loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}\n")


def parse_args():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="SupCon Stage 1 Training -- Khosla et al. (2020)"
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to CIFAR-10 data')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='Fraction of data to use (0.1 = 10%%)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows compatibility)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Base learning rate (SupCon paper: 0.05 for batch 512)')
    parser.add_argument('--use_lars', action='store_true', default=False,
                        help='Use LARS optimizer (if available)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
