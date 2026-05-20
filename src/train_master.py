"""
train_master.py — SimCLR Final-Phase Master Training Script
Group 20, CISC 867, Queen's University, Spring 2026
Author (Student B): Mahmoud Alyosify

PURPOSE
───────
This is THE production training script for the final 200-epoch ResNet-50
SimCLR run on the RTX 5000 Ada (32 GB VRAM). It also supports:
  • Quick hyperparameter sweeps (τ, batch size, LR)
  • The 8 midterm ablation experiments (--exp_id 1-8)
  • SupCon Bonus #4 (--loss supcon --label_fraction 0.1)

PERFORMANCE FEATURES ENABLED
─────────────────────────────
1. torch.amp.autocast    — FP16 activations, halves memory bandwidth
2. torch.compile()       — TorchInductor/Triton fusion, ~25-35% speedup
3. cudnn.benchmark=True  — fastest conv algorithm auto-select
4. pin_memory + non_blocking — overlap CPU→GPU transfers with compute
5. drop_last=True        — ensures every batch has exactly N samples
                           (critical for NT-Xent similarity matrix)

USAGE EXAMPLES
──────────────
# The 200-epoch master run (wait for Natalie's best aug combo):
python train_master.py --epochs 200 --batch_size 1024 --backbone resnet50

# Quick 8-experiment ablation (all 8 midterm configs in sequence):
python train_master.py --run_all_ablations --epochs 20 --batch_size 128

# Temperature sweep:
python train_master.py --temperature 0.1 --exp_id 8 --epochs 50
python train_master.py --temperature 0.5 --exp_id 8 --epochs 50
python train_master.py --temperature 1.0 --exp_id 8 --epochs 50

# Batch size sweep:
python train_master.py --batch_size 256 --exp_id 8 --epochs 50
python train_master.py --batch_size 512 --exp_id 8 --epochs 50
python train_master.py --batch_size 1024 --exp_id 8 --epochs 50

# Bonus #4 — SupCon with 10% labels:
python train_master.py --loss supcon --label_fraction 0.1 --epochs 200

HAND-OFFS
─────────
→ Mirna : checkpoints at outputs/<run_id>/checkpoints/  (.pth files)
           training CSV log at outputs/<run_id>/logs/training_log.csv
→ Natalie: training CSV log to verify convergence with her augmentations
"""

import os
import sys
import time

# Windows console encoding fix (tau character support)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import math
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from sklearn.manifold import TSNE

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model import get_simclr_model
from loss import get_loss
from dataset import get_train_dataloader, get_eval_dataloader


# ══════════════════════════════════════════════════════════════════════════
# 1. ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SimCLR Final-Phase Training — Group 20"
    )

    # ── Data ──────────────────────────────────────────────────────────
    p.add_argument("--data_dir",      type=str,   default="./data")
    p.add_argument("--output_dir",    type=str,   default="./outputs")
    p.add_argument("--exp_id",        type=int,   default=8,
        help="Augmentation config 1–8 (midterm ablations). "
             "Ignored if --aug_name is set.")
    p.add_argument("--aug_name",      type=str,   default=None,
        help="Named augmentation from Natalie's extended registry "
             "(e.g. 'randaugment', 'cutmix'). Overrides --exp_id.")

    # ── Model ─────────────────────────────────────────────────────────
    p.add_argument("--backbone",      type=str,   default="resnet50",
        choices=["resnet18", "resnet50"])
    p.add_argument("--projection_dim", type=int,  default=128)

    # ── Loss ──────────────────────────────────────────────────────────
    p.add_argument("--loss",          type=str,   default="ntxent",
        choices=["ntxent", "supcon"],
        help="'ntxent' for standard SimCLR. "
             "'supcon' for Bonus #4 hybrid training.")
    p.add_argument("--temperature",   type=float, default=0.5,
        help="NT-Xent τ. Sweep {0.1, 0.5, 1.0}.")
    p.add_argument("--label_fraction", type=float, default=0.1,
        help="Fraction of labels used for SupCon. Only active when "
             "--loss supcon. Default 0.10 = 10%%.")

    # ── Training ──────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=1024)
    p.add_argument("--lr",            type=float, default=0.03,
        help="Peak LR for cosine schedule. "
             "Linear scaling rule: lr = 0.03 * batch_size / 256")
    p.add_argument("--auto_lr",       action="store_true", default=True,
        help="Apply linear LR scaling rule automatically.")
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int,   default=10,
        help="Linear warmup. Paper uses 10 epochs for 200-epoch runs.")
    p.add_argument("--optimizer",     type=str,   default="adamw",
        choices=["adamw", "sgd", "lars"],
        help="'lars' recommended for very large batches (BS >= 4096). "
             "'adamw' works well for BS 1024 on CIFAR-10.")

    # ── Checkpointing ─────────────────────────────────────────────────
    p.add_argument("--save_every",    type=int,   default=50,
        help="Save checkpoint every N epochs. "
             "Always saves at 50/100/150/200 regardless.")
    p.add_argument("--resume",        type=str,   default=None,
        help="Path to checkpoint .pth to resume from.")

    # ── Compute ───────────────────────────────────────────────────────
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no_compile",    action="store_true", default=False,
        help="Disable torch.compile (useful for debugging).")

    # ── Convenience ───────────────────────────────────────────────────
    p.add_argument("--run_all_ablations", action="store_true", default=False,
        help="Run all 8 midterm experiments sequentially in subprocesses.")
    p.add_argument("--tsne_final_only",   action="store_true", default=True)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# 2. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════════════════
# 3. LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Linear warmup followed by cosine annealing (step-level granularity).
    Using step-level (not epoch-level) ensures smooth LR even for large batches
    where each epoch has very few steps.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear ramp from 0 → 1
            return float(step) / max(warmup_steps, 1)
        # Cosine annealing from 1 → 0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════
# 4. OPTIMIZER FACTORY
# ══════════════════════════════════════════════════════════════════════════

def build_optimizer(model, args):
    """
    Three optimizer options:
      adamw — best for BS 512–1024, stable, our primary choice
      sgd   — used in the original SimCLR paper with LARS for large batches
      lars  — wraps SGD with layer-adaptive rate scaling; needed for BS >=4096
    """
    params = model.parameters()

    if args.optimizer == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == "sgd":
        return optim.SGD(
            params, lr=args.lr,
            momentum=0.9, weight_decay=args.weight_decay, nesterov=True
        )

    if args.optimizer == "lars":
        # LARS is not in core PyTorch — use a pure-Python wrapper
        # that applies layer-wise adaptive rate scaling on top of SGD.
        try:
            from torch.optim import SGD

            class LARS(optim.Optimizer):
                """Simplified LARS (Layerwise Adaptive Rate Scaling)."""
                def __init__(self, params, lr, momentum=0.9,
                             weight_decay=1e-6, eta=1e-3):
                    defaults = dict(lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, eta=eta)
                    super().__init__(params, defaults)

                @torch.no_grad()
                def step(self, closure=None):
                    loss = None
                    if closure is not None:
                        with torch.enable_grad():
                            loss = closure()
                    for group in self.param_groups:
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            param_norm = p.data.norm(2)
                            grad_norm  = p.grad.norm(2)
                            if param_norm > 0 and grad_norm > 0:
                                adaptive_lr = (
                                    group["eta"] * param_norm / (
                                        grad_norm
                                        + group["weight_decay"] * param_norm
                                        + 1e-9
                                    )
                                )
                                adaptive_lr = min(adaptive_lr, group["lr"])
                            else:
                                adaptive_lr = group["lr"]
                            d_p = p.grad + group["weight_decay"] * p.data
                            if "momentum_buffer" not in self.state[p]:
                                self.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
                            buf = self.state[p]["momentum_buffer"]
                            buf.mul_(group["momentum"]).add_(d_p)
                            p.data.add_(buf, alpha=-adaptive_lr)
                    return loss

            return LARS(params, lr=args.lr, weight_decay=args.weight_decay)

        except Exception as e:
            print(f"[WARN] LARS build failed ({e}), falling back to AdamW.")
            return optim.AdamW(params, lr=args.lr,
                               weight_decay=args.weight_decay)

    raise ValueError(f"Unknown optimizer '{args.optimizer}'.")


# ══════════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    print(f"  [Resume] Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_loss   = ckpt.get("best_loss", float("inf"))
    loss_history = ckpt.get("loss_history", [])
    print(f"  [Resume] Resuming from epoch {start_epoch}, best loss {best_loss:.4f}")
    return start_epoch, best_loss, loss_history


# ══════════════════════════════════════════════════════════════════════════
# 6. t-SNE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_tsne(model, device, data_dir, epoch, output_dir,
                  num_workers=4, run_id="run"):
    print(f"\n  [t-SNE] Generating for epoch {epoch}...")
    model.eval()

    loader = get_eval_dataloader(
        data_dir, train=False, batch_size=512, num_workers=num_workers
    )
    all_feats, all_labels = [], []
    for imgs, lbls in loader:
        feats, _ = model(imgs.to(device))
        all_feats.append(feats.cpu().numpy())
        all_labels.append(lbls.numpy())
        if sum(x.shape[0] for x in all_feats) >= 2000:
            break

    feats  = np.concatenate(all_feats)[:2000]
    labels = np.concatenate(all_labels)[:2000]
    
    if np.isnan(feats).any():
        print("[t-SNE] Skipped because of NaN features")
        return

    emb = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    ).fit_transform(feats)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10",
                    alpha=0.6, s=8)
    cb = plt.colorbar(sc, ax=ax, ticks=range(10))
    cb.ax.set_yticklabels(class_names)
    ax.set_title(f"t-SNE — Epoch {epoch}  ({run_id})", fontsize=13,
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    save_path = os.path.join(output_dir, "plots", f"tsne_epoch_{epoch:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE] Saved → {save_path}")
    model.train()


# ══════════════════════════════════════════════════════════════════════════
# 7. LOSS CURVE PLOT
# ══════════════════════════════════════════════════════════════════════════

def plot_loss_curve(loss_history, best_loss, output_dir, run_id):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(loss_history) + 1), loss_history,
            lw=2, marker="o", markersize=4)
    ax.axhline(y=best_loss, ls="--", alpha=0.6,
               label=f"Best loss = {best_loss:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent / SupCon Loss")
    ax.set_title(f"Training Loss — {run_id}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "plots", "training_loss_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Loss curve saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 8. DATASET HELPERS FOR SUPCON (10% LABEL FILTERING)
# ══════════════════════════════════════════════════════════════════════════

def get_supcon_dataloader(data_dir, batch_size, num_workers,
                          label_fraction=0.1, exp_id=8, seed=42):
    """
    For Bonus #4: returns a DataLoader that yields
        (view1, view2), label
    but only for `label_fraction` of CIFAR-10 training images
    (stratified per class so class balance is maintained).
    """
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader, Subset
    from dataset import AugmentedDataset, EXP_REGISTRY
    import random as _random

    rng = np.random.RandomState(seed)

    base = CIFAR10(root=data_dir, train=True, download=True)
    labels_array = np.array(base.targets)

    # Stratified sampling — keep `label_fraction` per class
    selected_indices = []
    for cls in range(10):
        cls_idx = np.where(labels_array == cls)[0].tolist()
        n_keep  = max(1, int(len(cls_idx) * label_fraction))
        chosen  = rng.choice(cls_idx, size=n_keep, replace=False).tolist()
        selected_indices.extend(chosen)

    subset    = Subset(base, selected_indices)
    transform = EXP_REGISTRY[exp_id]["transform"]
    augmented = AugmentedDataset(subset, transform)

    print(f"  [SupCon] Using {len(selected_indices):,} images "
          f"({label_fraction*100:.0f}% of CIFAR-10)")

    return DataLoader(
        augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# 9. RUN ALL ABLATIONS (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════

def run_all_ablations(args):
    """
    Spawns 8 subprocesses — one per midterm augmentation config.
    Each uses --exp_id 1..8 and the epoch/batch settings you passed.
    Useful for regenerating all 8 t-SNE plots and loss curves at once.
    """
    print("\n" + "=" * 60)
    print("Running all 8 ablation experiments sequentially")
    print("=" * 60)

    for exp in range(1, 9):
        print(f"\n{'─'*50}")
        print(f"  Ablation Exp {exp}/8")
        print(f"{'─'*50}")

        cmd = [
            sys.executable, __file__,
            "--exp_id",       str(exp),
            "--epochs",       str(args.epochs),
            "--batch_size",   str(args.batch_size),
            "--backbone",     args.backbone,
            "--loss",         "ntxent",
            "--temperature",  str(args.temperature),
            "--output_dir",   os.path.join(args.output_dir, f"exp_{exp}"),
            "--data_dir",     args.data_dir,
            "--num_workers",  str(args.num_workers),
            "--seed",         str(args.seed),
        ]
        if args.no_compile:
            cmd.append("--no_compile")

        try:
            subprocess.run(cmd, check=True)
            print(f"  [OK] Experiment {exp} finished.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Experiment {exp} crashed — code {e.returncode}")

    print("\n" + "=" * 60)
    print("All 8 ablations done.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# 10. MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def train(args):

    # ── Seed & cuDNN ────────────────────────────────────────────────
    set_seed(args.seed)
    cudnn.benchmark = True    # fastest conv algo selection

    # ── Run ID for organized output dirs ────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_tag = args.aug_name if args.aug_name else f"exp{args.exp_id}"
    run_id  = (
        f"{args.backbone}_{aug_tag}_"
        f"{args.loss}_τ{args.temperature}_"
        f"bs{args.batch_size}_ep{args.epochs}_{ts}"
    )
    output_dir = os.path.join(args.output_dir, run_id)

    for sub in ("checkpoints", "plots", "logs"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # ── Save run config to JSON (for reproducibility) ───────────────
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n  [Config] Saved run config → {config_path}")

    # ── Device ──────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"  [Device] {device}  |  AMP={use_amp}")
    if device.type == "cuda":
        print(f"  [GPU]    {torch.cuda.get_device_name(0)}")
        print(f"  [VRAM]   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Auto LR scaling ─────────────────────────────────────────────
    # Linear scaling rule: lr = base_lr × batch_size / 256
    # Reference: Goyal et al. (2017) "Accurate, Large Minibatch SGD"
    if args.auto_lr:
        args.lr = 0.03 * args.batch_size / 256
        print(f"  [LR]     Auto-scaled to {args.lr:.4f} "
              f"(0.03 × {args.batch_size} / 256)")

    # ── Data ────────────────────────────────────────────────────────
    if args.loss == "supcon":
        train_loader = get_supcon_dataloader(
            args.data_dir,
            args.batch_size,
            args.num_workers,
            label_fraction=args.label_fraction,
            exp_id=args.exp_id,
            seed=args.seed,
        )
    else:
        train_loader = get_train_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            exp_id=args.exp_id,
            aug_name=args.aug_name,   # None → falls back to exp_id
        )

    steps_per_epoch = len(train_loader)
    total_samples   = len(train_loader.dataset)

    print(f"\n  [Data]   {total_samples:,} training samples")
    print(f"           {steps_per_epoch} steps/epoch  |  BS={args.batch_size}")

    # ── Model ───────────────────────────────────────────────────────
    model = get_simclr_model(
        backbone=args.backbone,
        projection_dim=args.projection_dim,
    ).to(device)

    # torch.compile: TorchInductor fuses kernels → ~25-35% speedup on Ada
    # Disabled in first-run debugging with --no_compile
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  [Compile] torch.compile() enabled (TorchInductor backend)")
        except Exception as e:
            print(f"  [Compile] torch.compile skipped: {e}")
    else:
        print("  [Compile] torch.compile disabled")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [Model]  {args.backbone} — {total_params:.2f}M parameters")

    # ── Loss ────────────────────────────────────────────────────────
    criterion = get_loss(args.loss, args.temperature).to(device)
    print(f"  [Loss]   {args.loss.upper()}  τ={args.temperature}")

    # ── Optimizer + Scheduler + Scaler ──────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume from checkpoint ──────────────────────────────────────
    start_epoch  = 1
    best_loss    = float("inf")
    loss_history = []

    if args.resume:
        start_epoch, best_loss, loss_history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    # ── Training loop ───────────────────────────────────────────────
    log_path = os.path.join(output_dir, "logs", "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss,lr,time_s\n")

    print(f"\n{'─'*55}")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'LR':>12}  {'Time':>8}")
    print(f"{'─'*55}")

    # Fixed checkpoint milestone epochs for Mirna's linear probe
    MILESTONE_EPOCHS = {50, 100, 150, 200}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            (x1, x2), labels_batch = batch

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                _, z1 = model(x1)
                _, z2 = model(x2)

                if args.loss == "ntxent":
                    loss = criterion(z1, z2)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] NaN loss at epoch {epoch}")
                    continue
                elif args.loss == "supcon":
                    # SupCon expects (N, n_views, D)
                    import torch.nn.functional as F
                    z1n = F.normalize(z1, dim=1)
                    z2n = F.normalize(z2, dim=1)
                    features = torch.stack([z1n, z2n], dim=1)  # (N, 2, D)
                    lbl = labels_batch.to(device, non_blocking=True)
                    loss = criterion(features, labels=lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale_before:
                scheduler.step()

            running_loss += loss.item()

        avg_loss   = running_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        loss_history.append(avg_loss)

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        print(f"  {epoch:>6}  {avg_loss:>10.4f}  {current_lr:>12.6f}  {elapsed:>7.1f}s"
              f"{'  ★' if is_best else ''}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{current_lr:.8f},{elapsed:.2f}\n")

        # ── Save checkpoints ──────────────────────────────────────────
        # Save at: milestone epochs, best loss, user-defined interval
        should_save = (
            epoch in MILESTONE_EPOCHS
            or epoch % args.save_every == 0
            or epoch == args.epochs
            or is_best
        )

        if should_save:
            tag    = "BEST" if is_best else f"ep{epoch:03d}"
            ckpt_path = os.path.join(
                output_dir, "checkpoints",
                f"simclr_{args.backbone}_{aug_tag}_{tag}.pth"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "loss_history": loss_history,
                    "args": vars(args),
                    "run_id": run_id,
                },
                ckpt_path,
            )

        # ── t-SNE (final epoch only by default) ──────────────────────
        if not args.tsne_final_only or epoch == args.epochs:
            generate_tsne(
                model, device, args.data_dir, epoch,
                output_dir, args.num_workers, run_id
            )

    # ── Final loss curve ────────────────────────────────────────────
    plot_loss_curve(loss_history, best_loss, output_dir, run_id)

    # ── Save encoder-only weights (for Mirna + ONNX export) ─────────
    encoder_path = os.path.join(output_dir, "simclr_encoder_final.pth")
    torch.save(model.state_dict(), encoder_path)
    print(f"\n  [Export] Encoder weights → {encoder_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Run complete: {run_id}")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Log CSV    : {log_path}")
    print(f"  Checkpoints: {output_dir}/checkpoints/")
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# 11. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args.run_all_ablations:
        run_all_ablations(args)
    else:
        train(args)
