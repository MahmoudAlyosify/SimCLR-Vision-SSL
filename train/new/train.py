"""
train.py — SimCLR Pre-Training (MIDTERM VERSION)
=================================================
نسخة الميدتيرم: تشغيل على CPU/Colab لإثبات إن النموذج بيتعلم.

الفرق عن نسخة الفاينال:
  ✗ لا Mixed Precision (AMP)   — بتحتاج GPU بـ Tensor Cores
  ✗ لا Gaussian Blur           — محجوزة للفاينال
  ✗ لا t-SNE                  — بياخد وقت طويل على CPU
  ✓ 20 epochs فقط              — كافية لإثبات إن الـ Loss بتنخفض
  ✓ batch_size صغير (256)      — مناسب للـ RAM العادية
  ✓ Loss curve محفوظة          — الدليل البصري في الريبورت

Usage:
    python train.py                        # defaults: 20 epochs, batch=256
    python train.py --epochs 5             # تجربة سريعة جداً
    python train.py --data_dir ./data      # لو عندك الداتا في مكان تاني
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")              # بدون GUI — مهم لـ Colab و server
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from src.model   import SimCLR_ResNet50
from src.loss    import NTXentLoss
from src.dataset import get_train_dataloader


# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="SimCLR Midterm Pre-training — CIFAR-10")
    p.add_argument("--data_dir",       type=str,   default="./data",
                   help="مكان تحميل/قراءة CIFAR-10")
    p.add_argument("--output_dir",     type=str,   default="./outputs",
                   help="مكان حفظ الـ checkpoints والـ plots")
    p.add_argument("--epochs",         type=int,   default=20,
                   help="عدد الـ epochs (20 كافية للميدتيرم)")
    p.add_argument("--batch_size",     type=int,   default=256,
                   help="حجم الـ batch — قلل لو بيطلع memory error")
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--temperature",    type=float, default=0.5,
                   help="τ — الـ temperature للـ NT-Xent loss")
    p.add_argument("--projection_dim", type=int,   default=128)
    p.add_argument("--warmup_epochs",  type=int,   default=5,
                   help="epochs للـ LR warmup (من 0 لـ lr المحدد)")
    p.add_argument("--save_every",     type=int,   default=10,
                   help="احفظ checkpoint كل كام epoch")
    p.add_argument("--num_workers",    type=int,   default=0,
                   help="اتركه 0 على Windows و Colab")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  LR Scheduler: Linear Warmup + Cosine Decay
# ══════════════════════════════════════════════════════════════════
def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    الـ LR بيبدأ من 0 ويوصل للـ lr المحدد في خلال warmup_epochs،
    وبعدين بيتناقص cosine حتى ~0 في آخر epoch.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════
#  Loss Curve Plot
# ══════════════════════════════════════════════════════════════════
def save_loss_curve(loss_history: list, output_dir: str):
    """
    بيحفظ رسمة الـ loss curve — ضروري للريبورت.
    لازم يظهر فيها إن الـ loss بتنخفض مع الـ epochs.
    """
    epochs = range(1, len(loss_history) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, loss_history, linewidth=2.0, color="#2563EB", marker="o",
            markersize=4, label="NT-Xent Loss")

    # خط الـ moving average علشان يبيّن الاتجاه بوضوح
    if len(loss_history) >= 5:
        window = 3
        ma = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(loss_history) + 1), ma,
            linewidth=2.0, color="#DC2626", linestyle="--",
            alpha=0.8, label=f"Moving avg (w={window})"
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title(
        "SimCLR Pre-Training Loss — CIFAR-10 (Midterm)\n"
        "ResNet50 (modified stem) + 2-Layer MLP Projection Head",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    path = os.path.join(output_dir, "plots", "loss_curve_midterm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════
def train(args):
    # ── Directories ───────────────────────────────────────────────
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"),       exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header = "=" * 60
    print(f"\n{header}")
    print("  SimCLR Pre-Training on CIFAR-10  [MIDTERM VERSION]")
    print(header)
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs         : {args.epochs}  (midterm PoC)")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Learning rate  : {args.lr}  (AdamW + cosine schedule)")
    print(f"  Temperature τ  : {args.temperature}")
    print(f"  Projection dim : {args.projection_dim}")
    print(f"  Warmup epochs  : {args.warmup_epochs}")
    print(f"  Mixed Prec.    : DISABLED (midterm version)")
    print(f"  Gaussian Blur  : DISABLED (midterm version)")
    print(f"{header}\n")

    # ── Data ──────────────────────────────────────────────────────
    print("  Loading CIFAR-10...")
    train_loader = get_train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    steps_per_epoch = len(train_loader)
    n_train = len(train_loader.dataset)
    print(f"  Train samples  : {n_train:,}")
    print(f"  Steps/epoch    : {steps_per_epoch}")
    print(f"  Effective neg. : {2 * (args.batch_size - 1)} per positive pair\n")

    # ── Model ─────────────────────────────────────────────────────
    model = SimCLR_ResNet50(projection_dim=args.projection_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params   : {total_params:.1f}M\n")

    # ── Loss & Optimizer ──────────────────────────────────────────
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch
    )

    # ── Training ──────────────────────────────────────────────────
    print(f"  {'Epoch':>6} | {'Loss':>8} | {'LR':>10} | {'Time':>8} | {'Note'}")
    print("  " + "-" * 55)

    loss_history = []
    best_loss    = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (x_i, x_j), _ in train_loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            optimizer.zero_grad()

            # ── Forward ───────────────────────────────────────────
            _, z_i = model(x_i)   # نحتاج z فقط للـ loss (مش h)
            _, z_j = model(x_j)

            loss = criterion(z_i, z_j)

            # ── Backward ──────────────────────────────────────────
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # ── Epoch Summary ─────────────────────────────────────────
        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        note = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            note = "★ best"

        print(
            f"  {epoch:>6} | {avg_loss:>8.4f} | {lr_now:>10.6f} | "
            f"{elapsed:>6.1f}s | {note}"
        )

        # ── Checkpoint ────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
                "loss_history":         loss_history,
                "args":                 vars(args),
            }
            ckpt_path = os.path.join(
                args.output_dir, "checkpoints",
                f"simclr_midterm_epoch{epoch:02d}.pth"
            )
            torch.save(ckpt, ckpt_path)
            print(f"         [SAVED] → {ckpt_path}")

    # ── Loss Curve ────────────────────────────────────────────────
    plot_path = save_loss_curve(loss_history, args.output_dir)

    # ── Final Encoder (بدون الـ projection head) ──────────────────
    # ده اللي هنستخدمه في الـ linear probe في الفاينال
    encoder_path = os.path.join(args.output_dir, "simclr_encoder_midterm.pth")
    torch.save(model.encoder.state_dict(), encoder_path)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{header}")
    print("  TRAINING COMPLETE — Midterm Summary")
    print(header)
    print(f"  Initial loss   : {loss_history[0]:.4f}")
    print(f"  Final loss     : {loss_history[-1]:.4f}")
    loss_drop = loss_history[0] - loss_history[-1]
    pct_drop  = loss_drop / loss_history[0] * 100
    print(f"  Loss drop      : {loss_drop:.4f}  ({pct_drop:.1f}% decrease) ✓")
    print(f"  Loss curve     : {plot_path}")
    print(f"  Encoder saved  : {encoder_path}")
    print(f"  (Next step: Linear Probe evaluation in Final Report)")
    print(f"{header}\n")

    # طباعة الأرقام بشكل جاهز للريبورت
    print("  ┌─── Paste this table in your Midterm Report ────────┐")
    print("  │  Epoch │  NT-Xent Loss  │                          │")
    print("  │ ────── │ ────────────── │                          │")
    for i, L in enumerate(loss_history, 1):
        marker = " ← start" if i == 1 else (" ← final" if i == len(loss_history) else "")
        print(f"  │  {i:>5} │    {L:.4f}      │{marker:26}│")
    print("  └────────────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
