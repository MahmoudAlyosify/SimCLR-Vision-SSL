import os
import sys
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast
from sklearn.manifold import TSNE

# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_simclr_model
from src.loss import NTXentLoss
from src.dataset import get_train_dataloader, get_eval_dataloader


# =========================================================
# Arguments
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train SimCLR on CIFAR-10"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Dataset location"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Folder used for saving outputs"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="NT-Xent temperature"
    )

    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=10
    )

    parser.add_argument(
        "--tsne_final_only",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--exp_id",
        type=int,
        default=8,
        help="Experiment number"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):

    import random

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True


# =========================================================
# Experiment Labels
# =========================================================

def get_experiment_name(exp_id):

    experiments = {
        1: ("Exp 1", "Crop Only"),
        2: ("Exp 2", "Crop + Flip"),
        3: ("Exp 3", "Crop + Color"),
        4: ("Exp 4", "Crop + Grayscale"),
        5: ("Exp 5", "Crop + Flip + Color"),
        6: ("Exp 6", "Crop + Flip + Grayscale"),
        7: ("Exp 7", "Crop + Color + Grayscale"),
        8: ("Exp 8", "Full SimCLR")
    }

    return experiments.get(exp_id, ("Unknown", "Unknown"))


# =========================================================
# Scheduler
# =========================================================

def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):

        if step < warmup_steps:
            return step / max(warmup_steps, 1)

        progress = (step - warmup_steps) / max(
            total_steps - warmup_steps,
            1
        )

        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


# =========================================================
# t-SNE Visualization
# =========================================================

@torch.no_grad()
def generate_tsne_plot(
    model,
    device,
    data_dir,
    epoch,
    output_dir,
    num_workers=0,
    exp_id=1
):

    exp_label, exp_name = get_experiment_name(exp_id)

    print(f"\nGenerating t-SNE for epoch {epoch}...")
    print(f"{exp_label} -> {exp_name}")

    model.eval()

    loader = get_eval_dataloader(
        data_dir,
        train=False,
        batch_size=512,
        num_workers=num_workers
    )

    all_features = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)

        features, _ = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

        total_samples = sum(x.shape[0] for x in all_features)

        if total_samples >= 2000:
            break

    features = np.concatenate(all_features)[:2000]
    labels = np.concatenate(all_labels)[:2000]

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )

    embedding = tsne.fit_transform(features)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=8
    )

    colorbar = plt.colorbar(
        scatter,
        ax=ax,
        ticks=range(10)
    )

    colorbar.ax.set_yticklabels(class_names)

    ax.set_title(
        f"{exp_label}: {exp_name}\nEpoch {epoch}",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    ax.set_xticks([])
    ax.set_yticks([])

    save_path = os.path.join(
        output_dir,
        "plots",
        f"tsne_epoch_{epoch:03d}.png"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved t-SNE plot to:\n{save_path}")


# =========================================================
# Training
# =========================================================

def train(args):

    set_seed(args.seed)

    os.makedirs(
        os.path.join(args.output_dir, "checkpoints"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(args.output_dir, "plots"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(args.output_dir, "logs"),
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = torch.cuda.is_available()

    # ================= Data =================

    train_loader = get_train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        exp_id=args.exp_id
    )

    steps_per_epoch = len(train_loader)

    print(f"\nTraining samples : {len(train_loader.dataset):,}")
    print(f"Steps per epoch  : {steps_per_epoch}")
    print(f"Batch size       : {args.batch_size}")

    # ================= Model =================

    model = get_simclr_model(
        args.backbone,
        args.projection_dim
    ).to(device)

    criterion = NTXentLoss(
        temperature=args.temperature
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = build_scheduler(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        steps_per_epoch
    )

    scaler = GradScaler(enabled=use_amp)

    total_params = sum(
        p.numel() for p in model.parameters()
    ) / 1e6

    print(f"Model parameters : {total_params:.2f}M\n")

    # ================= Training Loop =================

    loss_history = []

    best_loss = float("inf")

    log_file = os.path.join(
        args.output_dir,
        "logs",
        "training_log.csv"
    )

    with open(log_file, "w") as f:
        f.write("epoch,loss,lr,time\n")

    print(f"{'Epoch':>6} {'Loss':>10} {'LR':>12} {'Time':>10}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0.0

        start_time = time.time()

        for (x1, x2), _ in train_loader:

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):

                _, z1 = model(x1)
                _, z2 = model(x2)

                loss = criterion(z1, z2)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / steps_per_epoch

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed_time = time.time() - start_time

        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(
            f"{epoch:>6} "
            f"{avg_loss:>10.4f} "
            f"{current_lr:>12.6f} "
            f"{elapsed_time:>8.1f}s"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{epoch},"
                f"{avg_loss:.6f},"
                f"{current_lr:.8f},"
                f"{elapsed_time:.2f}\n"
            )

        # ================= Save Checkpoint =================

        if epoch % args.save_every == 0 or epoch == args.epochs:

            checkpoint_path = os.path.join(
                args.output_dir,
                "checkpoints",
                f"simclr_exp{args.exp_id}_{args.backbone}_epoch_{epoch:03d}.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args)
                },
                checkpoint_path
            )

            print(f"Checkpoint saved:\n{checkpoint_path}")

    # ================= Visualizations =================

    print("\n" + "=" * 60)
    print("Generating final plots...")
    print("=" * 60)

    generate_tsne_plot(
        model=model,
        device=device,
        data_dir=args.data_dir,
        epoch=args.epochs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        exp_id=args.exp_id
    )

    # ================= Loss Curve =================

    exp_label, exp_name = get_experiment_name(args.exp_id)

    print("\nGenerating loss curve...")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        linewidth=2,
        marker="o",
        markersize=4
    )

    ax.axhline(
        y=best_loss,
        linestyle="--",
        alpha=0.6,
        label=f"Best loss = {best_loss:.4f}"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_title(
        f"{exp_label}: {exp_name}\nTraining Loss",
        fontsize=13,
        fontweight="bold"
    )

    ax.legend()

    ax.grid(True, alpha=0.3)

    loss_curve_path = os.path.join(
        args.output_dir,
        "plots",
        "training_loss_curve.png"
    )

    fig.savefig(
        loss_curve_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Loss curve saved to:\n{loss_curve_path}")

    # ================= Save Encoder =================

    encoder_path = os.path.join(
        args.output_dir,
        "simclr_encoder.pth"
    )

    torch.save(
        model.state_dict(),
        encoder_path
    )

    print(f"\nEncoder weights saved to:\n{encoder_path}")

    print("\n" + "=" * 60)
    print("Training finished successfully.")
    print(f"Best loss : {best_loss:.4f}")
    print(f"Logs file : {log_file}")
    print("=" * 60 + "\n")



if __name__ == "__main__":

    args = parse_args()

    train(args)