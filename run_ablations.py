import os
import sys
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run_all_experiments():

    total_epochs = 20
    batch_size = 128
    model_backbone = "resnet18"

    print(f"\nStarting all SimCLR experiments using {model_backbone}\n")

    for exp_num in range(1, 9):

        print("-" * 60)
        print(f"Running Experiment #{exp_num}")
        print("-" * 60)

        save_dir = f"./outputs/exp_{exp_num}"

        cmd = [
            sys.executable,
            "src/train.py",
            "--exp_id", str(exp_num),
            "--epochs", str(total_epochs),
            "--batch_size", str(batch_size),
            "--backbone", model_backbone,
            "--output_dir", save_dir
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Experiment {exp_num} finished successfully.\n")

        except subprocess.CalledProcessError as err:
            print(f"Experiment {exp_num} crashed! Error code: {err.returncode}\n")

    print("=" * 60)
    print("All experiments are done.")
    print("=" * 60)

    create_comparison_plots()


def create_comparison_plots():

    experiments = {
        1: "Crop Only",
        2: "Crop + Flip",
        3: "Crop + Color",
        4: "Crop + Grayscale",
        5: "Crop + Flip + Color",
        6: "Crop + Flip + Grayscale",
        7: "Crop + Color + Grayscale",
        8: "Full SimCLR"
    }

    # t-SNE Results

    print("\nCreating t-SNE comparison figure...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    fig.suptitle(
        "t-SNE Visualization Comparison (Epoch 20)",
        fontsize=16,
        fontweight="bold"
    )

    for index, (exp_num, exp_name) in enumerate(experiments.items()):

        row = index // 4
        col = index % 4

        ax = axes[row][col]

        image_path = f"./outputs/exp_{exp_num}/plots/tsne_epoch_020.png"

        if os.path.exists(image_path):

            image = mpimg.imread(image_path)

            ax.imshow(image)

            ax.set_title(
                f"Exp {exp_num} - {exp_name}",
                fontsize=10,
                fontweight="bold"
            )

        else:
            ax.text(
                0.5,
                0.5,
                f"Experiment {exp_num}\nImage Missing",
                ha="center",
                va="center",
                fontsize=11
            )

        ax.axis("off")

    plt.tight_layout()

    tsne_output = "./outputs/tsne_comparison_all_experiments.png"

    fig.savefig(tsne_output, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved t-SNE comparison to:\n{tsne_output}")

    # Loss Curves

    print("\nCreating loss curve comparison figure...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    fig.suptitle(
        "Training Loss Comparison",
        fontsize=16,
        fontweight="bold"
    )

    for index, (exp_num, exp_name) in enumerate(experiments.items()):

        row = index // 4
        col = index % 4

        ax = axes[row][col]

        image_path = f"./outputs/exp_{exp_num}/plots/training_loss_curve.png"

        if os.path.exists(image_path):

            image = mpimg.imread(image_path)

            ax.imshow(image)

            ax.set_title(
                f"Exp {exp_num} - {exp_name}",
                fontsize=10,
                fontweight="bold"
            )

        else:
            ax.text(
                0.5,
                0.5,
                f"Experiment {exp_num}\nImage Missing",
                ha="center",
                va="center",
                fontsize=11
            )

        ax.axis("off")

    plt.tight_layout()

    loss_output = "./outputs/loss_comparison_all_experiments.png"

    fig.savefig(loss_output, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved loss comparison to:\n{loss_output}")

    print("\nFinished generating all comparison plots.")


if __name__ == "__main__":
    run_all_experiments()