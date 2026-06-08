"""
================================================================================
  Sync SimCLR Ablation Experiments (38-42) to Weights & Biases
  ------------------------------------------------------------
  Reads training_log.csv and run_config.json from each experiment's
  output folder and logs them as individual W&B runs under one project.
  Also uploads loss curves, t-SNE plots, and the final encoder checkpoint
  as artifacts.

  Prerequisites:
      pip install wandb
      wandb login          <-- must be done first!

  Usage:
      python sync_to_wandb.py
================================================================================
"""

import os
import sys
import csv
import json
import glob
import time

# -- Dependency check --------------------------------------------------------
try:
    import wandb
except ImportError:
    print("[ERROR] wandb not found. Install with:  pip install wandb")
    sys.exit(1)


# ==========================================================================
# Configuration
# ==========================================================================
WANDB_PROJECT = "SimCLR-Augmentation-Ablation"
WANDB_ENTITY  = None   # Uses your default entity (mahmoudalyosify)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.join(BASE_DIR, "All Experiment SimCLR 18 May 2026")

# -- Experiment registry: (folder_name, exp_id, human-readable name) -----
EXPERIMENTS = [
    ("v9_rotation_jitter_exp38",       38, "Exp38: Pure Rotation + Jitter"),
    ("v10_weakbaseline_jitter_exp39",  39, "Exp39: Weak Baseline + Jitter"),
    ("v11_blur_jitter_exp40",          40, "Exp40: Crop+Blur + Jitter"),
    ("v12_flipblur_jitter_exp41",      41, "Exp41: Crop+Flip+Blur + Jitter"),
    ("v13_cutout_jitter_exp42",        42, "Exp42: Crop+Cutout + Jitter"),
]

# -- Linear probe results (from evaluate_and_plot_beast.py) --------------
PROBE_ACCURACY = {
    38: 51.21,
    39: 80.53,
    40: 80.65,
    41: 84.30,
    42: 81.21,
}

# -- Corresponding base experiment accuracy (without jitter) -------------
BASE_ACCURACY = {
    38: 34.40,   # Exp 36: Pure Rotation
    39: 59.22,   # Exp 35: Weak Baseline
    40: 63.01,   # Exp 9:  Crop + Blur
    41: 64.49,   # Exp 13: Crop + Flip + Blur
    42: 66.27,   # Exp 10: Crop + Cutout
}


def find_run_dir(exp_folder):
    """Find the single run output directory inside an experiment folder."""
    outputs_dir = os.path.join(exp_folder, "outputs")
    if not os.path.isdir(outputs_dir):
        return None
    subdirs = [d for d in os.listdir(outputs_dir) 
               if os.path.isdir(os.path.join(outputs_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(outputs_dir, subdirs[0])
    # If multiple, pick the one matching the pattern
    for d in subdirs:
        if d.startswith("resnet50_exp"):
            return os.path.join(outputs_dir, d)
    return None


def load_config(run_dir):
    """Load run_config.json."""
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def load_training_log(run_dir):
    """Load training_log.csv and return list of dicts."""
    log_path = os.path.join(run_dir, "logs", "training_log.csv")
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "loss":  float(row["loss"]),
                "lr":    float(row["lr"]),
                "time_s": float(row["time_s"]),
            })
    return rows


def sync_experiment(folder_name, exp_id, display_name):
    """Sync a single experiment to W&B."""
    exp_folder = os.path.join(EXPERIMENT_ROOT, folder_name)
    if not os.path.isdir(exp_folder):
        print(f"  [SKIP] Folder not found: {folder_name}")
        return False

    run_dir = find_run_dir(exp_folder)
    if run_dir is None:
        print(f"  [SKIP] No run output found in: {folder_name}")
        return False

    # -- Load data -------------------------------------------------------
    config = load_config(run_dir)
    training_log = load_training_log(run_dir)

    if not training_log:
        print(f"  [SKIP] No training log found in: {run_dir}")
        return False

    # -- Compute summary metrics -----------------------------------------
    final_loss = training_log[-1]["loss"]
    best_loss = min(row["loss"] for row in training_log)
    init_loss = training_log[0]["loss"]
    total_time_h = sum(row["time_s"] for row in training_log) / 3600
    probe_acc = PROBE_ACCURACY.get(exp_id, None)
    base_acc  = BASE_ACCURACY.get(exp_id, None)
    jitter_gain = (probe_acc - base_acc) if (probe_acc and base_acc) else None

    # -- Initialize W&B run ----------------------------------------------
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=display_name,
        config={
            **config,
            "experiment_id": exp_id,
            "pipeline": folder_name,
            "has_color_jitter": True,
            "probe_accuracy": probe_acc,
            "base_accuracy_no_jitter": base_acc,
            "jitter_gain_pp": jitter_gain,
        },
        tags=["ablation", "color-jitter", f"exp{exp_id}", "resnet50", "200ep"],
        notes=f"SimCLR ablation experiment {exp_id}: {display_name}. "
              f"200 epochs, BS=512, ResNet-50 on CIFAR-10.",
        reinit=True,
    )

    # -- Log epoch-level metrics -----------------------------------------
    print(f"  Logging {len(training_log)} epochs...")
    for row in training_log:
        wandb.log({
            "epoch":         row["epoch"],
            "train/loss":    row["loss"],
            "train/lr":      row["lr"],
            "train/time_s":  row["time_s"],
        }, step=row["epoch"])

    # -- Log summary metrics ---------------------------------------------
    run.summary["final_loss"]     = final_loss
    run.summary["best_loss"]      = best_loss
    run.summary["initial_loss"]   = init_loss
    run.summary["loss_reduction"] = init_loss - best_loss
    run.summary["total_time_h"]   = round(total_time_h, 2)
    if probe_acc is not None:
        run.summary["probe_accuracy"] = probe_acc
    if base_acc is not None:
        run.summary["base_accuracy_no_jitter"] = base_acc
    if jitter_gain is not None:
        run.summary["jitter_gain_pp"] = round(jitter_gain, 2)

    # -- Upload plots as images ------------------------------------------
    plots_dir = os.path.join(run_dir, "plots")
    if os.path.isdir(plots_dir):
        for img_file in glob.glob(os.path.join(plots_dir, "*.png")):
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            wandb.log({f"plots/{img_name}": wandb.Image(img_file)})
            print(f"    Uploaded: {img_name}.png")

    # -- Upload final encoder as artifact --------------------------------
    encoder_path = os.path.join(run_dir, "simclr_encoder_final.pth")
    if os.path.exists(encoder_path):
        artifact = wandb.Artifact(
            name=f"simclr-encoder-exp{exp_id}",
            type="model",
            description=f"SimCLR ResNet-50 encoder from {display_name}",
            metadata={
                "exp_id": exp_id,
                "best_loss": best_loss,
                "probe_accuracy": probe_acc,
            }
        )
        artifact.add_file(encoder_path)
        run.log_artifact(artifact)
        print(f"    Uploaded artifact: simclr-encoder-exp{exp_id}")

    run.finish()
    return True


def main():
    print("=" * 72)
    print("  SimCLR Ablation -> Weights & Biases Sync")
    print("  Project: " + WANDB_PROJECT)
    print("=" * 72)
    print(f"  Experiment root: {EXPERIMENT_ROOT}")
    print(f"  Experiments to sync: {len(EXPERIMENTS)}")
    print("-" * 72)

    success_count = 0
    t_start = time.time()

    for folder_name, exp_id, display_name in EXPERIMENTS:
        print(f"\n[Exp {exp_id}] {display_name}")
        print(f"  Folder: {folder_name}")
        ok = sync_experiment(folder_name, exp_id, display_name)
        if ok:
            success_count += 1
            print(f"  [OK] Synced successfully!")
        else:
            print(f"  [FAIL] Could not sync.")

    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"  SYNC COMPLETE: {success_count}/{len(EXPERIMENTS)} experiments")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  View at: https://wandb.ai/mahmoudalyosify/{WANDB_PROJECT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
