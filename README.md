# SimCLR-Vision-SSL 🔍

> Self-Supervised Contrastive Learning for Visual Representations  
> A research-level implementation of SimCLR with ablations and linear evaluation on CIFAR-10.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Midterm%20Completed-success)

---

##  Overview

This repository implements **SimCLR** (Chen et al., ICML 2020) — a simple framework for 
contrastive self-supervised learning of visual representations — tailored for CIFAR-10.

**Key Implementations:**
- ✅ **Data Augmentation:** Full stochastic pipeline (Crop, Flip, Color Jitter, Grayscale) with an 8-experiment ablation study.
- ✅ **Architecture:** ResNet-18 (for fast ablations) and ResNet-50 (for full pretraining), modified for 32x32 inputs (3x3 conv, no max-pooling).
- ✅ **Contrastive Framework:** Two-layer MLP projection head and NT-Xent loss with temperature scaling.
- ✅ **Evaluation:** Supervised baseline, linear probe protocol, and t-SNE clustering visualization.
- ⏳ **Upcoming (Finals):** 200-epoch full pretraining, hyperparameter sweeps (learning rate, temperature), and potential BYOL/Transfer learning comparisons.

> **Course:** CISC 867 Deep Learning S26 — Group 20  
> **Team:** Natalie Nashed · Mahmoud Alyosify · Mirna Imbabi  
> **Hardware:** NVIDIA RTX 5000 Ada Generation

---

## Results (Midterm Benchmarks)

| Method | Backbone | Dataset | Top-1 Accuracy |
|---|---|---|---|
| Supervised (End-to-End, 90 ep) | ResNet-50 | CIFAR-10 | **93.77%** |
| Linear Probe (Supervised Enc) | ResNet-50 | CIFAR-10 | **93.89%** |
| SimCLR Ablations (8 Exps, 20 ep)| ResNet-18 | CIFAR-10 | *Evaluated via t-SNE* |
| SimCLR Full Pretraining (200 ep)| ResNet-50 | CIFAR-10 | *Pending* |

*Note: In contrastive learning, a higher NT-Xent loss in the full augmentation pipeline (Exp 8) indicates a more challenging and effective pretext task, leading to superior t-SNE cluster separation compared to spatial-only augmentations.*

---

## Repository Structure
```text
SimCLR-Vision-SSL/
├── src/                           # Core modules
│   ├── augmentations.py           # Augmentation pipelines & registries
│   ├── dataset.py                 # CIFAR-10 loading & custom AugmentedDataset wrapper
│   ├── loss.py                    # NT-Xent loss function
│   ├── model.py                   # ResNet backbones & Projection Heads
│   └── train.py                   # Main contrastive training loop
├── run_ablations.py               # Automated script for the 8-experiment ablation study
├── mirna_baseline_and_probe.ipynb # Supervised baseline & Linear Probe evaluation
├── outputs/                       # Auto-generated experiment results
│   ├── exp_1/ ... exp_8/          # Model checkpoints and plots (t-SNE, Loss curves)
├── LOG.md                         # Weekly development log
├── requirements.txt
└── README.md
