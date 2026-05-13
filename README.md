# SimCLR-Vision-SSL 🔍

> Self-Supervised Contrastive Learning for Visual Representations  
> A research-level implementation of SimCLR with ablations and linear evaluation on CIFAR-10.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Midterm%20Completed-success)
![Course](https://img.shields.io/badge/CISC_867-Group_20-blueviolet)

---

## Overview

This repository implements **SimCLR** ([Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709)) —
a simple framework for contrastive self-supervised learning of visual representations —
tailored for CIFAR-10.

**Key Implementations:**
- ✅ **Data Augmentation:** Full stochastic pipeline (Crop, Flip, ColorJitter, Grayscale) with an 8-experiment ablation study
- ✅ **Architecture:** ResNet-18 (fast ablations) and ResNet-50 (full pretraining), both modified for 32×32 inputs (3×3 conv, no max-pooling)
- ✅ **Contrastive Framework:** Two-layer MLP projection head + NT-Xent loss with temperature scaling (τ = 0.5)
- ✅ **Evaluation:** Supervised baseline (93.77%), linear probe sanity check (93.89%), t-SNE clustering, and SimCLR proof-of-concept probe (**72.14%**)
- ⏳ **Upcoming (Finals):** 200-epoch ResNet-50 pretraining, hyperparameter sweeps (LR, temperature, batch size)

> **Course:** CISC 867 Deep Learning, Spring 2026 — Group 20  
> **Team:** Natalie Nashed · Mahmoud Alyosify · Mirna Imbabi  
> **Hardware:** NVIDIA RTX 5000 Ada Generation (32 GB VRAM)  
> **Progress Log:** [`LOG.md`](LOG.md)

---

## Results (Midterm Benchmarks)

| Method | Backbone | Epochs | Top-1 Acc |
|--------|----------|--------|-----------|
| Supervised End-to-End | ResNet-50 | 90 | **93.77%** |
| Linear Probe — Supervised Encoder *(sanity check)* | ResNet-50 | 50 (probe) | **93.89%** |
| SimCLR Linear Probe — Exp 8 *(proof-of-concept)* | ResNet-18 | 20 (pretrain) | **72.14%** |
| SimCLR Full Pretraining *(pending)* | ResNet-50 | 200 | ⏳ Pending |
| SimCLR — Chen et al. (paper) | ResNet-50 | 1000 | 94.00% |
| Supervised Ceiling — Chen et al. (paper) | ResNet-50 | — | 95.10% |

> **Note on NT-Xent Loss:** A higher NT-Xent loss in the full augmentation
> pipeline (Exp 8, `L₂₀ = 3.968`) reflects a *harder* contrastive task,
> not inferior optimization. This is confirmed by the superior t-SNE
> clustering and the 72.14% linear probe accuracy achieved by Exp 8.

> **Note on Architecture:** The 72.14% proof-of-concept uses ResNet-18
> (512-dim, 20 epochs). The final evaluation will use ResNet-50
> (2048-dim, 200 epochs), expected to approach the paper's 94.00% target.

---

## Augmentation Ablation Study

Eight configurations evaluated over 20 epochs (ResNet-18, NVIDIA RTX 5000 Ada):

| Exp | Augmentations | Batch | L₁ | L₂₀ | ΔL | t/ep (s) |
|-----|--------------|-------|----|-----|-----|----------|
| 1 | Crop only | 128 | 4.283 | 3.728 | 0.556 | 29.1 |
| 2 | Crop + Flip | 128 | 4.296 | 3.730 | 0.566 | 39.9 |
| 3 | Crop + ColorJitter | 128 | 4.658 | 3.867 | 0.791 | 67.8 |
| 4 | Crop + Grayscale | 128 | 4.498 | 3.865 | 0.633 | 37.9 |
| 5 | Crop + Flip + ColorJitter | 128 | 4.663 | 3.872 | 0.791 | 68.4 |
| 6 | Crop + Flip + Grayscale | 128 | 4.508 | 3.875 | 0.633 | 40.3 |
| 7 | Crop + ColorJitter + Grayscale | 128 | 4.874 | 3.949 | 0.925 | 57.5 |
| **8** | **Full Pipeline (all 4)** | **1024** | **4.884** | **3.968** | **0.916** | **721.8** |

Three difficulty tiers emerge:
- 🟢 **Tier 1** — Spatial only (Exp 1–2): `L₂₀ ≈ 3.73`
- 🟡 **Tier 2** — One chromatic perturbation (Exp 3–6): `L₂₀ ≈ 3.87`
- 🔴 **Tier 3** — Joint chromatic perturbation (Exp 7–8): `L₂₀ ≈ 3.95–3.97`

---

## Repository Structure

```text
SimCLR-Vision-SSL/
├── configs/                       # YAML configs for augmentation hyperparameters
├── notebooks/                     # Jupyter notebooks
│   └── mirna_baseline_and_probe.ipynb  # Supervised baseline, linear probe & SimCLR PoC
├── outputs/                       # Auto-generated experiment results
│   ├── exp_1/ … exp_8/            # Checkpoints, loss curves, t-SNE plots per experiment
│   └── plots/
│       ├── loss_comparison_all_experiments.png
│       └── tsne_comparison_all_experiments.png
├── src/                           # Core source modules
│   ├── augmentations.py           # Augmentation pipelines & 8-config registry
│   ├── dataset.py                 # CIFAR-10 loader & AugmentedDataset wrapper
│   ├── loss.py                    # NT-Xent loss (temperature-scaled)
│   ├── model.py                   # ResNet-18/50 encoders & MLP projection head
│   └── train.py                   # Contrastive training loop w/ CSV logging
├── .gitignore
├── LICENSE                        # MIT License
├── LOG.md                         # Weekly development & progress log
├── README.md
├── requirements.txt
└── run_ablations.py               # Automated 8-experiment ablation runner
