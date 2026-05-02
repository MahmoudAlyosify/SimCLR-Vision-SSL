# SimCLR-Vision-SSL 🔍

> Self-Supervised Contrastive Learning for Visual Representations  
> A research-level implementation of SimCLR with ablations, BYOL comparison, and downstream evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Overview

This repository implements **SimCLR** (Chen et al., ICML 2020) — a simple framework for 
contrastive self-supervised learning of visual representations — along with:

- ✅ Full data augmentation pipeline with ablation studies
- ✅ NT-Xent loss with temperature scaling
- ✅ Linear evaluation protocol & KNN evaluation
- ✅ BYOL comparison baseline
- ✅ Transfer learning evaluation on STL-10
- ✅ t-SNE embedding visualization
- 🔄 Interactive similarity search demo *(coming soon)*

> **Course:** Machine Learning Capstone — Group 20  
> **Team:** Mahmoud Alyosify · Mirna Embaby · Natalie Nashed  
> **GPU:** NVIDIA RTX 5000 Ada Generation

---

## 📊 Results (Preliminary)

| Method | Backbone | Dataset | Linear Probe Top-1 |
|---|---|---|---|
| Supervised (baseline) | ResNet-18 | CIFAR-10 | ~94% |
| SimCLR (200 epochs) | ResNet-18 | CIFAR-10 | TBD |
| SimCLR (500 epochs) | ResNet-18 | CIFAR-10 | TBD |
| BYOL | ResNet-18 | CIFAR-10 | TBD |
| SimCLR → Transfer | ResNet-18 | STL-10 | TBD |

*Results will be updated as experiments complete.*

---

## 🗂️ Repository Structure
SimCLR-Vision-SSL/
├── configs/                  # YAML experiment configs
│   ├── simclr_cifar10.yaml
│   ├── byol_cifar10.yaml
│   └── supervised_baseline.yaml
├── src/
│   ├── models/
│   │   ├── encoder.py        # ResNet-18/50 backbone
│   │   └── projection_head.py
│   ├── losses/
│   │   └── nt_xent.py        # NT-Xent contrastive loss
│   ├── augmentations/
│   │   └── simclr_augs.py    # Full augmentation pipeline
│   └── eval/
│       ├── linear_probe.py
│       └── knn_eval.py
├── experiments/              # Experiment logs & results
├── notebooks/
│   ├── augmentation_viz.ipynb
│   └── tsne_visualization.ipynb
├── train_simclr.py
├── train_byol.py
├── train_supervised.py
├── evaluate.py
├── LOG.md                    # Weekly development log
├── requirements.txt
└── README.md

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/SimCLR-Vision-SSL.git
cd SimCLR-Vision-SSL
pip install -r requirements.txt
```

**Requirements:**
torch>=2.1.0
torchvision>=0.16.0
wandb
numpy
matplotlib
scikit-learn
faiss-cpu
tqdm
pyyaml

---

## 🚀 Quick Start

**1. Train SimCLR on CIFAR-10:**
```bash
python train_simclr.py --config configs/simclr_cifar10.yaml
```

**2. Linear evaluation:**
```bash
python evaluate.py --mode linear_probe --checkpoint checkpoints/simclr_epoch200.pt
```

**3. KNN evaluation:**
```bash
python evaluate.py --mode knn --k 20 --checkpoint checkpoints/simclr_epoch200.pt
```

**4. Run supervised baseline:**
```bash
python train_supervised.py --config configs/supervised_baseline.yaml
```

---

## 🔬 Augmentation Pipeline

The augmentation pipeline is the most critical component of SimCLR.  
Each image is independently transformed twice to produce a positive pair:

| Transform | Parameters | Critical? |
|---|---|---|
| RandomResizedCrop | scale=(0.2, 1.0) | ★ Yes |
| RandomHorizontalFlip | p=0.5 | ★ Yes |
| ColorJitter | strength=0.8, p=0.8 | ★ Yes |
| RandomGrayscale | p=0.2 | ★ Yes |
| GaussianBlur | σ ∈ [0.1, 2.0], p=0.5 | Important |
| Normalize | CIFAR-10 mean/std | ★ Yes |

---

## 📈 Experiments & Ablations

| Experiment | Variable | Status |
|---|---|---|
| EXP-01 | Supervised baseline | 🔄 |
| EXP-02 | SimCLR linear probe (100% labels) | 🔄 |
| EXP-03/04 | Semi-supervised (1% / 10% labels) | 🔄 |
| EXP-05 | Ablation: No color jitter | 🔄 |
| EXP-06 | Ablation: No random crop | 🔄 |
| EXP-07 | Ablation: Linear vs non-linear head | 🔄 |
| EXP-08 | Ablation: Batch size sweep | 🔄 |
| EXP-09 | Ablation: Temperature τ sweep | 🔄 |
| EXP-10 | BYOL vs SimCLR comparison | 🔄 |
| EXP-11 | Transfer learning → STL-10 | 🔄 |
| EXP-12 | t-SNE embedding visualization | 🔄 |

---

## 📖 References

```bibtex
@inproceedings{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020}
}
```

---

## 👥 Team

| Member | Role |
|---|---|
| **Mahmoud Alyosify** | Modeling & Training Lead (SimCLR/BYOL framework) |
| **Mirna Embaby** | Evaluation & Reporting Lead (linear probe, figures, report) |
| **Natalie Nashed** | Data & Augmentation Lead (pipeline, ablations) |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
