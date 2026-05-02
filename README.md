# SimCLR-Vision-SSL

> Self-Supervised Contrastive Learning for Visual Representations  
> A research-level implementation of SimCLR with ablations, BYOL comparison, and downstream evaluation.

[Python 3.10+] | [PyTorch 2.1+] | [License: MIT] | [Status: In Progress]

---

## Overview

This repository implements **SimCLR** (Chen et al., ICML 2020) — a simple framework for contrastive self-supervised learning of visual representations — along with:

- Full data augmentation pipeline with ablation studies
- NT-Xent loss with temperature scaling
- Linear evaluation protocol and KNN evaluation
- BYOL comparison baseline
- Transfer learning evaluation on STL-10
- t-SNE embedding visualization
- Interactive similarity search demo *(coming soon)*

> **Course:** Machine Learning Capstone — Group 20  
> **Team:** Mahmoud Alyosify · Mirna Embaby · Natalie Nashed  
> **GPU:** NVIDIA RTX 5000 Ada Generation

---

## Results (Preliminary)

| Method | Backbone | Dataset | Linear Probe Top-1 |
|---|---|---|---|
| Supervised (baseline) | ResNet-18 | CIFAR-10 | ~94% |
| SimCLR (200 epochs) | ResNet-18 | CIFAR-10 | TBD |
| SimCLR (500 epochs) | ResNet-18 | CIFAR-10 | TBD |
| BYOL | ResNet-18 | CIFAR-10 | TBD |
| SimCLR to Transfer | ResNet-18 | STL-10 | TBD |

*Results will be updated as experiments complete.*

---

## Repository Structure
```text
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
