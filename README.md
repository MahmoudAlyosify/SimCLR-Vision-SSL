# SimCLR-Vision-SSL

> **Self-Supervised & Semi-Supervised Contrastive Learning for Visual Representations**  
> A research-grade implementation of SimCLR and SupCon with 42-experiment augmentation ablations, shortcut-learning analysis, linear evaluation protocol, and a live real-time visual similarity search engine on CIFAR-10.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Final%20Completed-success)]()
[![Course](https://img.shields.io/badge/CISC_867-Group_20-blueviolet)]()
[![HuggingFace](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine)
[![W&B](https://img.shields.io/badge/Experiment%20Tracking-W%26B-orange)](https://wandb.ai/mahmoudalyosify/SimCLR-Vision-SSL/workspace)

---

## Live Demo

**Try the visual search engine live — no installation required:**

**[https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine](https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine)**

Upload any image or pick a random CIFAR-10 test image and retrieve the 5 most semantically similar images in < 5 ms using our ONNX-exported ResNet-50 encoder + FAISS index.

---

## Overview

This repository implements **SimCLR** ([Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709)) and **Supervised Contrastive Learning (SupCon)** ([Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362)) — powerful frameworks for self-supervised and semi-supervised visual representation learning — tailored and optimized for CIFAR-10.

**Key Implementations:**

- **42-Experiment Augmentation Ablation Suite:** Systematic study spanning spatial, photometric, and structural transformations, identifying `Exp 41` (Crop + Flip + GaussianBlur + ColorJitter) as the optimal configuration.
- **Color-Jitter Shortcut Mitigation Study:** 5 paired 200-epoch ResNet-50 experiments proving that NT-Xent loss alone is an unreliable proxy for representation quality — Color Jitter yields a mean **+18.10 pp** accuracy gain with zero architectural changes.
- **Architecture:** ResNet-50 modified with a custom small-image stem (`Conv2d 3x3, stride 1`, no max-pooling) following Chen et al. Appendix B.9, preserving spatial features for 32x32 inputs.
- **Contrastive Frameworks:**
  - *Unsupervised (SimCLR):* Two-layer MLP projection head + NT-Xent loss (tau = 0.5).
  - *Semi-supervised (SupCon):* Supervised Contrastive Loss (tau = 0.1) on a 10% stratified subset (5,000 labeled samples). Six iterative debugging runs required to stabilize training — see [SupCon Diagnostics](#supcon-training-diagnostics--6-debugging-runs) below.
- **Real-Time Visual Search Engine:** PyTorch ResNet-50 weights exported to **ONNX** (max deviation `3.81e-6` from PyTorch reference) paired with **FAISS IndexFlatIP** for exact sub-5ms similarity retrieval.
- **Interactive Web GUI:** Streamlit application with real-time visual similarity search + Ablation Dashboard.

---

## Results (Final Benchmarks)

| Method | Backbone | Labels | Epochs | Top-1 Acc | Notes |
|--------|----------|--------|--------|-----------|-------|
| Supervised End-to-End | ResNet-50 | 100% | 90 | **93.77%** | Supervised ceiling |
| Linear Probe — Supervised Encoder | ResNet-50 | 100% | 50 (probe) | **93.89%** | Sanity check |
| **SimCLR — Exp 41 (Best)** | **ResNet-50** | **0%** | **200** | **84.30%** | Crop+Flip+Blur+Jitter |
| **SupCon — Semi-supervised** | **ResNet-50** | **10%** | **100** | **75.20%** | 5,000 stratified labels |
| SimCLR — Midterm PoC (Exp 8) | ResNet-18 | 0% | 20 | **72.14%** | Proof-of-concept |
| CLIP Zero-Shot (Upper Bound) | ViT-B/32 | 0% | — | **88.80%** | 400M image-text pairs |
| SimCLR — Chen et al. (paper) | ResNet-50 | 0% | 1000 | 94.00% | Academic benchmark |
| Supervised Ceiling — Chen et al. | ResNet-50 | — | — | 95.10% | Academic ceiling |

**Key insight:** Our SimCLR encoder, trained on 50,000 unlabeled 32x32 CIFAR-10 images, achieves **95% of CLIP ViT-B/32 zero-shot accuracy** while using **8,000x less training data**.

---

## Augmentation Ablation Study (Color Jitter Shortcut Mitigation)

To isolate the exact impact of Color Jitter, we trained five structural augmentation pipelines **twice** at full scale (ResNet-50, 200 epochs, BS=512): once without and once with `RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)` prepended.

| Exp | Pipeline | Base Acc | +Jitter Acc | Gain (pp) | Rel. |
|-----|----------|----------|-------------|-----------|------|
| 36 to 38 | Pure Discrete Rotation | 34.40% | 51.21% | **+16.81** | +48.9% |
| 35 to 39 | Weak Spatial Baseline | 59.22% | 80.53% | **+21.31** | +36.0% |
| 9 to 40 | Crop + GaussianBlur | 63.01% | 80.65% | **+17.64** | +28.0% |
| **13 to 41** | **Crop + Flip + Blur (Best)** | **64.49%** | **84.30%** | **+19.81** | **+30.7%** |
| 10 to 42 | Crop + Random Cutout | 66.27% | 81.21% | **+14.94** | +22.5% |
| — | **Mean** | — | — | **+18.10** | **+33.2%** |

**Critical finding:** All five base pipelines converge to nearly identical NT-Xent losses (~4.95–4.96) yet span a **31.87 pp accuracy range** (34.40% to 66.27%). This proves that **NT-Xent loss alone is an unreliable proxy for representation quality** — the encoder solves the contrastive task via color-histogram shortcuts without learning semantic structure.

---

## SupCon Training Diagnostics — 6 Debugging Runs

Implementing SupCon from scratch required 6 iterative diagnostic runs to identify and resolve all failure modes. Each failure independently prevented any learning from occurring.

| Run | Key Change | L1 | L100 | Status |
|-----|------------|----|------|--------|
| 1 | AdamW lr=0.5, FP16 | 6.93 | NaN | FAIL — FP16 Overflow |
| 2 | AdamW lr=0.5, FP32 | 2.31 | 2.31 | FAIL — Stuck at log(10) |
| 3 | SGD lr=0.05, wrong loss | 2.31 | 2.31 | FAIL — Plateau |
| 4 | Correct loss, final BN | 7.44 | 6.93 | FAIL — Collapse |
| 5 | Official loss, tau=0.07 | 7.44 | 6.93 | FAIL — Collapse |
| **6** | **Official loss, tau=0.1, per-step warmup** | **6.93** | **4.98** | **SUCCESS** |

**Root causes of each failure:**

- **Run 1 (NaN):** `exp(1.0/0.1) = e^10 ~= 22,026`. FP16 overflows at `e^11.09 ~= 65,504`. Solution: disable AMP entirely (FP32).
- **Run 2–3 (stuck at 2.31 = log(10)):** AdamW lr=0.5 is ~167x too high; gradient updates destroy all learned structure from step 1, leaving the model at the uniform-similarity degenerate fixed point.
- **Run 4–5 (collapse to 6.93 = log(1023)):** Final BatchNorm on the projection output + subsequent L2-normalization = **double normalization** that over-constrains the representation manifold. All embeddings collapse to a single point.
- **Run 6:** Remove final BN, set tau=0.1, use per-step SGD warmup from lr=0.01 — solves all failure modes simultaneously.

---

## Architecture

**ResNet-50 with CIFAR-10 stem** (following Chen et al. Appendix B.9):

```text
Input (32x32x3)
    |
    |-- [Augmentation t  ~ T]  -->  x_i --|
    |-- [Augmentation t' ~ T]  -->  x_j --|
                                          |
                              Encoder f(.) -- ResNet-50
                              |--------------------------|
                              | Conv2d(3->64, 3x3, s=1)  |  <- 7x7 stride-2 replaced
                              | BatchNorm -> ReLU        |
                              | MaxPool -> Identity      |  <- removed per B.9
                              | Layer1 -> 2 -> 3 -> 4    |
                              | AvgPool                  |
                              |--------------------------|
                                          |  h in R^2048
                                          |
                              Projection Head g(.) -- MLP
                              |--------------------------|
                              | Linear(2048->2048)       |
                              | BatchNorm -> ReLU        |  <- BN in hidden layer ONLY
                              | Linear(2048->128)        |  <- No final BN (causes collapse)
                              |--------------------------|
                                          |  z in R^128  ->  L2-norm
                                          |
                              NT-Xent Loss (tau=0.5, 1023 in-batch negatives)

    -- After pretraining: discard g(.), freeze f(.) --
                                          |  h in R^2048
                                          |
                              nn.Linear(2048->10)  ->  84.30% Top-1
```



https://github.com/user-attachments/assets/54d31900-914f-4111-af39-356b416ed9ec



---

## Quick Start: Running the Interactive Web GUI

The visual search engine is fully compiled with pre-extracted database embeddings. No retraining required.

### 1. Install Dependencies

```bash
pip install streamlit onnxruntime faiss-cpu numpy pillow torch torchvision matplotlib scikit-learn
```

### 2. Start the Streamlit Application

```bash
streamlit run app.py
```

---

## Training and Re-running Experiments

### Standard SimCLR Pretraining (Best Config — Exp 41)

```bash
python src/train_master.py \
  --epochs 200 \
  --batch_size 512 \
  --backbone resnet50 \
  --exp_id 41
```

### Supervised Contrastive Learning (SupCon) — 10% Labels

```bash
python train_supcon.py \
  --epochs 100 \
  --batch_size 512 \
  --fraction 0.1 \
  --learning_rate 0.05
```

---

## Repository Structure

```text
SimCLR-Vision-SSL/
|-- app.py                          # Streamlit visual search engine & ablation GUI
|-- build_faiss.py                  # Extracts 2048-d features and builds FAISS index
|-- export_onnx.py                  # Exports Exp 41 ResNet-50 weights to ONNX
|-- train_supcon.py                 # SupCon Stage 1 pretraining orchestrator
|-- loss_supcon.py                  # Supervised Contrastive Loss implementation
|-- dataset_subset.py               # Stratified data sampler (10% / 100% labels)
|-- run_ablations.py                # Run all ablation configs sequentially
|
|-- src/                            # Core source modules
|   |-- augmentations.py            # 42 augmentation pipelines (Exps 1-42)
|   |-- dataset.py                  # DataLoader builders for training & evaluation
|   |-- loss.py                     # NT-Xent loss implementation
|   |-- model.py                    # ResNet-50/18 encoder with custom CIFAR-10 stem
|   |-- train_master.py             # Master contrastive training loop (AMP, W&B)
|
|-- deployment/                     # Production assets for Streamlit GUI
|   |-- simclr_encoder_exp41.onnx   # ONNX encoder (~89.6 MB, error: 3.81e-6)
|   |-- cifar10_index.faiss         # FAISS IndexFlatIP (10,000 vectors)
|   |-- metadata.json               # Vector ID -> class name / image path
|   |-- test_images/                # 10,000 reference CIFAR-10 PNG images
|
|-- outputs/                        # Checkpoints from training runs
|   |-- supcon_resnet50_frac10_.../  # SupCon 10% checkpoint
|   |-- supcon_resnet50_frac100_.../ # SupCon 100% checkpoint
|
|-- main-Final-SimCLR Report.tex    # LaTeX source of the final academic report
|-- requirements.txt                # Full dependency list
|-- LOG.md                          # Detailed progress log with AI usage disclosure
```

---

## Training Configuration

| Parameter | Supervised | SimCLR | SupCon |
|-----------|------------|--------|--------|
| Backbone | ResNet-50 | ResNet-50 | ResNet-50 |
| Batch size | 256 | 512 | 512 |
| Optimizer | SGD + momentum | AdamW | SGD + momentum |
| Peak LR | 0.1 | 0.06 | 0.05 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Warmup epochs | 5 | 10 | 10 |
| LR schedule | Cosine | Cosine | Cosine |
| Temperature tau | — | 0.5 | 0.1 |
| Epochs | 90 | 200 | 100 |
| Mixed precision | FP16 | FP16 | **FP32** |
| Random seed | 42 | 42 | 42 |
| GPU | RTX 5000 Ada | RTX 5000 Ada | RTX 5000 Ada |

**Note — SupCon uses FP32:** with tau=0.1, the exponential term `exp(z·z/tau)` can reach `e^10 ~= 22,026`, approaching the FP16 overflow threshold of `65,504`.

---

## Experiment Tracking

All 42 experiments tracked in real-time on **Weights & Biases**:

- Per-epoch NT-Xent loss, learning rate schedule, GPU utilization, wall-clock time
- t-SNE projection histories at epoch checkpoints
- Linear probe accuracy trajectories
- Full reproducibility with fixed seed 42 across Python, NumPy, and PyTorch

**[View W&B Dashboard]([https://wandb.ai/mahmoudalyosify/SimCLR-Vision-SSL/workspace](https://api.wandb.ai/links/models-queen-s-university/6f2t0db2))**

---

## Team

| Name | Role |
|------|------|
| **Natalie Nashed** | Data Augmentation Lead — 42-experiment pipeline design, positive-pair visualization, 3-tier difficulty hierarchy analysis |
| **Mahmoud Alyosify** | Contrastive Framework Lead — ResNet-50/18, NT-Xent loss, SupCon (6 diagnostic runs), ONNX+FAISS+Streamlit deployment, W&B tracking |
| **Mirna Imbabi** | Linear Evaluation & Reporting Lead — supervised baseline (93.77%), linear probe protocol, confusion matrix, final report |

<img width="1918" height="1078" alt="Team" src="https://github.com/user-attachments/assets/835dd738-3247-43e5-8a0f-e1df8f0c8341" />

**Course:** CISC 867 Deep Learning, Queen's University, Spring 2026  
**Hardware:** NVIDIA RTX 5000 Ada Generation (34.4 GB VRAM)

---

## References and License

```bibtex
@inproceedings{chen2020simple,
  title     = {A Simple Framework for Contrastive Learning of Visual Representations},
  author    = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle = {ICML},
  year      = {2020}
}

@inproceedings{khosla2020supervised,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Priyank and Teterwak, Piotr and Wang, Chen and others},
  booktitle = {NeurIPS},
  volume    = {33},
  year      = {2020}
}
```

This project is licensed under the [MIT License](LICENSE).
