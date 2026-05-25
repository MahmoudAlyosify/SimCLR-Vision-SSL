# SimCLR-Vision-SSL 🔍

> **Self-Supervised & Semi-Supervised Contrastive Learning for Visual Representations** > A research-grade implementation of SimCLR and SupCon with 42-experiment augmentation ablations, shortcut-learning analysis, linear evaluation protocol, and a live real-time visual similarity search engine on CIFAR-10.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Final%20Completed-success)]()
[![Course](https://img.shields.io/badge/CISC_867-Group_20-blueviolet)]()
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine)
[![W&B](https://img.shields.io/badge/Experiment%20Tracking-W%26B-orange)](https://wandb.ai/mahmoudalyosify/SimCLR-Vision-SSL/workspace)

---

##  Live Demo

**Try the visual search engine live — no installation required:**
>  **[https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine](https://huggingface.co/spaces/mahmoudalyosify/SimCLR-Visual-Search-Engine)**

Upload any image or pick a random CIFAR-10 test image and retrieve the 5 most semantically similar images in **< 5 ms** using our ONNX-exported ResNet-50 encoder + FAISS index.

---

## 📖 Overview

This repository implements **SimCLR** ([Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709)) and **Supervised Contrastive Learning (SupCon)** ([Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362)) — powerful frameworks for self-supervised and semi-supervised visual representation learning — tailored and optimized for CIFAR-10.

**Key Implementations:**
- ✅ **Color-Jitter Ablation Study:** A thorough analysis of 5 final 200-epoch ResNet-50 experiments to solve the "Shortcut Learning" problem where the encoder exploits low-level color histograms instead of learning invariant semantic features.
- ✅ **Architecture:** ResNet-50 modified with a custom small-image stem (conv1 3×3, stride 1, padding 1, and no max-pooling) to preserve spatial features for 32×32 CIFAR-10 images.
- ✅ **Contrastive Frameworks:** - *Unsupervised (SimCLR):* Two-layer MLP projection head + NT-Xent loss with temperature scaling (τ = 0.5).
  - *Semi-supervised (SupCon - Bonus #4):* Pretraining with Supervised Contrastive Loss (τ = 0.1) on a 10% stratified subset (5,000 labeled samples) of CIFAR-10.
- ✅ **Real-Time Visual Search Engine:** PyTorch ResNet-50 weights exported to an optimized **ONNX** inference session paired with a **FAISS (IndexFlatIP)** vector database for exact sub-millisecond similarity retrieval.
- ✅ **Premium Web GUI:** Streamlit application providing interactive real-time visual similarity search (latency < 5 ms) and an Ablation Dashboard to showcase our scientific findings.

---

## 🏆 Results (Final Benchmarks)

Our final 200-epoch pretraining runs resolved the color-shortcut limitations, leading to outstanding linear probe downstream performance:

| Method | Backbone | Pretrain Epochs | Top-1 Acc | Notes |
|--------|----------|-----------------|-----------|-------|
| Supervised End-to-End Baseline | ResNet-50 | 90 | **93.77%** | Standard fully-supervised training |
| Linear Probe — Supervised Encoder | ResNet-50 | 50 (probe) | **93.89%** | Sanity check (frozen supervised features) |
| **SimCLR Linear Probe — Exp 41 (Champion)** | **ResNet-50** | **200** | **84.30%** | **With Color Jitter (+19.81% Gain over midterm!)** |
| **SupCon Linear Probe — Bonus #4** | **ResNet-50** | **100** | **75.20%** | **Stage 1 trained on just 10% stratified subset** |
| Midterm Proof-of-Concept — Exp 8 | ResNet-18 | 20 | **72.14%** | Midterm proof-of-concept run |
| Zero-Shot CLIP Foundation Model | ViT-B/32 | — | **88.80%** | Academic Upper Bound (8,000x more data) |
| SimCLR — Chen et al. (paper) | ResNet-50 | 1000 | 94.00% | Full academic benchmark |
| Supervised Ceiling — Chen et al. (paper) | ResNet-50 | — | 95.10% | Fully supervised academic ceiling |

> **💡 The Power of Color Jitter:** Without color jittering, contrastive models exploit simple pixel-level color distribution shortcuts. Adding a photometric distortion "shield" forced our ResNet-50 encoder to focus on robust spatial contours, boundaries, and shapes, pushing our Top-1 performance from **64.49%** to a stellar **84.30%** (achieving **95% of CLIP ViT-B/32 zero-shot performance** while using 8,000x less training data!).

---

## 🔬 Augmentation Ablation Study (Color Jitter Re-runs)

To isolate the exact impact of color jittering, we ran five comparative experiments for 200 epochs on our ResNet-50 architecture:

| Experiment ID | Augmentation Combination | Without Jitter Acc | With Jitter Acc | Downstream Gain |
|---------------|--------------------------|--------------------|-----------------|-----------------|
| **Exp 38 vs 36** | Pure Discrete Rotation | 34.40% | 51.21% | **+16.81 pp** |
| **Exp 39 vs 35** | Weak Spatial Baseline | 59.22% | 80.53% | **+21.31 pp** |
| **Exp 40 vs 9** | Crop + Gaussian Blur | 63.01% | 80.65% | **+17.64 pp** |
| **Exp 41 vs 13** | **Crop + Flip + Blur (Champion)** | **64.49%** | **84.30%** | **+19.81 pp** |
| **Exp 42 vs 10** | Crop + Random Cutout | 66.27% | 81.21% | **+14.94 pp** |
| | **Mean Gain** | | | **+18.10 pp** |

*Note: All base pipelines converge to nearly identical NT-Xent losses (~4.95–4.96) yet span a 31.87 pp accuracy range — proving that NT-Xent loss alone is an unreliable proxy for representation quality.*

---

## 🏗️ Architecture

**ResNet-50 with CIFAR-10 stem** (following Chen et al. Appendix B.9):
<img width="8191" height="2555" alt="architecture_diagram" src="https://github.com/user-attachments/assets/72864cf7-75d2-4738-b727-d602f5dd249f" />

```text
Input (32×32×3)
    │
    ├── [Augmentation t  ~ T]  →  x̃ᵢ ─┐
    └── [Augmentation t' ~ T]  →  x̃ⱼ ─┤
                                        ↓
                              Encoder f(·) — ResNet-50
                              ┌─────────────────────────┐
                              │ Conv2d(3→64, 3×3, s=1)  │  ← 7×7 stride-2 replaced
                              │ BatchNorm → ReLU         │
                              │ MaxPool → Identity ✗     │  ← removed per B.9
                              │ Layer1 → 2 → 3 → 4      │
                              │ AvgPool                  │
                              └─────────────────────────┘
                                        │  h ∈ ℝ²⁰⁴⁸
                                        ↓
                              Projection Head g(·) — MLP
                              ┌─────────────────────────┐
                              │ Linear(2048→2048)        │
                              │ BatchNorm → ReLU         │  ← BN hidden layer only
                              │ Linear(2048→128)         │  ← No final BN (collapse!)
                              └─────────────────────────┘
                                        │  z ∈ ℝ¹²⁸  →  ℓ₂-norm
                                        ↓
                              NT-Xent Loss (τ=0.5, 1023 negatives)

    ── After pretraining: discard g(·), freeze f(·) ──
                                        │  h ∈ ℝ²⁰⁴⁸
                                        ↓
                              nn.Linear(2048→10)  →  84.30% Top-1

```

---

## 🚀 Quick Start: Running the Interactive Web GUI

The visual search engine is fully compiled and packed with pre-extracted database embeddings! You do not need to retrain the ResNet-50 to experience the interactive app.

### 1. Install Dependencies

```bash
pip install streamlit onnxruntime faiss-cpu numpy pillow torch torchvision matplotlib scikit-learn

```

### 2. Start the Streamlit Application

```bash
streamlit run app.py

```

---

## 🧠 Training & Re-running Experiments

### Standard SimCLR Pretraining

To run standard unsupervised SimCLR pretraining with our best parameters:

```bash
python src/train_master.py --epochs 200 --batch_size 512 --backbone resnet50 --exp_id 41

```

### Supervised Contrastive Learning (SupCon) Stage 1

To run pretraining on the 10% stratified subset under Supervised Contrastive Loss (Bonus #4):

```bash
python train_supcon.py --epochs 100 --batch_size 512 --fraction 0.1 --learning_rate 0.05

```

---

## 📁 Repository Structure

```text
SimCLR-Vision-SSL/
├── app.py                         # Premium Streamlit visual search engine & ablation GUI
├── build_faiss.py                 # Extracts 2048-d features, exports PNGs, and builds FAISS database
├── export_onnx.py                 # Exports Exp 41 ResNet-50 weights to optimized ONNX format
├── train_supcon.py                # SupCon Stage 1 pretraining orchestrator (Khosla et al.)
├── loss_supcon.py                 # Supervised Contrastive Loss implementation
├── dataset_subset.py              # Stratified data sampler (10% or 100% labels)
├── run_ablations.py               # Run all ablation configs sequentially
│
├── src/                           # Core source modules
│   ├── augmentations.py           # Color jitter & spatial augmentation pipelines
│   ├── dataset.py                 # Dataloader builders for training & linear evaluation
│   ├── loss.py                    # NT-Xent loss (Normalized Temp-scaled Cross-Entropy)
│   ├── model.py                   # ResNet-50 encoder architecture with custom stem
│   └── train_master.py            # Master contrastive training orchestrator with AMP
│
├── deployment/                    # Compiled assets for GUI production deployment
│   ├── simclr_encoder_exp41.onnx  # Exported ONNX encoder model (~90 MB)
│   ├── cifar10_index.faiss        # Pre-computed FAISS IndexFlatIP (10,000 vectors)
│   ├── metadata.json              # Mapping vector ID to class name/path
│   └── test_images/               # 10,000 reference PNG test images
│
├── outputs/                       # Experimental trained checkpoints and metrics
│   ├── supcon_resnet50_frac10_.../ # Pre-trained SupCon 10% checkpoint
│   └── supcon_resnet50_frac100_.../# Pre-trained SupCon 100% checkpoint
│
├── main-Final-SimCLR Report.tex   # Source LaTeX code of the final academic report
├── requirements.txt               # Dependencies list
└── LOG.md                         # Progress log

```

---

## ⚙️ Training Configuration

| Parameter | Supervised | SimCLR | SupCon |
| --- | --- | --- | --- |
| Backbone | ResNet-50 | ResNet-50 | ResNet-50 |
| Batch size | 256 | 512 | 512 |
| Optimizer | SGD + momentum | AdamW | SGD + momentum |
| Peak LR | 0.1 | 0.06 | 0.05 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Warmup epochs | 5 | 10 | 10 |
| LR schedule | Cosine | Cosine | Cosine |
| Temperature τ | — | 0.5 | 0.1 |
| Epochs | 90 | 200 | 100 |
| Mixed precision | FP16 | FP16 | FP32 |

> ⚠️ **SupCon uses FP32** — with τ=0.1, the exponential term exp(z·z/τ) can reach e¹⁰ ≈ 22,026, causing FP16 overflow.

---

## 🧪 Experiment Tracking

All 42 experiments tracked on **Weights & Biases**:

> 📊 **[https://wandb.ai/mahmoudalyosify/SimCLR-Vision-SSL/workspace](https://wandb.ai/mahmoudalyosify/SimCLR-Vision-SSL/workspace)**

Per-epoch logging: NT-Xent loss, learning rate, wall-clock time, GPU utilization, checkpoint paths.

---

## 👥 Team

| Name | Role |
| --- | --- |
| **Natalie Nashed** | Data Augmentation Lead — 8-config pipeline, positive-pair visualization |
| **Mahmoud Sayed Youssef** | Contrastive Framework Lead — ResNet-50/18, NT-Xent, SupCon, ONNX+FAISS deployment |
| **Mirna Imbabi** | Linear Evaluation & Reporting Lead — supervised baseline, linear probe, final report |

> **Course:** CISC 867 Deep Learning, Queen's University, Spring 2026
> **Hardware:** NVIDIA RTX 5000 Ada Generation (34.4 GB VRAM)

---

## 📚 Citation & License

```bibtex
@inproceedings{chen2020simple,
  title     = {A Simple Framework for Contrastive Learning of Visual Representations},
  author    = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {1597--1607},
  year      = {2020},
  organization = {PMLR}
}

@inproceedings{khosla2020supervised,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Priyank and Teterwak, Piotr and Wang, Chen and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {18661--18773},
  year      = {2020}
}

```

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).



```
