# SimCLR-Vision-SSL 🔍

> Self-Supervised Contrastive Learning & Supervised Contrastive Learning (SupCon) for Visual Representations  
> A research-level implementation of SimCLR and SupCon with extensive color-jitter ablations, linear evaluation, and a real-time visual similarity search engine on CIFAR-10.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Final%20Completed-success)
![Course](https://img.shields.io/badge/CISC_867-Group_20-blueviolet)

---

## Overview

This repository implements **SimCLR** ([Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709)) and **Supervised Contrastive Learning (SupCon)** ([Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362)) — powerful frameworks for self-supervised and semi-supervised visual representation learning — tailored and optimized for CIFAR-10.

**Key Implementations:**
- ✅ **Color-Jitter Ablation Study:** A thorough analysis of 5 final 200-epoch ResNet-50 experiments (Exp 38–42) to solve the "Shortcut Learning" problem where the encoder exploits low-level color histograms instead of learning invariant semantic features.
- ✅ **Architecture:** ResNet-50 modified with a custom small-image stem (conv1 $3\times3$, stride 1, padding 1, and no max-pooling) to preserve spatial features for $32\times32$ CIFAR-10 images.
- ✅ **Contrastive Frameworks:** 
  - *Unsupervised (SimCLR):* Two-layer MLP projection head + NT-Xent loss with temperature scaling ($\tau = 0.5$).
  - *Semi-supervised (SupCon - Bonus #4):* Pretraining with Supervised Contrastive Loss ($\tau = 0.1$) on a 10% stratified subset (5,000 labeled samples) of CIFAR-10.
- ✅ **Real-Time Visual Search Engine:** PyTorch ResNet-50 weights exported to an optimized **ONNX** inference session (with dynamic batching) paired with a **FAISS (IndexFlatIP)** vector database for exact sub-millisecond similarity retrieval.
- ✅ **Premium Web GUI:** Streamlit application providing interactive real-time visual similarity search (latency $< 5\text{ ms}$) and an Ablation Dashboard to showcase our scientific findings.

> **Course:** CISC 867 Deep Learning, Spring 2026 — Group 20  
> **Team:** Natalie Nashed · Mahmoud Alyosify · Mirna Imbabi  
> **Hardware:** NVIDIA RTX 5000 Ada Generation (32 GB VRAM)  
> **Progress Log:** [`LOG.md`](LOG.md)

---

## Results (Final Benchmarks)

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

> **The Power of Color Jitter:** Without color jittering, contrastive models exploit simple pixel-level color distribution shortcuts. Adding a photometric distortion "shield" forced our ResNet-50 encoder to focus on robust spatial contours, boundaries, and shapes, pushing our Top-1 performance from **64.49%** to a stellar **84.30%** (achieving **95% of CLIP ViT-B/32 zero-shot performance** while using 8,000x less training data!).

---

## Augmentation Ablation Study (Color Jitter Re-runs)

To isolate the exact impact of color jittering, we ran five comparative experiments for 200 epochs on our ResNet-50 architecture:

| Experiment ID | Augmentation Combination | Without Jitter Acc (Midterm) | With Jitter Acc (Finals) | Downstream Performance Gain |
|---------------|--------------------------|------------------------------|--------------------------|-----------------------------|
| **Exp 38 vs 36** | Pure Discrete Rotation | 34.40% | 51.21% | **+16.81 pp** |
| **Exp 39 vs 35** | Weak Spatial Baseline | 59.22% | 80.53% | **+21.31 pp** |
| **Exp 40 vs 9**  | Crop + Gaussian Blur | 63.01% | 80.65% | **+17.64 pp** |
| **Exp 41 vs 13** | **Crop + Flip + Blur (Champion)** | **64.49%** | **84.30%** | **+19.81 pp** |
| **Exp 42 vs 10** | Crop + Random Cutout | 66.27% | 81.21% | **+14.94 pp** |

*Note: Across all ablation pipelines, adding photometric distortion yielded an average downstream linear probe performance boost of **+18.10 pp**.*

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

### 3. Usage
- **Real-Time Search**: Upload any local photo or click **Pick a Random Test Image** to select one of the 10,000 reference images. The GUI preprocesses the image, runs dynamic-batch inference through the **ONNX ResNet-50 encoder** to extract a 2048-dimensional embedding, L2-normalizes it, and queries the **FAISS Inner Product (Cosine) Index**—all in **less than 5 ms**!
- **Ablation Study Dashboard**: An interactive scientific portal showcasing our final results, key project takeaways, and the mathematical formulations behind NT-Xent and SupCon losses.

---

## 🧠 Training & Re-running Experiments

### Standard SimCLR Pretraining
To run standard unsupervised SimCLR pretraining with our best parameters:
```bash
python src/train_master.py --epochs 200 --batch_size 1024 --backbone resnet50
```
*Features LARS/AdamW support, automatic linear learning rate scaling (`lr = 0.03 * batch_size / 256`), and TorchInductor compile integration.*

### Supervised Contrastive Learning (SupCon) Stage 1
To run pretraining on the 10% stratified subset under Supervised Contrastive Loss (Bonus #4):
```bash
python train_supcon.py --epochs 200 --batch_size 512 --fraction 0.1
```
*Performs stratified subset selection of CIFAR-10 and computes SupCon Stage 1 Loss.*

---

## Repository Structure

```text
SimCLR-Vision-SSL/
├── app.py                         # Premium Streamlit visual search engine & ablation GUI
├── build_faiss.py                 # Extracts 2048-d features, exports PNGs, and builds FAISS database
├── export_onnx.py                 # Exports Exp 41 ResNet-50 weights to optimized ONNX format
├── train_supcon.py                # SupCon Stage 1 pretraining orchestrator (Khosla et al.)
├── loss_supcon.py                 # Supervised Contrastive Loss implementation
├── dataset_subset.py              # Stratified data sampler (10% or 100% labels)
│
├── src/                           # Core source modules
│   ├── augmentations.py           # Natalie's color jitter & spatial augmentation pipelines
│   ├── dataset.py                 # Dataloader builders for training & linear evaluation
│   ├── loss.py                    # NT-Xent loss (Normalized Temp-scaled Cross-Entropy)
│   ├── model.py                   # ResNet-50 encoder architecture with custom stem
│   └── train_master.py            # Master contrastive training orchestrator with AMP & Compile
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
├── All Experiment SimCLR 18 May 2026/ # Training logs, curves & checkpoints for Exp 38-42
│   ├── v9_rotation_jitter_exp38/
│   ├── v10_weakbaseline_jitter_exp39/
│   ├── v11_blur_jitter_exp40/
│   ├── v12_flipblur_jitter_exp41/ # Best performing model
│   └── v13_cutout_jitter_exp42/
│
├── main-Final-SimCLR Report.tex   # Source LaTeX code of the final academic report
├── requirements.txt               # Dependencies list
└── LOG.md                         # Progress log
```

---

## 📚 Citation

```bibtex
@article{khosla2020supervised,
  title={Supervised contrastive learning},
  author={Khosla, Priyank and Teterwak, Piotr and Wang, Chen and others},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
  volume={33},
  pages={18661--18773}
}

@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}
```
