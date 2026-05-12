# CISC 867 Project 20 - Group 20 Development Log

This log tracks weekly progress, key decisions, issues encountered, and the individual contributions of each team member. This log is consistent with the Git commit history.

---

## Week 1 (May 4, 2026 - May 10, 2026)

### 🎯 Weekly Goals
- Understand SimCLR framework and set up the repository structure.
- Implement initial data augmentation pipeline for CIFAR-10.
- Build ResNet-50 modified backbone, Projection Head, and NT-Xent loss.
- Train supervised baseline and run a preliminary contrastive training loop to verify loss convergence.

---
### 👩‍💻 Natalie Nashed (Data Augmentation Pipeline Lead)
* **Progress:** Created `augmentations.py` using `torchvision.transforms`. Implemented Random Resized Crop, Color Jitter, Grayscale, and Horizontal Flip. Created visualizations of positive pairs to include in the midterm report.
* **Key Decisions:** Explicitly excluded Gaussian Blur from the default pipeline for CIFAR-10, aligning with the SimCLR paper's appendix recommendations for small-resolution images.
* **Issues Encountered:** Needed to ensure that the augmentation pipeline generates two *independent* views for the exact same image in a single pass; built a custom Dataset wrapper `SimCLRDataset` to handle returning tuples of augmented images.
* **Key Commits:**
  * `[49c97c1193f9d633fdd8c1966d1836c4f709232b]` - Add file : Add YAML config for augmentation hyperparameters and normalization stats.
  * `[82c6360487e224526fababd685b2f3a916c2240f]` - feat(data): setup augmentation configs and core SimCLR view generator class
  * `[47d67f65c8476f43fee994e11202dce16444f3ca]` - feat(data): implement baseline spatial augmentations [ Exp 1 and 2 ]
  * `[83fed8aee4ea78bdceac1d6bb35cded765f64737]` - feat(data): integrate photometric distortions and hybrid pipelines [ Exp 3-6 ]
  * `[518d35e4e0a528a4755872fb2a2063028e2bdf1c]` - feat(data): finalize comprehensive contrastive augmentation suite [ Exp 7-8 ]
  * `[c509405245433993e3d5b3f38ba7959ba3fa9573]` - feat(data): create Custom AugmentedDataset wrapper and initialize dataloaders
  * `[d73db8faf5eae69ce7d5a5db69035473099d371a]` - chore(vis): add Jupyter notebook for qualitative visualization of augmentation experiments
---

### 👨‍💻 Mahmoud Alyosify (Contrastive Learning Framework Lead)
* **Progress:** Implemented the SimCLR core architecture. Modified the ResNet-50 stem for $32\times32$ CIFAR-10 images (replaced $7\times7$ Conv with $3\times3$ Conv stride 1, and removed MaxPool). Built the 2-layer MLP projection head and the NT-Xent loss function. Ran a preliminary training loop to verify loss convergence.
* **Key Decisions:** Set NT-Xent temperature parameter $\tau=0.5$ based on optimal CIFAR-10 settings from the original paper. Used a smaller batch size temporarily for local testing before scaling to the RTX 5000 Ada GPU.
* **Issues Encountered:** Implementing the NT-Xent mask to exclude self-similarity correctly required careful handling of matrix operations; resolved using `torch.eye` as a boolean mask.
* **Key Commits:**
  * `a2410aa7e0271425374f452171dffdd8a4948007` - feat(model): modify ResNet-50 stem for 32x32 (CIFAR-10) images and implement 2-layer MLP projection head
  * `13c987f05627ad0702220c4574ade70ee7d87a4f` - feat(loss): implement NT-Xent loss function with temperature scaling and self-similarity masking
  * `b051778e7beb454c3915af5dbe3696430a588e08` - feat(train): build SimCLR contrastive training loop and logging setup

---
### 👩‍💻 Mirna Imbabi (Linear Evaluation & Reporting Lead)
* **Progress:** Set up the initial `LOG.md` and `README.md` structure. Trained the supervised ResNet-50 baseline model on CIFAR-10 to establish a comparative benchmark.
* **Key Decisions:** Used standard Cross-Entropy loss and default PyTorch parameters for the supervised baseline to ensure a fair evaluation later.
* **Issues Encountered:** Minor dependency issues with `torchvision` during baseline training, resolved by pinning the specific library versions in `requirements.txt`.
* **Key Commits:**
  * `[Insert-Commit-Hash-Here]` - Initialized project structure, README, and LOG.md.
  * `[Insert-Commit-Hash-Here]` - Added baseline ResNet-50 supervised training script.
