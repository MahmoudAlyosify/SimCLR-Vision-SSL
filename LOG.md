# CISC 867 - Group 20 Development Log

This log tracks weekly progress, key decisions, issues encountered, and the individual contributions of each team member. This log is consistent with the Git commit history.

---
### Natalie Nashed (Data Augmentation Pipeline Lead)
* **Progress:** Created `augmentations.py` using `torchvision.transforms`. Implemented Random Resized Crop, Color Jitter, Grayscale, and Horizontal Flip. Designed and structured the 8-experiment ablation study to isolate the contribution of spatial vs. photometric invariances. Created visualizations of positive pairs to include in the midterm report.
* **Key Decisions:** Explicitly excluded Gaussian Blur from the default pipeline for CIFAR-10, aligning with the SimCLR paper's appendix recommendations for small-resolution images ($32\times32$).
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

### Mahmoud Alyosify (Contrastive Learning Framework Lead)
* **Progress:** Implemented the SimCLR core architecture. Modified both ResNet-18 (for rapid ablations) and ResNet-50 (for full pretraining) stems for $32\times32$ CIFAR-10 images (replaced $7\times7$ Conv with $3\times3$ Conv stride 1, and removed MaxPool). Built the MLP projection heads and the NT-Xent loss function. Wrote the `run_ablations.py` script to automate the execution of the 8 experiments and executed them successfully on an RTX 5000 Ada GPU.
* **Key Decisions:** Used ResNet-18 for the ablation study to efficiently run 20 epochs across 8 configurations, while reserving ResNet-50 for the final pretraining phase. Set NT-Xent temperature parameter $\tau=0.5$ based on optimal CIFAR-10 settings.
* **Issues Encountered:** * Implementing the NT-Xent mask to exclude self-similarity correctly required careful handling of matrix operations; resolved using `torch.eye` as a boolean mask.
  * Faced CUDA/PyTorch incompatibility with Python 3.14 (pre-release); resolved by rebuilding a clean virtual environment using Python 3.11 and fixing `subprocess` module pathing in the ablation runner to enforce `sys.executable`.
* **Key Commits:**
  * `a2410aa7e0271425374f452171dffdd8a4948007` - feat(model): modify ResNet-18/50 stems for 32x32 images and implement MLP projection heads
  * `13c987f05627ad0702220c4574ade70ee7d87a4f` - feat(loss): implement NT-Xent loss function with temperature scaling and self-similarity masking
  * `b051778e7beb454c3915af5dbe3696430a588e08` - feat(train): build SimCLR contrastive training loop and logging setup
  * `[6efe1f5d4462698599667fd375ba30a30fbb40c4]` - feat(loss): implement NT-Xent loss function with temperature scaling and self-similarity masking
  * `[849cab84157c9f1fa84d613e4ac48a30edb8bd40]` - Modify: Natalie file and add the "AugmentedDataset" function or Custom SimCLR Dataset Wrapper, and perform an Experiment
  * `[19685806ffabd7ef3ea403c94f5a5496e100bf5d]` - add: Our model output for the 8 experiment
  Registry.
  * `[860fb9f84665e7edb2f782d965ae447f3e21829a]` - Update augmentations.py

    

---

### Mirna Imbabi (Linear Evaluation & Reporting Lead)
* **Progress:** Trained the supervised ResNet-50 baseline model on CIFAR-10 for 90 epochs using mixed-precision (FP16), achieving a peak top-1 test accuracy of **93.77%** to establish the performance ceiling. Implemented the complete Linear Probe evaluation protocol (feature caching + 50-epoch training on a frozen encoder), validating it with a **93.89%** sanity check accuracy. Authored the comprehensive IEEE-format Midterm Report in LaTeX, analyzing the ablation loss curves and t-SNE projections.
* **Key Decisions:** Designed the linear probe to strictly freeze the encoder (0 gradients propagated) to genuinely evaluate the representation quality without fine-tuning cheating.
* **Issues Encountered:** Managing Jupyter Notebook evaluation paths to dynamically locate the pre-trained checkpoints from the automated ablation outputs (`exp_8/checkpoints/simclr_epoch_020.pth`); resolved by writing a dynamic repository root locator.
* **Key Commits:**
  * `[Insert-Commit-Hash-Here]` - Initialized project structure, README, and LOG.md.
  * `[Insert-Commit-Hash-Here]` - feat(eval): Add baseline ResNet-50 supervised training script achieving 93.77% accuracy.
  * `[Insert-Commit-Hash-Here]` - feat(eval): Implement Linear Probe protocol and feature caching for representation evaluation.
  * `[Insert-Commit-Hash-Here]` - docs(report): Draft IEEE Midterm Report with Ablation analysis and t-SNE plots.
