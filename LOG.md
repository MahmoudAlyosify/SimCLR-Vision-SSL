# CISC 867 - Group 20 Development Log

This log tracks weekly progress, key decisions, issues encountered, and the individual contributions of each team member. This log is consistent with the Git commit history.

---
### Natalie Nashed (Data Augmentation Pipeline Lead)
* **Progress:** 
  * **Midterm:** Created `augmentations.py` using `torchvision.transforms`. Implemented Random Resized Crop, Color Jitter, Grayscale, and Horizontal Flip. Designed and structured the 8-experiment ablation study to isolate the contribution of spatial vs. photometric invariances. Created visualizations of positive pairs to include in the midterm report.
  * **Final:** Expanded the ablation suite to a comprehensive 42-experiment study. Implemented advanced structural corruptions (Gaussian Blur, Random Erasing, Sobel Edge, Gaussian Noise) to rigorously test the shortcut learning hypothesis. Designed the Color Jitter shortcut ablation sweep.
* **Key Decisions & Trials:** 
  * Explicitly excluded Gaussian Blur from the default pipeline for CIFAR-10 after multiple failed trial runs showed that default ImageNet kernel sizes completely destroyed the $32\times32$ images. Hand-tuned the kernel size to $k=3$ and $\sigma \in [0.1, 2.0]$.
  * Grouped the 42 experiments into logical tiers (Spatial, Chromatic, Structural, Hybrid, Conflict) to systematically analyze augmentation synergies rather than blindly testing random combinations.
* **Issues Encountered & Debugging:** 
  * **Tuple Augmentation:** Needed to ensure that the augmentation pipeline generates two *independent* views for the exact same image in a single pass; standard PyTorch datasets only return one. Built a custom Dataset wrapper `SimCLRDataset` to handle returning tuples of augmented images.
  * **Discrete vs. Continuous Rotation:** Designing the pure discrete rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) required custom tensor manipulations and disabling interpolation grids, since standard `RandomRotation` is continuous and introduces unwanted spatial blurring artifacts.
* **Key Commits:**
  *(Midterm Commits)*
  * `[49c97c1193f9d633fdd8c1966d1836c4f709232b]` - Add file : Add YAML config for augmentation hyperparameters and normalization stats.
  * `[82c6360487e224526fababd685b2f3a916c2240f]` - feat(data): setup augmentation configs and core SimCLR view generator class
  * `[47d67f65c8476f43fee994e11202dce16444f3ca]` - feat(data): implement baseline spatial augmentations [ Exp 1 and 2 ]
  * `[83fed8aee4ea78bdceac1d6bb35cded765f64737]` - feat(data): integrate photometric distortions and hybrid pipelines [ Exp 3-6 ]
  * `[518d35e4e0a528a4755872fb2a2063028e2bdf1c]` - feat(data): finalize comprehensive contrastive augmentation suite [ Exp 7-8 ]
  * `[c509405245433993e3d5b3f38ba7959ba3fa9573]` - feat(data): create Custom AugmentedDataset wrapper and initialize dataloaders
  * `[d73db8faf5eae69ce7d5a5db69035473099d371a]` - chore(vis): add Jupyter notebook for qualitative visualization of augmentation experiments
  *(Final Phase Commits - To be pushed)*
  * `[ ]` - feat(data): implement advanced structural augmentations (Blur, Sobel, Noise, Cutout)
  * `[ ]` - feat(data): design and integrate 42-experiment ablation configuration suite
  * `[ ]` - fix(data): tune Gaussian Blur kernel parameters specifically for 32x32 resolution

---

### Mahmoud Alyosify (Contrastive Learning Framework Lead)
* **Progress:** 
  * **Midterm (Architecture):** Implemented the SimCLR core architecture. Modified both ResNet-18 (for rapid ablations) and ResNet-50 (for full pretraining) stems for $32\times32$ CIFAR-10 images (replaced $7\times7$ Conv with $3\times3$ Conv stride 1, and removed MaxPool). Built the MLP projection heads and the NT-Xent loss function. Wrote the `run_ablations.py` script to automate the execution of the 8 experiments and executed them successfully on an RTX 5000 Ada GPU.
  * **Final (Infrastructure & Scale-up):** Developed a dynamic, automated ablation engine capable of sequentially executing all 42 experiments without manual intervention. Integrated Automatic Mixed Precision (AMP), step-level cosine annealing schedulers, and periodic checkpointing to maximize GPU utilization.
  * **Final (Semi-Supervised SupCon):** Spearheaded the Supervised Contrastive Learning (SupCon) extension. Implemented stratified dataset sampling for the 10% label fraction (5,000 images) and integrated the generalized SupCon loss function to exploit label structures during pretraining.
  * **Final (Production Deployment):** Exported the champion ResNet-50 SimCLR encoder to ONNX format, optimizing inference latency to under 5ms. Developed a vector similarity search engine using FAISS `IndexFlatIP`, indexing 10,000 CIFAR-10 test vectors. Built and deployed a fully interactive Streamlit GUI to Hugging Face Spaces for real-time semantic image querying.
* **Key Decisions & Trials:** 
  * Used ResNet-18 for the ablation study to efficiently run 20 epochs across 8 configurations, while reserving ResNet-50 for the final 200-epoch pretraining phase. Set NT-Xent temperature parameter $\tau=0.5$ based on optimal CIFAR-10 settings.
  * **SupCon Optimizer Sweep:** Ran extensive hyperparameter trials for SupCon. Switched the optimizer from AdamW to SGD (lr=0.05) with linear warmup for the SupCon runs. AdamW (lr=0.5) proved to be 4 orders of magnitude too high and destroyed all learned structure immediately.
  * **FAISS Metric Selection:** Tested both L2 distance and Cosine Similarity in FAISS. Decided to L2-normalize all vectors manually and use `IndexFlatIP` (Inner Product) to achieve mathematically exact cosine similarities.
* **Issues Encountered & Intensive Debugging:** 
  * Implementing the NT-Xent mask to exclude self-similarity correctly required careful handling of matrix operations; resolved using `torch.eye` as a boolean mask.
  * Faced CUDA/PyTorch incompatibility with Python 3.14 (pre-release); resolved by rebuilding a clean virtual environment using Python 3.11 and fixing `subprocess` module pathing in the ablation runner to enforce `sys.executable`.
  * **The 6 SupCon Failure Modes (Massive Debugging Effort):** 
    1. *FP16 Overflow:* Caused NaN losses due to extreme exponentiation values at $\tau=0.1$ ($e^{10} \approx 22,026$, pushing FP16 limits). Resolved by forcing FP32 precision.
    2. *Degenerate Plateau:* Loss stuck exactly at $\approx 2.31$ ($\log(10)$). Discovered this meant the model assigned equal probability to all 10 classes. Fixed by adding SGD linear warmup.
    3. *Representation Collapse:* Loss collapsed to $6.93$ (uniform similarity) when using a final BatchNorm layer followed by L2 normalization, which over-constrained the manifold. Solved by stripping final BN from the MLP.
* **Key Commits:**
  *(Midterm Commits)*
  * `a2410aa7e0271425374f452171dffdd8a4948007` - feat(model): modify ResNet-18/50 stems for 32x32 images and implement MLP projection heads
  * `13c987f05627ad0702220c4574ade70ee7d87a4f` - feat(loss): implement NT-Xent loss function with temperature scaling and self-similarity masking
  * `b051778e7beb454c3915af5dbe3696430a588e08` - feat(train): build SimCLR contrastive training loop and logging setup
  * `[71edb77d6f265b8a3f1b102b9c1ea60caf2d7a77]` - Add: Python script code to perform training on the 8 experiments and output the graphs.
  * `[cc24c17484b47c13c0dcfa0f45c75b4e1473c883]` - feat(train): build SimCLR contrastive training loop and logging setup for work in 8 Experiments
  * `[6efe1f5d4462698599667fd375ba30a30fbb40c4]` - feat(loss): implement NT-Xent loss function with temperature scaling and self-similarity masking
  * `[19685806ffabd7ef3ea403c94f5a5496e100bf5d]` - add: Our model output for the 8 experiment Registry.
  *(Final Phase Commits - To be pushed)*
  * `[ ]` - refactor(train): scale automated training engine to reliably execute 42-experiment suite
  * `[ ]` - feat(supcon): implement Supervised Contrastive loss and 10% stratified sampling
  * `[ ]` - fix(supcon): resolve FP16 overflow and BatchNorm representation collapse after 6 trial runs
  * `[ ]` - feat(deploy): export champion ResNet-50 encoder to ONNX for low-latency inference
  * `[ ]` - feat(deploy): build FAISS vector index (IndexFlatIP) for real-time 10k image retrieval
  * `[ ]` - feat(app): develop and deploy interactive Streamlit GUI to Hugging Face Spaces

---

### Mirna Imbabi (Linear Evaluation & Reporting Lead)
* **Progress:** 
  * **Midterm:** Trained the supervised ResNet-50 baseline model on CIFAR-10 for 90 epochs using mixed-precision (FP16), achieving a peak top-1 test accuracy of **93.77%** to establish the performance ceiling. Implemented the complete Linear Probe evaluation protocol (feature caching + 50-epoch training on a frozen encoder), validating it with a **93.89%** sanity check accuracy. Authored the comprehensive IEEE-format Midterm Report in LaTeX, analyzing the ablation loss curves and t-SNE projections.
  * **Final:** Ran linear probe evaluations across the entire 42-experiment suite. Generated confusion matrices, t-SNE embeddings, and OOD (Out-of-Distribution) failure case analyses for the champion model. Executed the OpenAI CLIP (ViT-B/32) zero-shot evaluation to establish a foundation model upper bound (88.80%).
* **Key Decisions & Trials:** 
  * **Strict Linear Probe Protocol:** Designed the linear probe to strictly freeze the encoder (using `requires_grad=False` on the entire trunk) to genuinely evaluate the representation quality without fine-tuning cheating. Trialed several weight decay values ($10^{-4}$) to prevent the linear classifier from overfitting the cached features.
  * **CLIP Implementation:** Decided to evaluate CLIP in a pure zero-shot manner without any fine-tuning. Wrote a custom loop using the Hugging Face `transformers` API to extract raw 512D text and image embeddings and compute exact cosine similarities against the prompt `"a photo of a {class_name}"`.
* **Issues Encountered & Debugging:** 
  * Managing Jupyter Notebook evaluation paths to dynamically locate the pre-trained checkpoints from the automated ablation outputs (`exp_8/checkpoints/simclr_epoch_020.pth`); resolved by writing a dynamic repository root locator.
  * **OOD Failure Hunting:** Spent hours manually testing diverse edge-case images on the live Hugging Face deployment to discover the model's blind spots. Successfully identified two specific failure mechanics:
    1. *Resolution-induced distribution shift* (Bicubic downsampling destroys facial features of high-res dogs, making them look like horses).
    2. *Background bias* (Camouflaged frogs retrieved as birds due to shared green/brown backgrounds).
* **Key Commits:**
  *(Midterm Commits)*
  * `[9960f373906ae6c56033335344129a221f7be380]` - Baseline added.
  * `[9036e4acf736f8e3a30f30c20881dfb323ef3899]` - bug fixes.
  * `[61fec6124907426f2e1f7684f6d5d7c42979962c]` - bug fixes.
  * `[2d5d43af5230bc26f20bb9ac173b39d030da21c0]` - Implement baseline.
  * `[f44c6720ff05bcf30280dfe6c5ca4861b065f349]` - Implement supervised baseline and linear probe evaluation.
  * `[e774bff9e38d7213e8175c3e85a96dcb2f1bae09]` - Add linear probe training and evaluation.
  *(Final Phase Commits - To be pushed)*
  * `[ ]` - feat(eval): automate and debug linear probe evaluation across all 42 experiments
  * `[ ]` - feat(vis): generate confusion matrices and final t-SNE projections
  * `[ ]` - feat(eval): implement custom CLIP ViT-B/32 zero-shot inference pipeline
  * `[ ]` - test(deploy): conduct extensive OOD live testing to isolate failure mode statistics
