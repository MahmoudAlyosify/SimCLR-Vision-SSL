# Detailed Project Structured Timeline (Gantt Chart)

Below is the interactive Gantt chart illustrating task allocation, the structured timeline, cross-task dependencies, and responsibilities across the five weeks of the project, strictly mirroring the detailed data provided in the report.

```mermaid
gantt
    title SimCLR & SupCon Project Structured Timeline (Gantt Chart)
    dateFormat  YYYY-MM-DD
    %% The axis format displays "Week" followed by the week number
    axisFormat  Week %U
    tickInterval 1week

    %% -----------------------------------------------------------
    %% ASSIGNEE KEY (implied via naming in tasks)
    %% N.N & M.I = N. Nashed & M. Imbabi
    %% M.A       = M. Alyosify
    %% N.N & M.A = N. Nashed & M. Alyosify
    %% M.I & M.A = M. Imbabi & M. Alyosify
    %% M.I       = M. Imbabi
    %% N.N       = N. Nashed
    %% All       = All Members
    %% -----------------------------------------------------------

    section Week 1: Setup & Baselines
    %% [crit] marks critical path tasks, [active/done] sets status
    Repo Init, CIFAR-10 Loader setup (N.N & M.I)   :active, w1_t1, 2026-06-01, 7d
    Impl. Supervised ResNet-50 baseline (N.N & M.I):w1_t2, after w1_t1, 5d
    ResNet-18/50 Stem Modifications (M.A)           :crit, w1_t3, after w1_t1, 6d

    section Week 2: Augmentations & Core
    Program Stochastic Augmentation Module (N.N & M.A):crit, w2_t1, after w1_t3, 7d
    Impl. MLP Head & NT-Xent Loss logic (N.N & M.A)  :w2_t2, after w1_t3, 7d
    Execute 20-epoch Ablations (M.I & M.A)            :w2_t3, after w2_t1 w2_t2, 5d
    MIDTERM REPORT SUBMISSION                         :milestone, w2_m, after w2_t3, 0d

    section Week 3: SimCLR Training & Eval
    Execute 200-epoch SimCLR Final Run (Exp 41) (M.A):crit, active, w3_t1, after w2_m, 7d
    Analyze Augmentation Difficulty Hierarchy (N.N)   :w3_t2, after w2_m, 5d
    SimCLR Linear Probe Evaluation (M.I)             :w3_t3, after w3_t1, 4d

    section Week 4: SupCon Adaptation
    Impl. SupCon Adaptation & resolve failures (M.A)  :w4_t1, after w3_t1, 7d
    SupCon Linear Probe Eval & Results Table (M.I)   :w4_t2, after w4_t1, 5d
    Shortcut Visualization Figure Generation (N.N)   :w4_t3, 2026-06-22, 6d %% Dependent on earlier Color Jitter ablations

    section Week 5: Deployment & Wrap-up
    Model Export (ONNX) & Streamlit Deployment (M.A) :w5_t1, after w4_t1, 7d
    Class-Level Diagnostics & Confusion Matrix (M.I) :w5_t2, after w4_t2, 5d
    Presentation Prep & Recording (All)              :w5_t3, after w4_t2, 5d %% Post results-freeze
    LOG.md Sync & IEEE Manuscript Assembly (All)     :crit, active, w5_t4, after w4_t2, 7d
    FINAL PROJECT SUBMISSION                          :milestone, final_m, after w5_t4, 0d
