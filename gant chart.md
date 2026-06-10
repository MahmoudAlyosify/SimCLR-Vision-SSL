# Detailed Project Structured Timeline (Gantt Chart)

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#ebf5fb', 'primaryBorderColor': '#2980b9', 'textColor': '#000000', 'lineColor': '#7f8c8d'}}}%%
gantt
    title SimCLR & SupCon Project Structured Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m
    tickInterval 7day
    todayMarker off

    section Week 1 (Setup)
    Repo Init & CIFAR-10 Loader (N.N & M.I)    :w1_1, 2026-06-01, 3d
    Supervised ResNet-50 Baseline (N.N & M.I)  :w1_2, 2026-06-04, 4d
    ResNet-18/50 Stem Modifications (M.A)      :crit, w1_3, 2026-06-01, 7d

    section Week 2 (Core Head)
    Stochastic Augmentation Module (N.N & M.A) :crit, w2_1, 2026-06-08, 4d
    MLP Head & NT-Xent Loss (N.N & M.A)        :w2_2, 2026-06-08, 4d
    20-epoch Ablations & Validation (M.I & M.A):w2_3, 2026-06-12, 3d
    MIDTERM SUBMISSION                         :milestone, m1, 2026-06-15, 0d

    section Week 3 (SimCLR)
    200-epoch SimCLR Final Run (M.A)           :crit, active, w3_1, 2026-06-15, 7d
    Augmentation Difficulty Analysis (N.N)     :w3_2, 2026-06-15, 4d
    SimCLR Linear Probe Eval (M.I)             :w3_3, 2026-06-19, 3d

    section Week 4 (SupCon)
    SupCon Adaptation & Fixes (M.A)            :w4_1, 2026-06-22, 7d
    Shortcut Visualization Fig (N.N)           :w4_2, 2026-06-22, 4d
    SupCon Linear Probe Eval (M.I)             :w4_3, 2026-06-26, 3d

    section Week 5 (Deployment)
    ONNX Export & Streamlit (M.A)              :w5_1, 2026-06-29, 7d
    Class Diagnostics & Conf. Matrix (M.I)     :w5_2, 2026-06-29, 4d
    Presentation Recording (All)               :w5_3, 2026-07-03, 3d
    IEEE Manuscript & LOG.md (All)             :crit, active, w5_4, 2026-07-03, 3d
    FINAL SUBMISSION                           :milestone, m2, 2026-07-06, 0d
