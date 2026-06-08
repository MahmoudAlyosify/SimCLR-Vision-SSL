"""
loss.py — NT-Xent + SupConLoss
Group 20, CISC 867, Queen's University, Spring 2026
Author (Student B): Mahmoud Alyosify

Two loss functions in one file:
  • NTXentLoss   — used for ALL self-supervised (unlabeled) pretraining
  • SupConLoss   — Bonus #4, Khosla et al. (NeurIPS 2020)
                   used for the hybrid 10%-label supervised-contrastive run

SupCon paper: https://arxiv.org/abs/2004.11362

Key design difference vs NT-Xent
────────────────────────────────
NT-Xent defines ONE positive per anchor (its augmented twin).
SupCon defines ALL same-class embeddings in the batch as positives.
When labels are present, the model can exploit the richer supervision
signal while still using unlabeled negatives for discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 1. NT-Xent Loss  (unchanged from midterm, kept here for single import)
# ══════════════════════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy loss.
    For a batch of N images → 2N views (two augmented copies each).
    Each view's positive is its twin; all 2(N-1) other views are negatives.

    Args:
        temperature (float): τ scaling factor. Default 0.5 (paper default).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: projection embeddings, shape (N, D). NOT yet normalized.
        Returns:
            Scalar NT-Xent loss averaged over both views.
        """
        N = z1.shape[0]
        device = z1.device

        # ℓ₂-normalize so cosine sim = dot product
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Stack into (2N, D) — [z1_0, z1_1, …, z2_0, z2_1, …]
        z = torch.cat([z1, z2], dim=0)                         # (2N, D)

        # Full similarity matrix scaled by temperature
        sim = torch.mm(z, z.T) / self.temperature             # (2N, 2N)

        # Mask out self-similarity (diagonal)
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Positive targets: view i pairs with view i+N (and vice versa)
        targets = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        return F.cross_entropy(sim, targets)


# ══════════════════════════════════════════════════════════════════════
# 2. SupCon Loss  (Bonus #4)  — Khosla et al., NeurIPS 2020
# ══════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Supports two modes controlled by `labels`:
      • labels=None  → self-supervised mode (same as NT-Xent with two views)
      • labels given → supervised mode: all same-class embeddings are positives

    For Bonus #4 we use SUPERVISED mode with 10% of CIFAR-10 labels.

    Mathematical formulation (paper Eq. 2):
    ┌──────────────────────────────────────────────────────────────────┐
    │  L = Σᵢ  -1/|P(i)| · Σₚ∈P(i)  log  exp(zᵢ·zₚ/τ)             │
    │                                      ──────────────────────     │
    │                                      Σₐ∈A(i) exp(zᵢ·zₐ/τ)     │
    │                                                                  │
    │  P(i) = set of positives for anchor i (same class, not self)    │
    │  A(i) = all indices except i (the denominator negatives)        │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        temperature (float): τ. Recommended 0.07–0.5 for SupCon.
        contrast_mode (str): 'all'  — use all views as anchors (default)
                             'one'  — use only first view as anchor
        base_temperature (float): τ_base for loss re-scaling. Default 0.07
                                   as in the original paper.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ):
        super().__init__()
        assert temperature > 0, "Temperature must be positive."
        assert contrast_mode in ("all", "one"), \
            "contrast_mode must be 'all' or 'one'."

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features: shape (N, n_views, D) — ℓ₂-normalized embeddings.
                      For two augmented views: n_views = 2.
                      Each slice features[:, v, :] is one view of the batch.
            labels:   shape (N,) — class labels for the N images.
                      Pass None for self-supervised mode (treats each sample
                      as its own class, recovers standard contrastive loss).
            mask:     shape (N, N) — binary mask where mask[i,j]=1 means i
                      and j form a positive pair. Overrides labels if given.
        Returns:
            Scalar supervised contrastive loss.
        """
        device = features.device

        # ── Input validation ─────────────────────────────────────────
        if features.ndim == 2:
            # Caller passed (N, D) — assume single view, unsqueeze
            features = features.unsqueeze(1)

        batch_size, n_views, feat_dim = features.shape

        # ── Build positive-pair mask ──────────────────────────────────
        if mask is not None:
            # Explicit mask provided — trust the caller
            mask = mask.float().to(device)

        elif labels is not None:
            # Supervised: mask[i,j] = 1 iff label[i] == label[j]
            labels = labels.contiguous().view(-1, 1)          # (N, 1)
            mask = torch.eq(labels, labels.T).float().to(device)  # (N, N)

        else:
            # Self-supervised fallback: each sample is its own class
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # ── Flatten views into one big (N*n_views, D) tensor ─────────
        # Anchor features — determined by contrast_mode
        if self.contrast_mode == "one":
            anchor_feat = features[:, 0, :]                   # (N, D)
            anchor_count = 1
        else:  # "all"
            # Stack all views: each becomes an anchor
            anchor_feat = features.view(
                batch_size * n_views, feat_dim
            )                                                  # (N*v, D)
            anchor_count = n_views

        # Contrast features = ALL views of ALL samples
        contrast_feat = features.view(
            batch_size * n_views, feat_dim
        )                                                      # (N*v, D)
        contrast_count = n_views

        # ── Similarity matrix ─────────────────────────────────────────
        # anchor_dot_contrast: (N*anchor_count, N*contrast_count)
        anchor_dot_contrast = (
            torch.mm(anchor_feat, contrast_feat.T) / self.temperature
        )

        # For numerical stability: subtract row-wise max
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # ── Expand mask to cover all view combinations ─────────────────
        # Original mask is (N, N). We tile it to (N*anchor_count, N*contrast_count)
        # so that every view of sample i treats every view of a same-class
        # sample j as a positive.
        mask_tiled = mask.repeat(anchor_count, contrast_count)   # (N*a, N*c)

        # Remove self-contrast (an anchor must not compare with itself)
        self_contrast_mask = torch.scatter(
            torch.ones_like(mask_tiled),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask_tiled = mask_tiled * self_contrast_mask

        # ── Log-softmax over negatives ────────────────────────────────
        exp_logits = torch.exp(logits) * self_contrast_mask    # zero out self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # ── Mean over positives ───────────────────────────────────────
        # For each anchor, average log-prob over its positives
        mean_log_prob_pos = (mask_tiled * log_prob).sum(1) / (
            mask_tiled.sum(1) + 1e-9
        )

        # ── Final loss (temperature re-scaling as in paper Eq. 2) ─────
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ══════════════════════════════════════════════════════════════════════
# 3. Loss Factory  — used by train_master.py and train_supcon.py
# ══════════════════════════════════════════════════════════════════════

def get_loss(loss_type: str, temperature: float = 0.5) -> nn.Module:
    """
    Factory function so all training scripts import losses by name.

    Args:
        loss_type   : 'ntxent' | 'supcon'
        temperature : τ value (same for both losses)
    Returns:
        Instantiated loss module.
    """
    if loss_type == "ntxent":
        return NTXentLoss(temperature=temperature)

    if loss_type == "supcon":
        # SupCon paper uses τ=0.07 by default; we allow override
        return SupConLoss(
            temperature=temperature,
            contrast_mode="all",
            base_temperature=0.07,
        )

    raise ValueError(
        f"Unknown loss '{loss_type}'. Choose 'ntxent' or 'supcon'."
    )


# ══════════════════════════════════════════════════════════════════════
# 4. Smoke Tests
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("Smoke Test — NTXentLoss")
    print("=" * 60)

    crit = NTXentLoss(temperature=0.5)
    N, D = 32, 128

    z1 = torch.randn(N, D)
    z2 = torch.randn(N, D)
    loss_rand = crit(z1, z2)
    loss_expected = math.log(2 * N - 1)

    print(f"Random embeddings loss : {loss_rand.item():.4f}")
    print(f"Expected (uniform)     : {loss_expected:.4f}")

    loss_perfect = crit(z1, z1.clone())
    print(f"Identical embeddings   : {loss_perfect.item():.6f}  (should be ~0)")
    assert loss_rand.shape == torch.Size([]), "Not a scalar!"
    print("PASSED ✓\n")

    # ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Smoke Test — SupConLoss (supervised mode, Bonus #4)")
    print("=" * 60)

    supcon = SupConLoss(temperature=0.07)
    N_sup = 16
    n_views = 2
    D_sup = 128
    n_classes = 10

    # Simulate two augmented views: features shape (N, n_views, D)
    feats = F.normalize(
        torch.randn(N_sup, n_views, D_sup), dim=2
    )
    labels = torch.randint(0, n_classes, (N_sup,))

    loss_sup = supcon(feats, labels=labels)
    print(f"SupCon loss (supervised, 2 views) : {loss_sup.item():.4f}")
    assert loss_sup.shape == torch.Size([]), "Not a scalar!"
    assert not torch.isnan(loss_sup), "NaN detected!"
    print("PASSED ✓\n")

    # Self-supervised fallback (no labels)
    loss_self = supcon(feats, labels=None)
    print(f"SupCon loss (self-supervised mode) : {loss_self.item():.4f}")
    assert not torch.isnan(loss_self), "NaN detected!"
    print("PASSED ✓\n")

    print("=" * 60)
    print("All loss tests passed.")
    print("=" * 60)
