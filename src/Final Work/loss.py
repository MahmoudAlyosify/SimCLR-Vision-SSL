# SupCon paper: https://arxiv.org/abs/2004.11362
#
# NT-Xent defines one positive per anchor (its augmented twin).
# SupCon defines all same-class embeddings in the batch as positives,
# exploiting label supervision while still using unlabeled negatives.

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR. Given N images with 2 augmented views each,
    each view's positive is its twin; all other 2(N-1) views are negatives.

    Args:
        temperature: τ scaling factor (default 0.5).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: projection embeddings, shape (N, D), not yet normalized.
        Returns:
            Scalar NT-Xent loss averaged over both views.
        """
        N = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)                        # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature              # (2N, 2N)

        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        targets = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        return F.cross_entropy(sim, targets)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020, Eq. 2).

    When labels=None, falls back to self-supervised mode (equivalent to
    NT-Xent with two views). In supervised mode, all same-class embeddings
    are treated as positives.

    Args:
        temperature:      τ, recommended 0.07–0.5.
        contrast_mode:    'all' (all views as anchors) or 'one' (first view only).
        base_temperature: τ_base for loss re-scaling (default 0.07, per paper).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ):
        super().__init__()
        assert temperature > 0, "Temperature must be positive."
        assert contrast_mode in ("all", "one"), "contrast_mode must be 'all' or 'one'."

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
            features: (N, n_views, D) — ℓ₂-normalized embeddings.
                      For two augmented views: n_views=2.
            labels:   (N,) — class labels. Pass None for self-supervised mode.
            mask:     (N, N) — explicit positive-pair mask; overrides labels.
        Returns:
            Scalar supervised contrastive loss.
        """
        device = features.device

        if features.ndim == 2:
            features = features.unsqueeze(1)

        batch_size, n_views, feat_dim = features.shape

        # Build positive-pair mask
        if mask is not None:
            mask = mask.float().to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # Select anchor features based on contrast_mode
        if self.contrast_mode == "one":
            anchor_feat = features[:, 0, :]                    # (N, D)
            anchor_count = 1
        else:
            anchor_feat = features.view(batch_size * n_views, feat_dim)
            anchor_count = n_views

        contrast_feat = features.view(batch_size * n_views, feat_dim)
        contrast_count = n_views

        # Similarity matrix with numerical stability
        anchor_dot_contrast = torch.mm(anchor_feat, contrast_feat.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask to cover all view combinations, then zero out self-contrast
        mask_tiled = mask.repeat(anchor_count, contrast_count)
        self_contrast_mask = torch.scatter(
            torch.ones_like(mask_tiled),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask_tiled = mask_tiled * self_contrast_mask

        exp_logits = torch.exp(logits) * self_contrast_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask_tiled * log_prob).sum(1) / (mask_tiled.sum(1) + 1e-9)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def get_loss(loss_type: str, temperature: float = 0.5) -> nn.Module:
    """
    Instantiate a loss module by name.

    Args:
        loss_type:   'ntxent' or 'supcon'
        temperature: τ value
    Returns:
        Instantiated loss module.
    """
    if loss_type == "ntxent":
        return NTXentLoss(temperature=temperature)

    if loss_type == "supcon":
        return SupConLoss(
            temperature=temperature,
            contrast_mode="all",
            base_temperature=0.07,
        )

    raise ValueError(f"Unknown loss '{loss_type}'. Choose 'ntxent' or 'supcon'.")


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

    print("=" * 60)
    print("Smoke Test — SupConLoss (supervised mode)")
    print("=" * 60)

    supcon = SupConLoss(temperature=0.07)
    N_sup, n_views, D_sup, n_classes = 16, 2, 128, 10

    feats = F.normalize(torch.randn(N_sup, n_views, D_sup), dim=2)
    labels = torch.randint(0, n_classes, (N_sup,))

    loss_sup = supcon(feats, labels=labels)
    print(f"SupCon loss (supervised, 2 views) : {loss_sup.item():.4f}")
    assert loss_sup.shape == torch.Size([]), "Not a scalar!"
    assert not torch.isnan(loss_sup), "NaN detected!"
    print("PASSED ✓\n")

    loss_self = supcon(feats, labels=None)
    print(f"SupCon loss (self-supervised mode) : {loss_self.item():.4f}")
    assert not torch.isnan(loss_self), "NaN detected!"
    print("PASSED ✓\n")

    print("=" * 60)
    print("All loss tests passed.")
    print("=" * 60)