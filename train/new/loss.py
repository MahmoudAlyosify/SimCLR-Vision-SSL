"""
src/loss.py — NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
=========================================================================
الـ contrastive loss الأساسي في SimCLR.

الفكرة:
  - عندنا batch فيه N صور → بعد الـ augmentation بيبقى 2N صورة
  - كل صورة وتوأمها (positive pair) المفروض يكونوا قريبين
  - كل الصور التانية في الـ batch هي negatives

المعادلة (من البيبر، معادلة 1):
  ℓ(i,j) = -log [ exp(sim(zᵢ, zⱼ)/τ) / Σₖ₌₁²ᴺ 1[k≠i] exp(sim(zᵢ, zₖ)/τ) ]

  حيث sim(u,v) = uᵀv / (‖u‖ · ‖v‖)  ← cosine similarity
  وτ (tau) = temperature hyperparameter

المراجع: SimCLR paper (Chen et al., 2020) - Section 2.1, Equation 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR.

    Args:
        temperature (float): τ في المعادلة. القيمة الموصى بها = 0.5 لـ CIFAR-10.
                             قيمة صغيرة جداً → overconfident (bad)
                             قيمة كبيرة جداً → uniform distribution (bad)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (N, D) — projections من الـ augmentation الأولى
            z_j: (N, D) — projections من الـ augmentation التانية

        Returns:
            loss: scalar
        """
        N = z_i.size(0)  # batch size
        device = z_i.device

        # ── Step 1: L2 Normalize ──────────────────────────────────
        # علشان sim تبقى cosine similarity
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # ── Step 2: دمج الاتنين في matrix واحدة ──────────────────
        # z shape: (2N, D)
        # الترتيب: [z_i[0], z_j[0], z_i[1], z_j[1], ...]
        # ده بيخلي الـ positive pair لكل صورة في الصف التاني منها
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # ── Step 3: Similarity Matrix ─────────────────────────────
        # sim_matrix[i][j] = cosine_similarity(z[i], z[j])
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # ── Step 4: Mask الـ diagonal (i=j) ──────────────────────
        # عشان ميحسبش similarity الصورة مع نفسها
        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # ── Step 5: Labels (الـ positive pair لكل صورة) ──────────
        # z_i[k] → positive هو z_j[k] اللي index بتاعه = k + N
        # z_j[k] → positive هو z_i[k] اللي index بتاعه = k
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),   # للـ z_i: positive في النص التاني
            torch.arange(0, N,     device=device),   # للـ z_j: positive في النص الأول
        ])  # (2N,)

        # ── Step 6: Cross Entropy ─────────────────────────────────
        # F.cross_entropy بيعمل softmax + log + nll في خطوة واحدة
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    criterion = NTXentLoss(temperature=0.5)

    # لو z_i == z_j (perfect representations) → loss المفروض تكون صغيرة
    z = F.normalize(torch.randn(8, 128), dim=1)
    loss_perfect = criterion(z, z.clone())

    # لو z_i و z_j random → loss المفروض تكون أكبر
    loss_random = criterion(
        F.normalize(torch.randn(8, 128), dim=1),
        F.normalize(torch.randn(8, 128), dim=1),
    )

    print("=" * 40)
    print("  NT-Xent Loss — Sanity Check")
    print("=" * 40)
    print(f"  τ (temperature) : 0.5")
    print(f"  Batch size      : 8")
    print(f"  Loss (identical): {loss_perfect.item():.4f}  ← should be low")
    print(f"  Loss (random)   : {loss_random.item():.4f}  ← should be high")
    print("=" * 40)
