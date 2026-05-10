"""
src/model.py — SimCLR Encoder + Projection Head
================================================
التعديلات على ResNet50 علشان CIFAR-10 (32×32):
  - الطبقة الأولى: 7×7 conv stride=2  →  3×3 conv stride=1
  - حذف الـ MaxPool (كان بيعمل downsampling زيادة على صور صغيرة)
  - الـ Projection Head: Linear(2048→2048) → ReLU → Linear(2048→128)

ليه التعديل ده مهم؟
  ImageNet صورها 224×224، فالـ stem الأصلي صمم يقلل الحجم بسرعة.
  CIFAR-10 صورها 32×32، لو اتبعنا الـ stem الأصلي هتبقى 4×4 بعد الـ stem!
  الهدف: نوصل للـ average pool بحجم معقول (~8×8).

المراجع: SimCLR paper (Chen et al., 2020) - Appendix B.9
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR_ResNet50(nn.Module):
    """
    SimCLR framework built on ResNet50.

    الـ forward بيرجع:
        h  — representation (2048-dim) — ده اللي بنستخدمه في الـ linear probe
        z  — projection (128-dim)      — ده اللي بنحسب عليه الـ NT-Xent loss
    """

    def __init__(self, projection_dim: int = 128):
        super().__init__()

        # ── Base encoder: ResNet50 ────────────────────────────────
        backbone = models.resnet50(weights=None)

        # ── تعديل الـ Stem لـ CIFAR-10 ───────────────────────────
        # الأصلي:  Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # الجديد:  Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # حذف الـ MaxPool (كان بيقلص الحجم من 32→16، ده كتير)
        backbone.maxpool = nn.Identity()

        # نأخذ كل الـ ResNet ماعدا الـ final FC layer
        self.encoder = nn.Sequential(
            backbone.conv1,    # 3×3, stride=1
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # Identity (محذوف فعلياً)
            backbone.layer1,   # residual blocks
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,  # Global Average Pooling → (B, 2048, 1, 1)
            nn.Flatten(),      # → (B, 2048)
        )

        # ── Projection Head: 2-Layer MLP ─────────────────────────
        # Linear → BN → ReLU → Linear
        # (بنضيف BatchNorm زي ما الـ SimCLR paper بيوصي)
        hidden_dim = backbone.fc.in_features  # 2048

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        """
        x: (B, 3, 32, 32)
        returns:
            h: (B, 2048)  ← representation — للـ linear probe
            z: (B, 128)   ← projection     — للـ NT-Xent loss
        """
        h = self.encoder(x)   # (B, 2048)
        z = self.projector(h) # (B, 128)
        return h, z


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    model = SimCLR_ResNet50(projection_dim=128)

    dummy = torch.randn(4, 3, 32, 32)
    h, z = model(dummy)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 45)
    print("  SimCLR ResNet50 — CIFAR-10 Adapted")
    print("=" * 45)
    print(f"  Input  shape : {tuple(dummy.shape)}")
    print(f"  h (repr) shape: {tuple(h.shape)}   ← encoder output")
    print(f"  z (proj) shape: {tuple(z.shape)}    ← loss input")
    print(f"  Total params : {total_params:.1f}M")
    print("=" * 45)
