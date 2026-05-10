"""
src/dataset.py — CIFAR-10 with SimCLR Augmentations
=====================================================
نسخة الميدتيرم: بدون Gaussian Blur (متاحة في النسخة النهائية)

الـ augmentation pipeline للميدتيرم:
  ✓ RandomResizedCrop(32)   — الأهم
  ✓ RandomHorizontalFlip
  ✓ ColorJitter             — ضروري جداً (بدونه الموديل بيتعلم colors بس)
  ✓ RandomGrayscale(p=0.2)
  ✗ GaussianBlur           — محجوزة للفاينال

المراجع: SimCLR paper - Appendix B.9 (CIFAR-10 experiments)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ════════════════════════════════════════════════════════════════
#  SimCLR Augmentation Pipeline
# ════════════════════════════════════════════════════════════════
def get_simclr_transform(image_size: int = 32, strength: float = 0.5) -> transforms.Compose:
    """
    SimCLR augmentation pipeline مخصوص لـ CIFAR-10.

    Args:
        image_size: حجم الصورة بعد الـ crop (32 لـ CIFAR-10)
        strength (s): قوة الـ color distortion.
                      البيبر بيوصي بـ s=0.5 لـ CIFAR-10 (مش 1.0 زي ImageNet)
    """
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength,
    )

    return transforms.Compose([
        # ① أهم augmentation: random crop + resize
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.2, 1.0),   # البيبر بيستخدم 0.08-1.0 لـ ImageNet
                                 # لـ CIFAR-10 نبدأ من 0.2 عشان الصور صغيرة
        ),
        # ② flip
        transforms.RandomHorizontalFlip(p=0.5),

        # ③ Color distortion (الجزء الثاني الأهم بعد الـ crop)
        transforms.RandomApply([color_jitter], p=0.8),

        # ④ Grayscale
        transforms.RandomGrayscale(p=0.2),

        # ⑤ تحويل لـ tensor وتطبيع
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
            std= [0.2023, 0.1994, 0.2010],  # CIFAR-10 std
        ),
    ])


# ════════════════════════════════════════════════════════════════
#  Dataset Wrappers
# ════════════════════════════════════════════════════════════════
class SimCLRDataset(torch.utils.data.Dataset):
    """
    يلف CIFAR-10 ويرجع (x_i, x_j) بدل صورة واحدة.
    x_i و x_j: نفس الصورة بعد augmentations مختلفة → الـ positive pair.
    """

    def __init__(self, root: str, train: bool = True, image_size: int = 32):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True
        )
        self.transform = get_simclr_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # نطبق الـ transform مرتين → augmentations مختلفة
        x_i = self.transform(image)
        x_j = self.transform(image)
        # بنرجع الـ label كمان (مش محتاجينه في الـ pretraining
        # بس محتاجينه في الـ t-SNE visualization)
        return (x_i, x_j), label


class EvalDataset(torch.utils.data.Dataset):
    """
    Dataset عادي (صورة واحدة + label) للـ evaluation و t-SNE.
    """
    _EVAL_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std= [0.2023, 0.1994, 0.2010],
        ),
    ])

    def __init__(self, root: str, train: bool = False):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self._EVAL_TRANSFORM(image), label


# ════════════════════════════════════════════════════════════════
#  DataLoader Factories
# ════════════════════════════════════════════════════════════════
def get_train_dataloader(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SimCLRDataset(root=data_dir, train=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # مهم! NT-Xent بيحتاج batch size ثابت
    )


def get_eval_dataloader(
    data_dir: str,
    train: bool = False,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    dataset = EvalDataset(root=data_dir, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ds = SimCLRDataset(root=tmp, train=True)
        (xi, xj), lbl = ds[0]

        print("=" * 40)
        print("  Dataset Sanity Check")
        print("=" * 40)
        print(f"  Dataset size : {len(ds)}")
        print(f"  x_i shape   : {tuple(xi.shape)}")
        print(f"  x_j shape   : {tuple(xj.shape)}")
        print(f"  Label        : {lbl}")
        print(f"  Are equal?   : {torch.allclose(xi, xj)} (should be False)")
        print("=" * 40)
