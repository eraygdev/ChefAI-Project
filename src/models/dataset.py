"""ChefAI - PyTorch Dataset ve transform'lar."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    # 200/sinif icin daha yumusak augmentation
    return transforms.Compose(
        [
            # 256x185 crop'lu siniflar var -> scale (0.85,1.0) hepsini kapsar
            # Ama daha agresif crop ile model detaydan çok şekle odaklanır
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),  # 0.7→0.85
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # 15→10
            # RandomAffine KALDIRILDI (rotation zaten var, shear zararli)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # RandomGrayscale KALDIRILDI (yemek rengi kritik)
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),  # 0.3→0.15
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FoodDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        class_to_idx: dict[str, int],
        transform=None,
        image_size: int = 224,
    ):
        self.root = Path(root) / split
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Split bulunamadi: {self.root}")

        for class_name, label in class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXT:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            # Bozuk: -1 label ile skip edilecek
            return torch.zeros(3, self.image_size, self.image_size), -1
