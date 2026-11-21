"""Test-set evaluation with 4-view TTA, classification report, and confusion matrix.

Usage:
    poetry run python -m src.models.21_evaluate
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, PROCESSED_DIR, ModelConfig
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on test split with TTA.")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: {args.checkpoint} (val_acc={ckpt.get('val_acc', '?')})")

    model_cfg = ModelConfig()
    model = build_model(
        model_name=model_cfg.model_name,
        num_classes=model_cfg.num_classes,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    test_ds = FoodDataset(PROCESSED_DIR, "test", CLASS_TO_IDX, get_eval_transforms())
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_ds)}")

    preds_list: list[int] = []
    labels_list: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            mask = labels >= 0
            if not mask.any():
                continue
            images = images[mask].to(device)
            labels = labels[mask].to(device)

            h, w = images.shape[-2:]
            crop = int(h * 0.88)
            top = (h - crop) // 2
            left = (w - crop) // 2
            center_crop = images[..., top : top + crop, left : left + crop]
            center_crop = F.interpolate(
                center_crop, size=(h, w), mode="bilinear", align_corners=False
            )
            views = [
                images,
                torch.flip(images, dims=[3]),
                center_crop,
                torch.flip(center_crop, dims=[3]),
            ]
            probs_list = [model(v).softmax(dim=1) for v in views]
            avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

            preds = avg_probs.argmax(dim=1)
            preds_list.extend(int(x) for x in preds.cpu().tolist())
            labels_list.extend(int(x) for x in labels.cpu().tolist())

    all_preds = np.array(preds_list)
    all_labels = np.array(labels_list)

    acc = float(np.mean(all_preds == all_labels))
    print(f"\nTest accuracy: {acc:.4f}\n")

    class_indices = list(range(len(CLASS_TO_IDX)))
    class_names = [IDX_TO_CLASS[i] for i in class_indices]
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=class_indices,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(class_indices)
    ax.set_yticks(class_indices)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={acc:.4f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\nConfusion matrix saved: {out}")


if __name__ == "__main__":
    main()
