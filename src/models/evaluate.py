"""ChefAI - Test degerlendirmesi (confusion matrix + per-class metrics)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, PROCESSED_DIR
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: {args.checkpoint} (val_acc={ckpt.get('val_acc', '?')})")

    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    test_ds = FoodDataset(PROCESSED_DIR, "test", CLASS_TO_IDX, get_eval_transforms())
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test: {len(test_ds)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            mask = labels >= 0
            if not mask.any():
                continue
            images = images[mask].to(device)
            labels = labels[mask].to(device)

            # TTA: 5 farklı görünüm
            logits_sum = model(images).softmax(dim=1)
            logits_sum += model(torch.flip(images, dims=[3])).softmax(dim=1)  # yatay flip
            logits_sum += model(torch.flip(images, dims=[2])).softmax(dim=1)  # dikey flip

            preds = logits_sum.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float(np.mean(all_preds == all_labels))
    print(f"\nTest accuracy: {acc:.4f}")

    print("\nSinif bazli rapor:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(CLASS_TO_IDX))),
            target_names=[IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))],
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_TO_IDX))))
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_TO_IDX)))
    ax.set_yticks(range(len(CLASS_TO_IDX)))
    ax.set_xticklabels([IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))], rotation=90, fontsize=8)
    ax.set_yticklabels([IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))], fontsize=8)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(f"Confusion Matrix (acc={acc:.4f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=100)
    print(f"\nConfusion matrix: {out}")


if __name__ == "__main__":
    main()
