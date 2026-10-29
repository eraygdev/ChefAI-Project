"""ChefAI - SHAP aciklanabilirlik (heatmap)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, PROCESSED_DIR
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pt"))
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    test_ds = FoodDataset(PROCESSED_DIR, "test", CLASS_TO_IDX, get_eval_transforms())
    loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)

    images_list, labels_list = [], []
    for i, (img, label) in enumerate(loader):
        if i >= args.n_samples:
            break
        if label.item() < 0:
            continue
        images_list.append(img)
        labels_list.append(label)

    if not images_list:
        print("HATA: ornek bulunamadi.")
        return

    images = torch.cat(images_list, dim=0).to(device)
    labels = torch.cat(labels_list).to(device)
    print(f"SHAP icin {len(images)} ornek")

    # GradientExplainer (background = ilk 10 ornek)
    explainer = shap.GradientExplainer(model, images[:10])
    print("SHAP degerleri hesaplaniyor...")
    shap_values = explainer.shap_values(images[:4])

    # 4 ornek icin heatmap ciz
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    for i in range(min(4, len(images))):
        # Orijinal goruntu
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # SHAP degeri
        if isinstance(shap_values, list):
            sv = shap_values[labels[i].item()][i]
        else:
            sv = shap_values[i]

        # 3 kanalin mutlak ortalamasi -> (H, W)
        heatmap = np.abs(sv).mean(axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        true_name = IDX_TO_CLASS[labels[i].item()]

        # Sol: orijinal
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Gercek: {true_name}")
        axes[i, 0].axis("off")

        # Sag: heatmap overlay
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[i, 1].set_title("SHAP - model nereye bakiyor?")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out = MODELS_DIR / "shap_heatmap.png"
    plt.savefig(out, dpi=100)
    print(f"SHAP heatmap: {out}")


if __name__ == "__main__":
    main()
