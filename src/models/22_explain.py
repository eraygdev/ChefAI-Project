"""GradCAM explainability heatmap generator (CPU-friendly for low-VRAM GPUs).

Usage:
    CUDA_VISIBLE_DEVICES="" poetry run python -m src.models.22_explain --n-samples 8
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, PROCESSED_DIR
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model, get_target_layer


class GradCAM:
    """Minimal GradCAM attribution wrapper for convolutional layers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(
        self,
        module: nn.Module,
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        self.activations = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def generate(self, x: torch.Tensor, class_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Compute normalized GradCAM heatmaps for input batch x."""
        self.model.zero_grad()
        out: torch.Tensor = self.model(x)
        target_idx: torch.Tensor = out.argmax(dim=1) if class_idx is None else class_idx

        scores = out.gather(1, torch.reshape(target_idx, (-1, 1))).squeeze(1)
        scores.sum().backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks failed to capture activations or gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))

        cam_flat = torch.flatten(cam, start_dim=1)
        cmin = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cmax = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        return (cam - cmin) / (cmax - cmin + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GradCAM heatmaps on test samples.")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pt"))
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: {args.checkpoint} (val_acc={ckpt.get('val_acc', '?')})")

    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    test_ds = FoodDataset(PROCESSED_DIR, "test", CLASS_TO_IDX, get_eval_transforms())
    loader = DataLoader(test_ds, batch_size=args.n_samples, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    mask = labels >= 0
    images = images[mask]
    labels = labels[mask]
    print(f"Generating GradCAM for {len(images)} samples")

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(images.to(device))

    cam_resized = (
        F.interpolate(cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
        .squeeze(1)
        .cpu()
        .numpy()
    )

    n_plot = len(images)
    n_rows = (n_plot + 3) // 4
    fig, axes = plt.subplots(n_rows * 2, 4, figsize=(16, 4 * n_rows * 2))

    for i in range(n_plot):
        row = (i // 4) * 2
        col = i % 4

        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        true_name = IDX_TO_CLASS[int(labels[i].item())]

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"True: {true_name}", fontsize=8)
        axes[row, col].axis("off")

        axes[row + 1, col].imshow(img)
        axes[row + 1, col].imshow(cam_resized[i], cmap="jet", alpha=0.5)
        axes[row + 1, col].set_title("GradCAM attribution", fontsize=8)
        axes[row + 1, col].axis("off")

    plt.tight_layout()
    out = MODELS_DIR / "shap_heatmap.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGradCAM heatmap saved: {out}")


if __name__ == "__main__":
    main()
