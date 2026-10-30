"""ChefAI - GradCAM aciklanabilirlik (SHAP yerine, ~100x hizli).

GradCAM: son conv katmanindaki aktivasyon + gradient ile heatmap uretir.
Her gorsel icin 1 forward + 1 backward (SHAP 200x yapar).
Cikti: models/shap_heatmap.png (dosya adi korundu, teslim icin ayni)

Kullanim:
    poetry run python -m src.models.explain
    make explain
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, PROCESSED_DIR
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model


class GradCAM:
    """Minimal GradCAM implementasyonu."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1)

        # Batch icin: her ornek kendi class_idx skorunu al, sonra topla
        # out: (B, C), class_idx: (B,)
        scores = out.gather(1, class_idx.view(-1, 1)).squeeze(1)  # (B,)
        loss = scores.sum()
        loss.backward(retain_graph=False)

        act = self.activations
        grad = self.gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1)
        cam = F.relu(cam)

        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cmin = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cmax = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cmin) / (cmax - cmin + 1e-8)
        return cam


def find_target_layer(model):
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Target layer bulunamadi")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pt"))
    parser.add_argument("--n-samples", type=int, default=8)
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
    loader = DataLoader(test_ds, batch_size=args.n_samples, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    mask = labels >= 0
    images = images[mask]
    labels = labels[mask]
    print(f"GradCAM icin {len(images)} ornek")

    target_layer = find_target_layer(model)
    print(f"Target layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(images.to(device))
    print(f"CAM shape: {tuple(cam.shape)}")

    # (B, H', W') -> (B, 224, 224)
    cam_resized = (
        F.interpolate(cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
        .squeeze(1)
        .cpu()
        .numpy()
    )

    # 4 sutunlu grid
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
        axes[row, col].set_title(f"Gercek: {true_name}", fontsize=8)
        axes[row, col].axis("off")

        axes[row + 1, col].imshow(img)
        axes[row + 1, col].imshow(cam_resized[i], cmap="jet", alpha=0.5)
        axes[row + 1, col].set_title("GradCAM - model nereye bakiyor?", fontsize=8)
        axes[row + 1, col].axis("off")

    plt.tight_layout()
    out = MODELS_DIR / "shap_heatmap.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nGradCAM heatmap: {out}")


if __name__ == "__main__":
    main()
