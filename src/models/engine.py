"""Shared training and evaluation loops, MixUp/CutMix, and EMA helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    use_mixup: bool = True,
    use_cutmix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, float] | None]:
    """Apply MixUp or CutMix randomly (50/50 when both are enabled)."""
    if not use_mixup and not use_cutmix:
        return images, labels, None

    use_cut = use_cutmix and (not use_mixup or np.random.rand() < 0.5)

    if use_cut:
        lam = float(np.random.beta(1.0, 1.0))
        b, _, h, w = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))
        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))
        index = torch.randperm(b, device=images.device)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        return mixed, labels, (labels[index], lam)

    lam = float(np.random.beta(0.2, 0.2))
    b = images.size(0)
    index = torch.randperm(b, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, (labels[index], lam)


def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    global_step: int,
    decay_max: float = 0.999,
) -> None:
    """Update exponential moving average weights with a smooth warmup decay."""
    decay = min(decay_max, 0.9 + 0.099 * min(1.0, global_step / 500.0))
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters(), strict=False):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers(), strict=False):
            ema_b.data.copy_(b.data)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    accum_steps: int,
    use_mixup: bool = True,
    ema_model: nn.Module | None = None,
    global_step: int = 0,
    backbone_frozen: bool = False,
) -> tuple[float, float, int]:
    """Run one training epoch with optional AMP, MixUp/CutMix, and EMA updates."""
    model.train()
    if backbone_frozen:
        try:
            head_module = model.get_classifier()  # type: ignore[operator]
            head_module_ids = {id(m) for m in head_module.modules()}
        except AttributeError:
            head_module_ids = {
                id(m)
                for n, m in model.named_modules()
                if n.startswith(("classifier", "head", "fc"))
            }
        for _, module in model.named_modules():
            if id(module) not in head_module_ids:
                module.eval()
    else:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    correct = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)

    for step, (images, labels) in enumerate(loader):
        mask = labels >= 0
        if not mask.any():
            continue
        images = images[mask].to(device, non_blocking=True)
        labels = labels[mask].to(device, non_blocking=True)

        if use_mixup:
            mixed_images, _, mix_info = mix_batch(images, labels, use_mixup=True, use_cutmix=True)
        else:
            mixed_images, mix_info = images, None

        with autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(mixed_images)
            if mix_info is not None:
                labels_b, lam = mix_info
                loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last = (step + 1) == num_batches
        if (step + 1) % accum_steps == 0 or is_last:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema_model is not None:
                update_ema(ema_model, model, global_step=global_step)
                global_step += 1

        total_loss += float(loss.item()) * accum_steps * images.size(0)
        preds = logits.argmax(dim=1)
        if mix_info is not None:
            labels_b, lam = mix_info
            correct += float(
                (lam * (preds == labels).float() + (1.0 - lam) * (preds == labels_b).float())
                .sum()
                .item()
            )
        else:
            correct += float((preds == labels).sum().item())
        total += int(labels.size(0))

    return total_loss / max(total, 1), correct / max(total, 1), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model loss and top-1 accuracy over a validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        mask = labels >= 0
        if not mask.any():
            continue
        images = images[mask].to(device, non_blocking=True)
        labels = labels[mask].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)
