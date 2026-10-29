"""ChefAI - Egitim scripti (MLflow + AMP + MixUp + EMA + Label Smoothing)."""

from __future__ import annotations

import argparse
import random
import time
import warnings

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, ModelConfig, TrainConfig
from src.models.dataset import FoodDataset, get_eval_transforms, get_train_transforms
from src.models.model import build_model


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_batch(images, labels, alpha=0.2):
    """MixUp: iki goruntuyu karistirir. alpha<=0 ise devre disi."""
    if alpha <= 0:
        return images, labels, None
    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, (labels[index], lam)


def update_ema(ema_model, model, global_step, decay_max=0.999):
    """
    EMA with warmup: decay baslangicta dusuk, zamanla decay_max'a yaklasir.
    Formul (MoCo/BYOL standardi): decay = min(decay_max, (1 + step) / (10 + step))
    """
    decay = min(decay_max, (1.0 + global_step) / (10.0 + global_step))
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters(), strict=False):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers(), strict=False):
            ema_b.data.copy_(b.data)


# -------------------------------------------------------------------
# TRAIN / EVAL
# -------------------------------------------------------------------
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accum_steps,
    use_mixup=True,
    ema_model=None,
    global_step=0,
    backbone_frozen=False,
):
    model.train()
    # Freeze sirasinda backbone'u eval() modunda tut -> BN running stats korunur
    if backbone_frozen:
        for name, module in model.named_modules():
            if name and not name.startswith("classifier"):
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
            mixed_images, _, mix_info = mixup_batch(images, labels, alpha=0.2)
        else:
            mixed_images, mix_info = images, None

        with autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(mixed_images)

            if mix_info is not None:
                labels_b, lam = mix_info
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
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

        total_loss += loss.item() * accum_steps * images.size(0)

        preds = logits.argmax(dim=1)
        if mix_info is not None:
            labels_b, lam = mix_info
            correct += (
                (lam * (preds == labels).float() + (1 - lam) * (preds == labels_b).float())
                .sum()
                .item()
            )
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    # PyTorch LR scheduler deprecation uyarisini sustur (SequentialLR internal)
    warnings.filterwarnings(
        "ignore",
        message=r".*epoch parameter.*scheduler\.step.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-mixup", action="store_true", help="MixUp devre disi")
    parser.add_argument("--no-ema", action="store_true", help="EMA devre disi")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=1,
        help="Ilk N epoch backbone dondur (0=kapali, default=1)",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    model_cfg = ModelConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    use_mixup = not args.no_mixup
    use_ema = not args.no_ema

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Model: {model_cfg.model_name}")
    print(f"MixUp: {'ON' if use_mixup else 'OFF'} | EMA: {'ON' if use_ema else 'OFF'}")

    # Data (persistent_workers + prefetch_factor hiz icin)
    train_ds = FoodDataset(cfg.data_dir, "train", CLASS_TO_IDX, get_train_transforms())
    val_ds = FoodDataset(cfg.data_dir, "val", CLASS_TO_IDX, get_eval_transforms())
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.num_workers > 0,
        "prefetch_factor": 2 if cfg.num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    print(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
        f"workers={cfg.num_workers} pin_mem={cfg.pin_memory}"
    )

    # Model
    model = build_model(
        model_name=model_cfg.model_name,
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        dropout=model_cfg.dropout,
    ).to(device)

    # EMA model (kopya)
    ema_model = None
    if use_ema:
        ema_model = build_model(
            model_name=model_cfg.model_name,
            num_classes=model_cfg.num_classes,
            pretrained=False,
            dropout=model_cfg.dropout,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = GradScaler(device.type) if cfg.mixed_precision and device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 1) FREEZE (once tanimla — scheduler buna bagli)
    # ------------------------------------------------------------------
    FREEZE_EPOCHS = args.freeze_epochs
    if FREEZE_EPOCHS > 0:
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
                frozen += 1
            else:
                trainable += 1
        print(
            f"Backbone donduruldu: {frozen} param, "
            f"{trainable} trainable head ({FREEZE_EPOCHS} epoch)"
        )
        if trainable == 0:
            print("UYARI: 'classifier' prefix'i eslesmedi! Son paramlar:")
            for n, _ in list(model.named_parameters())[-5:]:
                print(f"  {n}")

    # ------------------------------------------------------------------
    # 2) LR SCHEDULE (freeze ile cakismasin)
    #    - Freeze ON  -> sade Cosine (tam lr, head hizli ogrenir)
    #    - Freeze OFF -> Warmup + Cosine (backbone korunur)
    # ------------------------------------------------------------------
    if FREEZE_EPOCHS > 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
        print(f"LR schedule: Cosine (T_max={cfg.epochs}), warmup YOK (freeze aktif)")
    else:
        warmup_epochs = max(1, min(2, cfg.epochs // 10))
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup_epochs, 1))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(f"LR schedule: Warmup({warmup_epochs}) + Cosine (freeze kapali)")

    # MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run(run_name=cfg.mlflow_run_name or None)
        # Crash olsa bile run'i kapatmak icin finally'de end_run cagrilir (asagida)
        mlflow.log_params(
            {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "grad_accum_steps": cfg.grad_accum_steps,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "model": model_cfg.model_name,
                "mixed_precision": cfg.mixed_precision,
                "num_classes": model_cfg.num_classes,
                "label_smoothing": 0.1,
                "mixup": use_mixup,
                "ema": use_ema,
                "freeze_epochs": FREEZE_EPOCHS,
            }
        )

    best_val_acc = 0.0
    patience_counter = 0
    global_step = 0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / cfg.checkpoint_name

    for epoch in range(cfg.epochs):
        # Freeze bitince backbone'u ac
        if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS:
            for p in model.parameters():
                p.requires_grad = True
            model.train()  # BN stats tekrar guncellensin
            print(f"Backbone acildi (epoch {epoch + 1}'den itibaren tam fine-tune)")

        t0 = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg.grad_accum_steps,
            use_mixup=use_mixup,
            ema_model=ema_model if use_ema else None,
            global_step=global_step,
            backbone_frozen=(FREEZE_EPOCHS > 0 and epoch < FREEZE_EPOCHS),
        )

        val_model = ema_model if (use_ema and ema_model is not None) else model

        val_loss, val_acc = evaluate(val_model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if not args.no_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": lr,
                },
                step=epoch,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": val_model.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": model_cfg.num_classes,
                },
                best_path,
            )
            print(f"  Yeni en iyi model: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping ({patience_counter} epoch iyilesme yok)")
                break

    print(f"\nEn iyi val_acc: {best_val_acc:.4f}")

    if not args.no_mlflow:
        try:
            log_model = ema_model if (use_ema and ema_model is not None) else model
            log_model.eval()
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_artifact(str(best_path), artifact_path="checkpoint")
            log_model_cpu = log_model.cpu()
            mlflow.pytorch.log_model(
                log_model_cpu,
                "model",
                registered_model_name=f"chefai-{model_cfg.model_name}",
            )
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    main()
