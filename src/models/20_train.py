"""ChefAI model training script with MLflow tracking, AMP, EMA, and SWA.

Usage:
    poetry run python -m src.models.20_train --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import time
import warnings

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, ModelConfig, TrainConfig
from src.models.dataset import FoodDataset, get_eval_transforms, get_train_transforms
from src.models.engine import evaluate, set_seed, train_one_epoch
from src.models.model import build_model


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*epoch parameter.*scheduler\.step.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(description="Train ChefAI classification model.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-mixup", action="store_true", help="Disable MixUp and CutMix")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA model tracking")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=3,
        help="Freeze backbone for initial N epochs (0 disables)",
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
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Model : {model_cfg.model_name}")
    print(f"MixUp : {'ON' if use_mixup else 'OFF'} | EMA: {'ON' if use_ema else 'OFF'}")

    train_ds = FoodDataset(cfg.data_dir, "train", CLASS_TO_IDX, get_train_transforms())
    val_ds = FoodDataset(cfg.data_dir, "val", CLASS_TO_IDX, get_eval_transforms())

    persistent = cfg.num_workers > 0
    prefetch = 2 if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | workers={cfg.num_workers}")

    model = build_model(
        model_name=model_cfg.model_name,
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        dropout=model_cfg.dropout,
    ).to(device)

    ema_model: nn.Module | None = None
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

    freeze_epochs = int(args.freeze_epochs)
    if freeze_epochs > 0:
        try:
            head = model.get_classifier()  # type: ignore[operator]
            head_param_ids = {id(p) for p in head.parameters()}
        except AttributeError:
            prefixes = ("classifier", "head", "fc")
            head_param_ids = {id(p) for n, p in model.named_parameters() if n.startswith(prefixes)}

        frozen = trainable = 0
        for _, p in model.named_parameters():
            if id(p) in head_param_ids:
                p.requires_grad = True
                trainable += 1
            else:
                p.requires_grad = False
                frozen += 1
        print(f"Backbone frozen: {frozen} params, {trainable} head params ({freeze_epochs} epochs)")

    scheduler: LRScheduler
    if freeze_epochs > 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
        print(f"LR schedule: Cosine (T_max={cfg.epochs})")
    else:
        warmup_epochs = max(1, min(2, cfg.epochs // 10))
        warmup_sched = LinearLR(optimizer, start_factor=0.5, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup_epochs, 1))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(f"LR schedule: Warmup({warmup_epochs}) + Cosine")

    if not args.no_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run(run_name=cfg.mlflow_run_name or None)
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
                "freeze_epochs": freeze_epochs,
            }
        )

    best_val_acc = 0.0
    best_train_acc = 0.0
    patience_counter = 0
    global_step = 0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / cfg.checkpoint_name

    swa_start = cfg.epochs - 10
    swa_model: AveragedModel | None = None

    for epoch in range(cfg.epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for p in model.parameters():
                p.requires_grad = True
            try:
                head_params = list(model.get_classifier().parameters())  # type: ignore[operator]
                head_ids = {id(p) for p in head_params}
            except AttributeError:
                head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]
                head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": cfg.learning_rate * 0.1},
                    {"params": head_params, "lr": cfg.learning_rate},
                ],
                weight_decay=cfg.weight_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - epoch, 1))

            bn_frozen = 0
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
                    module.eval()
                    module.momentum = 0.0
                    bn_frozen += 1
            print(
                f"Backbone unfrozen (epoch {epoch + 1}): "
                f"backbone_lr={cfg.learning_rate * 0.1:.1e}, "
                f"head_lr={cfg.learning_rate:.1e}, bn_frozen={bn_frozen}"
            )

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
            backbone_frozen=(freeze_epochs > 0 and epoch < freeze_epochs),
        )

        val_model = ema_model if (use_ema and ema_model is not None) else model
        val_loss, val_acc = evaluate(val_model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = float(optimizer.param_groups[0]["lr"])

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

        if epoch == swa_start:
            swa_model = AveragedModel(val_model)
            print(f"SWA started (epoch {epoch + 1})")
        elif epoch > swa_start and swa_model is not None:
            swa_model.update_parameters(val_model)

        is_improvement = val_acc > best_val_acc
        is_tie_break = (val_acc == best_val_acc) and (train_acc > best_train_acc)

        if is_improvement or is_tie_break:
            if is_improvement:
                best_val_acc = val_acc
                best_train_acc = train_acc
                patience_counter = 0
                print(f"  New best model: {best_path}")
            else:
                best_train_acc = train_acc
                patience_counter = 0
                print(f"  Tie-break best model (train_acc={train_acc:.4f})")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": val_model.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "num_classes": model_cfg.num_classes,
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping ({patience_counter} epochs without improvement)")
                break

    print(f"\nBest val_acc: {best_val_acc:.4f}")

    if swa_model is not None:
        print("Updating SWA BatchNorm statistics...")
        update_bn(train_loader, swa_model, device=device)
        _, swa_val_acc = evaluate(swa_model, val_loader, criterion, device)
        print(f"SWA val_acc: {swa_val_acc:.4f}")
        if swa_val_acc > best_val_acc:
            best_val_acc = swa_val_acc
            torch.save(
                {
                    "epoch": cfg.epochs - 1,
                    "model_state": swa_model.module.state_dict(),
                    "val_acc": swa_val_acc,
                    "train_acc": 0.0,
                    "num_classes": model_cfg.num_classes,
                },
                best_path,
            )
            print(f"SWA checkpoint saved: {best_path}")

    if not args.no_mlflow:
        try:
            log_model = ema_model if (use_ema and ema_model is not None) else model
            log_model.eval()
            mlflow.log_metric("best_val_acc", best_val_acc)
            try:
                mlflow.log_artifact(str(best_path), artifact_path="checkpoint")
                mlflow.pytorch.log_model(
                    log_model.cpu(),
                    "model",
                    registered_model_name=f"chefai-{model_cfg.model_name}",
                )
                print("MLflow artifact and model logged.")
            except Exception as e:
                print(f"WARN: MLflow artifact upload failed (local checkpoint safe): {e}")
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    main()
