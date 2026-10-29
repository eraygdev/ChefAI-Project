"""ChefAI - Optuna hiperparametre optimizasyonu (MX330 uyumlu)."""

from __future__ import annotations

import argparse
import time

import mlflow
import optuna
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, MODELS_DIR, TrainConfig
from src.models.dataset import FoodDataset, get_eval_transforms, get_train_transforms
from src.models.model import build_model
from src.models.train import evaluate, set_seed, train_one_epoch

HPO_EPOCHS = 5


def objective(trial: optuna.Trial, cfg: TrainConfig, device: torch.device) -> float:
    # Batch size 2 GB VRAM icin [4, 6, 8] ile sinirli; AMP ile 8 calisir.
    lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    set_seed(cfg.seed)
    train_ds = FoodDataset(cfg.data_dir, "train", CLASS_TO_IDX, get_train_transforms())
    val_ds = FoodDataset(cfg.data_dir, "val", CLASS_TO_IDX, get_eval_transforms())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(pretrained=True, dropout=dropout).to(device)
    # Ana egitimle ayni kayip: label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS)
    # MX330 icin AMP zorunlu
    scaler = GradScaler(device.type) if device.type == "cuda" else None

    best_val_acc = 0.0
    for epoch in range(HPO_EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, 1)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        best_val_acc = max(best_val_acc, val_acc)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow_experiment}-optuna")

    # mlflow.start_run exception durumunda zombie bırakmasın diye context manager
    run_name = f"optuna-hpo-{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(
            lambda t: objective(t, cfg, device),
            n_trials=args.n_trials,
            timeout=args.timeout,
        )

        print(f"\nEn iyi trial: {study.best_trial.number}")
        print(f"En iyi val_acc: {study.best_value:.4f}")
        print("En iyi parametreler:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_acc", study.best_value)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_config_path = MODELS_DIR / "best_hyperparams.txt"
        with open(best_config_path, "w") as f:
            f.write(f"# Optuna en iyi trial: {study.best_trial.number}\n")
            f.write(f"# En iyi val_acc: {study.best_value:.4f}\n")
            f.write(f"# HPO sureci: {HPO_EPOCHS} epoch, label_smoothing=0.1\n")
            f.write("# NOT: ana egitim 35 epoch + warmup + EMA kullanir;")
            f.write(" LR/WD'yi kucuk farklarla kendiniz de deneyin.\n")
            for k, v in study.best_params.items():
                f.write(f"{k} = {v}\n")
        print(f"\nEn iyi parametreler: {best_config_path}")

        mlflow.log_artifact(str(best_config_path), artifact_path="hyperparams")


if __name__ == "__main__":
    main()
