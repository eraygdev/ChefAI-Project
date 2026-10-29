"""ChefAI - Optuna hiperparametre optimizasyonu."""

from __future__ import annotations

import argparse

import mlflow
import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, MODELS_DIR, TrainConfig
from src.models.dataset import FoodDataset, get_eval_transforms, get_train_transforms
from src.models.model import build_model
from src.models.train import evaluate, set_seed, train_one_epoch


def objective(trial: optuna.Trial, cfg: TrainConfig, device: torch.device) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    set_seed(cfg.seed)
    train_ds = FoodDataset(cfg.data_dir, "train", CLASS_TO_IDX, get_train_transforms())
    val_ds = FoodDataset(cfg.data_dir, "val", CLASS_TO_IDX, get_eval_transforms())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(pretrained=True, dropout=dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(5):
        train_one_epoch(model, train_loader, criterion, optimizer, None, device, 1)
        _, val_acc = evaluate(model, val_loader, criterion, device)
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

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow_experiment}-optuna")
    mlflow.start_run(run_name="optuna-hpo")

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

    # En iyi parametreleri dosyaya da yaz
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_config_path = MODELS_DIR / "best_hyperparams.txt"
    with open(best_config_path, "w") as f:
        f.write(f"# Optuna en iyi trial: {study.best_trial.number}\n")
        f.write(f"# En iyi val_acc: {study.best_value:.4f}\n")
        for k, v in study.best_params.items():
            f.write(f"{k} = {v}\n")
    print(f"\nEn iyi parametreler: {best_config_path}")

    mlflow.log_artifact(str(best_config_path), artifact_path="hyperparams")
    mlflow.end_run()


if __name__ == "__main__":
    main()
