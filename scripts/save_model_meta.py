"""Best model metadata'sini JSON olarak kaydeder."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch

CKPT = Path("models/champion_08717.pt")
OUT = Path("models/champion_08717_meta.json")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()[:16]

meta = {
    "checkpoint_file": str(CKPT),
    "checkpoint_size_mb": round(CKPT.stat().st_size / 1024 / 1024, 2),
    "checkpoint_sha256_16": sha,
    "saved_at": datetime.now().isoformat(),
    "training": {
        "run_name": "painted-swan-832",
        "run_id": "a1fb96e8d9da4d95bfe8165636ba6dbb",
        "mlflow_url": "http://localhost:5000/#/experiments/1/runs/a1fb96e8d9da4d95bfe8165636ba6dbb",
        "best_epoch": int(ckpt["epoch"]) + 1,
        "best_val_acc": float(ckpt["val_acc"]),
        "num_classes": int(ckpt["num_classes"]),
        "total_epochs": 35,
        "early_stopping": "not triggered",
        "hyperparams": {
            "model": "tf_efficientnetv2_b0.in1k",
            "batch_size": 8,
            "grad_accum_steps": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "freeze_epochs": 3,
            "backbone_lr_after_unfreeze": 1e-4,
            "head_lr_after_unfreeze": 1e-3,
            "bn_frozen": 59,
            "mixed_precision": True,
            "label_smoothing": 0.1,
            "mixup": True,
            "ema": True,
            "scheduler": "CosineAnnealingLR T_max=35",
            "patience": 12,
            "seed": 42,
        },
        "data": {
            "train_samples": 4800,
            "val_samples": 600,
            "test_samples": 600,
            "classes": 30,
            "split_ratio": "80/10/10",
        },
    },
    "evaluation": {
        "val_acc": 0.8717,
        "test_acc": 0.8650,
        "macro_f1": 0.86,
        "weighted_f1": 0.86,
        "tta_views": 4,
        "confusion_matrix": "models/confusion_matrix.png",
    },
    "weak_classes": {
        "tavuk_sote": {"recall": 0.60, "confused_with": "izgara_tavuk"},
        "imam_bayildi": {"recall": 0.60, "confused_with": "karniyarik"},
        "pide": {"recall": 0.65, "confused_with": "pizza"},
        "mercimek_corbasi": {"recall": 0.75, "confused_with": "ezogelin_corbasi"},
    },
    "known_issues": [
        "MLflow artifact upload failed (S3 credentials) - metrics saved, model file local only",
        "MLflow log_model bug: final epoch model logged, not best epoch",
    ],
}

OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"Yazildi: {OUT}")
print(json.dumps(meta, indent=2, ensure_ascii=False))
