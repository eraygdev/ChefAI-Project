"""
En iyi dogruluk oranlarini gosterir:
  1. Local checkpoint (models/best_model.pt)
  2. MLflow top runs

Kullanim:
    poetry run python scripts/best_acc.py
    veya
    make best-acc
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT = Path("models/best_model.pt")

print("=== Local checkpoint (models/best_model.pt) ===")
if CHECKPOINT.exists():
    try:
        import torch

        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        size_mb = CHECKPOINT.stat().st_size / 1024 / 1024
        acc = ckpt.get("val_acc", 0)
        epoch = ckpt.get("epoch", 0) + 1
        num_classes = ckpt.get("num_classes", "?")
        print(f"  Size        : {size_mb:.1f} MB")
        print(f"  Epoch       : {epoch}")
        print(f"  Num classes : {num_classes}")
        print(f"  Best val_acc: {acc:.4f} ({acc * 100:.2f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  NOT FOUND (henuz egitim yapilmamis)")

print()
print("=== MLflow top 5 runs ===")
try:
    from mlflow.tracking import MlflowClient

    client = MlflowClient("http://localhost:5000")
    exp = client.get_experiment_by_name("chefai-food-classification")
    if exp is None:
        print("  (experiment bulunamadi - henuz egitim yok)")
    else:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["metrics.best_val_acc DESC"],
            max_results=5,
        )
        if not runs:
            print("  (run bulunamadi)")
        else:
            for i, r in enumerate(runs, 1):
                acc = r.data.metrics.get("best_val_acc", 0)
                status = r.info.status
                rid = r.info.run_id[:8]
                print(f"  {i}. {rid} | best_val_acc={acc * 100:.2f}% | {status}")
except Exception:
    print("  MLflow erisilemedi (make docker-up calistir)")
