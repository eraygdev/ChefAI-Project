"""ChefAI - Model konfigurasyonu."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# -------------------------------------------------------------------
# CLASSES — data/raw/ altindan dinamik okunur
# Sadece gercek sinif klasorleri (gizli/temp/silinen haric)
# -------------------------------------------------------------------
RAW_DIR = DATA_DIR / "raw"


def _discover_classes() -> list[str]:
    """data/raw/ altindaki sinif klasorlerini alfabetik dondur."""
    if not RAW_DIR.exists():
        return []
    return sorted(
        p.name
        for p in RAW_DIR.iterdir()
        if p.is_dir()
        and not p.name.startswith((".", "_"))  # .gitkeep, _tmp vb atla
        and p.name not in {"removed", "removed_v1", "suspicious", "reports"}
    )


CLASSES: list[str] = _discover_classes()
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = dict(enumerate(CLASSES))


@dataclass
class ModelConfig:
    model_name: str = "tf_efficientnetv2_b0.in1k"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    dropout: float = 0.2
    image_size: int = 224


@dataclass
class TrainConfig:
    # Data
    data_dir: Path = PROCESSED_DIR
    num_workers: int = 1  # RAM tasarrufu
    pin_memory: bool = False  # 7.5GB RAM'de pin_mem swap'e iter, kapat

    # Training
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 4  # efektif batch = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 12

    # Optimizations
    mixed_precision: bool = True

    # Paths
    output_dir: Path = MODELS_DIR
    checkpoint_name: str = "best_model.pt"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "chefai-food-classification"
    mlflow_run_name: str = ""

    # Seed
    seed: int = 42
