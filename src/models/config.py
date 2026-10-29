"""ChefAI - Model konfigurasyonu."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CLASSES: list[str] = [
    # Corbalar (4)
    "mercimek_corbasi",
    "ezogelin_corbasi",
    "domates_corbasi",
    "tavuk_suyu_corba",
    # Et/Tavuk (6)
    "izgara_tavuk",
    "kofte",
    "tavuk_sote",
    "etli_nohut",
    "karniyarik",
    "manti",
    # Sebze (5)
    "zeytinyagli_fasulye",
    "imam_bayildi",
    "menemen",
    "kisir",
    "firin_sebze",
    # Hamur (5)
    "pizza",
    "pide",
    "borek",
    "pogaca",
    "gozleme",
    # Pilav/Makarna (2)
    "pirinc_pilavi",
    "bulgur_pilavi",
    # Kahvaltilik (3)
    "omlet",
    "pankek",
    "avokado_tost",
    # Tatlilar (3)
    "kek",
    "kurabiye",
    "sutlac",
    # Ekstra (2)
    "patates_kizartmasi",
    "midye",
]

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
    num_workers: int = 2
    pin_memory: bool = False

    # Training
    epochs: int = 35
    batch_size: int = 8
    grad_accum_steps: int = 4  # efektif batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8

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
