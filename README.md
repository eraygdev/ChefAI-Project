# ChefAI

ChefAI is an end-to-end MLOps food recognition and allergen warning system.
Users upload a food image, and the system predicts the dish class, checks it
against user-defined allergens, and returns potential risk warnings.

The project covers the full MLOps lifecycle: environment setup, data
cleaning and versioning, model training and tuning, API deployment, and
service monitoring.

## Architecture Overview

| Layer | Technology |
| --- | --- |
| Deep Learning | PyTorch, torchvision, timm (`tf_efficientnetv2_b0.in1k`) |
| Data Pipeline | OpenCV, Pillow, icrawler, DVC |
| Experiment Tracking | MLflow, PostgreSQL, MinIO (S3-compatible storage) |
| Hyperparameter Tuning | Optuna (`MedianPruner`) |
| Explainability | GradCAM attribution heatmaps |
| API Serving | FastAPI, Gunicorn, Uvicorn, Pydantic |
| Monitoring | Prometheus, Grafana (`prometheus-fastapi-instrumentator`) |
| Code Quality & CI | Poetry, Ruff, Mypy, Pytest, pre-commit, GitHub Actions |

## Key Technical Features

- **Dynamic Class Discovery:** Classes are automatically discovered from
  subdirectories under `data/raw/`, allowing the dataset to scale or change
  without hardcoding class lists.
- **Configurable Training Engine:** Supports configurable batch size, gradient
  accumulation, AMP mixed precision, Label Smoothing, AdamW, Cosine / Warmup
  LR schedules, MixUp/CutMix, EMA, SWA, and early stopping.
- **Evaluation & Explainability:** Includes 4-view Test-Time Augmentation
  (TTA), per-class classification metrics, confusion matrix export, and
  GradCAM visualizations.
- **Standalone API Container:** Uses a lightweight CPU-only PyTorch Docker
  image for inference without external base-image dependencies.

## Prerequisites

- Python `3.12` (`>=3.12,<3.13`)
- Poetry `2.x`
- Docker and Docker Compose

## Installation

```bash
poetry install
poetry run pre-commit install
```

## Project Structure

```text
MLOps_Project/
├── .github/workflows/    CI pipeline (lint, typecheck, test, docker build)
├── data/                 Data directories (raw, processed, staging folders)
├── docker/               Standalone API Dockerfile and Docker Compose stack
├── docs/                 Pipeline documentation
├── models/               Saved checkpoints, plots, and hyperparameter reports
├── monitoring/           Prometheus scrape configuration
├── scripts/              Helper inspection and metadata scripts
├── src/
│   ├── utils.py          Shared paths, file discovery, size, and safe move/copy helpers
│   ├── data/             Numbered data pipeline scripts (01-11)
│   ├── models/           Core modules (config, dataset, model, engine) and scripts (20-23)
│   └── api/              FastAPI service (prediction, allergen check, metrics)
├── tests/                Pytest test suite
├── export.py             Source code exporter
└── pyproject.toml        Dependencies and tool configuration
```

## Docker Services

All compose commands can be run from the `docker/` directory.

### Default Stack (MLflow + MinIO + PostgreSQL)

```bash
cd docker && docker compose up -d
```

### Full Stack (Default + API + Monitoring)

```bash
cd docker && docker compose --profile app --profile monitor up -d
```

### Monitoring Stack Only (Prometheus + Grafana)

```bash
cd docker && docker compose --profile monitor up -d
```

### Stop Services

```bash
cd docker && docker compose down
```

| Service | URL | Default Credentials |
| --- | --- | --- |
| API Docs | `http://localhost:8000/docs` | - |
| MLflow UI | `http://localhost:5000` | - |
| MinIO Console | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| PostgreSQL | `localhost:5432` | `mlflow` / `mlflow` |
| Prometheus | `http://localhost:9090` | - |
| Grafana | `http://localhost:3000` | `admin` / `admin` |

## Data Pipeline

All commands are executed from the project root.

| Step | Script | Purpose |
| --- | --- | --- |
| `01` | `01_download.py` | Download missing class images using multi-engine web crawling |
| `02` | `02_filter_suspicious.py` | Detect watermark and menu-card candidates via Canny edges and Hough lines |
| `03` | `03_filter_broken.py` | Move corrupted or undersized images to `data/brokendata/` |
| `04` | `04_deduplicate.py` | Remove exact pixel-level duplicates using RGB MD5 hashes |
| `05` | `05_trim.py` | Move excess images per class to `data/extradata/` |
| `06` | `06_split.py` | Stratified `80% / 10% / 10%` split into `data/processed/{train,val,test}` |
| `07` | `07_renumber.py` | Rename images sequentially (`000001`, `000002`, ...) |
| `10` | `10_status.py` | Print image counts and disk usage across all data folders |
| `11` | `11_manage.py` | Restore quarantined images or clean staging directories |

### Common Data Commands

```bash
# Check current dataset status
poetry run python -m src.data.10_status

# Preview suspicious images (dry-run by default)
poetry run python -m src.data.02_filter_suspicious --preview
poetry run python -m src.data.02_filter_suspicious --move

# Filter broken/small images and remove duplicates
poetry run python -m src.data.03_filter_broken
poetry run python -m src.data.04_deduplicate

# Trim class counts (configurable target) and renumber
poetry run python -m src.data.05_trim --target 200
poetry run python -m src.data.07_renumber --raw-only

# Create stratified train/val/test split
poetry run python -m src.data.06_split

# Restore quarantined data or clear processed splits
poetry run python -m src.data.11_manage restore broken
poetry run python -m src.data.11_manage restore extra
poetry run python -m src.data.11_manage clear processed
```

## Model Pipeline

| Step | Script | Purpose |
| --- | --- | --- |
| `20` | `20_train.py` | Train model with MLflow logging, AMP, MixUp/CutMix, EMA, and SWA |
| `21` | `21_evaluate.py` | Evaluate test split using 4-view TTA and save `confusion_matrix.png` |
| `22` | `22_explain.py` | Generate GradCAM attribution heatmaps (`shap_heatmap.png`) |
| `23` | `23_tune.py` | Run Optuna hyperparameter search and log results to MLflow |

### Training Examples

```bash
# Quick verification run without MLflow
poetry run python -m src.models.20_train --epochs 2 --no-mlflow

# Custom training parameters
poetry run python -m src.models.20_train --epochs 35 --batch-size 8 --lr 1e-3

# Disable optional regularizers
poetry run python -m src.models.20_train --no-mixup --no-ema
```

### Evaluation, Explainability, and Tuning

```bash
# Test-set evaluation + confusion matrix
poetry run python -m src.models.21_evaluate

# GradCAM explainability (CPU mode supported)
CUDA_VISIBLE_DEVICES="" poetry run python -m src.models.22_explain --n-samples 8

# Optuna hyperparameter tuning
poetry run python -m src.models.23_tune --n-trials 5 --timeout 3600

# Inspect best local checkpoint and top MLflow runs
poetry run python scripts/best_acc.py
```

## API Endpoints

The FastAPI service (`src/api/main.py`) loads the checkpoint defined by
`MODEL_PATH` (default: `models/best_model.pt`).

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Service status, loaded device, and model state |
| `/predict` | `POST` | Upload an image to get top-1 prediction and top-5 probabilities |
| `/check-allergy` | `POST` | Upload an image with comma-separated allergens to receive safety warnings |
| `/metrics` | `GET` | Prometheus metrics endpoint |
| `/docs` | `GET` | Interactive OpenAPI (Swagger) documentation |

Supported allergen keys: `gluten`, `sut`, `yumurta`, `et`, `tavuk`,
`avokado`, `nohut`, `balik`, `findik`.

## Code Quality and Testing

```bash
# Lint and format checks
poetry run ruff check src/ tests/ scripts/
poetry run ruff format --check src/ tests/ scripts/

# Static type checking
poetry run mypy src/

# Run tests with coverage
poetry run pytest tests/ -v

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on push and pull
requests targeting `main`:

1. **lint-and-test:** Executes `ruff check`, `ruff format --check`, `mypy`,
   and `pytest`.
2. **docker-build:** Verifies that `docker/Dockerfile` builds cleanly.

## Export

Export all text-based project source files into a single file:

```bash
poetry run python export.py -o project_export.txt
```

## Authors

Eray, Burak

## License

MIT License (see [LICENSE](LICENSE))
