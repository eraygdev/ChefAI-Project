# ChefAI

Food recognition + allergy warning system.

## Kurulum

```bash
poetry install
```

## Proje Yapisi

- `data/raw/` - Ham goruntuler
- `data/processed/` - Train/val/test
- `src/data/` - Veri pipeline
- `src/models/` - Model egitimi
- `src/api/` - REST API
- `tests/` - Testler
- `docker/` - Containerization
- `.github/workflows/` - CI/CD

## Pipeline

1. Setup (Docker, Poetry, pre-commit)
2. Data (DVC, Validation)
3. Model (MLflow, Optuna, SHAP)
4. Deploy (FastAPI, Docker, GitHub Actions)
5. Monitor (Evidently, Prometheus, Grafana)
