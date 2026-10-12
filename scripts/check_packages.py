"""Check that all required packages are installed."""

import importlib.util

PACKAGES = [
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "PIL",
    "pandas",
    "sklearn",
    "fastapi",
    "uvicorn",
    "gunicorn",
    "pydantic",
    "mlflow",
    "optuna",
    "shap",
    "dvc",
    "great_expectations",
    "evidently",
    "prometheus_client",
    "pre_commit",
    "ruff",
    "mypy",
    "pytest",
    "icrawler",
    "matplotlib",
    "seaborn",
]

missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]

if missing:
    for p in missing:
        print(f"  FAIL {p}")
    print(f"  {len(PACKAGES) - len(missing)}/{len(PACKAGES)} packages installed")
else:
    print(f"  OK   {len(PACKAGES)}/{len(PACKAGES)} packages installed")
