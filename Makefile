# ============================================================
# ChefAI - Makefile
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help install setup check test lint format clean docker-up docker-down docker-logs data-status

# ------------------------------------------------------------
# HELP
# ------------------------------------------------------------
help:  ## Show this help message
	@echo "ChefAI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
install:  ## Install dependencies with Poetry
	poetry install

setup: install  ## Full setup: install + pre-commit + docker
	poetry run pre-commit install
	@echo "Setup complete."

# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------
check:  ## Run full setup check (packages, CUDA, docker, data, git)
	@echo "=========================================="
	@echo "  ChefAI SETUP CHECK"
	@echo "=========================================="
	@echo ""
	@echo "[1] Poetry + venv"
	@poetry --version
	@poetry env info --path
	@echo ""
	@echo "[2] Python packages"
	@poetry run python scripts/check_packages.py
	@echo ""
	@echo "[3] CUDA"
	@poetry run python -c "import torch; print('  torch:', torch.__version__); print('  cuda:', torch.cuda.is_available()); print('  gpu:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
	@echo ""
	@echo "[4] pre-commit"
	@test -f .git/hooks/pre-commit && echo "  OK   hook installed" || echo "  FAIL hook missing"
	@echo ""
	@echo "[5] Docker services"
	@cd docker && docker compose ps
	@echo ""
	@echo "[6] Git"
	@echo "  branch: $$(git branch --show-current)"
	@echo "  remote: $$(git remote get-url origin 2>/dev/null)"
	@echo "  commits: $$(git rev-list --count HEAD)"
	@echo ""
	@echo "[7] Data"
	@echo "  raw classes: $$(find data/raw -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)"
	@echo "  raw images:  $$(find data/raw -type f 2>/dev/null | wc -l)"
	@echo "  processed:   $$(find data/processed -type f 2>/dev/null | wc -l)"
	@echo ""
	@echo "=========================================="
	@echo "  Check complete."
	@echo "=========================================="

# ------------------------------------------------------------
# CODE QUALITY
# ------------------------------------------------------------
lint:  ## Run ruff linter
	poetry run ruff check src/ tests/

format:  ## Format code with ruff
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

typecheck:  ## Run mypy type checker
	poetry run mypy src/

precommit:  ## Run all pre-commit hooks
	poetry run pre-commit run --all-files

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
test:  ## Run pytest
	poetry run pytest tests/ -v

test-cov:  ## Run pytest with coverage
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

# ------------------------------------------------------------
# DOCKER
# ------------------------------------------------------------
docker-up:  ## Start Docker services (MLflow, MinIO, PostgreSQL)
	cd docker && docker compose up -d

docker-down:  ## Stop Docker services
	cd docker && docker compose down

docker-logs:  ## Show Docker service logs
	cd docker && docker compose logs -f

docker-ps:  ## Show Docker service status
	cd docker && docker compose ps

docker-full:  ## Start full stack (API + monitoring + mlops)
	cd docker && docker compose up -d

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
data-status:  ## Show data status report
	poetry run python src/data/report_status.py

data-pipeline:  ## Run full data pipeline (move_broken -> split)
	poetry run python src/data/move_broken.py
	poetry run python src/data/remove_duplicates.py
	poetry run python src/data/trim_to_extradata.py
	poetry run python src/data/split_dataset.py

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------
clean:  ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean  ## Remove cache + venv + docker volumes
	rm -rf .venv
	cd docker && docker compose down -v
	@echo "Full clean done."
