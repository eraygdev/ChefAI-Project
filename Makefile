# ============================================================
# ChefAI - Makefile
# ============================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON  := poetry run python
COMPOSE := cd docker && docker compose

EPOCHS     ?= 50
BATCH_SIZE ?= 8
OUTPUT     ?= project_export.txt
N_TRIALS   ?= 5

.PHONY: help install setup check \
        lint format typecheck precommit test test-cov \
        up docker-up docker-down docker-logs docker-ps \
        clean-docker docker-clean-all \
        data-status data-pipeline data-move-broken data-move-suspicious data-remove-dups data-trim data-split data-renumber data-merge-move data-merge-dry \
        train train-quick train-bg logs stop \
        evaluate explain tune best-acc open-confusion open-shap \
        mlflow-ui mlflow-runs mlflow-promote \
        export export-clean \
        clean clean-all rebuild docker-up-monitor

# ------------------------------------------------------------
# HELP
# ------------------------------------------------------------
help:  ## Bu yardim mesajini goster
	@echo "ChefAI - Kullanilabilir komutlar:"
	@echo ""
	@echo "  KURULUM"
	@echo "    make install              Bagimliliklari kur"
	@echo "    make setup                Kurulum + pre-commit hook"
	@echo ""
	@echo "  KONTROL"
	@echo "    make check                Genel setup kontrolu"
	@echo "    make lint                 Ruff ile kod kontrolu"
	@echo "    make format               Ruff ile formatla"
	@echo "    make typecheck            mypy tip kontrolu"
	@echo "    make precommit            Tum pre-commit hook'lari"
	@echo "    make test                 pytest testleri"
	@echo "    make test-cov             pytest + coverage"
	@echo ""
	@echo "  DOCKER"
	@echo "    make up                   Stack baslat + MLflow UI ac"
	@echo "    make docker-up            Servisleri baslat (MLflow + MinIO + Postgres)"
	@echo "    make docker-up-monitor    Monitoring baslat (Prometheus + Grafana)"
	@echo "    make docker-down          Servisleri durdur"
	@echo "    make docker-logs          Docker log'lari izle"
	@echo "    make docker-ps            Servis durumu"
	@echo "    make clean-docker         Kullanilmayan image/container/cache sil"
	@echo "    make docker-clean-all     TUM Docker sil (TEHLIKE, onay ister)"
	@echo "    make rebuild              API rebuild + tum stack (API + monitoring)"
	@echo ""
	@echo "  DATA"
	@echo "    make data-status          Durum raporu"
	@echo "    make data-pipeline        Tum pipeline"
	@echo "    make data-move-broken     Bozuk/kucuk ayir"
	@echo "    make data-move-suspicious Filigran/menu supheli ayir (dry-run)"
	@echo "    make data-remove-dups     Duplicate sil"
	@echo "    make data-trim            Fazlaliklari tasi"
	@echo "    make data-split           Train/val/test bol"
	@echo "    make data-clear           processed'i sil (raw guvende)"
	@echo "    make data-clear-dry       processed silme on izleme"
	@echo "    make data-merge-move      processed -> raw TASI (DUPLICATE RISKI)"
	@echo "    make data-merge-dry       processed -> raw on izleme"
	@echo "    make data-renumber        Tum goruntuleri (raw + processed) 000001'den baslat"
	@echo ""
	@echo "  MODEL"
	@echo "    make train                Tam egitim (35 epoch)"
	@echo "    make train-quick          Hizli test (2 epoch)"
	@echo "    make train-bg             Arka planda egitim"
	@echo "    make logs                 Egitim log izle"
	@echo "    make stop                 Egitimi durdur"
	@echo "    make evaluate             Test degerlendirmesi"
	@echo "    make explain              SHAP aciklanabilirlik"
	@echo "    make tune                 Optuna hiperparametre"
	@echo "    make best-acc             En iyi dogruluk"
	@echo "    make open-confusion       Confusion matrix ac"
	@echo "    make open-shap            SHAP heatmap ac"
	@echo ""
	@echo "  MLFLOW"
	@echo "    make mlflow-ui            MLflow UI ac"
	@echo "    make mlflow-runs          Run'lari listele"
	@echo "    make mlflow-promote       Modeli Production'a al"
	@echo ""
	@echo "  EXPORT & CLEAN"
	@echo "    make export               Kodu export et"
	@echo "    make export-clean         Eski sil + yeni olustur"
	@echo "    make clean                Cache temizle"
	@echo "    make clean-all            Tam temizlik"
	@echo ""
	@echo "Degiskenler (ornek: make train EPOCHS=30 BATCH_SIZE=4):"
	@echo "  EPOCHS     = $(EPOCHS)"
	@echo "  BATCH_SIZE = $(BATCH_SIZE)"
	@echo "  OUTPUT     = $(OUTPUT)"
	@echo "  N_TRIALS   = $(N_TRIALS)"
	@echo ""

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
install:  ## Bagimliliklari kur (poetry install)
	poetry install

setup: install  ## Kurulum + pre-commit hook
	poetry run pre-commit install
	@echo "Setup tamamlandi."

# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------
check:  ## Genel setup kontrolu (paket, CUDA, docker, data, git)
	@echo "=========================================="
	@echo "  ChefAI SETUP CHECK"
	@echo "=========================================="
	@echo "[1] Poetry + venv"
	@poetry --version
	@poetry env info --path
	@echo ""
	@echo "[2] Python packages"
	@$(PYTHON) scripts/check_packages.py
	@echo ""
	@echo "[3] CUDA"
	@$(PYTHON) -c "import torch; print('  torch:', torch.__version__); print('  cuda:', torch.cuda.is_available()); print('  gpu:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
	@echo ""
	@echo "[4] pre-commit"
	@test -f .git/hooks/pre-commit && echo "  OK   hook installed" || echo "  FAIL hook missing"
	@echo ""
	@echo "[5] Docker services"
	@$(COMPOSE) ps
	@echo ""
	@echo "[6] Git"
	@echo "  branch: $$(git branch --show-current)"
	@echo "  commits: $$(git rev-list --count HEAD)"
	@echo ""
	@echo "[7] Data"
	@echo "  raw classes: $$(find data/raw -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)"
	@echo "  raw images:  $$(find data/raw -type f 2>/dev/null | wc -l)"
	@echo "  processed:   $$(find data/processed -type f 2>/dev/null | wc -l)"
	@echo ""
	@echo "=========================================="

# ------------------------------------------------------------
# CODE QUALITY
# ------------------------------------------------------------
lint:  ## Ruff ile kod kontrolu
	poetry run ruff check src/ tests/ scripts/

format:  ## Ruff ile kodu formatla
	poetry run ruff format src/ tests/ scripts/
	poetry run ruff check --fix src/ tests/ scripts/

typecheck:  ## mypy tip kontrolu
	poetry run mypy src/

precommit:  ## Tum pre-commit hook'lari calistir
	poetry run pre-commit run --all-files

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
test:  ## pytest testleri
	poetry run pytest tests/ -v

test-cov:  ## pytest + coverage
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

# ------------------------------------------------------------
# DOCKER
# ------------------------------------------------------------
docker-up:  ## Tum stack baslat (API + MLflow + MinIO + Postgres + Monitoring)
	$(COMPOSE) --profile app --profile monitor up -d
	@sleep 3
	@$(MAKE) --no-print-directory mlflow-ui
	@echo "Servisler hazir:"
	@echo "  MLflow    : http://localhost:5000"
	@echo "  MinIO     : http://localhost:9001"
	@echo "  Grafana   : http://localhost:3000"

docker-up:  ## Servisleri baslat (MLflow + MinIO + PostgreSQL)
	$(COMPOSE) up -d

docker-up-monitor:  ## Monitoring stack baslat (Prometheus + Grafana)
	cd docker && docker compose --profile monitor up -d
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana   : http://localhost:3000 (admin/admin)"

docker-down:  ## Servisleri durdur
	$(COMPOSE) down

docker-logs:  ## Docker log'larini canli izle
	$(COMPOSE) logs -f

docker-ps:  ## Docker servis durumu
	$(COMPOSE) ps

clean-docker:  ## Kullanilmayan Docker image/container/builder cache sil (filtresiz)
	@echo "Docker build cache siliniyor..."
	-docker builder prune -f
	@echo "Kullanilmayan image'lar siliniyor..."
	-docker image prune -a -f
	@echo "Durmus container'lar siliniyor..."
	-docker container prune -f
	@echo "Docker temizligi tamamlandi."
	@echo ""
	@echo "Not: Calisan container'lar ve volume'lar KORUNDU."
	@echo "  MLflow verisi : make docker-clean-all (onayli, TEHLIKELI)"

docker-clean-all:  ## TEHLIKE: Tum Docker container/image/volume/network sil
	@echo "UYARI: Sistemdeki TUM Docker verileri silinecek."
	@echo "Bu islem geri alinamaz."
	@read -p "Onaylamak icin 'yes' yaz: " c && [ "$$c" = "yes" ] || (echo "Iptal."; exit 1)
	-docker ps -aq | xargs -r docker rm -f
	-docker images -aq | xargs -r docker rmi -f
	-docker volume ls -q | xargs -r docker volume rm -f
	-docker network prune -f
	@echo "Tum Docker verileri silindi."

rebuild:  ## API image rebuild + tum stack (API + monitoring) baslat
	@echo "chefai-api image yeniden build ediliyor..."
	cd docker && docker compose build chefai-api
	@echo "Tum stack baslatiliyor (API + MLflow + MinIO + Postgres + Prometheus + Grafana)..."
	cd docker && docker compose --profile app --profile monitor up -d
	@echo "Rebuild tamam. Servisler:"
	@echo "  API        : http://localhost:8000/docs"
	@echo "  MLflow     : http://localhost:5000"
	@echo "  MinIO      : http://localhost:9001"
	@echo "  Prometheus : http://localhost:9090"
	@echo "  Grafana    : http://localhost:3000 (admin/admin)"


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
data-status:  ## Data durum raporu
	$(PYTHON) -m src.data.report_status

data-pipeline:  ## Tum data pipeline (suspicious -> broken -> split)
	$(PYTHON) -m src.data.move_suspicious
	$(PYTHON) -m src.data.move_broken
	$(PYTHON) -m src.data.remove_duplicates
	$(PYTHON) -m src.data.trim_to_extradata
	$(PYTHON) -m src.data.split_dataset

data-move-suspicious:  ## Filigran/menu karti supheli gorselleri ayir
	$(PYTHON) -m src.data.move_suspicious --dry-run
	@echo ""
	@echo "DRY RUN tamam. Gercekten tasimak icin:"
	@echo "  poetry run python -m src.data.move_suspicious"

data-move-broken:  ## Bozuk/kucuk gorselleri ayir
	$(PYTHON) -m src.data.move_broken

data-remove-dups:  ## Duplicate gorselleri sil
	$(PYTHON) -m src.data.remove_duplicates

data-trim:  ## Fazlaliklari extradata'ya tasi
	$(PYTHON) -m src.data.trim_to_extradata

data-split:  ## Train/val/test bol
	$(PYTHON) -m src.data.split_dataset

data-clear:  ## processed'i sil (raw guvende, temiz baslangic)
	$(PYTHON) -m src.data.clear_processed --yes

data-clear-dry:  ## processed silme on izleme (hicbir sey silinmez)
	$(PYTHON) -m src.data.clear_processed --dry-run

data-merge-move:  ## processed -> raw TASI (DUPLICATE RISKI, nadir kullanim)
	$(PYTHON) -m src.data.clear_processed --merge --move

data-merge-dry:  ## processed -> raw on izleme (hicbir sey yapmaz)
	$(PYTHON) -m src.data.clear_processed --merge --dry-run

data-renumber:  ## Tum goruntuleri (raw + processed) 000001... formatina cevir
	$(PYTHON) -m src.data.renumber_images

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------
train:
	$(PYTHON) -m src.models.train --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --freeze-epochs 3

train-quick:  ## Hizli test egitimi (2 epoch, MLflow yok, EMA kapali, freeze 1, MixUp kapali)
	poetry run python -m src.models.train --epochs 3 --freeze-epochs 1 --no-mlflow --no-ema --no-mixup

train-bg:  ## Egitimi arka planda baslat (train.log'a yazar)
	@nohup $(PYTHON) -m src.models.train --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --freeze-epochs 3 --no-mixup > train.log 2>&1 &
	@echo "Egitim arka planda basladi (PID: $$!)"
	@echo "Log dosyasi: train.log"
	@echo "Canli izle : make logs"
	@echo "Durdur     : make stop"

logs:  ## Egitim log'unu canli izle (Ctrl+C ile cik)
	@test -f train.log && tail -f train.log || echo "train.log bulunamadi (once 'make train-bg' calistir)"

stop:  ## Arka plandaki egitimi durdur
	@pkill -f "src.models.train" && echo "Egitim durduruldu." || echo "Calisan egitim yok."

evaluate:  ## Test set degerlendirmesi (confusion matrix + TTA)
	$(PYTHON) -m src.models.evaluate

explain:  ## GradCAM aciklanabilirlik (CPU - 2GB VRAM OOM yok)
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m src.models.explain --n-samples 8

tune:  ## Optuna hiperparametre (N_TRIALS=5)
	$(PYTHON) -m src.models.tune --n-trials $(N_TRIALS)

best-acc:  ## En iyi dogruluk orani (checkpoint + MLflow)
	@$(PYTHON) scripts/best_acc.py

open-confusion:  ## Confusion matrix gorselini ac
	@xdg-open models/confusion_matrix.png 2>/dev/null || echo "models/confusion_matrix.png yok (once 'make evaluate')"

open-shap:  ## SHAP heatmap gorselini ac
	@xdg-open models/shap_heatmap.png 2>/dev/null || echo "models/shap_heatmap.png yok (once 'make explain')"

# ------------------------------------------------------------
# MLFLOW
# ------------------------------------------------------------
mlflow-ui:  ## MLflow UI'yi tarayicida ac (localhost:5000)
	@$(PYTHON) -c "import webbrowser; webbrowser.open('http://localhost:5000')" 2>/dev/null || \
	 xdg-open http://localhost:5000 2>/dev/null || \
	 sensible-browser http://localhost:5000 2>/dev/null || \
	 (echo "Manuel ac: http://localhost:5000")

mlflow-runs:  ## Son MLflow run'larini listele
	@$(PYTHON) scripts/best_acc.py

mlflow-promote:  ## En iyi modeli Production stage'e al
	@$(PYTHON) -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://localhost:5000'); name='chefai-tf_efficientnetv2_b0.in1k'; vs=c.search_model_versions(f'name={name}'); v=max(vs, key=lambda x: int(x.version)); c.transition_model_version_stage(name, v.version, 'Production'); print(f'Model {name} v{v.version} -> Production')" 2>/dev/null || echo "MLflow erisilemedi veya model yok"

# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------
export:  ## Tum kodu tek dosyaya export et (OUTPUT=project_export.txt)
	@$(PYTHON) export.py -o $(OUTPUT)
	@echo "Export: $(OUTPUT)"

export-clean:  ## Eski export'u sil + yenisini olustur
	@rm -f $(OUTPUT)
	@$(PYTHON) export.py -o $(OUTPUT)
	@echo "Export: $(OUTPUT)"

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------
clean:  ## Python cache temizle
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache temizlendi."

clean-all: clean  ## Cache + proje Docker image + log temizle (volume'lar KORUNUR)
	@echo "Proje image'lari siliniyor..."
	-docker rmi -f chefai-api:latest 2>/dev/null || true
	@echo "Docker build cache temizleniyor..."
	-docker builder prune -f 2>/dev/null || true
	@echo "Log temizleniyor..."
	rm -f train.log
	@echo "Cache + image + log temizlendi (MLflow volume'lari KORUNDU)."
