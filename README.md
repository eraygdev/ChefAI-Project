# ChefAI

Akilli yemek analiz ve alerji uyari sistemi.

Kullanicinin cektigi yemek fotografi analiz edilerek yemek taninir, alerjen
profili ile karsilastirilir ve risk uyarisi verilir. Proje, MLOps yasam
dongusunun tamamini kapsar: Setup, Data, Model, Deploy, Monitor.

## Donanim ve Kisitlar

| Bilesen | Deger |
|---------|-------|
| GPU | NVIDIA GeForce MX330 (2 GB VRAM) |
| RAM | 7.5 GB (swap 6 GB) |
| OS | Linux Mint |
| Python | 3.12 |

Bu donanim nedeniyle: batch size 8 + gradient accumulation 4 (efektif 32),
AMP mixed precision, num_workers 2.

## Veri Seti

30 yemek sinifi, hedef 250 gorsel/sinif, toplam 7500 gorsel.

| Kategori | Siniflar | Adet |
|----------|----------|------|
| Corbalar | mercimek, ezogelin, domates, tavuk_suyu | 4 |
| Et/Tavuk | izgara_tavuk, kofte, tavuk_sote, etli_nohut, karniyarik, manti | 6 |
| Sebze | zeytinyagli_fasulye, imam_bayildi, menemen, kisir, firin_sebze | 5 |
| Hamur | pizza, pide, borek, pogaca, gozleme | 5 |
| Pilav/Makarna | pirinc_pilavi, bulgur_pilavi | 2 |
| Kahvaltilik | omlet, pankek, avokado_tost | 3 |
| Tatlilar | kek, kurabiye, sutlac | 3 |
| Ekstra | patates_kizartmasi, midye | 2 |

Bolme orani: %80 train, %10 val, %10 test (her sinif kendi icinde stratified).

## Model

| Bilesen | Deger |
|---------|-------|
| Mimari | EfficientNetV2-B0 (timm) |
| Framework | PyTorch 2.4.1 + CUDA 12.1 |
| Input | 224x224 RGB |
| Output | 30 sinif (softmax) |
| Loss | CrossEntropyLoss + Label Smoothing (0.1) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Epoch | 35 |
| Batch | 8 (grad_accum 4, efektif 32) |
| Augmentation | RandomResizedCrop, HorizontalFlip, Rotation, Affine, ColorJitter, Erasing |
| Regularization | MixUp (alpha=0.2), EMA (decay=0.999) |

## Pipeline

### 1. Setup

- Poetry (bagimlilik yonetimi)
- Docker + Docker Compose
- pre-commit (ruff, mypy, kalite kontrolleri)

### 2. Data

- DVC (veri versiyonlama)
- Gorsel indirme: icrawler (Google, Bing, Baidu, Flickr)
- Temizlik: bozuk, kucuk, duplicate, supheli (filigran/menu) ayiklama
- Otomatik yeniden numaralandirma (000001, 000002, ...)
- Train/val/test split

### 3. Model

- MLflow (deney takibi + model registry)
- Optuna (hiperparametre optimizasyonu)
- SHAP (aciklanabilirlik)
- Model Registry: Development, Staging, Production

### 4. Deploy

- FastAPI (REST servisi)
- Docker multi-stage build
- Gunicorn + Uvicorn (4 worker)
- GitHub Actions CI/CD

### 5. Monitor

- Evidently (data/model drift)
- Prometheus (metrik toplama)
- Grafana (gorsellestirme)

## Kurulum

### On Kosullar

- Python 3.12
- Poetry 2.x
- Docker + Docker Compose

### Adimlar

    poetry install
    poetry run pre-commit install

## Docker Servisleri

Tum komutlar `docker/` dizininden veya proje kokunden calisir.

### Varsayilan stack (MLflow + MinIO + PostgreSQL)

    cd docker && docker compose up -d

| Servis | URL | Kullanici |
|--------|-----|-----------|
| MLflow UI | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| PostgreSQL | localhost:5432 | mlflow / mlflow |

### Monitoring stack (Prometheus + Grafana)

    cd docker && docker compose --profile monitor up -d

| Servis | URL | Kullanici |
|--------|-----|-----------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

### Durdurma

    cd docker && docker compose down

### Temizlik

    # Proje container + volume (MLflow verisi silinir)
    cd docker && docker compose down -v

    # Kullanilmayan Docker image/container/cache (filtresiz)
    docker builder prune -f
    docker image prune -a -f
    docker container prune -f

## Veri Pipeline

Tum komutlar proje kokunden (`~/Masaustu/MLOps_Project`) calisir.

### Durum raporu

    poetry run python -m src.data.report_status

### Tam pipeline (suspicious -> broken -> duplicates -> trim -> split)

    poetry run python -m src.data.move_suspicious --dry-run
    poetry run python -m src.data.move_broken
    poetry run python -m src.data.remove_duplicates
    poetry run python -m src.data.trim_to_extradata
    poetry run python -m src.data.split_dataset

### Tek tek

    # Filigran/menu supheli gorseller (once dry-run)
    poetry run python -m src.data.move_suspicious --dry-run
    poetry run python -m src.data.move_suspicious

    # Bozuk/kucuk gorseller
    poetry run python -m src.data.move_broken

    # Duplicate gorseller (hash bazli)
    poetry run python -m src.data.remove_duplicates

    # Fazlaliklari extradata'ya tasi (250/sinif hedefi)
    poetry run python -m src.data.trim_to_extradata

    # Train/val/test bol
    poetry run python -m src.data.split_dataset

    # Yeniden numaralandir (000001, ...)
    poetry run python -m src.data.renumber_images

### Geri alma (yanlis karar verildiyse)

    # brokendata'dan raw'a geri tasi
    poetry run python src/data/restore_broken.py

    # processed'i raw'a birlestir (test/val/train -> raw)
    poetry run python -m src.data.merge_to_raw

## Model Egitimi

### 2 epoch hizli test (MLflow yok, VRAM/sure testi)

    poetry run python -m src.models.train --epochs 2 --no-mlflow

### Tam egitim (35 epoch, MLflow ile)

    poetry run python -m src.models.train --epochs 35 --batch-size 8

### Arka planda egitim

    nohup poetry run python -m src.models.train --epochs 35 > train.log 2>&1 &

    # Log izle (Ctrl+C ile cik, egitim devam eder)
    tail -f train.log

    # Durdur
    pkill -f "src.models.train"

### Parametre override

    poetry run python -m src.models.train --epochs 30 --batch-size 4 --lr 5e-5

### Opsiyonel kapatma

    # MixUp olmadan
    poetry run python -m src.models.train --epochs 35 --no-mixup

    # EMA olmadan
    poetry run python -m src.models.train --epochs 35 --no-ema

    # Ikisi de kapali
    poetry run python -m src.models.train --epochs 35 --no-mixup --no-ema

## Degerlendirme

### Test set + confusion matrix + TTA

    poetry run python -m src.models.evaluate

Cikti: `models/confusion_matrix.png` (sinif bazli rapor + heatmap).

### SHAP aciklanabilirlik

    poetry run python -m src.models.explain

Cikti: `models/shap_heatmap.png` (model her gorselin neresine bakiyor).

### Optuna hiperparametre optimizasyonu

    # 5 trial, 1 saat max
    poetry run python -m src.models.tune --n-trials 5 --timeout 3600

Cikti: `models/best_hyperparams.txt` + MLflow'da en iyi parametreler.

### En iyi dogruluk raporu

    poetry run python scripts/best_acc.py

Checkpoint val_acc + MLflow top 5 run gosterir.

## Kalite ve Test

    # Ruff lint
    poetry run ruff check src/ tests/ scripts/

    # Ruff format
    poetry run ruff format src/ tests/ scripts/
    poetry run ruff check --fix src/ tests/ scripts/

    # mypy tip kontrolu
    poetry run mypy src/

    # Tum pre-commit hook'lari
    poetry run pre-commit run --all-files

    # pytest
    poetry run pytest tests/ -v
    poetry run pytest tests/ -v --cov=src --cov-report=term-missing

## Export

Tum kodu tek dosyaya yaz (sunum/arsiv icin):

    poetry run python export.py -o project_export.txt

Cikti: `project_export.txt` (~40-50 KB).

## Proje Yapisi

    MLOps_Project/
    ├── data/
    │   ├── raw/            Ham gorseller (30 sinif)
    │   ├── processed/      Train/val/test split
    │   ├── brokendata/     Bozuk/kucuk gorseller
    │   ├── extradata/      Fazlaliklar
    │   ├── suspicious/     Filigran/menu supheliler
    │   └── reports/        CSV raporlari
    ├── src/
    │   ├── data/           Veri pipeline scriptleri
    │   ├── models/         Model egitimi ve degerlendirme
    │   └── api/            FastAPI (deploy asamasinda)
    ├── monitoring/         Prometheus config + drift check
    ├── docker/             Dockerfile + docker-compose.yml
    ├── scripts/            Yardimci scriptler
    ├── tests/              pytest testleri
    ├── docs/               Dokumantasyon
    ├── models/             Egitilmis model agirliklari
    ├── mlruns/             MLflow local run dizini
    ├── .github/workflows/  CI/CD
    ├── pyproject.toml      Bagimliliklar + tool config
    └── poetry.lock         Kilit dosyasi

## API (planlanan)

| Endpoint | Metod | Aciklama |
|----------|-------|----------|
| `/predict` | POST | Fotograf yukle, yemek tahmini + guven skoru |
| `/check-allergy` | POST | Fotograf + kullanici alerjenleri, uyari |
| `/health` | GET | Servis saglik kontrolu |
| `/docs` | GET | Otomatik OpenAPI dokumantasyonu |

Alerji modulu 30 sinif icin alerjen haritasini icerir (gluten, sut, yumurta,
et, tavuk, avokado, nohut). Kullanici alerjenleri ile kesisim varsa uyari
dondurulur.

## Veri Kalitesi

Her gorsel su asamalardan gecer:

1. **move_suspicious**: Hough cizgi analizi ile filigran/menu karti tespit
2. **move_broken**: PIL ile bozukluk + boyut kontrolu (200x200 minimum)
3. **remove_duplicates**: MD5 hash ile birebir ayni gorseller
4. **trim_to_extradata**: Sinif basina 250 gorsel hedefi
5. **split_dataset**: Stratified %80/%10/%10
6. **renumber_class**: 000001, 000002, ... formatinda

Pipeline deterministik (SEED=42), tekrar calistirilabilir.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):

- **lint-and-test**: ruff lint, ruff format check, mypy, pytest
- **docker-build**: multi-stage Dockerfile build testi

Her `git push` ve `pull_request`'te otomatik calisir.

## Monitoring (planlanan)

- **Evidently**: Referans (train) vs. canli veri drift karsilastirmasi
- **Prometheus**: `http_requests_total`, `http_request_duration_seconds`,
  `model_predictions_total` gibi metrikler
- **Grafana**: Dashboard (latency p50/p95/p99, request rate, error rate,
  prediction distribution)

## Bilinen Kisitlar

- MX330 2 GB VRAM: batch 8, AMP zorunlu
- Web scraping etiket gurultusu: %5-15 yanlis etiket kabul edildi
- 30 sinif, bazi siniflar gorsel olarak cok yakin (corbalar, hamur isleri):
  confusion matrix'te karsilikli karisim normal
- Model tek GPU'da egitilir (distributed training yok)
- Beklenen test dogrulugu: %78-85 (Food-101 benchmark ile kiyasla makul)

## Gelistirici

Eray, Burak - MLOps ve AI Systems dersi

## Lisans

MIT (bkz. LICENSE)
