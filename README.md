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

Bolme orani: %80 train, %10 val, %10 test (stratified, her sinif kendi icinde).

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

### 1. Setup (tamamlandi)

- Poetry (bagimlilik yonetimi)
- Docker + Docker Compose
- pre-commit (ruff, mypy, kalite kontrolleri)

### 2. Data (tamamlandi)

- DVC (veri versiyonlama)
- Gorsel indirme: icrawler (Google, Bing, Baidu, Flickr)
- Temizlik: bozuk, kucuk, duplicate, supheli (filigran/menu) ayiklama
- Otomatik yeniden numaralandirma (000001, 000002, ...)
- Train/val/test split

### 3. Model (devam ediyor)

- MLflow (deney takibi + model registry)
- Optuna (hiperparametre optimizasyonu)
- SHAP (aciklanabilirlik)
- Model Registry: Development, Staging, Production

### 4. Deploy (planlanan)

- FastAPI (REST servisi)
- Docker multi-stage build
- Gunicorn + Uvicorn (4 worker)
- GitHub Actions CI/CD

### 5. Monitor (planlanan)

- Evidently (data/model drift)
- Prometheus (metrik toplama)
- Grafana (gorsellestirme)

## Kurulum

### On Kosullar

- Python 3.12
- Poetry 2.x
- Docker + Docker Compose
- CUDA 12.1 (opsiyonel, CPU'da da calisir)

### Adimlar

```bash
# 1. Bagimliliklari kur
poetry install

# 2. pre-commit hook
poetry run pre-commit install

# 3. Docker stack baslat (MLflow + MinIO + PostgreSQL)
make up

# 4. Kontrol
make check
