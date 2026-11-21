# ChefAI - Data Pipeline Yonergesi

Bu dosya, ham gorselden egitime hazir veri setine kadar olan adimlari
sirasiyla gosterir. Her adimi sirayla calistir.

---

## On Kosul

Proje koku: `~/Masaustu/MLOps_Project`
Poetry aktif olmali: `poetry --version`
Veri konumu: `data/raw/` (30 sinif, hedef 200 gorsel/sinif)

Kontrol:

    poetry run python -m src.data.10_status

---

## ADIM 1: Gorsel Indirme (Opsiyonel)

Script : `src/data/01_download.py`
Amac   : Eksik siniflari hedefe tamamlar.

    poetry run python -m src.data.01_download

---

## ADIM 2: Supheli (Filigran / Menu Karti) Gorselleri Ayir

Script : `src/data/02_filter_suspicious.py`
Amac   : Filigran ve menu karti adaylarini `data/suspicious/` altina tasir.

    poetry run python -m src.data.02_filter_suspicious --preview
    poetry run python -m src.data.02_filter_suspicious --move

Geri almak icin:

    poetry run python -m src.data.11_manage restore suspicious

---

## ADIM 3: Bozuk ve Kucuk Gorselleri Ayir

Script : `src/data/03_filter_broken.py`
Amac   : Bozuk, acilamayan veya 200x200'den kucuk gorselleri `data/brokendata/` altina tasir.

    poetry run python -m src.data.03_filter_broken

Geri almak icin:

    poetry run python -m src.data.11_manage restore broken

---

## ADIM 4: Duplicate (Ayni) Gorselleri Sil

Script : `src/data/04_deduplicate.py`
Amac   : MD5 piksel hash karsilastirmasi ile birebir ayni gorselleri siler.

    poetry run python -m src.data.04_deduplicate

---

## ADIM 5: Fazlaliklari Extradata'ya Tasi

Script : `src/data/05_trim.py`
Amac   : Her sinifta 200'den fazla gorsel varsa, fazlasini `data/extradata/` altina tasir.

    poetry run python -m src.data.05_trim

Geri almak icin:

    poetry run python -m src.data.11_manage restore extra

---

## ADIM 6: Yeniden Numaralandirma ve Durum Kontrolu

Script : `src/data/07_renumber.py`, `src/data/10_status.py`
Amac   : `data/raw/` gorsellerini 000001.. formatina cevirir ve guncel durumu raporlar.

    poetry run python -m src.data.07_renumber --raw-only
    poetry run python -m src.data.10_status

---

## ADIM 7: Train / Val / Test Bolumu

Script : `src/data/06_split.py`
Amac   : `data/raw/` icindeki gorselleri %80/%10/%10 oraninda `data/processed/{train,val,test}/<sinif>/` altina kopyalar.

    poetry run python -m src.data.06_split

Beklenen (30 sinif x 200 gorsel):
- train : 4800
- val   : 600
- test  : 600
- TOPLAM: 6000

---

## ADIM 8: DVC ile Versiyonlama

    poetry run dvc add data/raw
    poetry run dvc add data/processed
    git add data/*.dvc data/.gitignore
    git commit -m "Add dataset to DVC"

---

## Ozet Tablo

| #  | Script                   | Amac                        | Cikti              |
| -- | ------------------------ | --------------------------- | ------------------ |
| 01 | `01_download.py`         | Eksik gorselleri indir      | `data/raw/`        |
| 02 | `02_filter_suspicious.py`| Filigran/menu ayir          | `data/suspicious/` |
| 03 | `03_filter_broken.py`    | Bozuk/kucuk ayir            | `data/brokendata/` |
| 04 | `04_deduplicate.py`      | Duplicate sil               | (silinir)          |
| 05 | `05_trim.py`             | Fazlaliklari tasi           | `data/extradata/`  |
| 06 | `06_split.py`            | Train/val/test bol          | `data/processed/`  |
| 07 | `07_renumber.py`         | `000001...` numaralandir    | `raw/` & `proc/`   |
| 10 | `10_status.py`           | Durum ve disk raporu        | (ekran)            |
| 11 | `11_manage.py`           | Geri alma / temizlik CLI    | `raw/` / temizlik  |
