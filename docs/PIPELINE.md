# ChefAI - Data Pipeline Yonergesi

Bu dosya, ham gorselden egitime hazir veri setine kadar olan adimlari
sirasiyla gosterir. Her adimi sirayla calistir.

---

## On Kosul

Proje koku: `~/Masaustu/MLOps_Project`
Poetry aktif olmali: `poetry --version`
Veri konumu: `data/raw/` (30 sinif, hedef 200 gorsel/sinif)

Kontrol:
poetry run python src/data/report_status.py

---

## ADIM 1: Gorsel Indirme (Zaten Yapildi)

Script : `src/data/more_to_target.py`
Amac : Eksik siniflari hedefe (250) tamamlar.
Not : Zaten 9668 gorsel var. Eksik varsa calistir.

    poetry run python src/data/more_to_target.py

---

## ADIM 2: Bozuk ve Kucuk Gorselleri Ayir

Script : `src/data/move_broken.py`
Amac : Bozuk, acilamayan, 200x200'den kucuk gorselleri
`data/brokendata/` altina tasir.
Cikti : `data/brokendata/<sinif>/`

    poetry run python src/data/move_broken.py

Geri almak istersen (yanlis karar verdiysen):

    poetry run python src/data/restore_broken.py

---

## ADIM 3: Duplicate (Ayni) Gorselleri Sil

Script : `src/data/remove_duplicates.py`
Amac : Hash bazli karsilastirma ile ayni gorselleri siler.

    poetry run python src/data/remove_duplicates.py

---

## ADIM 4: Fazlaliklari Extradata'ya Tasi

Script : `src/data/trim_to_extradata.py`
Amac : Her sinifta 200'den fazla gorsel varsa, fazlasini
`data/extradata/` altina tasir.
Cikti : `data/extradata/<sinif>/`

    poetry run python src/data/trim_to_extradata.py

---

## ADIM 5: Durum Kontrolu

Script : `src/data/report_status.py`
Amac : Ham, brokendata, extradata, processed klasorlerinin
guncel durumunu raporlar.

    poetry run python src/data/report_status.py

Beklenen:

- raw/ : 30 sinif, her biri ~200 gorsel
- brokendata/: bozuk/kucuk gorseller
- extradata/ : fazla gorseller

---

## ADIM 6: Train / Val / Test Bolumu

Script : `src/data/split_dataset.py`
Amac : `data/raw/` icindeki gorselleri %80/%10/%10 oraninda
train/val/test olarak `data/processed/` altina kopyalar.
Cikti : `data/processed/{train,val,test}/<sinif>/`

    poetry run python src/data/split_dataset.py

Beklenen (30 sinif x 200 gorsel):

- train: 30 sinif x 160 gorsel = 4800
- val : 30 sinif x 20 gorsel = 600
- test : 30 sinif x 20 gorsel = 600
- TOPLAM: 6000

---

## ADIM 7: DVC ile Versiyonlama

Ilk kurulum (bir kere):

    poetry run dvc init
    git add .dvc .dvcignore
    git commit -m "Init DVC"

Veriyi takip et:

    poetry run dvc add data/raw
    poetry run dvc add data/processed

Git'e ekle:

    git add data/raw.dvc data/processed.dvc data/.gitignore
    git commit -m "Add data to DVC"

---

## Ozet Tablo

| #   | Script               | Amac                        | Cikti            |
| --- | -------------------- | --------------------------- | ---------------- |
| 1   | more_to_target.py    | Eksik gorselleri indir      | data/raw/        |
| 2   | move_broken.py       | Bozuk/kucuk gorselleri ayir | data/brokendata/ |
| 3   | remove_duplicates.py | Duplicate sil               | (silinir)        |
| 4   | trim_to_extradata.py | Fazlaliklari tasi           | data/extradata/  |
| 5   | report_status.py     | Durum raporu                | (ekran)          |
| 6   | split_dataset.py     | Train/val/test bol          | data/processed/  |
| 7   | (DVC)                | Versiyonla                  | .dvc dosyalari   |
| -   | restore_broken.py    | brokendata'dan geri tasi    | data/raw/        |

---

## Tam Pipeline (Tek Seferde)

    # 1. Durum kontrolu
    poetry run python src/data/report_status.py

    # 2. Bozuk/kucuk ayir
    poetry run python src/data/move_broken.py

    # 3. Duplicate sil
    poetry run python src/data/remove_duplicates.py

    # 4. Fazlaliklari tasi
    poetry run python src/data/trim_to_extradata.py

    # 5. Tekrar durum kontrolu
    poetry run python src/data/report_status.py

    # 6. Train/val/test bol
    poetry run python src/data/split_dataset.py

    # 7. DVC ile versiyonla
    poetry run dvc add data/raw
    poetry run dvc add data/processed
    git add data/*.dvc data/.gitignore .dvc .dvcignore
    git commit -m "Add dataset to DVC"
