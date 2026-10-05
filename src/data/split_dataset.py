"""
ChefAI - data/raw/ icindeki gorselleri train/val/test olarak boler.
Cikti: data/processed/{train,val,test}/<sinif>/

Bolme orani:
    train: %80
    val  : %10
    test : %10

Kullanim:
    poetry run python src/data/split_dataset.py
"""

import random
import shutil
from pathlib import Path

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Varolan processed/ klasorunu sil (temiz baslangic)
CLEAN_FIRST = True


def copy_or_move(src: Path, dst_dir: Path, move: bool = False) -> None:
    """Dosyayi kopyalar veya tasir. Cakisma olursa isim degistirir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name

    if target.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while (dst_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = dst_dir / f"{stem}_{i}{suffix}"

    if move:
        shutil.move(str(src), str(target))
    else:
        shutil.copy2(str(src), str(target))


def split_class(class_dir: Path, class_name: str, move: bool = False) -> dict:
    """Bir sinifi train/val/test olarak boler."""
    images = [
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train : n_train + n_val]
    test_imgs = images[n_train + n_val :]

    for img in train_imgs:
        copy_or_move(img, PROCESSED_DIR / "train" / class_name, move)
    for img in val_imgs:
        copy_or_move(img, PROCESSED_DIR / "val" / class_name, move)
    for img in test_imgs:
        copy_or_move(img, PROCESSED_DIR / "test" / class_name, move)

    return {
        "total": n,
        "train": len(train_imgs),
        "val": len(val_imgs),
        "test": len(test_imgs),
    }


def main() -> None:
    print(f"Raw dir       : {RAW_DIR}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Ratios        : train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    if CLEAN_FIRST and PROCESSED_DIR.exists():
        print(f"Temizleniyor: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(f"{'Sinif':<25} {'Toplam':>7} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("-" * 60)

    totals = {"total": 0, "train": 0, "val": 0, "test": 0}

    for class_dir in classes:
        s = split_class(class_dir, class_dir.name, move=False)
        for k in totals:
            totals[k] += s[k]

        print(
            f"{class_dir.name:<25} " f"{s['total']:>7} {s['train']:>7} {s['val']:>6} {s['test']:>6}"
        )

    print("-" * 60)
    print(
        f"{'TOPLAM':<25} "
        f"{totals['total']:>7} {totals['train']:>7} "
        f"{totals['val']:>6} {totals['test']:>6}"
    )
    print()
    print("Sonraki adim: DVC ile versiyonla")
    print("  dvc init")
    print("  dvc add data/processed")


if __name__ == "__main__":
    main()
