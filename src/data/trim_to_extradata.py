"""
ChefAI - Hedefin uzerindeki fazla gorselleri data/extradata/ altina tasir.
Her sinif icin TARGET_COUNT kadar gorsel birakir, fazlasini tasir.

Kullanim:
    poetry run python src/data/trim_to_extradata.py
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
EXTRA_DIR = DATA_DIR / "extradata"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
TARGET_COUNT = 200
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def trim_class(class_dir: Path, class_name: str) -> dict:
    """Bir siniftan fazla gorselleri extradata'ya tasir."""
    stats = {"before": 0, "after": 0, "moved": 0}

    images = [
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    stats["before"] = len(images)

    if len(images) <= TARGET_COUNT:
        stats["after"] = len(images)
        return stats

    random.seed(SEED)
    random.shuffle(images)

    keep = images[:TARGET_COUNT]
    move = images[TARGET_COUNT:]

    target_dir = EXTRA_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for img_path in move:
        target = target_dir / img_path.name
        if target.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            i = 1
            while (target_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            target = target_dir / f"{stem}_{i}{suffix}"

        try:
            shutil.move(str(img_path), str(target))
            stats["moved"] += 1
        except Exception as e:
            print(f"    TASIMA HATASI: {img_path.name}: {e}")

    stats["after"] = len(keep)
    return stats


def main() -> None:
    print(f"Raw dir   : {RAW_DIR}")
    print(f"Extra dir : {EXTRA_DIR}")
    print(f"Target    : {TARGET_COUNT} gorsel/sinif")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(f"{'Sinif':<25} {'Once':>6} {'Sonra':>6} {'Tasinan':>8}")
    print("-" * 55)

    totals = {"before": 0, "after": 0, "moved": 0}

    for class_dir in classes:
        s = trim_class(class_dir, class_dir.name)
        for k in totals:
            totals[k] += s[k]

        print(f"{class_dir.name:<25} " f"{s['before']:>6} {s['after']:>6} {s['moved']:>8}")

    print("-" * 55)
    print(f"{'TOPLAM':<25} " f"{totals['before']:>6} {totals['after']:>6} {totals['moved']:>8}")
    print()
    print(f"Toplam tasinan: {totals['moved']}")
    print(f"Kalan gorsel  : {totals['after']}")
    print()
    print("Sonraki adim: poetry run python src/data/split_dataset.py")


if __name__ == "__main__":
    main()
