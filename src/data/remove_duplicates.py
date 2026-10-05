"""
ChefAI - Duplicate (ayni) gorselleri siler.
Hash bazli karsilastirma yapar.

Kullanim:
    poetry run python src/data/cleaner.py
"""

import hashlib
import warnings
from pathlib import Path

from PIL import Image

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
TARGET_COUNT = None  # None = fazlaliklari silme
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# -------------------------------------------------------------------
# TEMIZLEME
# -------------------------------------------------------------------
def clean_class(folder: Path) -> dict:
    """Bir sinifi duplicate'lerden temizler."""
    stats = {"before": 0, "after": 0, "dup": 0, "errors": 0}

    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    stats["before"] = len(images)

    seen = set()
    unique = []

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                h = hashlib.md5(img_rgb.tobytes()).hexdigest()

            if h in seen:
                img_path.unlink()
                stats["dup"] += 1
            else:
                seen.add(h)
                unique.append(img_path)
        except Exception:
            stats["errors"] += 1
            continue

    if TARGET_COUNT and len(unique) > TARGET_COUNT:
        import random

        random.seed(SEED)
        random.shuffle(unique)
        for p in unique[TARGET_COUNT:]:
            p.unlink()
        unique = unique[:TARGET_COUNT]

    stats["after"] = len(unique)
    return stats


def main() -> None:
    print(f"Raw dir : {RAW_DIR}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(f"{'Sinif':<25} {'Once':>6} {'Sonra':>6} {'Dup':>5} {'Hata':>5}")
    print("-" * 55)

    totals = {"before": 0, "after": 0, "dup": 0, "errors": 0}

    for class_dir in classes:
        s = clean_class(class_dir)
        for k in totals:
            totals[k] += s[k]

        print(
            f"{class_dir.name:<25} "
            f"{s['before']:>6} {s['after']:>6} "
            f"{s['dup']:>5} {s['errors']:>5}"
        )

    print("-" * 55)
    print(
        f"{'TOPLAM':<25} "
        f"{totals['before']:>6} {totals['after']:>6} "
        f"{totals['dup']:>5} {totals['errors']:>5}"
    )
    print()
    print(f"Silinen duplicate: {totals['dup']}")
    print(f"Kalan gorsel     : {totals['after']}")
    print()
    print("Sonraki adim: poetry run python src/data/trim_to_extradata.py")


if __name__ == "__main__":
    main()
