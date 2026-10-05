"""
ChefAI - Gorsel temizleme.
- Bozuk, cok kucuk, duplicate gorselleri siler.
- Hedefin uzerindeki fazlaliklari siler (opsiyonel).
- Hata olursa gor se atlar, sinifi atlamaz.
"""

import hashlib
import warnings
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Uyarilari bastir (Palette images vs.)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMAGES_DIR = PROJECT_ROOT / "images"

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
MIN_SIZE = 200
TARGET_COUNT = None  # None = fazlaliklari silme, sayi = o sayiya dusur
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -------------------------------------------------------------------
# TEMIZLEME FONKSIYONU
# -------------------------------------------------------------------
def clean_class(folder: Path) -> dict:
    """Bir sinifi temizler. Istatistik dondurur."""
    stats = {
        "broken": 0,
        "small": 0,
        "dup": 0,
        "excess": 0,
        "kept": 0,
        "before": 0,
        "after": 0,
        "errors": 0,
    }

    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    stats["before"] = len(images)

    # 1. Bozuk ve kucuk sil
    valid = []
    for img_path in images:
        try:
            # load() ile piksel verisini oku -> bozuklari yakalar
            with Image.open(img_path) as img:
                img.load()
                w, h = img.size

            if w < MIN_SIZE or h < MIN_SIZE:
                img_path.unlink()
                stats["small"] += 1
                continue

            valid.append(img_path)

        except Exception:
            # Hata olursa gor se atla, sinifi atlama
            try:
                img_path.unlink()
            except Exception:
                pass
            stats["broken"] += 1
            continue

    # 2. Duplicate sil (hash bazli)
    seen = set()
    unique = []
    for img_path in valid:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                # Kucuk boyutlu hash (hizli)
                h = hashlib.md5(img.tobytes()).hexdigest()

            if h in seen:
                img_path.unlink()
                stats["dup"] += 1
            else:
                seen.add(h)
                unique.append(img_path)

        except Exception:
            # Hata olursa gor se atla
            try:
                img_path.unlink()
            except Exception:
                pass
            stats["broken"] += 1
            continue

    # 3. Fazlalik sil (TARGET_COUNT ayarliysa)
    if TARGET_COUNT and len(unique) > TARGET_COUNT:
        import random

        random.seed(SEED)
        random.shuffle(unique)
        for p in unique[TARGET_COUNT:]:
            p.unlink()
            stats["excess"] += 1
        unique = unique[:TARGET_COUNT]

    stats["kept"] = len(unique)
    stats["after"] = len(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )
    return stats


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print(f"Images dir : {IMAGES_DIR}")
    print(f"Min size   : {MIN_SIZE}")
    print(f"Target/class: {TARGET_COUNT if TARGET_COUNT else 'yok (fazlalik silinmez)'}")
    print()

    if not IMAGES_DIR.exists():
        print("HATA: images/ klasoru yok!")
        return

    classes = sorted([p for p in IMAGES_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(
        f"{'Sinif':<25} {'Once':>6} {'Sonra':>6} {'Bozuk':>6} {'Kucuk':>6} {'Dup':>5} {'Fazla':>6}"
    )
    print("-" * 78)

    totals = {"before": 0, "after": 0, "broken": 0, "small": 0, "dup": 0, "excess": 0}

    for class_dir in classes:
        s = clean_class(class_dir)
        for k in totals:
            totals[k] += s[k]

        print(
            f"{class_dir.name:<25} "
            f"{s['before']:>6} {s['after']:>6} "
            f"{s['broken']:>6} {s['small']:>6} {s['dup']:>5} {s['excess']:>6}"
        )

    print("-" * 78)
    print(
        f"{'TOPLAM':<25} "
        f"{totals['before']:>6} {totals['after']:>6} "
        f"{totals['broken']:>6} {totals['small']:>6} "
        f"{totals['dup']:>5} {totals['excess']:>6}"
    )
    print()

    total_removed = totals["before"] - totals["after"]
    print(f"Silinen toplam gorsel: {total_removed}")
    print(f"Kalan toplam gorsel  : {totals['after']}")


if __name__ == "__main__":
    main()
