"""
ChefAI - Bozuk ve cok kucuk gorselleri data/brokendata/ altina tasir.
Her sinif kendi alt klasorunde korunur: brokendata/<sinif_adi>/

Kullanim:
    poetry run python src/data/move_broken.py
"""

import shutil
from pathlib import Path

from PIL import Image

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # src/data/ -> src/ -> proje koku
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BROKEN_DIR = DATA_DIR / "brokendata"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
MIN_SIZE = 200
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# -------------------------------------------------------------------
# CHECK FUNCTIONS
# -------------------------------------------------------------------
def is_broken(path: Path) -> bool:
    """Gorsel bozuk mu? (tum piksel verisini okuyarak)"""
    try:
        with Image.open(path) as img:
            img.load()
            img.verify()
        return False
    except Exception:
        return True


def is_too_small(path: Path) -> bool:
    """Gorsel cok kucuk mu?"""
    try:
        with Image.open(path) as img:
            w, h = img.size
        return bool(w < MIN_SIZE or h < MIN_SIZE)
    except Exception:
        return True


# -------------------------------------------------------------------
# MOVE FUNCTION
# -------------------------------------------------------------------
def move_to_broken(path: Path, class_name: str, reason: str, counter: dict) -> None:
    """Gorseli brokendata/<class_name>/ altina tasir."""
    target_dir = BROKEN_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / path.name

    if target.exists():
        stem = path.stem
        suffix = path.suffix
        i = 1
        while (target_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = target_dir / f"{stem}_{i}{suffix}"

    try:
        shutil.move(str(path), str(target))
        counter[reason] = counter.get(reason, 0) + 1
    except Exception as e:
        print(f"    TASIMA HATASI: {path.name}: {e}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    print(f"Raw dir     : {RAW_DIR}")
    print(f"Broken dir  : {BROKEN_DIR}")
    print(f"Min size    : {MIN_SIZE}x{MIN_SIZE}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    grand_total = {"broken": 0, "small": 0, "kept": 0}

    for class_dir in classes:
        class_name = class_dir.name
        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        counter = {"broken": 0, "small": 0, "kept": 0}
        total = len(images)

        for img_path in images:
            if is_broken(img_path):
                move_to_broken(img_path, class_name, "broken", counter)
                continue
            if is_too_small(img_path):
                move_to_broken(img_path, class_name, "small", counter)
                continue
            counter["kept"] += 1

        moved = counter["broken"] + counter["small"]
        print(
            f"[{class_name}] {total} gorsel -> "
            f"Kalan: {counter['kept']}, "
            f"Tasinan: {moved} "
            f"(bozuk={counter['broken']}, kucuk={counter['small']})"
        )

        for k in grand_total:
            grand_total[k] += counter[k]

    print()
    print("=" * 60)
    print("TOPLAM")
    print("=" * 60)
    print(f"Kalan (raw/)           : {grand_total['kept']}")
    print(f"Tasinan (brokendata/)  : {grand_total['broken'] + grand_total['small']}")
    print(f"  - Bozuk              : {grand_total['broken']}")
    print(f"  - Cok kucuk          : {grand_total['small']}")
    print()
    print(f"Brokendata yolu: {BROKEN_DIR}")
    print()
    print("Sonraki adim: poetry run python src/data/cleaner.py")


if __name__ == "__main__":
    main()
