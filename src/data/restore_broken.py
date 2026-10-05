"""
ChefAI - data/brokendata/ icindeki TUM gorselleri data/raw/ altina geri tasir.
Hicbir kontrol yapmaz.

Kullanim:
    poetry run python src/data/restore_broken.py
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BROKEN_DIR = DATA_DIR / "brokendata"


def main() -> None:
    print(f"Broken dir: {BROKEN_DIR}")
    print(f"Raw dir   : {RAW_DIR}")
    print()

    if not BROKEN_DIR.exists():
        print("brokendata/ yok.")
        return

    class_dirs = sorted([p for p in BROKEN_DIR.iterdir() if p.is_dir()])
    if not class_dirs:
        print("brokendata/ bos.")
        return

    print(f"{'Sinif':<25} {'Tasinan':>10}")
    print("-" * 40)

    grand_total = 0
    for class_dir in class_dirs:
        target_dir = RAW_DIR / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for img_path in class_dir.iterdir():
            if not img_path.is_file():
                continue
            try:
                shutil.move(str(img_path), str(target_dir / img_path.name))
                moved += 1
            except Exception as e:
                print(f"  HATA: {img_path.name}: {e}")

        grand_total += moved
        print(f"{class_dir.name:<25} {moved:>10}")

        try:
            class_dir.rmdir()
        except Exception:
            pass

    print("-" * 40)
    print(f"{'TOPLAM':<25} {grand_total:>10}")
    print()

    try:
        BROKEN_DIR.rmdir()
        print("brokendata/ silindi.")
    except Exception:
        print("brokendata/ hala bos degil.")

    print()
    print("Sonraki adim: poetry run python src/data/report_status.py")


if __name__ == "__main__":
    main()
