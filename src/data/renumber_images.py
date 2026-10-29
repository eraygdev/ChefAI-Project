"""
ChefAI - data/raw/ altindaki tum siniflari sirayla yeniden numaralandirir.
Ornek: 000001, 000002, 000003, ...

Kullanim:
    poetry run python src/data/renumber_images.py
"""

from pathlib import Path

from src.data.utils import count_images, renumber_class

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    print(f"Raw dir : {RAW_DIR}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(f"{'Sinif':<25} {'Once':>6} {'Sonra':>6}")
    print("-" * 45)

    total = 0
    for class_dir in classes:
        before = count_images(class_dir)
        after = renumber_class(class_dir)
        total += after
        print(f"{class_dir.name:<25} {before:>6} {after:>6}")

    print("-" * 45)
    print(f"{'TOPLAM':<25} {'':>6} {total:>6}")
    print()
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
