"""
ChefAI - brokendata/ icindeki TUM gorselleri images/ altina geri tasir.
Hicbir kontrol yapmaz. Ayni isimde dosya varsa uzerine yazar.
"""

import shutil
from pathlib import Path

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMAGES_DIR = PROJECT_ROOT / "images"
BROKEN_DIR = PROJECT_ROOT / "brokendata"


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print(f"Broken dir: {BROKEN_DIR}")
    print(f"Images dir: {IMAGES_DIR}")
    print()

    if not BROKEN_DIR.exists():
        print("brokendata/ yok, islem yapilamiyor.")
        return

    class_dirs = sorted([p for p in BROKEN_DIR.iterdir() if p.is_dir()])

    if not class_dirs:
        print("brokendata/ bos, tasinacak gorsel yok.")
        return

    print(f"brokendata/ icinde {len(class_dirs)} sinif klasoru var.")
    print()

    print(f"{'Sinif':<25} {'Tasinan':>10}")
    print("-" * 40)

    grand_total = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        target_dir = IMAGES_DIR / class_name
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
        print(f"{class_name:<25} {moved:>10}")

        # Bosalan klasoru sil
        try:
            class_dir.rmdir()
        except Exception:
            pass

    print("-" * 40)
    print(f"{'TOPLAM':<25} {grand_total:>10}")
    print()

    # brokendata/ bosaldiysa sil
    try:
        BROKEN_DIR.rmdir()
        print("brokendata/ silindi.")
    except Exception:
        print("brokendata/ hala bos degil (bazi dosyalar tasinamadi).")

    print()
    print("Sonraki adim: python Meta/info.py")


if __name__ == "__main__":
    main()