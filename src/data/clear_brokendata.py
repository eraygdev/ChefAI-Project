"""
ChefAI - data/brokendata/ klasorunu tamamen temizler.
Iki mod:

  --soft  : Icerikleri siler, brokendata/ klasoru kalir (bos).
  --hard  : brokendata/ klasorunu de siler.

Varsayilan: --soft (guvenli).

Kullanim:
    poetry run python -m src.data.clear_brokendata
    poetry run python -m src.data.clear_brokendata --hard
    poetry run python -m src.data.clear_brokendata --yes    # onay sormadan
"""

import argparse
import shutil
import sys
from pathlib import Path

# Proje kokunu sys.path'e ekle
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.utils import count_images  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BROKEN_DIR = DATA_DIR / "brokendata"


def folder_size_bytes(folder: Path) -> int:
    if not folder.exists():
        return 0
    total = 0
    for p in folder.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="brokendata/ klasorunu temizle.")
    parser.add_argument("--hard", action="store_true", help="brokendata/ klasorunu de sil")
    parser.add_argument("--yes", action="store_true", help="Onay sormadan sil")
    args = parser.parse_args()

    print(f"Broken dir : {BROKEN_DIR}")
    print()

    if not BROKEN_DIR.exists():
        print("brokendata/ yok. Silinecek bir sey yok.")
        return

    classes = sorted([p for p in BROKEN_DIR.iterdir() if p.is_dir()])
    total_count = sum(count_images(c) for c in classes)
    total_size = folder_size_bytes(BROKEN_DIR)

    print(f"Sinif sayisi   : {len(classes)}")
    print(f"Toplam gorsel  : {total_count}")
    print(f"Toplam boyut   : {human_mb(total_size)}")
    print(f"Mod            : {'HARD (klasor de silinir)' if args.hard else 'SOFT (icerik silinir)'}")
    print()

    if total_count == 0 and not args.hard:
        print("Silinecek gorsel yok.")
        return

    if not args.yes:
        answer = input("Devam edilsin mi? (y/n): ").strip().lower()
        if answer != "y":
            print("Iptal edildi.")
            return

    if args.hard:
        try:
            shutil.rmtree(BROKEN_DIR)
            print(f"Silindi: {BROKEN_DIR}")
        except Exception as e:
            print(f"HATA: {e}")
            return
    else:
        # Soft: sadece icerikleri sil, klasoru birak
        for class_dir in classes:
            try:
                shutil.rmtree(class_dir)
            except Exception as e:
                print(f"  HATA: {class_dir.name}: {e}")
        # Klasorde kalan dosyalari da sil (varsa)
        for p in BROKEN_DIR.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception as e:
                print(f"  HATA: {p.name}: {e}")
        BROKEN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Temizlendi: {BROKEN_DIR} (bos)")

    print()
    print("Sonraki adim: poetry run python -m src.data.report_status")


if __name__ == "__main__":
    main()
