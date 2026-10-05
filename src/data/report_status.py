"""
ChefAI - data/ klasorunun durumunu raporlar.

Kullanim:
    poetry run python src/data/info.py
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def report_raw() -> None:
    print("=" * 60)
    print("RAW DURUMU (data/raw/)")
    print("=" * 60)

    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        print(f"  {raw_dir} yok!")
        return

    classes = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    print(f"{'Sinif':<25} {'Gorsel':>8}")
    print("-" * 35)

    total = 0
    for c in classes:
        n = count_images(c)
        total += n
        flag = "  <-- AZ" if n < 200 else ""
        print(f"{c.name:<25} {n:>8}{flag}")

    print("-" * 35)
    print(f"{'TOPLAM':<25} {total:>8}")
    print()


def report_split() -> None:
    print("=" * 60)
    print("SPLIT DURUMU (data/processed/)")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / "processed" / split
        if not split_dir.exists():
            print(f"  {split}: YOK")
            continue

        classes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        total = sum(count_images(c) for c in classes)
        print(f"  {split:5s}: {len(classes):3d} sinif, {total:6d} gorsel")
    print()


def report_other() -> None:
    print("=" * 60)
    print("DIGER KLASORLER")
    print("=" * 60)

    for name in ["brokendata", "extradata"]:
        d = DATA_DIR / name
        if not d.exists():
            print(f"  {name}: YOK")
            continue
        total = sum(count_images(c) for c in d.iterdir() if c.is_dir())
        classes = len([c for c in d.iterdir() if c.is_dir()])
        print(f"  {name}: {classes} sinif, {total} gorsel")
    print()


def main() -> None:
    print()
    print(f"Proje koku : {PROJECT_ROOT}")
    print(f"Data dir   : {DATA_DIR}")
    print()

    report_raw()
    report_split()
    report_other()


if __name__ == "__main__":
    main()
