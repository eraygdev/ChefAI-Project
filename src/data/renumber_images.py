"""
ChefAI - Tum gorselleri sirali yeniden numaralandirir.

Varsayilan: HEM data/raw/ HEM data/processed/{train,val,test}/ altindaki
tum siniflari 000001, 000002, ... formatina cevirir.
Her split+sinif klasoru BAGIMSIZ numaralandirilir (train/class/000001-160,
val/class/000001-020 gibi).

Kullanim:
    # Hepsi (raw + train/val/test)
    poetry run python -m src.data.renumber_images

    # Sadece raw
    poetry run python -m src.data.renumber_images --raw-only

    # Sadece processed (tum split'ler)
    poetry run python -m src.data.renumber_images --processed-only

    # Sadece belirli split'ler
    poetry run python -m src.data.renumber_images --splits train,val

    # On izleme (hicbir sey degistirme)
    poetry run python -m src.data.renumber_images --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.utils import count_images, renumber_class

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ALL_SPLITS = ["train", "val", "test"]


def renumber_dir(base_dir: Path, dry_run: bool) -> int:
    """Bir klasordeki tum siniflari renumber et. Toplam rename sayisi doner."""
    if not base_dir.exists():
        print(f"  ({base_dir.name}/ yok, atlaniyor)")
        return 0

    classes = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not classes:
        print("  (bos)")
        return 0

    print(f"{'Sinif':<25} {'Once':>6} {'Yeni':>6}")
    print("-" * 40)

    total = 0
    for class_dir in classes:
        before = count_images(class_dir)
        renamed = renumber_class(class_dir, dry_run=dry_run)
        total += renamed
        print(f"  {class_dir.name:<23} {before:>6} {renamed:>6}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tum gorselleri (raw + processed) yeniden numaralandirir."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--raw-only", action="store_true", help="Sadece data/raw/")
    grp.add_argument("--processed-only", action="store_true", help="Sadece data/processed/")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Virgulle ayrilmis split listesi (default: train,val,test)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Hicbir dosyayi degistirme, sadece rapor"
    )
    args = parser.parse_args()

    do_raw = not args.processed_only
    do_processed = not args.raw_only

    print(f"Mod: {'DRY RUN' if args.dry_run else 'MOVE'}")
    print(f"Raw      : {'ACIK' if do_raw else 'kapali'}")
    print(f"Processed: {'ACIK' if do_processed else 'kapali'}")
    print()

    total = 0

    if do_raw:
        print("=" * 55)
        print("RAW (data/raw/)")
        print("=" * 55)
        total += renumber_dir(RAW_DIR, args.dry_run)
        print()

    if do_processed:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        invalid = [s for s in splits if s not in ALL_SPLITS]
        if invalid:
            print(f"HATA: gecersiz split(ler): {invalid}. Gecerli: {ALL_SPLITS}")
            return

        for split in splits:
            print("=" * 55)
            print(f"PROCESSED - {split} (data/processed/{split}/)")
            print("=" * 55)
            subtotal = renumber_dir(PROCESSED_DIR / split, args.dry_run)
            print(f"  SUBTOTAL: {subtotal}")
            print()
            total += subtotal

    print("=" * 55)
    print(f"TOPLAM yeniden adlandirilan: {total}")
    print("=" * 55)
    if args.dry_run:
        print("(DRY RUN: hicbir sey degistirilmedi)")


if __name__ == "__main__":
    main()
