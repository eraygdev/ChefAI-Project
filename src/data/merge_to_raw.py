"""
ChefAI - data/processed/{train,val,test}/ altindaki tum gorselleri
data/raw/<sinif>/ altinda yeniden birlestirir.

Kullanim:
    # Kopyala (default) - processed/ te kalir
    poetry run python -m src.data.merge_to_raw

    # Tasi - processed/ bosalir
    poetry run python -m src.data.merge_to_raw --move

    # On izleme
    poetry run python -m src.data.merge_to_raw --dry-run

    # Belirli split'leri dahil et
    poetry run python -m src.data.merge_to_raw --splits train,val
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.utils import list_images  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ALL_SPLITS = ["train", "val", "test"]


def merge_split(split: str, move: bool, dry_run: bool) -> dict:
    """Bir split'i (train/val/test) raw/ altinda birlestirir."""
    stats = {"split": split, "files": 0, "copied": 0, "moved": 0, "collisions": 0}

    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        print(f"  [{split}] YOK - atlaniyor")
        return stats

    classes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not classes:
        print(f"  [{split}] bos - atlaniyor")
        return stats

    for class_dir in classes:
        class_name = class_dir.name
        images = list_images(class_dir)
        if not images:
            continue

        target_dir = RAW_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            stats["files"] += 1

            target = target_dir / img_path.name
            # Cakisma: ayni isim varsa _1, _2 ekle
            if target.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                i = 1
                while (target_dir / f"{stem}_{i}{suffix}").exists():
                    i += 1
                target = target_dir / f"{stem}_{i}{suffix}"
                stats["collisions"] += 1

            if dry_run:
                continue

            try:
                if move:
                    shutil.move(str(img_path), str(target))
                    stats["moved"] += 1
                else:
                    shutil.copy2(str(img_path), str(target))
                    stats["copied"] += 1
            except Exception as e:
                print(f"    HATA: {img_path.name}: {e}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="processed/{train,val,test}/ altindaki gorselleri raw/ altinda birlestir."
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Kopyalamak yerine tasi (processed/ bosalir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hicbir sey yapma, sadece rapor ver",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Virgulle ayrilmis split listesi (default: train,val,test)",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    invalid = [s for s in splits if s not in ALL_SPLITS]
    if invalid:
        print(f"HATA: gecersiz split(ler): {invalid}")
        print(f"Gecerli: {ALL_SPLITS}")
        sys.exit(1)

    mode = "MOVE" if args.move else "COPY"
    print(f"Raw dir       : {RAW_DIR}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Mod           : {mode}{' (DRY RUN)' if args.dry_run else ''}")
    print(f"Split'ler     : {splits}")
    print()

    if not PROCESSED_DIR.exists():
        print(f"HATA: {PROCESSED_DIR} yok! (once 'make data-split' calistir)")
        return

    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Not: {RAW_DIR} olusturuldu.")

    grand = {"files": 0, "copied": 0, "moved": 0, "collisions": 0}

    for split in splits:
        s = merge_split(split, args.move, args.dry_run)
        for k in grand:
            grand[k] += s[k]

        action = "tasinacak" if args.move else "kopyalanacak"
        print(f"  [{split:5s}] {s['files']:>5} dosya {action} (cakisma: {s['collisions']})")

    print()
    print("=" * 60)
    print("TOPLAM")
    print("=" * 60)
    print(f"Bulunan dosya   : {grand['files']}")
    print(f"Kopyalanan      : {grand['copied']}")
    print(f"Tasinan         : {grand['moved']}")
    print(f"Adi degistirilen: {grand['collisions']} (cakisma)")

    if args.dry_run:
        print()
        print("DRY RUN: hicbir sey yapilmadi.")

    print()
    print("Sonraki adim: poetry run python -m src.data.report_status")


if __name__ == "__main__":
    main()
