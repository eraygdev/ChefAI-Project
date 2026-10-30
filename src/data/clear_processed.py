"""
ChefAI - data/processed/ yonetimi.

VARSAYILAN: processed'i siler (raw'a dokunmaz, guvenli).
Eski merge davranisi (processed -> raw) --merge flag'i ile opsiyonel.

Kullanim:
    # Default: processed'i sil (raw'dan yeniden split edilecek)
    poetry run python -m src.data.merge_to_raw
    poetry run python -m src.data.merge_to_raw --yes
    poetry run python -m src.data.merge_to_raw --dry-run

    # Eski davranis: processed -> raw birlestir (DUPLICATE RISKI)
    poetry run python -m src.data.merge_to_raw --merge --move
    poetry run python -m src.data.merge_to_raw --merge --copy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.utils import count_images, list_images  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ALL_SPLITS = ["train", "val", "test"]


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


def clear_processed(dry_run: bool, yes: bool) -> None:
    """data/processed/ icindeki her seyi siler. raw'a DOKUNMAZ."""
    if not PROCESSED_DIR.exists():
        print(f"{PROCESSED_DIR} yok. Silinecek bir sey yok.")
        return

    total_size = folder_size_bytes(PROCESSED_DIR)
    print(f"Hedef   : {PROCESSED_DIR}")
    print(f"Boyut   : {human_mb(total_size)}")
    print()

    for split in ALL_SPLITS:
        sd = PROCESSED_DIR / split
        if sd.exists():
            n = sum(count_images(c) for c in sd.iterdir() if c.is_dir())
            print(f"  {split:<6}: {n:>5} gorsel")
    print()

    if dry_run:
        print("DRY RUN: hicbir sey silinmedi.")
        return

    if not yes:
        ans = input(f"{PROCESSED_DIR} silinsin mi? (yes yaz): ").strip()
        if ans != "yes":
            print("Iptal edildi.")
            return

    shutil.rmtree(PROCESSED_DIR)
    print(f"Silindi: {PROCESSED_DIR}")
    print()
    print("Sonraki adim: make data-split  (raw'dan yeniden split)")


def merge_to_raw(move: bool, dry_run: bool) -> None:
    """ESKI davranis: processed -> raw birlestir. DUPLICATE RISKI."""
    print("DIKKAT: merge modu. processed'deki gorseller raw'a tasinacak.")
    print(f"  Raw dir : {RAW_DIR}")
    print(f"  Mod     : {'MOVE' if move else 'COPY'}")
    print()

    if not PROCESSED_DIR.exists():
        print(f"{PROCESSED_DIR} yok.")
        return

    grand = {"files": 0, "copied": 0, "moved": 0, "collisions": 0}

    for split in ALL_SPLITS:
        split_dir = PROCESSED_DIR / split
        if not split_dir.exists():
            continue

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = list_images(class_dir)
            if not images:
                continue

            target_dir = RAW_DIR / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                grand["files"] += 1
                target = target_dir / img_path.name

                if target.exists():
                    stem, suffix = img_path.stem, img_path.suffix
                    i = 1
                    while (target_dir / f"{stem}_{i}{suffix}").exists():
                        i += 1
                    target = target_dir / f"{stem}_{i}{suffix}"
                    grand["collisions"] += 1

                if dry_run:
                    continue
                try:
                    if move:
                        shutil.move(str(img_path), str(target))
                        grand["moved"] += 1
                    else:
                        shutil.copy2(str(img_path), str(target))
                        grand["copied"] += 1
                except Exception as e:
                    print(f"  HATA: {img_path.name}: {e}")

    print()
    print("=" * 60)
    print("TOPLAM")
    print("=" * 60)
    print(f"Bulunan dosya   : {grand['files']}")
    print(f"Kopyalanan      : {grand['copied']}")
    print(f"Tasinan         : {grand['moved']}")
    print(f"Adi degistirilen: {grand['collisions']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="processed'i temizle (default) veya raw'a birlestir (--merge)."
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="ESKI davranis: processed -> raw birlestir (DUPLICATE RISKI)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="(merge ile) tasi; default kopyala",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="(merge ile) kopyala (default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Hicbir sey yapma, sadece rapor")
    parser.add_argument("--yes", action="store_true", help="Onay sorma")
    args = parser.parse_args()

    if args.merge:
        move = args.move and not args.copy
        merge_to_raw(move=move, dry_run=args.dry_run)
    else:
        clear_processed(dry_run=args.dry_run, yes=args.yes)


if __name__ == "__main__":
    main()
