"""Maintenance CLI for restoring quarantined images or clearing staging folders.

Usage:
    poetry run python -m src.data.11_manage restore broken
    poetry run python -m src.data.11_manage restore extra
    poetry run python -m src.data.11_manage restore suspicious --dry-run
    poetry run python -m src.data.11_manage clear broken --hard
    poetry run python -m src.data.11_manage clear processed
    poetry run python -m src.data.11_manage clear processed --merge --move
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils import (
    BROKEN_DIR,
    EXTRA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SUSPICIOUS_DIR,
    count_images,
    folder_size_bytes,
    human_mb,
    list_images,
    safe_copy,
    safe_move,
)

ALL_SPLITS = ["train", "val", "test"]
RESTORE_TARGETS = {
    "broken": BROKEN_DIR,
    "extra": EXTRA_DIR,
    "suspicious": SUSPICIOUS_DIR,
}


def restore_to_raw(source_dir: Path, dry_run: bool) -> None:
    """Move all class images from source_dir back into data/raw/ safely."""
    print(f"Source: {source_dir}")
    print(f"Target: {RAW_DIR}")
    print(f"Mode  : {'DRY RUN' if dry_run else 'MOVE'}\n")

    if not source_dir.exists():
        print(f"{source_dir.name}/ does not exist.")
        return

    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"{source_dir.name}/ is empty.")
        return

    print(f"{'Class':<25} {'Files':>10}")
    print("-" * 38)

    total = 0
    for class_dir in class_dirs:
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file()]
        moved = 0
        if dry_run:
            moved = len(files)
        else:
            target_dir = RAW_DIR / class_dir.name
            for img_path in files:
                if safe_move(img_path, target_dir) is not None:
                    moved += 1
            try:
                class_dir.rmdir()
            except OSError:
                pass
        total += moved
        print(f"{class_dir.name:<25} {moved:>10}")

    print("-" * 38)
    print(f"{'TOTAL':<25} {total:>10}\n")

    if not dry_run:
        try:
            source_dir.rmdir()
            print(f"{source_dir.name}/ removed.")
        except OSError:
            print(f"{source_dir.name}/ is not empty yet.")


def clear_broken(hard: bool, yes: bool, dry_run: bool) -> None:
    """Delete contents of data/brokendata/ (soft) or the folder itself (hard)."""
    if not BROKEN_DIR.exists():
        print("brokendata/ does not exist.")
        return

    classes = sorted(p for p in BROKEN_DIR.iterdir() if p.is_dir())
    total_count = sum(count_images(c) for c in classes)
    total_size = folder_size_bytes(BROKEN_DIR)

    print(f"Target : {BROKEN_DIR}")
    print(f"Classes: {len(classes)} | Images: {total_count} | Size: {human_mb(total_size)}")
    print(f"Mode   : {'HARD' if hard else 'SOFT'}{' (DRY RUN)' if dry_run else ''}\n")

    if dry_run:
        return
    if total_count == 0 and not hard:
        print("Nothing to delete.")
        return

    if not yes and input("Continue? (y/n): ").strip().lower() != "y":
        print("Cancelled.")
        return

    if hard:
        shutil.rmtree(BROKEN_DIR)
        print(f"Removed: {BROKEN_DIR}")
        return

    for p in BROKEN_DIR.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    print(f"Cleared contents: {BROKEN_DIR}")


def clear_processed(yes: bool, dry_run: bool, merge: bool, move: bool) -> None:
    """Delete data/processed/ or merge splits back into data/raw/."""
    if not PROCESSED_DIR.exists():
        print("processed/ does not exist.")
        return

    if merge:
        print(f"Merge mode: {'MOVE' if move else 'COPY'} processed -> raw")
        counts = {"files": 0, "done": 0}
        for split in ALL_SPLITS:
            split_dir = PROCESSED_DIR / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for img_path in list_images(class_dir):
                    counts["files"] += 1
                    if dry_run:
                        continue
                    op = safe_move if move else safe_copy
                    if op(img_path, RAW_DIR / class_dir.name) is not None:
                        counts["done"] += 1
        print(f"Found: {counts['files']} | Processed: {counts['done']}")
        return

    total_size = folder_size_bytes(PROCESSED_DIR)
    print(f"Target: {PROCESSED_DIR} ({human_mb(total_size)})")
    for split in ALL_SPLITS:
        sd = PROCESSED_DIR / split
        if sd.exists():
            n = sum(count_images(c) for c in sd.iterdir() if c.is_dir())
            print(f"  {split:<6}: {n:>5} images")

    if dry_run:
        print("\nDRY RUN - nothing deleted.")
        return
    if not yes and input(f"\nDelete {PROCESSED_DIR}? (yes): ").strip() != "yes":
        print("Cancelled.")
        return

    shutil.rmtree(PROCESSED_DIR)
    print(f"Removed: {PROCESSED_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChefAI data management and recovery tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_restore = sub.add_parser("restore", help="Move images back to data/raw/")
    p_restore.add_argument("target", choices=sorted(RESTORE_TARGETS.keys()))
    p_restore.add_argument("--dry-run", action="store_true")

    p_clear = sub.add_parser("clear", help="Clear staging directories")
    p_clear.add_argument("target", choices=["broken", "processed"])
    p_clear.add_argument("--hard", action="store_true", help="Remove brokendata/ folder itself")
    p_clear.add_argument("--merge", action="store_true", help="Merge processed/ back into raw/")
    p_clear.add_argument("--move", action="store_true", help="Move instead of copy during --merge")
    p_clear.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p_clear.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "restore":
        restore_to_raw(RESTORE_TARGETS[args.target], dry_run=args.dry_run)
    elif args.command == "clear":
        if args.target == "broken":
            clear_broken(hard=args.hard, yes=args.yes, dry_run=args.dry_run)
        else:
            clear_processed(yes=args.yes, dry_run=args.dry_run, merge=args.merge, move=args.move)


if __name__ == "__main__":
    main()
