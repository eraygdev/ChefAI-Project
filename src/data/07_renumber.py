"""Sequentially renumber class images as 000001, 000002, ...

Usage:
    poetry run python -m src.data.07_renumber
    poetry run python -m src.data.07_renumber --raw-only
    poetry run python -m src.data.07_renumber --processed-only --splits train,val
    poetry run python -m src.data.07_renumber --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import PROCESSED_DIR, RAW_DIR, count_images, list_images

ALL_SPLITS = ["train", "val", "test"]


def renumber_class(
    class_dir: Path,
    dry_run: bool = False,
    prefix: str = "",
    pad: int = 6,
) -> int:
    """Rename all images in class_dir to sequential 1..N zero-padded filenames."""
    if not class_dir.exists() or not class_dir.is_dir():
        return 0

    images = list_images(class_dir)
    if not images:
        return 0

    expected_names = [f"{prefix}{i:0{pad}d}{img.suffix.lower()}" for i, img in enumerate(images, 1)]
    actual_names = [img.name for img in images]
    if actual_names == expected_names:
        return 0

    temp_paths: list[Path] = []
    for i, img in enumerate(images, 1):
        temp_path = class_dir / f"_tmp_{i:0{pad}d}{img.suffix.lower()}"
        if not dry_run:
            try:
                img.rename(temp_path)
            except Exception as e:
                print(f"  RENAME ERROR: {img.name}: {e}")
                continue
        temp_paths.append(temp_path)

    renamed = 0
    for i, temp in enumerate(temp_paths, 1):
        final_path = class_dir / f"{prefix}{i:0{pad}d}{temp.suffix.lower()}"
        if not dry_run:
            try:
                temp.rename(final_path)
                renamed += 1
            except Exception as e:
                print(f"  RENAME ERROR: {temp.name}: {e}")
        else:
            renamed += 1

    return renamed


def renumber_dir(base_dir: Path, dry_run: bool) -> int:
    """Renumber every class directory inside base_dir."""
    if not base_dir.exists():
        print(f"  ({base_dir.name}/ missing, skipped)")
        return 0

    classes = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if not classes:
        print("  (empty)")
        return 0

    print(f"{'Class':<25} {'Count':>6} {'Renamed':>8}")
    print("-" * 42)
    total = 0
    for class_dir in classes:
        before = count_images(class_dir)
        renamed = renumber_class(class_dir, dry_run=dry_run)
        total += renamed
        print(f"  {class_dir.name:<23} {before:>6} {renamed:>8}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Renumber raw and/or processed images.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--raw-only", action="store_true", help="Only process data/raw/")
    grp.add_argument("--processed-only", action="store_true", help="Only process data/processed/")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Processed splits")
    parser.add_argument("--dry-run", action="store_true", help="Preview without renaming")
    args = parser.parse_args()

    do_raw = not args.processed_only
    do_processed = not args.raw_only
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RENAME'} | Raw={do_raw} | Processed={do_processed}\n")

    total = 0
    if do_raw:
        print("=== RAW (data/raw/) ===")
        total += renumber_dir(RAW_DIR, args.dry_run)
        print()

    if do_processed:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        invalid = [s for s in splits if s not in ALL_SPLITS]
        if invalid:
            print(f"ERROR: invalid splits: {invalid}")
            return

        for split in splits:
            print(f"=== PROCESSED ({split}) ===")
            total += renumber_dir(PROCESSED_DIR / split, args.dry_run)
            print()

    print(f"TOTAL renamed: {total}")


if __name__ == "__main__":
    main()
