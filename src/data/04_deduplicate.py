"""Remove exact pixel-level duplicate images inside each raw class using MD5 hashes.

Usage:
    poetry run python -m src.data.04_deduplicate
    poetry run python -m src.data.04_deduplicate --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

from PIL import Image

from src.utils import RAW_DIR, list_images

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def clean_class(folder: Path, dry_run: bool) -> dict[str, int]:
    """Scan a class folder and remove duplicate images."""
    images = list_images(folder)
    stats = {"before": len(images), "after": 0, "dup": 0, "errors": 0}
    seen: set[str] = set()
    kept = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                digest = hashlib.md5(img.convert("RGB").tobytes()).hexdigest()

            if digest in seen:
                if not dry_run:
                    img_path.unlink()
                stats["dup"] += 1
            else:
                seen.add(digest)
                kept += 1
        except Exception:
            stats["errors"] += 1

    stats["after"] = kept
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate images per class.")
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without deleting")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DELETE'} | Classes: {len(classes)}\n")
    print(f"{'Class':<25} {'Before':>6} {'After':>6} {'Dup':>5} {'Err':>5}")
    print("-" * 55)

    totals = {"before": 0, "after": 0, "dup": 0, "errors": 0}
    for class_dir in classes:
        s = clean_class(class_dir, dry_run=args.dry_run)
        for k in totals:
            totals[k] += s[k]
        print(
            f"{class_dir.name:<25} {s['before']:>6} {s['after']:>6} "
            f"{s['dup']:>5} {s['errors']:>5}"
        )

    print("-" * 55)
    print(
        f"{'TOTAL':<25} {totals['before']:>6} {totals['after']:>6} "
        f"{totals['dup']:>5} {totals['errors']:>5}"
    )


if __name__ == "__main__":
    main()
