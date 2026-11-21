"""Move excess images beyond target per class from data/raw/ to data/extradata/.

Usage:
    poetry run python -m src.data.05_trim
    poetry run python -m src.data.05_trim --target 200 --dry-run
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils import EXTRA_DIR, RAW_DIR, list_images, safe_move

DEFAULT_TARGET = 200
SEED = 42


def trim_class(class_dir: Path, target_count: int, dry_run: bool) -> dict[str, int]:
    """Keep target_count images in class_dir and move excess to extradata/."""
    images = list_images(class_dir)
    stats = {"before": len(images), "after": len(images), "moved": 0}
    if len(images) <= target_count:
        return stats

    random.seed(SEED)
    random.shuffle(images)

    excess = images[target_count:]
    if dry_run:
        stats["moved"] = len(excess)
        stats["after"] = target_count
        return stats

    target_dir = EXTRA_DIR / class_dir.name
    moved = 0
    for img_path in excess:
        if safe_move(img_path, target_dir) is not None:
            moved += 1

    stats["moved"] = moved
    stats["after"] = len(images) - moved
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim excess class images to extradata/.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Max images per class")
    parser.add_argument("--dry-run", action="store_true", help="Report only without moving")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Target: {args.target}/class | Mode: {'DRY RUN' if args.dry_run else 'MOVE'}\n")
    print(f"{'Class':<25} {'Before':>6} {'After':>6} {'Moved':>8}")
    print("-" * 55)

    totals = {"before": 0, "after": 0, "moved": 0}
    for class_dir in classes:
        s = trim_class(class_dir, args.target, args.dry_run)
        for k in totals:
            totals[k] += s[k]
        print(f"{class_dir.name:<25} {s['before']:>6} {s['after']:>6} {s['moved']:>8}")

    print("-" * 55)
    print(f"{'TOTAL':<25} {totals['before']:>6} {totals['after']:>6} {totals['moved']:>8}")


if __name__ == "__main__":
    main()
