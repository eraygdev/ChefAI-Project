"""Split data/raw/ classes into data/processed/{train,val,test}/ (80/10/10).

Usage:
    poetry run python -m src.data.06_split
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from src.utils import PROCESSED_DIR, RAW_DIR, list_images, safe_copy

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42


def split_class(class_dir: Path) -> dict[str, int]:
    """Deterministically copy class images into train, val, and test splits."""
    images = list_images(class_dir)
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split_name, split_imgs in splits.items():
        dst_dir = PROCESSED_DIR / split_name / class_dir.name
        for img in split_imgs:
            safe_copy(img, dst_dir)

    return {
        "total": n,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified 80/10/10 split into data/processed/.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing processed/")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    if not args.no_clean and PROCESSED_DIR.exists():
        print(f"Cleaning existing directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO} | Classes: {len(classes)}\n")
    print(f"{'Class':<25} {'Total':>7} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("-" * 60)

    totals = {"total": 0, "train": 0, "val": 0, "test": 0}
    for class_dir in classes:
        s = split_class(class_dir)
        for k in totals:
            totals[k] += s[k]
        print(
            f"{class_dir.name:<25} {s['total']:>7} {s['train']:>7} {s['val']:>6} {s['test']:>6}"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':<25} {totals['total']:>7} {totals['train']:>7} "
        f"{totals['val']:>6} {totals['test']:>6}"
    )


if __name__ == "__main__":
    main()
