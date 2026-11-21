"""Detect corrupted or undersized (<200x200) images and move them to data/brokendata/.

Usage:
    poetry run python -m src.data.03_filter_broken --dry-run
    poetry run python -m src.data.03_filter_broken
    poetry run python -m src.data.03_filter_broken --quick
"""

from __future__ import annotations

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from tqdm import tqdm

from src.utils import BROKEN_DIR, RAW_DIR, REPORT_DIR, list_images, safe_move

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

MIN_SIZE = 200
DEFAULT_WORKERS = 4
SKIP_LOW_RES = {"kuru_fasulye", "yaprak_sarma", "simit", "lahmacun"}


def check_single_image(task: tuple[str, bool]) -> tuple[str, str, int, int, str]:
    """Validate image readability, EXIF dimensions, and pixel integrity."""
    warnings.filterwarnings("ignore", category=UserWarning)
    path_str, quick = task
    path = Path(path_str)

    if not path.exists():
        return ("missing", path_str, 0, 0, "file not found")

    try:
        with Image.open(path) as img:
            w, h = img.size
            try:
                transposed = ImageOps.exif_transpose(img)
                if transposed is not None:
                    w, h = transposed.size
            except Exception:
                pass

            if w < MIN_SIZE or h < MIN_SIZE:
                return ("small", path_str, w, h, f"{w}x{h}")

            if not quick:
                try:
                    img.load()
                except Exception as e:
                    return ("broken", path_str, w, h, f"load: {type(e).__name__}")

            return ("ok", path_str, w, h, "")

    except FileNotFoundError:
        return ("missing", path_str, 0, 0, "file not found")
    except Image.UnidentifiedImageError:
        return ("broken", path_str, 0, 0, "unidentified format")
    except OSError as e:
        return ("broken", path_str, 0, 0, f"OSError: {e}")
    except Exception as e:
        return ("broken", path_str, 0, 0, f"{type(e).__name__}: {e}")


def process_class(
    class_dir: Path,
    dry_run: bool,
    quick: bool,
    executor: ProcessPoolExecutor,
    csv_writer: Any = None,
) -> dict[str, int]:
    """Check all images in a class directory and move invalid ones."""
    class_name = class_dir.name
    images = list_images(class_dir)
    stats = {"total": len(images), "ok": 0, "small": 0, "broken": 0, "missing": 0}

    if not images:
        return stats

    tasks = [(str(p), quick) for p in images]
    results = list(
        tqdm(
            executor.map(check_single_image, tasks),
            total=len(images),
            desc=f"[{class_name}]",
            leave=False,
            ncols=80,
        )
    )

    for status, path_str, w, h, reason in results:
        stats[status] += 1
        if csv_writer:
            csv_writer.writerow([class_name, Path(path_str).name, status, w, h, reason])
        if status != "ok" and not dry_run:
            safe_move(Path(path_str), BROKEN_DIR / class_name)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Move broken or undersized images.")
    parser.add_argument("--dry-run", action="store_true", help="Report only without moving files")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Worker processes")
    parser.add_argument("--csv", type=str, default="", help="Custom CSV report path")
    parser.add_argument("--quick", action="store_true", help="Header check only (skip pixel load)")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    csv_file = None
    csv_writer = None
    if not args.dry_run:
        if args.csv:
            csv_path = Path(args.csv)
        else:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = REPORT_DIR / f"filter_broken_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["class", "filename", "status", "width", "height", "reason"])

    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name not in SKIP_LOW_RES)
    print(f"Classes: {len(classes)} (skipped low-res: {sorted(SKIP_LOW_RES)})")
    print(f"Mode   : {'DRY RUN' if args.dry_run else 'MOVE'} | Quick: {args.quick}\n")

    grand_total = {"total": 0, "ok": 0, "small": 0, "broken": 0, "missing": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for class_dir in classes:
            s = process_class(class_dir, args.dry_run, args.quick, executor, csv_writer)
            for k in grand_total:
                grand_total[k] += s[k]
            print(
                f"  {class_dir.name:<25} total={s['total']:>4} ok={s['ok']:>4} "
                f"small={s['small']:>3} broken={s['broken']:>3} missing={s['missing']:>2}"
            )

    moved = grand_total["small"] + grand_total["broken"] + grand_total["missing"]
    print("\n" + "=" * 60)
    print(f"Total: {grand_total['total']} | OK: {grand_total['ok']} | Flagged: {moved}")

    if csv_file is not None:
        csv_file.close()
        print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
