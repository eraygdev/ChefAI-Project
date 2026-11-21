"""Report image counts and disk usage across all data/ subdirectories.

Usage:
    poetry run python -m src.data.10_status
"""

from __future__ import annotations

from src.utils import (
    BROKEN_DIR,
    DATA_DIR,
    EXTRA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    SUSPICIOUS_DIR,
    count_images,
    folder_size_bytes,
    human_mb,
)


def report_raw() -> tuple[int, int]:
    """Report per-class image counts and sizes in data/raw/."""
    print("=" * 70)
    print("RAW STATUS (data/raw/)")
    print("=" * 70)
    if not RAW_DIR.exists():
        print(f"  {RAW_DIR} missing!")
        return 0, 0

    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if not classes:
        print("  (empty)")
        return 0, 0

    print(f"{'Class':<25} {'Images':>8} {'Size':>12}")
    print("-" * 50)

    total_count = total_size = 0
    for c in classes:
        n = count_images(c)
        s = folder_size_bytes(c)
        total_count += n
        total_size += s
        flag = "  <-- LOW" if n < 200 else ""
        print(f"{c.name:<25} {n:>8} {human_mb(s):>12}{flag}")

    print("-" * 50)
    print(f"{'TOTAL':<25} {total_count:>8} {human_mb(total_size):>12}\n")
    return total_count, total_size


def report_split() -> tuple[int, int]:
    """Report train/val/test split counts and sizes in data/processed/."""
    print("=" * 70)
    print("SPLIT STATUS (data/processed/)")
    print("=" * 70)
    if not PROCESSED_DIR.exists():
        print(f"  {PROCESSED_DIR} missing!\n")
        return 0, 0

    print(f"{'Split':<10} {'Classes':>7} {'Images':>8} {'Size':>12}")
    print("-" * 40)

    total_count = total_size = 0
    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split
        if not split_dir.exists():
            print(f"{split:<10} {'-':>7} {'0':>8} {'-':>12}")
            continue
        classes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        n = sum(count_images(c) for c in classes)
        s = folder_size_bytes(split_dir)
        total_count += n
        total_size += s
        print(f"{split:<10} {len(classes):>7} {n:>8} {human_mb(s):>12}")

    print("-" * 40)
    print(f"{'TOTAL':<10} {'':>7} {total_count:>8} {human_mb(total_size):>12}\n")
    return total_count, total_size


def report_other() -> dict[str, tuple[int, int]]:
    """Report brokendata, extradata, and suspicious folder statistics."""
    print("=" * 70)
    print("OTHER DIRECTORIES")
    print("=" * 70)

    folders = {
        "brokendata": BROKEN_DIR,
        "extradata": EXTRA_DIR,
        "suspicious": SUSPICIOUS_DIR,
    }
    results: dict[str, tuple[int, int]] = {}
    for name, path in folders.items():
        if not path.exists():
            print(f"{name:<25} {'MISSING':>8} {'-':>12}")
            results[name] = (0, 0)
            continue
        classes = sorted(p for p in path.iterdir() if p.is_dir())
        n = sum(count_images(c) for c in classes)
        s = folder_size_bytes(path)
        results[name] = (n, s)
        print(f"{name:<25} {n:>8} {human_mb(s):>12}  ({len(classes)} classes)")

    print()
    return results


def main() -> None:
    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}\n")

    raw_n, raw_s = report_raw()
    proc_n, proc_s = report_split()
    other = report_other()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Directory':<28} {'Images':>10} {'Size':>14}")
    print("-" * 55)
    print(f"{'raw':<28} {raw_n:>10} {human_mb(raw_s):>14}")
    print(f"{'processed (train+val+test)':<28} {proc_n:>10} {human_mb(proc_s):>14}")
    for name in ["brokendata", "extradata", "suspicious"]:
        print(f"{name:<28} {other[name][0]:>10} {human_mb(other[name][1]):>14}")
    print("-" * 55)

    grand_count = raw_n + proc_n + sum(v[0] for v in other.values())
    grand_size = raw_s + proc_s + sum(v[1] for v in other.values())
    print(f"{'TOTAL (disk)':<28} {grand_count:>10} {human_mb(grand_size):>14}\n")


if __name__ == "__main__":
    main()
