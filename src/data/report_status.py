"""
ChefAI - data/ klasorunun durumunu raporlar.
Gorsel sayisi + disk boyutu (MB) gosterir.

Kullanim:
    poetry run python -m src.data.report_status
"""

import sys
from pathlib import Path

# Proje kokunu sys.path'e ekle (src.data.utils importu icin)
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.utils import count_images  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def folder_size_bytes(folder: Path) -> int:
    """Klasordeki tum dosyalarin toplam boyutu (byte)."""
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


def human_mb(n_bytes: int) -> str:
    """Byte -> MB (1 ondalik)."""
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def class_row(name: str, count: int, size_bytes: int, width: int = 25) -> str:
    """Tek satir format: sinif adi, sayi, MB."""
    return f"{name:<{width}} {count:>8} {human_mb(size_bytes):>12}"


# -------------------------------------------------------------------
# RAW
# -------------------------------------------------------------------
def report_raw() -> tuple[int, int]:
    """data/raw/ altindaki siniflari raporlar. (count, size_bytes) doner."""
    print("=" * 70)
    print("RAW DURUMU (data/raw/)")
    print("=" * 70)

    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        print(f"  {raw_dir} yok!")
        return 0, 0

    classes = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not classes:
        print("  (bos)")
        return 0, 0

    print(f"{'Sinif':<25} {'Gorsel':>8} {'Boyut':>12}")
    print("-" * 50)

    total_count = 0
    total_size = 0
    for c in classes:
        n = count_images(c)
        s = folder_size_bytes(c)
        total_count += n
        total_size += s
        flag = "  <-- AZ" if n < 200 else ""
        print(class_row(c.name, n, s) + flag)

    print("-" * 50)
    print(f"{'TOPLAM':<25} {total_count:>8} {human_mb(total_size):>12}")
    print()
    return total_count, total_size


# -------------------------------------------------------------------
# PROCESSED (train / val / test)
# -------------------------------------------------------------------
def report_split() -> tuple[int, int]:
    """data/processed/{train,val,test} raporu. (count, size_bytes) doner."""
    print("=" * 70)
    print("SPLIT DURUMU (data/processed/)")
    print("=" * 70)

    processed_dir = DATA_DIR / "processed"
    if not processed_dir.exists():
        print(f"  {processed_dir} yok!")
        return 0, 0

    print(f"{'Split':<10} {'Sinif':>7} {'Gorsel':>8} {'Boyut':>12}")
    print("-" * 40)

    total_count = 0
    total_size = 0
    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            print(f"{split:<10} {'-':>7} {'0':>8} {'-':>12}")
            continue

        classes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        n = sum(count_images(c) for c in classes)
        s = folder_size_bytes(split_dir)
        total_count += n
        total_size += s
        print(f"{split:<10} {len(classes):>7} {n:>8} {human_mb(s):>12}")

    print("-" * 40)
    print(f"{'TOPLAM':<10} {'':>7} {total_count:>8} {human_mb(total_size):>12}")
    print()

    # Sinif bazli train/val/test dagilimi
    if any((processed_dir / s).exists() for s in ["train", "val", "test"]):
        print(f"{'Sinif':<25} {'Train':>7} {'Val':>6} {'Test':>6} {'Toplam':>8} {'Boyut':>12}")
        print("-" * 70)

        class_names: set[str] = set()
        for split in ["train", "val", "test"]:
            sd = processed_dir / split
            if sd.exists():
                class_names.update(p.name for p in sd.iterdir() if p.is_dir())

        grand = {"train": 0, "val": 0, "test": 0}
        for name in sorted(class_names):
            counts = {}
            cls_size = 0
            for split in ["train", "val", "test"]:
                sd = processed_dir / split / name
                if sd.exists():
                    counts[split] = count_images(sd)
                    cls_size += folder_size_bytes(sd)
                    grand[split] += counts[split]
                else:
                    counts[split] = 0
            total = sum(counts.values())
            print(
                f"{name:<25} {counts['train']:>7} {counts['val']:>6} "
                f"{counts['test']:>6} {total:>8} {human_mb(cls_size):>12}"
            )

        print("-" * 70)
        grand_total = sum(grand.values())
        print(
            f"{'TOPLAM':<25} {grand['train']:>7} {grand['val']:>6} "
            f"{grand['test']:>6} {grand_total:>8} {human_mb(total_size):>12}"
        )
        print()

    return total_count, total_size


# -------------------------------------------------------------------
# BROKENDATA + EXTRADATA
# -------------------------------------------------------------------
def report_other() -> dict[str, tuple[int, int]]:
    """brokendata ve extradata raporu. {name: (count, size)} doner."""
    print("=" * 70)
    print("DIGER KLASORLER")
    print("=" * 70)

    results: dict[str, tuple[int, int]] = {}

    for name in ["brokendata", "extradata"]:
        d = DATA_DIR / name
        if not d.exists():
            print(f"{name:<25} {'YOK':>8} {'-':>12}")
            results[name] = (0, 0)
            continue

        classes = sorted([p for p in d.iterdir() if p.is_dir()])
        n = sum(count_images(c) for c in classes)
        s = folder_size_bytes(d)
        results[name] = (n, s)
        print(f"{name:<25} {n:>8} {human_mb(s):>12}  ({len(classes)} sinif)")

    print()
    return results


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    print()
    print(f"Proje koku : {PROJECT_ROOT}")
    print(f"Data dir   : {DATA_DIR}")
    print()

    raw_n, raw_s = report_raw()
    proc_n, proc_s = report_split()
    other = report_other()

    # Genel ozet
    print("=" * 70)
    print("GENEL OZET")
    print("=" * 70)
    print(f"{'Klasor':<25} {'Gorsel':>10} {'Boyut':>14}")
    print("-" * 55)
    print(f"{'raw':<25} {raw_n:>10} {human_mb(raw_s):>14}")
    print(f"{'processed (train+val+test)':<25} {proc_n:>10} {human_mb(proc_s):>14}")
    print(f"{'brokendata':<25} {other['brokendata'][0]:>10} {human_mb(other['brokendata'][1]):>14}")
    print(f"{'extradata':<25} {other['extradata'][0]:>10} {human_mb(other['extradata'][1]):>14}")
    print("-" * 55)

    grand_count = raw_n + proc_n + other["brokendata"][0] + other["extradata"][0]
    grand_size = raw_s + proc_s + other["brokendata"][1] + other["extradata"][1]
    print(f"{'TOPLAM (disk)':<25} {grand_count:>10} {human_mb(grand_size):>14}")
    print()


if __name__ == "__main__":
    main()
