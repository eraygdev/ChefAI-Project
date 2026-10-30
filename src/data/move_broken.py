"""
ChefAI - Bozuk ve cok kucuk gorselleri data/brokendata/ altina tasir.

Iyilestirmeler:
  - Paralel kontrol (4 worker, RAM dostu)
  - Iki asama: once header (hizli boyut), sonra pixel (bozukluk)
  - Progress bar (tqdm)
  - --dry-run destegi
  - Detayli hata raporu (CSV)
  - EXIF rotation destegi
  - utils.py bagimliligi dogru

Kullanim:
    # Test (hicbir sey tasima)
    poetry run python src/data/move_broken.py --dry-run

    # Gercek tasima (default 4 worker)
    poetry run python src/data/move_broken.py

    # 8 worker (daha hizli, daha fazla RAM)
    poetry run python src/data/move_broken.py --workers 8

    # Sadece hizli tarama (pixel decode yok, bozukluk kacabilir)
    poetry run python src/data/move_broken.py --quick

    # Yeniden numaralandirmayi atla
    poetry run python src/data/move_broken.py --no-renumber
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import warnings

from PIL import Image, ImageOps
from tqdm import tqdm

# Pillow'un EXIF/corrupt metadata uyarilarini sustur
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# Proje kokunu sys.path'e ekle
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.utils import list_images, renumber_class  # noqa: E402

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BROKEN_DIR = DATA_DIR / "brokendata"
REPORT_DIR = DATA_DIR / "reports"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
MIN_SIZE = 200
DEFAULT_WORKERS = 4


# -------------------------------------------------------------------
# TEK GORSEL KONTROL (paralel worker)
# -------------------------------------------------------------------
def check_single_image(path_str: str) -> tuple[str, str, int, int, str]:
    """
    Tek gorsel kontrolu.

    Donus: (status, path_str, width, height, reason)
        status: "ok" | "small" | "broken" | "missing"
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    path = Path(path_str)

    if not path.exists():
        return ("missing", path_str, 0, 0, "dosya yok")

    try:
        # 1. Header ac (lazy - pixel decode yok)
        with Image.open(path) as img:
            w, h = img.size

            # 2. EXIF rotation (sadece dikey/yatay karisikligi icin)
            try:
                transposed = ImageOps.exif_transpose(img)
                if transposed is not None:
                    w, h = transposed.size
            except Exception:
                pass

            # 3. Boyut kontrolu (hizli eleme - pixel decode gerekmez)
            if w < MIN_SIZE or h < MIN_SIZE:
                return ("small", path_str, w, h, f"{w}x{h}")

            # 4. Pixel decode (bozukluk kontrolu - yavas)
            try:
                img.load()
            except Exception as e:
                return ("broken", path_str, w, h, f"load: {type(e).__name__}")

            return ("ok", path_str, w, h, "")

    except FileNotFoundError:
        return ("missing", path_str, 0, 0, "file not found")
    except Image.UnidentifiedImageError:
        return ("broken", path_str, 0, 0, "format taninmadi")
    except OSError as e:
        return ("broken", path_str, 0, 0, f"OSError: {e}")
    except Exception as e:
        return ("broken", path_str, 0, 0, f"{type(e).__name__}: {e}")


# -------------------------------------------------------------------
# TASIMA
# -------------------------------------------------------------------
def move_to_broken(src: Path, class_name: str) -> Path | None:
    """Gorseli brokendata/<class_name>/ altina tasir."""
    target_dir = BROKEN_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / src.name
    if target.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while (target_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = target_dir / f"{stem}_{i}{suffix}"

    try:
        shutil.move(str(src), str(target))
        return target
    except Exception as e:
        print(f"    TASIMA HATASI: {src.name}: {e}")
        return None


# -------------------------------------------------------------------
# BIR SINIFI ISLE
# -------------------------------------------------------------------
def process_class(
    class_dir: Path,
    dry_run: bool,
    executor: ProcessPoolExecutor,
    csv_writer: Any = None,
) -> dict:
    """Bir siniftaki tum gorselleri kontrol eder."""
    class_name = class_dir.name
    images = list_images(class_dir)

    stats = {"total": len(images), "ok": 0, "small": 0, "broken": 0, "missing": 0}

    if not images:
        return stats

    # Paralel kontrol
    results = list(
        tqdm(
            executor.map(check_single_image, [str(p) for p in images]),
            total=len(images),
            desc=f"[{class_name}]",
            leave=False,
            ncols=80,
        )
    )

    # Sonuclari isle
    for status, path_str, w, h, reason in results:
        stats[status] += 1

        if csv_writer:
            csv_writer.writerow([class_name, Path(path_str).name, status, w, h, reason])

        if status == "ok":
            continue

        # Tasima (sadece dry_run degilse)
        if not dry_run:
            move_to_broken(Path(path_str), class_name)

    return stats


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bozuk ve cok kucuk gorselleri tespit edip tasir."
    )
    parser.add_argument("--dry-run", action="store_true", help="Hicbir sey tasima, sadece rapor")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Paralel worker sayisi (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--no-renumber", action="store_true", help="Yeniden numaralandirmayi atla")
    parser.add_argument("--csv", type=str, default="", help="CSV rapor yolu")
    parser.add_argument("--quick", action="store_true", help="Sadece header kontrolu (hizli)")
    args = parser.parse_args()

    print(f"Raw dir     : {RAW_DIR}")
    print(f"Broken dir  : {BROKEN_DIR}")
    print(f"Min size    : {MIN_SIZE}x{MIN_SIZE}")
    print(f"Workers     : {args.workers}")
    print(f"Mode        : {'DRY RUN' if args.dry_run else 'MOVE'}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    # CSV rapor
    if args.csv:
        csv_path = Path(args.csv)
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPORT_DIR / f"move_broken_{timestamp}.csv"

    csv_file = None
    csv_writer = None
    if not args.dry_run:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["class", "filename", "status", "width", "height", "reason"])

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    # 256x185 crop'lu siniflar: MIN_SIZE 200'u gecmez, oldugu gibi birak
    SKIP_LOW_RES = {"kuru_fasulye", "yaprak_sarma", "simit", "lahmacun"}
    classes = [c for c in classes if c.name not in SKIP_LOW_RES]
    print(f"Sinif sayisi: {len(classes)} (atlana: {sorted(SKIP_LOW_RES)})")
    print()

    grand_total = {"total": 0, "ok": 0, "small": 0, "broken": 0, "missing": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for class_dir in classes:
            s = process_class(class_dir, args.dry_run, executor, csv_writer)
            for k in grand_total:
                grand_total[k] += s[k]

            print(
                f"  {class_dir.name:<25} "
                f"toplam={s['total']:>4} "
                f"ok={s['ok']:>4} "
                f"kucuk={s['small']:>3} "
                f"bozuk={s['broken']:>3} "
                f"eksik={s['missing']:>2}"
            )

    print()
    print("=" * 60)
    print("TOPLAM")
    print("=" * 60)
    print(f"Toplam gorsel   : {grand_total['total']}")
    print(f"OK (kalan)      : {grand_total['ok']}")
    print(f"Tasinan         : {grand_total['small'] + grand_total['broken'] + grand_total['missing']}")
    print(f"  - Cok kucuk   : {grand_total['small']}")
    print(f"  - Bozuk       : {grand_total['broken']}")
    print(f"  - Eksik       : {grand_total['missing']}")
    print()

    if csv_file:
        csv_file.close()
        print(f"CSV rapor : {csv_path}")

    # Yeniden numaralandirma
    if not args.no_renumber and not args.dry_run:
        print()
        print("Yeniden numaralandiriliyor (raw/ temiz siniflar)...")
        total_renamed = 0
        for class_dir in classes:
            if class_dir.exists():
                total_renamed += renumber_class(class_dir, dry_run=False)
        print(f"Toplam yeniden adlandirilan: {total_renamed}")

    print()
    print("Sonraki adim: poetry run python -m src.data.remove_duplicates")


if __name__ == "__main__":
    main()
