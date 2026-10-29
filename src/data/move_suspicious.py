"""
ChefAI - Duz cizgi / filigran / menu karti supheli gorselleri tespit eder.

Bir gorsel "supheli" sayilirsa:
  - En az MIN_PERP_PAIRS adet dik acili cizgi cifti varsa
  - Cizgiler MIN_LINE_LENGTH pikselden uzunsa
  - Aci toleransi +-ANGLE_TOLERANCE derece icindeyse

data/raw/<sinif>/  ->  data/suspicious/<sinif>/

Kullanim:
    # On izleme (hicbir sey tasima, sadece rapor)
    poetry run python src/data/move_suspicious.py --dry-run

    # Tasima
    poetry run python src/data/move_suspicious.py

    # Sadece belirli siniflar
    poetry run python src/data/move_suspicious.py --classes pizza,borek

    # Kopyalama (silmeden test)
    poetry run python src/data/move_suspicious.py --copy
"""

from __future__ import annotations

import argparse
import csv
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUSPICIOUS_DIR = DATA_DIR / "suspicious"
REPORT_DIR = DATA_DIR / "reports"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Canny edge detection
CANNY_LOW = 100
CANNY_HIGH = 250
APERTURE_SIZE = 3

# Hough line detection
HOUGH_THRESHOLD = 150
MIN_LINE_LENGTH = 150
MAX_LINE_GAP = 4

# Dik aci tolerans
ANGLE_TOLERANCE = 7
MIN_PERP_PAIRS = 8
MAX_IMAGE_SIDE = 1280  # daha yuksek cozunurluk

# -------------------------------------------------------------------
# CIKARIM: dik acili cizgi ciftlerini say
# -------------------------------------------------------------------
def count_perpendicular_pairs(image_path: Path) -> tuple[int, int]:
    """
    Gorseldeki dik acili cizgi ciftlerini sayar.

    Donus: (dik_cift_sayisi, toplam_cizgi_sayisi)
    Hata olursa (-1, -1)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return (-1, -1)

    # Hiz icin resize
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, apertureSize=APERTURE_SIZE)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if lines is None:
        return (0, 0)

    # Her cizginin aci ve uzunlugunu hesapla
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
        line_data.append((angle, length))

    if len(line_data) < 2:
        return (0, len(line_data))

    # Dik acili ciftleri say
    perp_count = 0
    for i in range(len(line_data)):
        for j in range(i + 1, len(line_data)):
            a1 = line_data[i][0]
            a2 = line_data[j][0]
            diff = abs(a1 - a2)
            if diff > 90:
                diff = 180 - diff
            # diff 90'a yakinsa dik aci
            if abs(diff - 90) <= ANGLE_TOLERANCE:
                perp_count += 1

    return (perp_count, len(line_data))


# -------------------------------------------------------------------
# TASIMA / KOPYALAMA
# -------------------------------------------------------------------
def move_or_copy(
    src: Path, dst_dir: Path, class_name: str, mode: str, dry_run: bool
) -> str | None:
    """
    Dosyayi tasir veya kopyalar. Cakisma olursa isim degistirir.

    mode: "move" | "copy"
    dry_run: True ise hicbir sey yapmaz
    """
    if dry_run:
        return str(dst_dir / src.name)

    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name

    if target.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while (dst_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = dst_dir / f"{stem}_{i}{suffix}"

    try:
        if mode == "copy":
            shutil.copy2(str(src), str(target))
        else:
            shutil.move(str(src), str(target))
        return str(target)
    except Exception as e:
        print(f"    TASIMA HATASI: {src.name}: {e}")
        return None


# -------------------------------------------------------------------
# TEK GORSEL KONTROL (paralel icin)
# -------------------------------------------------------------------
def check_single_image(img_path: Path) -> tuple[str, int, int]:
    """Tek gorsel kontrolu. Paralel cagrilir.

    Donus: (dosya_adi, perp_pairs, total_lines)
    Hata: (-1, -1)
    """
    perp, total = count_perpendicular_pairs(img_path)
    return (img_path.name, perp, total)


# -------------------------------------------------------------------
# BIR SINIFI ISLE
# -------------------------------------------------------------------
def process_class(
    class_dir: Path,
    class_name: str,
    mode: str,
    dry_run: bool,
    csv_writer: Any = None,
) -> dict:
    """Bir siniftaki tum gorselleri kontrol eder, suphelileri ayirir."""
    stats = {
        "total": 0,
        "suspicious": 0,
        "clean": 0,
        "errors": 0,
    }

    images = sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    stats["total"] = len(images)

    # Paralel kontrol (4 worker)
    results: list[tuple[Path, int, int]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_path = {
            executor.submit(check_single_image, img_path): img_path for img_path in images
        }
        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            try:
                name, perp, total = future.result()
                results.append((img_path, perp, total))
            except Exception as e:
                print(f"    [{class_name}] HATA: {img_path.name}: {e}")
                stats["errors"] += 1

    # Sonuclari isle
    for img_path, perp, total in results:
        if perp == -1:
            stats["errors"] += 1
            if csv_writer:
                csv_writer.writerow([class_name, img_path.name, "ERROR", "", ""])
            continue

        if perp >= MIN_PERP_PAIRS:
            stats["suspicious"] += 1
            dst_dir = SUSPICIOUS_DIR / class_name
            new_path = move_or_copy(img_path, dst_dir, class_name, mode, dry_run)
            if csv_writer:
                csv_writer.writerow([class_name, img_path.name, "SUSPICIOUS", perp, total])
            if new_path and not dry_run:
                print(f"    [{class_name}] supheli: {img_path.name} (perp={perp})")
        else:
            stats["clean"] += 1
            if csv_writer:
                csv_writer.writerow([class_name, img_path.name, "CLEAN", perp, total])

    return stats


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duz cizgi / filigran supheli gorselleri ayiklar."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hicbir sey tasima, sadece rapor ver",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Tasima yerine kopyala (orijinali korur)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Virgulle ayrilmis sinif listesi (bos = hepsi)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="CSV rapor yolu (bos = data/reports/suspicious_report.csv)",
    )
    args = parser.parse_args()

    mode = "copy" if args.copy else "move"
    dry_run = args.dry_run

    print(f"Raw dir        : {RAW_DIR}")
    print(f"Suspicious dir : {SUSPICIOUS_DIR}")
    print(f"Mode           : {mode.upper()}{' (DRY RUN)' if dry_run else ''}")
    print(f"Min perp pairs : {MIN_PERP_PAIRS}")
    print(f"Min line length: {MIN_LINE_LENGTH}")
    print(f"Angle tolerance: +-{ANGLE_TOLERANCE} deg")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok!")
        return

    # CSV raporu
    if args.csv:
        csv_path = Path(args.csv)
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = REPORT_DIR / "suspicious_report.csv"

    csv_file = None
    csv_writer = None
    if not dry_run:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["class", "filename", "status", "perp_pairs", "total_lines"])

    # Sinif listesi
    all_classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        classes = [c for c in all_classes if c.name in wanted]
        missing = wanted - {c.name for c in classes}
        if missing:
            print(f"UYARI: bulunamayan siniflar: {sorted(missing)}")
    else:
        classes = all_classes

    print(f"Sinif sayisi: {len(classes)}")
    print()

    print(f"{'Sinif':<25} {'Toplam':>7} {'Supheli':>8} {'Temiz':>7} {'Hata':>6}")
    print("-" * 60)

    totals = {"total": 0, "suspicious": 0, "clean": 0, "errors": 0}

    for class_dir in classes:
        s = process_class(class_dir, class_dir.name, mode, dry_run, csv_writer)
        for k in totals:
            totals[k] += s[k]

        print(
            f"{class_dir.name:<25} "
            f"{s['total']:>7} {s['suspicious']:>8} "
            f"{s['clean']:>7} {s['errors']:>6}"
        )

    print("-" * 60)
    print(
        f"{'TOPLAM':<25} "
        f"{totals['total']:>7} {totals['suspicious']:>8} "
        f"{totals['clean']:>7} {totals['errors']:>6}"
    )
    print()

    if csv_file:
        csv_file.close()
        print(f"CSV rapor : {csv_path}")

    if dry_run:
        print()
        print("DRY RUN: hicbir sey tasinmadi. Gercekten tasimak icin --dry-run'i kaldir.")
    else:
        print()
        print(f"Supheli gorseller : {SUSPICIOUS_DIR}")
        print(f"Tasima modu       : {mode}")
        print()
        print("Sonraki adim: poetry run python src/data/move_broken.py")


if __name__ == "__main__":
    main()
