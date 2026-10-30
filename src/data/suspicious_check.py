"""
ChefAI - Birlesik suspicious check (watermark + menu card).

ESKI move_suspicious.py'nin Hough cizgi mantigini + yeni watermark
(kose kenar yogunlugu + text-like pattern) mantigini birlestirir.

AMAC: Gercek suphelileri yakalamak, ama SAGLAM gorselleri YANLISLIKLA
tasimamak. Eski kod perp_pairs >= 10 ile cok agresifti (tabak kenari,
masa kenari, peçete yakalaniyordu). Simdi:

  Watermark icin:  kose kenar yogunlugu + text-like pattern (kucuk
                   connected components cok olmali = yazi/logo)

  Menu karti icin: perp_pairs >= 15 VE cizgiler eksen-hizali (0/90 +-5)
                   VE cizgilerin cogu sinir yakini (kenar bosluğu)

  Karar:           Bu kosullar SAGLANMADAN tasima yok.

VARSAYILAN: Hicbir sey tasinmaz. Sadece rapor + opsiyonel preview.
Tasimak icin ACIKCA --move.

Kullanim:
    # 1) Guvenli test + gorsel onizleme
    poetry run python -m src.data.suspicious_check --preview

    # 2) Muhafazakar mod (yanlis pozitif iyice azalir)
    poetry run python -m src.data.suspicious_check --preview --conservative

    # 3) Belirli siniflar
    poetry run python -m src.data.suspicious_check --classes pizza,pide --preview

    # 4) Sadece watermark / sadece menu test et
    poetry run python -m src.data.suspicious_check --preview --no-menu
    poetry run python -m src.data.suspicious_check --preview --no-watermark

    # 5) GERCEKTEN TASI (onay ister)
    poetry run python -m src.data.suspicious_check --move
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUSPICIOUS_DIR = DATA_DIR / "suspicious"
REPORT_DIR = DATA_DIR / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
MAX_SIDE = 1024
CORNER_RATIO = 0.22

# Watermark esikleri (DAHA KATI)
WM_STRONG_SCORE = 0.25            # 0.15 -> 0.25
WM_WEAK_SCORE   = 0.20            # 0.10 -> 0.20
WM_MIN_COMPONENTS = 25            # 15 -> 25
WM_COMPONENT_MIN_AREA = 8         # 5 -> 8 (kucuk gurultu elemek icin)
WM_COMPONENT_MAX_AREA = 300       # 500 -> 300 (buyuk objeleri elemek icin)
WM_CORNER_DOMINANCE = 1.8         # YENI: baskin kose digerlerinin 1.8x'i olmali
WM_DOMINANT_TEXT_MIN = 25         # YENI: baskin kosede min text-like component

# Menu karti esikleri (DAHA KATI)
MENU_MIN_PERP     = 25            # 15 -> 25
MENU_MIN_AXIS     = 6             # 4 -> 6
MENU_MIN_BORDER   = 5             # 3 -> 5
MENU_ANGLE_TOL    = 5       # eksen hizalamasi icin +-5 derece
MENU_BORDER_RATIO = 0.15    # sinirdan uzaklik esigi (resmin %15'i)
MENU_MIN_LINE_LEN = 150
MENU_HOUGH_THRESH = 150
MENU_MAX_LINE_GAP = 4

# -------------------------------------------------------------------
# WATERMARK TESPITI
# -------------------------------------------------------------------
def detect_watermark(path: Path) -> tuple[float, dict]:
    """
    Kose kenar yogunlugu + text-like pattern + KOSE BASKINLIGI.

    Gercek filigran: TEK kose yuksek, digerleri dusuk (dominance >= 1.8).
    Normal yemek:    4 kose benzer (dominance ~1.0).

    Return: (max_corner_density, details)
    details = {
        "corner_scores": [4 float],
        "text_like_corners": int,
        "dominance": float,               # max / diger 3'un ortalamasi
        "dominant_corner": int,           # 0=SU, 1=SD, 2=AU, 3=AD
        "dominant_text_components": int,  # baskin kosedeki text-like sayisi
    }
    """
    empty = {
        "corner_scores": [], "text_like_corners": 0,
        "dominance": 1.0, "dominant_corner": -1, "dominant_text_components": 0,
    }
    img = cv2.imread(str(path))
    if img is None:
        return -1.0, empty

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    ch = max(int(h * CORNER_RATIO), 20)
    cw = max(int(w * CORNER_RATIO), 20)
    corners = [
        edges[:ch, :cw],        # 0: sol-ust
        edges[:ch, -cw:],       # 1: sag-ust
        edges[-ch:, :cw],       # 2: sol-alt
        edges[-ch:, -cw:],      # 3: sag-alt
    ]

    scores: list[float] = []
    text_per_corner: list[int] = []

    for c in corners:
        density = float((c > 0).mean())
        scores.append(density)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
        tc = sum(
            1 for i in range(1, n_labels)
            if WM_COMPONENT_MIN_AREA <= int(stats[i, cv2.CC_STAT_AREA]) <= WM_COMPONENT_MAX_AREA
        )
        text_per_corner.append(tc)

    max_score = max(scores) if scores else 0.0
    max_idx = scores.index(max_score) if scores else -1
    others = [s for i, s in enumerate(scores) if i != max_idx]
    others_mean = sum(others) / len(others) if others else 0.01
    dominance = max_score / max(others_mean, 0.01)

    text_like_corners = sum(1 for t in text_per_corner if t >= WM_MIN_COMPONENTS)
    dominant_text = text_per_corner[max_idx] if max_idx >= 0 else 0

    return max_score, {
        "corner_scores": scores,
        "text_like_corners": text_like_corners,
        "dominance": dominance,
        "dominant_corner": max_idx,
        "dominant_text_components": dominant_text,
    }


# -------------------------------------------------------------------
# MENU KARTI TESPITI (Hough cizgi + eksen + sinir)
# -------------------------------------------------------------------
def detect_menu_card(path: Path) -> tuple[float, dict]:
    """
    Hough cizgileri + eksen hizalama + sinir yakini kontrolu.

    Menukarti gercek imzasi:
      - Cok sayida dik aci cifti (>= 15)
      - Bu cizgilerin cogu yatay veya dikey (0/90 +-5)
      - Cogunlugu resmin kenarina yakin
      - Dikdortgen olusturur

    Return: (menu_score_0_1, details)
    """
    img = cv2.imread(str(path))
    if img is None:
        return -1.0, {}

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 250, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=MENU_HOUGH_THRESH,
        minLineLength=MENU_MIN_LINE_LEN,
        maxLineGap=MENU_MAX_LINE_GAP,
    )

    if lines is None:
        return 0.0, {"perp_pairs": 0, "axis_aligned": 0, "border_lines": 0, "total_lines": 0}

    line_data: list[tuple[float, float, bool, bool]] = []
    min_dim = min(h, w)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0

        # Eksen hizali mi? 0, 90 veya 180'e yakin
        dist_horiz = min(angle, 180 - angle)
        dist_vert = abs(angle - 90)
        is_axis = (dist_horiz < MENU_ANGLE_TOL) or (dist_vert < MENU_ANGLE_TOL)

        # Sinir yakini mi? (cizginin orta noktasi resim kenarina yakin)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        border_dist = min(cx, cy, w - cx, h - cy)
        near_border = border_dist < min_dim * MENU_BORDER_RATIO

        line_data.append((angle, length, is_axis, near_border))

    if len(line_data) < 2:
        return 0.0, {
            "perp_pairs": 0,
            "axis_aligned": sum(1 for l in line_data if l[2]),
            "border_lines": sum(1 for l in line_data if l[3]),
            "total_lines": len(line_data),
        }

    # Dik aci ciftleri
    perp_count = 0
    for i in range(len(line_data)):
        for j in range(i + 1, len(line_data)):
            a1, a2 = line_data[i][0], line_data[j][0]
            diff = abs(a1 - a2)
            if diff > 90:
                diff = 180 - diff
            if abs(diff - 90) <= 7:
                perp_count += 1

    axis_count = sum(1 for l in line_data if l[2])
    border_count = sum(1 for l in line_data if l[3])

    # SKOR mantigi — cok katmanli:
    # 1) perp_pairs < MENU_MIN_PERP -> asla "menu" sayma
    if perp_count < MENU_MIN_PERP:
        score = min(0.4, perp_count / (2.0 * MENU_MIN_PERP))
    # 2) perp_pairs var AMA eksen + sinir YOK -> "yemek fotografi (tabak vs)"
    elif axis_count < MENU_MIN_AXIS or border_count < MENU_MIN_BORDER:
        score = 0.45  # tam ortada -> tasima icin yetmez
    # 3) Uc sart birlikte -> gercek menu karti sinyali
    else:
        # perp 15 -> 0.70, perp 25 -> 0.90, perp 35+ -> 1.0
        score = min(1.0, 0.5 + (perp_count - MENU_MIN_PERP) / 40.0)

    return score, {
        "perp_pairs": perp_count,
        "axis_aligned": axis_count,
        "border_lines": border_count,
        "total_lines": len(line_data),
    }


# -------------------------------------------------------------------
# KARAR MANTIGI (konservatif)
# -------------------------------------------------------------------
def decide(
    wm_score: float,
    wm_info: dict,
    menu_score: float,
    menu_info: dict,
    conservative: bool,
) -> tuple[bool, str]:
    text_corners = int(wm_info.get("text_like_corners", 0))
    dom_text = int(wm_info.get("dominant_text_components", 0))
    dominance = float(wm_info.get("dominance", 1.0))
    perp = int(menu_info.get("perp_pairs", 0))
    axis = int(menu_info.get("axis_aligned", 0))
    border = int(menu_info.get("border_lines", 0))

    reasons: list[str] = []

    # ------- WATERMARK: 3 sart birlikte -------
    # 1) skor yuksek  2) baskin kose  3) baskin kosede text pattern
    if conservative:
        # Cok emin
        if wm_score >= 0.28 and dominance >= 2.5 and dom_text >= 30:
            reasons.append(f"wm_vstrong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.22 and dominance >= 2.0 and dom_text >= 20:
            reasons.append(f"wm_strong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
    else:
        # Skor + baskin kose + o kosede text pattern
        if wm_score >= 0.25 and dominance >= 1.8 and dom_text >= 25:
            reasons.append(f"wm_strong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.22 and dominance >= 2.2 and dom_text >= 15:
            reasons.append(f"wm_dom({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.18 and text_corners >= 3 and dominance >= 1.5:
            reasons.append(f"wm_multi({wm_score:.2f},corners={text_corners})")

    # ------- MENU: 4 sart birlikte -------
    if conservative:
        if menu_score >= 0.85 and perp >= 30 and axis >= 6 and border >= 5:
            reasons.append(f"menu_strong(perp={perp},axis={axis},border={border})")
    else:
        if menu_score >= 0.75 and perp >= 25 and axis >= 5 and border >= 4:
            reasons.append(f"menu_strong(perp={perp},axis={axis},border={border})")

    return len(reasons) > 0, " + ".join(reasons) if reasons else ""


# -------------------------------------------------------------------
# PARALEL WORKER
# -------------------------------------------------------------------
def _scan_worker(args: tuple[str, bool, bool, bool]) -> tuple[str, float, dict, float, dict]:
    path_str, do_wm, do_menu, conservative = args
    wm_score, wm_info = detect_watermark(Path(path_str)) if do_wm else (0.0, {})
    menu_score, menu_info = detect_menu_card(Path(path_str)) if do_menu else (0.0, {})
    return path_str, wm_score, wm_info, menu_score, menu_info


# -------------------------------------------------------------------
# TASIMA
# -------------------------------------------------------------------
def move_file(src: Path, class_name: str) -> bool:
    target_dir = SUSPICIOUS_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / src.name
    if target.exists():
        i = 1
        while (target_dir / f"{src.stem}_{i}{src.suffix}").exists():
            i += 1
        target = target_dir / f"{src.stem}_{i}{src.suffix}"
    try:
        shutil.move(str(src), str(target))
        return True
    except Exception as e:
        print(f"  TASIMA HATASI: {src.name}: {e}")
        return False


# -------------------------------------------------------------------
# CONTACT SHEET
# -------------------------------------------------------------------
def build_preview_data(
    scan_results: dict[str, list[dict]],
    top_n: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    random.seed(seed)
    preview: dict[str, list[dict]] = {}
    for cname, items in scan_results.items():
        if not items:
            preview[cname] = []
            continue
        susp = sorted(
            [it for it in items if it["is_suspicious"]],
            key=lambda x: x["score"],
            reverse=True,
        )
        clean = [it for it in items if not it["is_suspicious"]]
        random.shuffle(clean)

        if len(susp) >= top_n:
            preview[cname] = susp[:top_n]
        else:
            preview[cname] = susp + clean[: top_n - len(susp)]
    return preview


def render_contact_sheet(preview_data: dict[str, list[dict]], out_path: Path, n_cols: int = 8) -> None:
    import matplotlib.pyplot as plt

    n_rows = len(preview_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    if n_rows == 1:
        axes = np.array([axes])

    for row, (cname, samples) in enumerate(preview_data.items()):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cname, rotation=0, labelpad=60, fontsize=8, va="center")
            if col < len(samples):
                s = samples[col]
                try:
                    with Image.open(s["path"]) as im:
                        ax.imshow(im.convert("RGB"))
                    color = "red" if s["is_suspicious"] else "gray"
                    label = f"{s['score']:.2f}"
                    if s.get("reason"):
                        label = "W" if "wm" in s["reason"] else "M"
                        label += f" {s['score']:.2f}"
                    ax.set_title(label, fontsize=7, color=color)
                except Exception:
                    ax.text(0.5, 0.5, "ERR", ha="center", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=70, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Birlesik watermark + menu karti supheli tespiti. TASIMA YOK (default)."
    )
    parser.add_argument("--move", action="store_true", help="Gercekten tasi")
    parser.add_argument("--preview", action="store_true", help="Contact sheet uret")
    parser.add_argument("--conservative", action="store_true",
                        help="Daha katı esikler (yanlis pozitif cok daha az)")
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--no-watermark", action="store_true", help="Watermark tespiti kapat")
    parser.add_argument("--no-menu", action="store_true", help="Menu karti tespiti kapat")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--csv", action="store_true", help="CSV raporu yaz")
    parser.add_argument("--yes", action="store_true", help="Tasima onayi sorma")
    parser.add_argument("--force", action="store_true", help="%%15 uyari esigini yoksay")
    args = parser.parse_args()

    do_wm = not args.no_watermark
    do_menu = not args.no_menu

    if not do_wm and not do_menu:
        print("HATA: en az bir tespit acik olmali (--no-watermark ve --no-menu birlikte verilemez)")
        return

    mode = "MOVE (onay ister)" if args.move else "DRY RUN (guvenli)"
    print("=" * 70)
    print("  ChefAI - Birlesik Suspicious Check")
    print("=" * 70)
    print(f"Mod          : {mode}")
    print(f"Watermark    : {'ACIK' if do_wm else 'kapali'}")
    print(f"Menu karti   : {'ACIK' if do_menu else 'kapali'}")
    print(f"Muhafazakar  : {'EVET' if args.conservative else 'HAYIR'}")
    print(f"Raw dir      : {RAW_DIR}")
    print(f"Cikti        : {SUSPICIOUS_DIR}")
    print()

    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} yok")
        return

    classes = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        classes = [c for c in classes if c.name in wanted]
        missing = wanted - {c.name for c in classes}
        if missing:
            print(f"UYARI: bulunamayan siniflar: {sorted(missing)}")

    print(f"Sinif sayisi: {len(classes)}")
    print()

    # ---------------- TARA ----------------
    all_paths: list[tuple[str, str]] = []  # (class_name, path_str)
    for class_dir in classes:
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                all_paths.append((class_dir.name, str(p)))

    if not all_paths:
        print("HATA: hic gorsel bulunamadi")
        return

    print(f"Toplam {len(all_paths)} gorsel taranacak ({args.workers} worker)...")
    print()

    scan_results: dict[str, list[dict]] = {c.name: [] for c in classes}
    class_of: dict[str, str] = {p: c for c, p in all_paths}

    tasks = [(p, do_wm, do_menu, args.conservative) for _, p in all_paths]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (path_str, wm_score, wm_info, menu_score, menu_info) in enumerate(
            ex.map(_scan_worker, tasks, chunksize=8), 1
        ):
            if i % 200 == 0:
                print(f"  ... {i}/{len(tasks)}")

            cname = class_of[path_str]

            if wm_score < 0 or menu_score < 0:
                continue

            is_susp, reason = decide(
                wm_score, wm_info, menu_score, menu_info, args.conservative
            )
            combined = max(wm_score, menu_score)

            scan_results[cname].append(
                {
                    "path": path_str,
                    "wm_score": wm_score,
                    "menu_score": menu_score,
                    "score": combined,
                    "is_suspicious": is_susp,
                    "reason": reason,
                }
            )

    print()

    # ---------------- OZET ----------------
    print(f"{'Sinif':<25} {'Toplam':>7} {'Supheli':>8} {'%':>7} {'OrtSkor':>9}")
    print("-" * 65)
    grand_total = grand_susp = 0
    for cname, items in scan_results.items():
        n = len(items)
        s = sum(1 for it in items if it["is_suspicious"])
        grand_total += n
        grand_susp += s
        if n == 0:
            print(f"{cname:<25} {0:>7} {0:>8} {'-':>7} {'-':>9}")
            continue
        pct = 100 * s / n
        avg = sum(it["score"] for it in items) / n
        print(f"{cname:<25} {n:>7} {s:>8} {pct:>6.1f}% {avg:>9.3f}")
    print("-" * 65)
    pct_all = 100 * grand_susp / max(grand_total, 1)
    print(f"{'TOPLAM':<25} {grand_total:>7} {grand_susp:>8} {pct_all:>6.1f}%")
    print()

    # ---------------- CSV ----------------
    if args.csv:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPORT_DIR / f"suspicious_check_{ts}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["class", "filename", "wm_score", "menu_score", "combined", "is_suspicious", "reason"]
            )
            for cname, items in scan_results.items():
                for it in items:
                    w.writerow([
                        cname,
                        Path(it["path"]).name,
                        f"{it['wm_score']:.3f}",
                        f"{it['menu_score']:.3f}",
                        f"{it['score']:.3f}",
                        int(it["is_suspicious"]),
                        it["reason"],
                    ])
        print(f"CSV raporu: {csv_path}")
        print()

    # ---------------- PREVIEW ----------------
    if args.preview:
        print("Contact sheet uretiliyor...")
        preview_data = build_preview_data(scan_results, top_n=args.top)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPORT_DIR / f"preview_{ts}.png"
        render_contact_sheet(preview_data, out)
        print(f"  -> {out}")
        print(f"  Ac : xdg-open '{out}'")
        print("  Kirmizi baslik = supheli (tasima adayi), gri = temiz")
        print("  W = watermark, M = menu karti")
        print()

        # ---------------- MOVE ----------------
    if args.move:
        if grand_susp == 0:
            print("Tasima adayi yok. Cikiliyor.")
            return

        # GUVENLIK: %15'ten fazla tasima -> yanlis pozitif olabilir
        if pct_all > 15 and not args.force:
            print("=" * 70)
            print(f"  REDDEDILDI: Supheli orani %{pct_all:.1f} (>15%)")
            print("=" * 70)
            print("  Bu oran cok yuksek -> threshold'lar cok gevsek olabilir.")
            print("  Secenekler:")
            print("    1) --conservative ekle ve tekrar dene")
            print("    2) --preview ile gozle dogrula")
            print("    3) Gercekten bu kadar supheli varsa: --force ekle")
            return

        print("=" * 70)
        print(f"  {grand_susp} gorsel tasinacak -> {SUSPICIOUS_DIR}")
        print("=" * 70)
        if not args.yes:
            ans = input("Devam edilsin mi? (yes yaz): ").strip()
            if ans != "yes":
                print("Iptal edildi.")
                return

        print("=" * 70)
        print(f"  {grand_susp} gorsel tasinacak -> {SUSPICIOUS_DIR}")
        print("=" * 70)
        if not args.yes:
            ans = input("Devam edilsin mi? (yes yaz): ").strip()
            if ans != "yes":
                print("Iptal edildi.")
                return

        moved = 0
        for cname, items in scan_results.items():
            for it in items:
                if not it["is_suspicious"]:
                    continue
                src = Path(it["path"])
                if not src.exists():
                    continue
                if move_file(src, cname):
                    moved += 1
        print(f"\nTasinan: {moved} gorsel")
        print()
        print("Sonraki adimlar:")
        print("  make data-move-broken")
        print("  make data-remove-dups")
        print("  make data-trim")
        print("  make data-split")
        print("  make data-renumber")
    else:
        print("=" * 70)
        print("  DRY RUN - HICBIR SEY TASINMADI")
        print("=" * 70)
        if not args.preview:
            print("  Preview icin: --preview ekle")
        print("  Preview'de kirmizi basliklilar GERCEKTEN filigran/menu mu bak")
        print("  Dogruysa tasimak icin: --move --yes")
        print()
        if pct_all > 15:
            print(f"  UYARI: Supheli %{pct_all:.1f} -> cok yuksek. --conservative dene.")
        elif pct_all < 1:
            print(f"  UYARI: Supheli %{pct_all:.1f} -> cok dusuk. Muhafazakar flag'i kapali mi?")
        else:
            print(f"  Supheli %{pct_all:.1f} -> normal aralik.")


if __name__ == "__main__":
    main()
