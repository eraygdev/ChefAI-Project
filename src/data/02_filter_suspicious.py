"""Detect watermark and menu-card candidates using edge and Hough line analysis.

Dry-run by default. Pass --move to relocate detected images to data/suspicious/.

Usage:
    poetry run python -m src.data.02_filter_suspicious --preview
    poetry run python -m src.data.02_filter_suspicious --preview --conservative
    poetry run python -m src.data.02_filter_suspicious --move --yes
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.utils import IMAGE_EXTENSIONS, RAW_DIR, REPORT_DIR, SUSPICIOUS_DIR, safe_move

MAX_SIDE = 1024
CORNER_RATIO = 0.22

WM_MIN_COMPONENTS = 25
WM_COMPONENT_MIN_AREA = 8
WM_COMPONENT_MAX_AREA = 300

MENU_MIN_PERP = 25
MENU_MIN_AXIS = 6
MENU_MIN_BORDER = 5
MENU_ANGLE_TOL = 5
MENU_BORDER_RATIO = 0.15
MENU_MIN_LINE_LEN = 150
MENU_HOUGH_THRESH = 150
MENU_MAX_LINE_GAP = 4


def detect_watermark(path: Path) -> tuple[float, dict[str, Any]]:
    """Estimate corner watermark probability via edge density and corner dominance."""
    empty: dict[str, Any] = {
        "corner_scores": [],
        "text_like_corners": 0,
        "dominance": 1.0,
        "dominant_corner": -1,
        "dominant_text_components": 0,
    }
    img = cv2.imread(str(path))
    if img is None:
        return -1.0, empty

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    ch = max(int(h * CORNER_RATIO), 20)
    cw = max(int(w * CORNER_RATIO), 20)
    corners = [
        edges[:ch, :cw],
        edges[:ch, -cw:],
        edges[-ch:, :cw],
        edges[-ch:, -cw:],
    ]

    scores: list[float] = []
    text_per_corner: list[int] = []

    for corner in corners:
        density = float((corner > 0).mean())
        scores.append(density)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(corner, connectivity=8)
        tc = sum(
            1
            for i in range(1, n_labels)
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


def detect_menu_card(path: Path) -> tuple[float, dict[str, int]]:
    """Detect menu cards via axis-aligned perpendicular Hough lines near borders."""
    img = cv2.imread(str(path))
    if img is None:
        return -1.0, {}

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
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

        dist_horiz = min(angle, 180 - angle)
        dist_vert = abs(angle - 90)
        is_axis = (dist_horiz < MENU_ANGLE_TOL) or (dist_vert < MENU_ANGLE_TOL)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        border_dist = min(cx, cy, w - cx, h - cy)
        near_border = border_dist < min_dim * MENU_BORDER_RATIO

        line_data.append((angle, length, is_axis, near_border))

    if len(line_data) < 2:
        return 0.0, {
            "perp_pairs": 0,
            "axis_aligned": sum(1 for item in line_data if item[2]),
            "border_lines": sum(1 for item in line_data if item[3]),
            "total_lines": len(line_data),
        }

    perp_count = 0
    for i in range(len(line_data)):
        for j in range(i + 1, len(line_data)):
            a1, a2 = line_data[i][0], line_data[j][0]
            diff = abs(a1 - a2)
            if diff > 90:
                diff = 180 - diff
            if abs(diff - 90) <= 7:
                perp_count += 1

    axis_count = sum(1 for item in line_data if item[2])
    border_count = sum(1 for item in line_data if item[3])

    if perp_count < MENU_MIN_PERP:
        score = min(0.4, perp_count / (2.0 * MENU_MIN_PERP))
    elif axis_count < MENU_MIN_AXIS or border_count < MENU_MIN_BORDER:
        score = 0.45
    else:
        score = min(1.0, 0.5 + (perp_count - MENU_MIN_PERP) / 40.0)

    return score, {
        "perp_pairs": perp_count,
        "axis_aligned": axis_count,
        "border_lines": border_count,
        "total_lines": len(line_data),
    }


def decide(
    wm_score: float,
    wm_info: dict[str, Any],
    menu_score: float,
    menu_info: dict[str, int],
    conservative: bool,
) -> tuple[bool, str]:
    """Decide whether an image is suspicious based on watermark and menu signals."""
    text_corners = int(wm_info.get("text_like_corners", 0))
    dom_text = int(wm_info.get("dominant_text_components", 0))
    dominance = float(wm_info.get("dominance", 1.0))
    perp = int(menu_info.get("perp_pairs", 0))
    axis = int(menu_info.get("axis_aligned", 0))
    border = int(menu_info.get("border_lines", 0))

    reasons: list[str] = []

    if conservative:
        if wm_score >= 0.28 and dominance >= 2.5 and dom_text >= 30:
            reasons.append(f"wm_vstrong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.22 and dominance >= 2.0 and dom_text >= 20:
            reasons.append(f"wm_strong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
    else:
        if wm_score >= 0.25 and dominance >= 1.8 and dom_text >= 25:
            reasons.append(f"wm_strong({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.22 and dominance >= 2.2 and dom_text >= 15:
            reasons.append(f"wm_dom({wm_score:.2f},dom={dominance:.1f},txt={dom_text})")
        elif wm_score >= 0.18 and text_corners >= 3 and dominance >= 1.5:
            reasons.append(f"wm_multi({wm_score:.2f},corners={text_corners})")

    if conservative:
        if menu_score >= 0.85 and perp >= 30 and axis >= 6 and border >= 5:
            reasons.append(f"menu_strong(perp={perp},axis={axis},border={border})")
    else:
        if menu_score >= 0.75 and perp >= 25 and axis >= 5 and border >= 4:
            reasons.append(f"menu_strong(perp={perp},axis={axis},border={border})")

    return len(reasons) > 0, " + ".join(reasons)


def _scan_worker(
    args: tuple[str, bool, bool],
) -> tuple[str, float, dict[str, Any], float, dict[str, int]]:
    path_str, do_wm, do_menu = args
    wm_score, wm_info = detect_watermark(Path(path_str)) if do_wm else (0.0, {})
    menu_score, menu_info = detect_menu_card(Path(path_str)) if do_menu else (0.0, {})
    return path_str, wm_score, wm_info, menu_score, menu_info


def build_preview_data(
    scan_results: dict[str, list[dict[str, Any]]],
    top_n: int,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Select top suspicious candidates and random clean samples per class for review."""
    random.seed(seed)
    preview: dict[str, list[dict[str, Any]]] = {}
    for cname, items in scan_results.items():
        if not items:
            preview[cname] = []
            continue
        susp = sorted(
            [it for it in items if it["is_suspicious"]],
            key=lambda x: float(x["score"]),
            reverse=True,
        )
        clean = [it for it in items if not it["is_suspicious"]]
        random.shuffle(clean)

        if len(susp) >= top_n:
            preview[cname] = susp[:top_n]
        else:
            preview[cname] = susp + clean[: top_n - len(susp)]
    return preview


def render_contact_sheet(
    preview_data: dict[str, list[dict[str, Any]]],
    out_path: Path,
    n_cols: int = 8,
) -> None:
    """Render a contact sheet image summarizing suspicious candidates per class."""
    import matplotlib.pyplot as plt

    n_rows = len(preview_data)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    if n_rows == 1:
        axes = np.array([axes])

    for row, (cname, samples) in enumerate(preview_data.items()):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cname, rotation=0, labelpad=60, fontsize=8, va="center")
            if col < len(samples):
                sample = samples[col]
                try:
                    with Image.open(sample["path"]) as im:
                        ax.imshow(im.convert("RGB"))
                    color = "red" if sample["is_suspicious"] else "gray"
                    label = f"{sample['score']:.2f}"
                    if sample.get("reason"):
                        prefix = "W" if "wm" in str(sample["reason"]) else "M"
                        label = f"{prefix} {sample['score']:.2f}"
                    ax.set_title(label, fontsize=7, color=color)
                except Exception:
                    ax.text(0.5, 0.5, "ERR", ha="center", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=70, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter watermark and menu-card candidates.")
    parser.add_argument("--move", action="store_true", help="Move flagged files to suspicious/")
    parser.add_argument("--preview", action="store_true", help="Generate preview contact sheet")
    parser.add_argument("--conservative", action="store_true", help="Use stricter thresholds")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated class filter")
    parser.add_argument("--no-watermark", action="store_true", help="Disable watermark detection")
    parser.add_argument("--no-menu", action="store_true", help="Disable menu-card detection")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--top", type=int, default=8, help="Preview columns per class")
    parser.add_argument("--csv", action="store_true", help="Export detailed CSV report")
    parser.add_argument("--yes", action="store_true", help="Skip move confirmation prompt")
    parser.add_argument("--force", action="store_true", help="Ignore >15%% safety threshold")
    args = parser.parse_args()

    do_wm = not args.no_watermark
    do_menu = not args.no_menu
    if not do_wm and not do_menu:
        print("ERROR: at least one detector must remain enabled.")
        return

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    classes = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        classes = [c for c in classes if c.name in wanted]
        missing = wanted - {c.name for c in classes}
        if missing:
            print(f"WARN: missing classes: {sorted(missing)}")

    all_paths: list[tuple[str, str]] = []
    for class_dir in classes:
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                all_paths.append((class_dir.name, str(p)))

    if not all_paths:
        print("ERROR: no images found.")
        return

    print(f"Scanning {len(all_paths)} images ({args.workers} workers)...")
    scan_results: dict[str, list[dict[str, Any]]] = {c.name: [] for c in classes}
    class_of = {p: c for c, p in all_paths}
    tasks = [(p, do_wm, do_menu) for _, p in all_paths]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, (path_str, wm_score, wm_info, menu_score, menu_info) in enumerate(
            executor.map(_scan_worker, tasks, chunksize=8), 1
        ):
            if idx % 200 == 0:
                print(f"  ... {idx}/{len(tasks)}")
            if wm_score < 0 or menu_score < 0:
                continue

            is_susp, reason = decide(
                wm_score, wm_info, menu_score, menu_info, args.conservative
            )
            scan_results[class_of[path_str]].append(
                {
                    "path": path_str,
                    "wm_score": wm_score,
                    "menu_score": menu_score,
                    "score": max(wm_score, menu_score),
                    "is_suspicious": is_susp,
                    "reason": reason,
                }
            )

    print(f"\n{'Class':<25} {'Total':>7} {'Susp':>8} {'%':>7} {'AvgScore':>9}")
    print("-" * 62)
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
        avg = sum(float(it["score"]) for it in items) / n
        print(f"{cname:<25} {n:>7} {s:>8} {pct:>6.1f}% {avg:>9.3f}")

    pct_all = 100 * grand_susp / max(grand_total, 1)
    print("-" * 62)
    print(f"{'TOTAL':<25} {grand_total:>7} {grand_susp:>8} {pct_all:>6.1f}%\n")

    if args.csv:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPORT_DIR / f"suspicious_check_{ts}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["class", "filename", "wm_score", "menu_score", "combined", "is_susp", "reason"]
            )
            for cname, items in scan_results.items():
                for it in items:
                    writer.writerow(
                        [
                            cname,
                            Path(it["path"]).name,
                            f"{it['wm_score']:.3f}",
                            f"{it['menu_score']:.3f}",
                            f"{it['score']:.3f}",
                            int(bool(it["is_suspicious"])),
                            it["reason"],
                        ]
                    )
        print(f"CSV report: {csv_path}")

    if args.preview:
        preview_data = build_preview_data(scan_results, top_n=args.top)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPORT_DIR / f"preview_{ts}.png"
        render_contact_sheet(preview_data, out)
        print(f"Preview contact sheet: {out}")

    if args.move:
        if grand_susp == 0:
            print("Nothing to move.")
            return
        if pct_all > 15 and not args.force:
            print(f"ABORTED: suspicious rate {pct_all:.1f}% > 15%. Use --conservative or --force.")
            return
        if not args.yes:
            ans = input(f"Move {grand_susp} images to {SUSPICIOUS_DIR}? (yes): ").strip()
            if ans != "yes":
                print("Cancelled.")
                return

        moved = 0
        for cname, items in scan_results.items():
            for it in items:
                if not it["is_suspicious"]:
                    continue
                src = Path(it["path"])
                if src.exists() and safe_move(src, SUSPICIOUS_DIR / cname) is not None:
                    moved += 1
        print(f"Moved: {moved} images")
    else:
        print("DRY RUN - no files moved. Pass --move to apply.")


if __name__ == "__main__":
    main()
