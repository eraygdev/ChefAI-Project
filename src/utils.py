"""Shared filesystem, image discovery, and safe file operation utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BROKEN_DIR = DATA_DIR / "brokendata"
EXTRA_DIR = DATA_DIR / "extradata"
SUSPICIOUS_DIR = DATA_DIR / "suspicious"
REPORT_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def list_images(folder: Path) -> list[Path]:
    """Return sorted image file paths inside a directory."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(folder: Path) -> int:
    """Return total image count inside a directory."""
    return len(list_images(folder))


def folder_size_bytes(folder: Path) -> int:
    """Return recursive total size of files in bytes."""
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


def human_size(n: float) -> str:
    """Format byte size into a human-readable unit."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_mb(n_bytes: int) -> str:
    """Format bytes in megabytes with one decimal place."""
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def resolve_unique_path(target_dir: Path, filename: str) -> Path:
    """Return a non-colliding file path inside target_dir."""
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    src_path = Path(filename)
    stem = src_path.stem
    suffix = src_path.suffix
    idx = 1
    while (target_dir / f"{stem}_{idx}{suffix}").exists():
        idx += 1
    return target_dir / f"{stem}_{idx}{suffix}"


def safe_move(src: Path, dst_dir: Path) -> Path | None:
    """Move src into dst_dir without overwriting existing files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_unique_path(dst_dir, src.name)
    try:
        shutil.move(str(src), str(target))
        return target
    except Exception as e:
        print(f"  MOVE ERROR: {src.name}: {e}")
        return None


def safe_copy(src: Path, dst_dir: Path) -> Path | None:
    """Copy src into dst_dir without overwriting existing files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_unique_path(dst_dir, src.name)
    try:
        shutil.copy2(str(src), str(target))
        return target
    except Exception as e:
        print(f"  COPY ERROR: {src.name}: {e}")
        return None
