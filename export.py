"""
ChefAI Projesi - Kaynak Kod Export Scripti

Projedeki tum metin tabanli kaynak dosyalarini tek bir .txt ciktisina doner.
Atlananlar:
  - Binary dosyalar (gorseller, modeller, .pt, .pth, .onnx, .pkl)
  - Veri klasorleri (data/raw, data/processed, data/brokendata, data/extradata)
  - MLflow ciktilari (mlruns/, mlartifacts/)
  - Sanal ortamlar (venv/, .venv/, env/)
  - Cache klasorleri (__pycache__, .pytest_cache, .mypy_cache)
  - Git klasoru (.git/)
  - Not defteri ciktilari (*.ipynb_checkpoints)
  - Lock dosyalari (poetry.lock, uv.lock)
  - Arff, csv, parquet, json gibi veri dosyalari (istege bagli)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --------------------------------------------------------------
# YAPILANDIRMA
# --------------------------------------------------------------

# Dahil edilecek uzantilar (nokta ile)
INCLUDE_EXTENSIONS: set[str] = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".sh",
    ".cfg",
    ".ini",
    ".env",
    ".dockerignore",
    ".gitignore",
    ".pre-commit-config.yaml",
}

# Uzantisiz ama dahil edilecek dosyalar (tam isim)
INCLUDE_FILENAMES: set[str] = {
    "Dockerfile",
    "Makefile",
    "LICENSE",
    ".gitignore",
    ".dockerignore",
    ".pre-commit-config.yaml",
    ".env.example",
    "pyproject.toml",
}

# Atlanacak klasor isimleri (her seviyede)
EXCLUDE_DIRS: set[str] = {
    ".git",
    ".github",  # istersen kaldir
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "mlruns",
    "mlartifacts",
    "data/raw",
    "data/processed",
    "data/brokendata",
    "data/extradata",
    ".ipynb_checkpoints",
    "htmlcov",
    "dist",
    "build",
    "*.egg-info",
}

# Atlanacak dosya uzantilari
EXCLUDE_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".svg",
    ".ico",
    ".pt",
    ".pth",
    ".onnx",
    ".pkl",
    ".joblib",
    ".h5",
    ".keras",
    ".tflite",
    ".pb",
    ".lock",
    ".log",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arff",
    ".npy",
    ".npz",
    ".bin",
    ".exe",
    ".o",
    ".a",
}

# Atlanacak dosya isimleri
EXCLUDE_FILENAMES: set[str] = {
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    ".DS_Store",
    "Thumbs.db",
    "project_export.txt",
    "export.py",  # kendini disarida birak
}

# Cok buyuk dosyalari atla (byte)
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

# Ayrac
SEPARATOR = "=" * 100


# --------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# --------------------------------------------------------------


def should_exclude_dir(path: Path) -> bool:
    """Klasor atlanmali mi?"""
    parts = set(path.parts)
    for ex in EXCLUDE_DIRS:
        if "/" in ex:
            # Yol iceren kural (ornek: data/raw)
            if str(path).endswith(ex) or f"/{ex}/" in f"/{path}/":
                return True
        else:
            if ex in parts:
                return True
    return False


def should_include_file(path: Path) -> bool:
    """Dosya dahil edilmeli mi?"""
    name = path.name
    suffix = path.suffix.lower()

    # Dosya ismi atlanacaklar listesinde mi?
    if name in EXCLUDE_FILENAMES:
        return False

    # Tam isimle dahil edilecekler
    if name in INCLUDE_FILENAMES:
        return True

    # Uzanti atlanacaklar listesinde mi?
    if suffix in EXCLUDE_EXTENSIONS:
        return False

    # Uzanti dahil edilecekler listesinde mi?
    if suffix in INCLUDE_EXTENSIONS:
        return True

    # .pre-commit-config.yaml gibi ozel durumlar
    if name.startswith(".pre-commit-config"):
        return True

    return False


def read_text_safe(path: Path) -> str | None:
    """Dosyayi guvenli sekilde oku. Hata olursa None doner."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def human_size(n: float) -> str:
    """Byte'i okunabilir hale getir."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------
# ANA FONKSIYON
# --------------------------------------------------------------


def collect_files(root: Path) -> list[Path]:
    """Projeyi tara, dahil edilecek dosyalari topla."""
    files: list[Path] = []

    for path in sorted(root.rglob("*")):
        # Klasorleri atla
        if path.is_dir():
            continue

        # Klasor yolu icinde atlanacak bir sey var mi?
        if any(should_exclude_dir(p) for p in path.parents):
            continue

        # Gorunmez klasorleri atla (.git zaten listede)
        if any(
            part.startswith(".") and part not in {".github", ".pre-commit-config.yaml"}
            for part in path.parts
        ):
            # Ama .gitignore, .dockerignore gibi dosyalari elle kontrol edecegiz
            pass

        # Dosya dahil edilmeli mi?
        if not should_include_file(path):
            continue

        # Boyut kontrolu
        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size > MAX_FILE_SIZE:
            print(f"  ATLANDI (buyuk): {path.relative_to(root)} ({human_size(size)})")
            continue

        files.append(path)

    return files


def export_project(root: Path, output: Path) -> None:
    """Projeyi tara ve tek dosyaya yaz."""
    print(f"Proje kok dizini: {root}")
    print(f"Cikti dosyasi   : {output}")
    print()

    files = collect_files(root)

    if not files:
        print("Hic dosya bulunamadi.")
        sys.exit(1)

    print(f"Toplam {len(files)} dosya bulundu.")
    print()

    total_bytes = 0

    with output.open("w", encoding="utf-8") as f:
        # Baslik
        f.write(SEPARATOR + "\n")
        f.write("ChefAI Projesi - Kaynak Kod Export\n")
        f.write(f"Proje koku: {root}\n")
        f.write(f"Dosya sayisi: {len(files)}\n")
        f.write(SEPARATOR + "\n\n")

        # Icindekiler
        f.write("ICINDEKILER\n")
        f.write("-" * 100 + "\n")
        for i, path in enumerate(files, 1):
            rel = path.relative_to(root)
            f.write(f"{i:4d}. {rel}\n")
        f.write("\n")

        # Dosya icerikleri
        for i, path in enumerate(files, 1):
            rel = path.relative_to(root)
            try:
                content = read_text_safe(path)
            except Exception as e:
                f.write(f"\n{SEPARATOR}\n")
                f.write(f"[{i}/{len(files)}] {rel}\n")
                f.write(f"OKUNAMADI: {e}\n")
                f.write(SEPARATOR + "\n")
                continue

            if content is None:
                f.write(f"\n{SEPARATOR}\n")
                f.write(f"[{i}/{len(files)}] {rel}\n")
                f.write("OKUNAMADI (binary veya encoding hatasi)\n")
                f.write(SEPARATOR + "\n")
                continue

            size = len(content.encode("utf-8"))
            total_bytes += size

            f.write(f"\n{SEPARATOR}\n")
            f.write(f"[{i}/{len(files)}] DOSYA: {rel}\n")
            f.write(f"BOYUT: {human_size(size)}\n")
            f.write(SEPARATOR + "\n")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

    print(f"Toplam icerik boyutu: {human_size(total_bytes)}")
    print(f"Cikti yazildi: {output}")
    print(f"Cikti boyutu: {human_size(output.stat().st_size)}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChefAI projesindeki kaynak kodlari tek dosyaya export eder."
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        default=Path(".").resolve(),
        help="Proje kok dizini (varsayilan: mevcut dizin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("project_export.txt"),
        help="Cikti dosyasi (varsayilan: project_export.txt)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"HATA: Proje koku bulunamadi: {args.root}")
        sys.exit(1)

    export_project(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
