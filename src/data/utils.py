"""
ChefAI - Ortak yardimci fonksiyonlar.

Kullanim:
    from src.data.utils import list_images, renumber_class
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def list_images(folder: Path) -> list[Path]:
    """Klasordeki tum gorselleri isim sirali dondurur."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def renumber_class(
    class_dir: Path,
    dry_run: bool = False,
    prefix: str = "",
    pad: int = 6,
) -> int:
    """
    Bir sinifin tum gorsellerini 000001, 000002, ... olarak yeniden adlandirir.

    Guvenli: once gecici isimlere tasir, sonra hedef isimlere (cakisma yok).

    Args:
        class_dir: Sinif klasoru
        dry_run: True ise sadece say, degistirme
        prefix: Dosya ismi on eki (orn: "img_")
        pad: Kac haneli olacak (default: 6)

    Returns:
        Yeniden adlandirilan dosya sayisi (dry_run'da tahmini sayi)
    """
    if not class_dir.exists() or not class_dir.is_dir():
        return 0

    images = list_images(class_dir)
    if not images:
        return 0

    # Zaten dogru mu? (optimizasyon: hepsi pad haneli sayi ise atla)
    all_ok = all(p.stem.isdigit() and len(p.stem) == pad for p in images)
    if all_ok:
        return 0

    # 1. Asama: gecici isimlere tasi
    temp_paths: list[Path] = []
    for i, img in enumerate(images, 1):
        temp_name = f"_tmp_{i:0{pad}d}{img.suffix.lower()}"
        temp_path = class_dir / temp_name
        if not dry_run:
            try:
                img.rename(temp_path)
            except Exception as e:
                print(f"  RENAME HATASI: {img.name}: {e}")
                continue
        temp_paths.append(temp_path)

    # 2. Asama: hedef isimlere tasi
    renamed = 0
    for i, temp in enumerate(temp_paths, 1):
        final_name = f"{prefix}{i:0{pad}d}{temp.suffix.lower()}"
        final_path = class_dir / final_name
        if not dry_run:
            try:
                temp.rename(final_path)
                renamed += 1
            except Exception as e:
                print(f"  RENAME HATASI: {temp.name}: {e}")
        else:
            renamed += 1

    return renamed


def human_size(n: float) -> str:
    """Byte'i okunabilir hale getir."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def count_images(folder: Path) -> int:
    """Klasordeki gorsel sayisini dondurur."""
    return len(list_images(folder))
