"""
ChefAI - images/ klasorunun durumunu raporlar.
Seçenek B uygulandıysa (data/raw/* -> images/) bu script sadece
images/ icindeki yemekleri ve sayilarini gösterir.

Eger data/ hala varsa uyari verir, silmek isteyip istemedigini sorar.
"""

from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJE_KOKU = SCRIPT_DIR.parent

GORSEL_UZANTI = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def gorsel_sayisi(klasor: Path) -> int:
    if not klasor.exists():
        return 0
    return len([p for p in klasor.iterdir() if p.suffix.lower() in GORSEL_UZANTI])


def main():
    print("=" * 60)
    print(f"Proje koku: {PROJE_KOKU}")
    print("=" * 60)
    print()

    images_dir = PROJE_KOKU / "images"
    data_dir = PROJE_KOKU / "data"
    raw_dir = data_dir / "raw"

    # --- images/ kontrolu ---
    if not images_dir.exists():
        print("HATA: images/ klasoru yok!")
        print(f"Beklenen yol: {images_dir}")
        return

    print(f"images/ yolu : {images_dir}")
    print()

    # --- images/ icindeki yemekler ---
    yemekler = sorted([p for p in images_dir.iterdir() if p.is_dir()])

    if not yemekler:
        print("UYARI: images/ icinde hic klasor yok!")
        return

    print(f"images/ icinde {len(yemekler)} yemek klasoru:")
    print()
    print(f"{'Yemek':<25} {'Gorsel':>8}")
    print("-" * 35)

    toplam = 0
    eksik_olanlar = []
    for y in yemekler:
        n = gorsel_sayisi(y)
        toplam += n
        isaret = ""
        if n < 100:
            isaret = "  <-- AZ"
            eksik_olanlar.append(y.name)
        print(f"{y.name:<25} {n:>8}{isaret}")

    print("-" * 35)
    print(f"{'TOPLAM':<25} {toplam:>8}")
    print()

    if eksik_olanlar:
        print(f"100'den az gorseli olan siniflar ({len(eksik_olanlar)}):")
        for ad in eksik_olanlar:
            print(f"  - {ad}")
        print()

    # --- data/ hala var mi? ---
    print("=" * 60)
    if data_dir.exists():
        print(f"UYARI: data/ klasoru hala var: {data_dir}")
        if raw_dir.exists():
            icerik = list(raw_dir.iterdir())
            if icerik:
                print(f"data/raw/ icinde {len(icerik)} oge var.")
                print("Bunlar yedek mi, yoksa kopya mi?")
                cevap = input("data/ klasorunu SILMEK ister misin? (e/h): ").strip().lower()
                if cevap == "e":
                    shutil.rmtree(data_dir)
                    print("OK: data/ silindi.")
                else:
                    print("data/ oldugu gibi birakildi.")
            else:
                print("data/raw/ bos. Silinebilir.")
                cevap = input("data/ klasorunu SILMEK ister misin? (e/h): ").strip().lower()
                if cevap == "e":
                    shutil.rmtree(data_dir)
                    print("OK: data/ silindi.")
        else:
            print("data/ icinde raw/ yok. Silinebilir.")
            cevap = input("data/ klasorunu SILMEK ister misin? (e/h): ").strip().lower()
            if cevap == "e":
                shutil.rmtree(data_dir)
                print("OK: data/ silindi.")
    else:
        print("data/ yok. Guzel, her sey tek yerde (images/).")

    print()
    print("=" * 60)
    print("SONUC")
    print("=" * 60)
    print(f"Toplam {len(yemekler)} yemek sinifi, {toplam} gorsel.")
    print(f"Ortalama: {toplam // max(len(yemekler), 1)} gorsel/sinif")
    print()
    print("Sonraki adim: python Meta/more_to_target.py")


if __name__ == "__main__":
    main()