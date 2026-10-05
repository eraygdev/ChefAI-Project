"""
ChefAI - Yemek Gorselleri Indirme Scripti
Her yemek kendi klasorune iner. Google ve Bing uzerinden cekim yapar.
"""

import os
from pathlib import Path

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

# -------------------------------------------------------------------
# AYARLAR
# -------------------------------------------------------------------
CIKTI_KLASORU = "data/raw"
HER_YEMEK_ICIN = 250  # Once 250 cek, sonra temizleyip 150-200'e dusur
KULLAN_GOOGLE = True  # Google kullan
KULLAN_BING = True  # Bing de kullan (yedek / ek kaynak)

# -------------------------------------------------------------------
# YEMEK LISTESI (klasor adi : arama terimi)
# Klasor adlari Turkce karaktersiz, kucuk harfli ve alt cizgili
# -------------------------------------------------------------------
YEMEKLER = {
    # --- Corbalar ---
    "mercimek_corbasi": "mercimek çorbası",
    "ezogelin_corbasi": "ezogelin çorbası",
    "domates_corbasi": "domates çorbası",
    "tavuk_suyu_corba": "tavuk suyu çorba",
    # --- Ana Yemek - Et ve Tavuk ---
    "izgara_tavuk": "ızgara tavuk",
    "kofte": "köfte",
    "tavuk_sote": "tavuk sote",
    "etli_nohut": "etli nohut",
    "karniyarik": "karnıyarık",
    "manti": "mantı",
    # --- Ana Yemek - Sebze ---
    "zeytinyagli_fasulye": "zeytinyağlı yeşil fasulye",
    "imam_bayildi": "imam bayıldı",
    "menemen": "menemen",
    "kisir": "kısır",
    "firin_sebze": "fırında sebze",
    # --- Hamur Isleri ---
    "pizza": "pizza",
    "pide": "türk pidesi",
    "borek": "börek",
    "pogaca": "poğaça",
    "gozleme": "gözleme",
    # --- Pilav ve Makarna ---
    "pirinc_pilavi": "pirinç pilavı",
    "bulgur_pilavi": "bulgur pilavı",
    "spagetti_bolonez": "spaghetti bolognese",
    "fettuccine_alfredo": "fettuccine alfredo",
    # --- Kahvaltilik ---
    "omlet": "omlet",
    "pankek": "pankek",
    "avokado_tost": "avocado toast",
    # --- Tatlilar ---
    "kek": "kek dilimi",
    "kurabiye": "kurabiye",
    "sutlac": "sütlaç",
}


# -------------------------------------------------------------------
# INDIRME FONKSIYONU
# -------------------------------------------------------------------
def gorsel_indir(klasor_adi: str, arama_terimi: str, adet: int):
    """
    Belirtilen yemek icin Google ve Bing'den gorsel indirir.
    Her yemek kendi klasorune iner.
    """
    klasor_yolu = Path(CIKTI_KLASORU) / klasor_adi
    klasor_yolu.mkdir(parents=True, exist_ok=True)

    print(f"\n[{klasor_adi}] indiriliyor -> '{arama_terimi}'")

    # --- Google ---
    if KULLAN_GOOGLE:
        try:
            google_crawler = GoogleImageCrawler(
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
                storage={"root_dir": str(klasor_yolu)},
            )
            google_crawler.crawl(
                keyword=arama_terimi,
                max_num=adet,
                min_size=(200, 200),
                file_idx_offset=0,
            )
            print(f"  Google: tamam")
        except Exception as e:
            print(f"  Google HATA: {e}")

    # --- Bing ---
    if KULLAN_BING:
        try:
            bing_crawler = BingImageCrawler(
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
                storage={"root_dir": str(klasor_yolu)},
            )
            bing_crawler.crawl(
                keyword=arama_terimi,
                max_num=adet // 2,
                min_size=(200, 200),
                file_idx_offset="auto",
            )
            print(f"  Bing: tamam")
        except Exception as e:
            print(f"  Bing HATA: {e}")

    toplam = len(list(klasor_yolu.glob("*.*")))
    print(f"  Toplam dosya: {toplam}")


# -------------------------------------------------------------------
# ANA DONGU
# -------------------------------------------------------------------
def main():
    Path(CIKTI_KLASORU).mkdir(parents=True, exist_ok=True)

    for klasor_adi, arama_terimi in YEMEKLER.items():
        gorsel_indir(klasor_adi, arama_terimi, HER_YEMEK_ICIN)

    print("\nTum yemekler indirildi.")
    print(f"Klasor: {CIKTI_KLASORU}")


if __name__ == "__main__":
    main()
