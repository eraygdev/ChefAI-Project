"""
ChefAI - Fills all classes up to the target count (250).
Stops downloading for a class once it reaches the target.
Does NOT delete existing excess images.
Uses: Google (with date filters) + Bing + Baidu + Flickr + DuckDuckGo.
"""

import time
import random
import requests
from pathlib import Path
from icrawler.builtin import (
    GoogleImageCrawler,
    BingImageCrawler,
    BaiduImageCrawler,
    FlickrImageCrawler,
)

# -------------------------------------------------------------------
# PATHS / YOLLAR
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "images"

# -------------------------------------------------------------------
# SETTINGS / AYARLAR
# -------------------------------------------------------------------
TARGET_COUNT = 220
EXTRA_RATIO = 1.0
MIN_DOWNLOAD = 50
WAIT_SECONDS = 2.5
DUCKDUCKGO_WAIT = 3.0
DUCKDUCKGO_TIMEOUT = 10

# DuckDuckGo destegi (opsiyonel, kuruluysa True yap)
USE_DUCKDUCKGO = True

# Tarih filtreleri: Google icin farkli yillar kullanilir (daha cesitli sonuclar)
# (baslangic_yil, baslangic_ay, baslangic_gun), (bitis_yil, bitis_ay, bitis_gun)
DATE_FILTERS = [
    ((2020, 1, 1), (2020, 12, 31)),
    ((2021, 1, 1), (2021, 12, 31)),
    ((2022, 1, 1), (2022, 12, 31)),
    ((2023, 1, 1), (2023, 12, 31)),
    ((2024, 1, 1), (2024, 12, 31)),
]

ENGINES = {
    "google":     True,
    "bing":       True,
    "baidu":      True,
    "flickr":     True,
    "duckduckgo": True,    # USE_DUCKDUCKGO ile birlikte calisir
}

# -------------------------------------------------------------------
# SELECTED CLASSES / SECILEN SINIFLAR
# Bos birakilirsa TUM 30 sinif taranir.
# -------------------------------------------------------------------
SELECTED = []   # bos = hepsi

# -------------------------------------------------------------------
# FOOD LIST / YEMEK LISTESI
# Her yemek icin coklu dil ve coklu aci kullanilir.
# -------------------------------------------------------------------
FOODS = {
    # =================================================================
    # CORBALAR
    # =================================================================
    "mercimek_corbasi": [
        "mercimek çorbası", "kırmızı mercimek",
        "türk mutfağı mercimek", "anadolu mercimek çorbası",
        "kış çorbası mercimek", "sıcak çorba mercimek",
        "refika mercimek çorbası", "nefis yemek mercimek",
        "mercimek çorbası porsiyon", "tek kişilik mercimek çorbası",
        "turkish red lentil", "lentil soup authentic",
        "best lentil soup", "easy lentil soup",
        "lentil soup winter", "lentil soup traditional",
        "Linsensuppe original", "soupe lentilles traditionnelle",
        "sopa lentejas casera", "zuppa lenticchie tradizionale",
    ],
    "ezogelin_corbasi": [
        "ezogelin çorbası", "ezogelin çorbası gelin",
        "türk mutfağı ezogelin", "anadolu ezogelin",
        "bulgurlu ezogelin", "naneli ezogelin",
        "refika ezogelin", "nefis yemek ezogelin",
        "ezogelin porsiyon", "tek kişilik ezogelin",
        "turkish ezogelin", "ezogelin authentic",
        "ezogelin traditional", "ezogelin homemade",
        "Ezogelin Suppe original", "soupe ezogelin traditionnelle",
    ],
    "domates_corbasi": [
        "domates çorbası", "kremalı domates çorbası",
        "türk mutfağı domates çorbası", "ege domates çorbası",
        "fesleğenli domates", "sarımsaklı domates çorbası",
        "refika domates çorbası", "nefis yemek domates",
        "domates çorbası porsiyon", "tek kişilik domates",
        "turkish tomato soup", "tomato soup authentic",
        "creamy tomato soup", "best tomato soup",
        "tomato soup winter", "Tomatensuppe original",
        "soupe tomates traditionnelle", "sopa tomate casera",
    ],
    "tavuk_suyu_corba": [
        "tavuk suyu çorba", "tavuk çorbası şehriyeli",
        "türk mutfağı tavuk çorbası", "anadolu tavuk çorbası",
        "limonlu tavuk çorbası", "sebzeli tavuk çorbası",
        "refika tavuk çorbası", "nefis yemek tavuk çorbası",
        "tavuk çorbası porsiyon", "tek kişilik tavuk çorbası",
        "turkish chicken soup", "chicken soup authentic",
        "best chicken soup", "easy chicken soup",
        "chicken noodle soup", "Hühnersuppe original",
        "soupe poulet traditionnelle", "sopa pollo casera",
    ],

    # =================================================================
    # ANA YEMEK - ET VE TAVUK
    # =================================================================
    "izgara_tavuk": [
        "ızgara tavuk", "tavuk göğsü ızgara",
        "tavuk but ızgara", "baharatlı tavuk ızgara",
        "limonlu tavuk ızgara", "türk usulü tavuk",
        "refika ızgara tavuk", "nefis yemek tavuk",
        "ızgara tavuk porsiyon", "tek kişilik ızgara tavuk",
        "grilled chicken", "grilled chicken authentic",
        "best grilled chicken", "easy grilled chicken",
        "grilled chicken dinner", "Grillhähnchen original",
        "poulet grillé traditionnel", "pollo parrilla casero",
    ],
    "kofte": [
        "köfte", "ızgara köfte", "türk köftesi",
        "baharatlı köfte", "soğanlı köfte", "maydanozlu köfte",
        "refika köfte", "nefis yemek köfte", "oktay usta köfte",
        "köfte porsiyon", "tek kişilik köfte",
        "turkish meatballs", "turkish kofte authentic",
        "best meatballs", "easy meatballs",
        "Frikadellen original", "boulettes traditionnelles",
        "albondigas caseras", "polpette tradizionali",
    ],
    "tavuk_sote": [
        "tavuk sote", "sebzeli tavuk sote", "biberli tavuk sote",
        "soğanlı tavuk sote", "sarımsaklı tavuk sote",
        "refika tavuk sote", "nefis yemek tavuk sote",
        "tavuk sote porsiyon", "tek kişilik tavuk sote",
        "chicken saute", "chicken saute authentic",
        "best chicken saute", "easy chicken saute",
        "Hähnchenpfanne original", "poulet sauté traditionnel",
        "pollo salteado casero", "pollo padella tradizionale",
    ],
    "etli_nohut": [
        "etli nohut", "kuzu etli nohut", "dana etli nohut",
        "salçalı nohut", "soğanlı nohut", "baharatlı nohut",
        "refika etli nohut", "nefis yemek nohut",
        "etli nohut porsiyon", "tek kişilik etli nohut",
        "chickpea stew meat", "chickpea stew authentic",
        "best chickpea stew", "easy chickpea stew",
        "Kichererbsen original", "ragoût pois chiches",
        "guiso garbanzos casero", "stufato ceci tradizionale",
    ],
    "karniyarik": [
        "karnıyarık", "fırında karnıyarık", "kıymalı karnıyarık",
        "domatesli karnıyarık", "biberli karnıyarık",
        "refika karnıyarık", "nefis yemek karnıyarık", "oktay usta karnıyarık",
        "karnıyarık porsiyon", "tek kişilik karnıyarık",
        "turkish stuffed eggplant", "karniyarik authentic",
        "best stuffed eggplant", "easy stuffed eggplant",
        "gefüllte Auberginen original", "aubergines farcies",
        "berenjenas rellenas caseras", "melanzane ripiene",
    ],
    "manti": [
        "mantı", "türk mantısı", "el açması mantı",
        "yoğurtlu mantı", "sarımsaklı mantı", "tereyağlı mantı",
        "refika mantı", "nefis yemek mantı", "oktay usta mantı",
        "mantı porsiyon", "tek kişilik mantı",
        "turkish manti", "turkish dumplings authentic",
        "best manti", "easy manti",
        "türkische Teigtaschen original", "raviolis turcs",
        "manti yogur casero", "manti yogurt tradizionale",
    ],

    # =================================================================
    # ANA YEMEK - SEBZE
    # =================================================================
    "zeytinyagli_fasulye": [
        "zeytinyağlı fasulye", "yeşil fasulye zeytinyağlı",
        "domatesli fasulye", "soğanlı fasulye", "sarımsaklı fasulye",
        "refika zeytinyağlı fasulye", "nefis yemek fasulye",
        "zeytinyağlı fasulye porsiyon", "tek kişilik fasulye",
        "turkish green beans", "green beans olive oil authentic",
        "best green beans", "easy green beans",
        "grüne Bohnen original", "haricots verts traditionnels",
        "judías verdes caseras", "fagiolini tradizionali",
    ],
    "imam_bayildi": [
        "imam bayıldı", "fırında imam bayıldı", "soğuk imam bayıldı",
        "domatesli imam bayıldı", "sarımsaklı imam bayıldı",
        "refika imam bayıldı", "nefis yemek imam bayıldı",
        "imam bayıldı porsiyon", "tek kişilik imam bayıldı",
        "turkish imam bayildi", "stuffed eggplant olive oil",
        "best imam bayildi", "easy imam bayildi",
        "Imam Bayildi original", "aubergines imam bayildi",
        "imam bayildi casero", "imam bayildi tradizionale",
    ],
    "menemen": [
        "menemen", "kahvaltılık menemen", "domatesli menemen",
        "biberli menemen", "soğanlı menemen", "sucuklu menemen",
        "refika menemen", "nefis yemek menemen", "oktay usta menemen",
        "menemen porsiyon", "tek kişilik menemen",
        "turkish menemen", "turkish scrambled eggs authentic",
        "best menemen", "easy menemen",
        "türkisches Rührei original", "oeufs brouillés",
        "huevos revueltos caseros", "uova strapazzate",
    ],
    "kisir": [
        "kısır", "türk kısırı", "maydanozlu kısır",
        "salçalı kısır", "limonlu kısır", "nar ekşili kısır",
        "refika kısır", "nefis yemek kısır", "oktay usta kısır",
        "kısır porsiyon", "tek kişilik kısır",
        "turkish kisir", "turkish bulgur salad authentic",
        "best kisir", "easy kisir",
        "Bulgursalat original", "salade boulgour",
        "ensalada bulgur casera", "insalata bulgur",
    ],
    "firin_sebze": [
        "fırında sebze", "fırında karışık sebze",
        "zeytinyağlı fırın sebze", "baharatlı fırın sebze",
        "refika fırın sebze", "nefis yemek fırın sebze",
        "fırında sebze porsiyon", "tek kişilik fırın sebze",
        "roasted vegetables", "oven roasted vegetables authentic",
        "best roasted vegetables", "easy roasted vegetables",
        "Ofengemüse original", "légumes rôtis",
        "verduras horno caseras", "verdure forno",
    ],

    # =================================================================
    # HAMUR ISLERI
    # =================================================================
    "pizza": [
        "pizza", "pizza margherita", "pepperoni pizza",
        "pizza dilimi", "bütün pizza", "pizza porsiyon",
        "ev yapımı pizza", "italyan pizza", "taş fırın pizza",
        "best pizza", "easy pizza", "homemade pizza",
        "authentic pizza", "italian pizza original",
        "Pizza original", "pizza traditionnelle",
        "pizza casera", "pizza tradizionale", "pizza napoletana",
    ],
    "pide": [
        "pide", "türk pidesi", "kıymalı pide", "kuşbaşılı pide",
        "peynirli pide", "sucuklu pide", "yumurtalı pide",
        "refika pide", "nefis yemek pide", "oktay usta pide",
        "pide porsiyon", "tek kişilik pide",
        "turkish pide", "turkish flatbread authentic",
        "best turkish pide", "easy turkish pide",
        "türkische Pizza original", "pide turque",
        "pide turca casera", "pide turca tradizionale",
    ],
    "borek": [
        "börek", "su böreği", "sigara böreği", "peynirli börek",
        "ıspanaklı börek", "kıymalı börek", "patatesli börek",
        "refika börek", "nefis yemek börek", "oktay usta börek",
        "börek porsiyon", "tek kişilik börek",
        "turkish borek", "turkish pastry authentic",
        "best turkish borek", "easy borek",
        "türkisches Gebäck original", "börek turc",
        "borek turco casero", "borek turca tradizionale",
    ],
    "pogaca": [
        "poğaça", "peynirli poğaça", "patatesli poğaça",
        "kıymalı poğaça", "zeytinli poğaça", "maydanozlu poğaça",
        "refika poğaça", "nefis yemek poğaça", "oktay usta poğaça",
        "poğaça porsiyon", "tek kişilik poğaça",
        "turkish pogaca", "turkish pastry authentic",
        "best turkish pogaca", "easy pogaca",
        "türkische Brötchen original", "petits pains turcs",
        "panecillos turcos caseros", "pogaca turca",
    ],
    "gozleme": [
        "gözleme", "peynirli gözleme", "ıspanaklı gözleme",
        "patatesli gözleme", "kıymalı gözleme", "yumurtalı gözleme",
        "refika gözleme", "nefis yemek gözleme", "oktay usta gözleme",
        "gözleme porsiyon", "tek kişilik gözleme",
        "turkish gozleme", "turkish flatbread authentic",
        "best turkish gozleme", "easy gozleme",
        "türkische Fladenbrot original", "gozleme turc",
        "gozleme turco casero", "gozleme turca",
    ],

    # =================================================================
    # PILAV VE MAKARNA
    # =================================================================
    "pirinc_pilavi": [
        "pirinç pilavı", "şehriyeli pilav", "tereyağlı pilav",
        "tavuklu pilav", "nohutlu pilav", "sebzeli pilav",
        "refika pilav", "nefis yemek pilav", "oktay usta pilav",
        "pilav porsiyon", "tek kişilik pilav",
        "turkish rice pilaf", "turkish rice authentic",
        "best turkish rice", "easy rice pilaf",
        "türkischer Reis original", "riz pilaf turc",
        "arroz pilaf casero", "riso pilaf",
    ],
    "bulgur_pilavi": [
        "bulgur pilavı", "domatesli bulgur pilavı",
        "salçalı bulgur pilavı", "soğanlı bulgur pilavı",
        "biberli bulgur pilavı", "sebzeli bulgur pilavı",
        "refika bulgur pilavı", "nefis yemek bulgur",
        "bulgur pilavı porsiyon", "tek kişilik bulgur pilavı",
        "turkish bulgur pilaf", "bulgur pilaf authentic",
        "best bulgur pilaf", "easy bulgur pilaf",
        "Bulgur Pilaw original", "pilaf boulgour",
        "pilaf bulgur casero", "pilaf bulgur",
    ],
    "spagetti_bolonez": [
        "spagetti bolonez", "kıymalı spagetti",
        "domates soslu spagetti", "italyan spagetti",
        "spagetti porsiyon", "tek kişilik spagetti",
        "best bolognese", "easy bolognese",
        "spaghetti bolognese authentic", "spaghetti bolognese original",
        "Spaghetti Bolognese original", "spaghetti bolognaise",
        "espaguetis boloñesa caseros", "spaghetti bolognese italiano",
        "homemade spaghetti bolognese", "italian spaghetti bolognese",
    ],
    "fettuccine_alfredo": [
        "fettuccine alfredo", "kremalı fettuccine",
        "sahneli makarna", "italyan fettuccine",
        "fettuccine porsiyon", "tek kişilik fettuccine",
        "best fettuccine alfredo", "easy fettuccine alfredo",
        "creamy fettuccine authentic", "creamy pasta original",
        "Fettuccine Alfredo original", "fettuccine alfredo français",
        "fetuccini alfredo caseros", "fettuccine alfredo italiano",
        "homemade fettuccine alfredo", "italian fettuccine alfredo",
    ],

    # =================================================================
    # KAHVALTILIK
    # =================================================================
    "omlet": [
        "omlet", "peynirli omlet", "sebzeli omlet",
        "mantarlı omlet", "sucuklu omlet", "kaşarlı omlet",
        "refika omlet", "nefis yemek omlet",
        "omlet porsiyon", "tek kişilik omlet",
        "omelette", "cheese omelette authentic",
        "best omelette", "easy omelette",
        "french omelette original", "Omelett original",
        "omelette française", "tortilla francesa casera",
        "omelette italiana", "omelette breakfast",
    ],
    "pankek": [
        "pankek", "ballı pankek", "çilekli pankek",
        "çikolatalı pankek", "muzlu pankek", "yaban mersinli pankek",
        "refika pankek", "nefis yemek pankek",
        "pankek porsiyon", "tek kişilik pankek",
        "pancakes", "fluffy pancakes authentic",
        "best pancakes", "easy pancakes",
        "american pancakes original", "Pfannkuchen original",
        "pancakes françaises", "tortitas americanas caseras",
        "pancake americani", "pancake breakfast",
    ],
    "avokado_tost": [
        "avokado tost", "yumurtalı avokado tost",
        "domatesli avokado tost", "feta peynirli avokado tost",
        "avokado tost porsiyon", "tek kişilik avokado tost",
        "avocado toast", "avocado toast authentic",
        "best avocado toast", "easy avocado toast",
        "avocado toast breakfast", "avocado toast egg",
        "Avocado Toast original", "toast avocat",
        "tostada aguacate casera", "toast avocado",
        "homemade avocado toast", "healthy avocado toast",
    ],

    # =================================================================
    # TATLILAR
    # =================================================================
    "kek": [
        "kek", "çikolatalı kek", "vanilyalı kek",
        "meyveli kek", "havuçlu kek", "fındıklı kek",
        "refika kek", "nefis yemek kek", "oktay usta kek",
        "kek porsiyon", "tek kişilik kek", "kek dilimi",
        "cake", "chocolate cake authentic",
        "best cake", "easy cake",
        "homemade cake", "Kuchen original",
        "gâteau original", "pastel chocolate casero",
        "torta cioccolato", "birthday cake",
    ],
    "kurabiye": [
        "kurabiye", "un kurabiyesi", "cevizli kurabiye",
        "çikolatalı kurabiye", "fındıklı kurabiye", "bademli kurabiye",
        "refika kurabiye", "nefis yemek kurabiye", "oktay usta kurabiye",
        "kurabiye porsiyon", "tek kişilik kurabiye",
        "cookies", "turkish cookies authentic",
        "best cookies", "easy cookies",
        "homemade cookies", "Kekse original",
        "biscuits original", "galletas caseras",
        "biscotti fatti in casa", "turkish biscuits",
    ],
    "sutlac": [
        "sütlaç", "fırında sütlaç", "fındıklı sütlaç",
        "tarçınlı sütlaç", "kremalı sütlaç", "vanilyalı sütlaç",
        "refika sütlaç", "nefis yemek sütlaç", "oktay usta sütlaç",
        "sütlaç porsiyon", "tek kişilik sütlaç",
        "turkish rice pudding", "rice pudding authentic",
        "best rice pudding", "easy rice pudding",
        "homemade rice pudding", "Milchreis original",
        "riz au lait original", "arroz con leche casero",
        "riso al latte", "creamy rice pudding",
    ],
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# -------------------------------------------------------------------
# HELPERS / YARDIMCI FONKSIYONLAR
# -------------------------------------------------------------------
def count_images(folder: Path) -> int:
    """Klasordeki gorsel sayisini dondurur."""
    return len([p for p in folder.glob("*.*")
                if p.suffix.lower() in IMAGE_EXTENSIONS])


def max_index(folder: Path) -> int:
    """Klasordeki en buyuk sayisal dosya adi (ornek: 00123.jpg -> 123)."""
    result = 0
    for p in folder.glob("*.*"):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            result = max(result, int(p.stem))
        except ValueError:
            continue
    return result


def pick_date_filter(offset: int):
    """
    Offset'e gore tarih filtresi secer (dongu halinde).
    Boylece ayni terim farkli yillarda aranir -> farkli sonuclar.
    """
    if not DATE_FILTERS:
        return None
    return DATE_FILTERS[offset % len(DATE_FILTERS)]


# -------------------------------------------------------------------
# PER-ENGINE DOWNLOAD FUNCTIONS
# -------------------------------------------------------------------
def download_google(folder: Path, query: str, count: int, offset: int,
                    date_filter=None):
    try:
        crawler = GoogleImageCrawler(
            feeder_threads=1, parser_threads=1, downloader_threads=2,
            storage={"root_dir": str(folder)},
        )
        kwargs = {
            "keyword": query,
            "max_num": count,
            "min_size": (200, 200),
            "file_idx_offset": offset,
        }
        if date_filter:
            kwargs["filters"] = {"date": date_filter}
        crawler.crawl(**kwargs)
    except Exception as e:
        print(f"    Google ERROR: {e}")


def download_bing(folder: Path, query: str, count: int, offset: int,
                  date_filter=None):
    try:
        crawler = BingImageCrawler(
            feeder_threads=1, parser_threads=1, downloader_threads=2,
            storage={"root_dir": str(folder)},
        )
        crawler.crawl(keyword=query, max_num=count, min_size=(200, 200),
                      file_idx_offset=offset)
    except Exception as e:
        print(f"    Bing ERROR: {e}")


def download_baidu(folder: Path, query: str, count: int, offset: int,
                   date_filter=None):
    try:
        crawler = BaiduImageCrawler(
            feeder_threads=1, parser_threads=1, downloader_threads=2,
            storage={"root_dir": str(folder)},
        )
        crawler.crawl(keyword=query, max_num=count, min_size=(200, 200),
                      file_idx_offset=offset)
    except Exception as e:
        print(f"    Baidu ERROR: {e}")


def download_flickr(folder: Path, query: str, count: int, offset: int,
                    date_filter=None):
    try:
        crawler = FlickrImageCrawler(
            feeder_threads=1, parser_threads=1, downloader_threads=2,
            storage={"root_dir": str(folder)},
        )
        crawler.crawl(keyword=query, max_num=count, min_size=(200, 200),
                      file_idx_offset=offset)
    except Exception as e:
        print(f"    Flickr ERROR: {e}")


def download_duckduckgo(folder: Path, query: str, count: int, offset: int,
                        date_filter=None):
    """
    DuckDuckGo gorsellerini indirir.
    icrawler'da hazir crawler yok, kendi implementasyonumuz.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("    DuckDuckGo ERROR: 'duckduckgo-search' kurulu degil")
        return

    try:
        with DDGS() as ddgs:
            results = ddgs.images(
                keywords=query,
                max_results=count,
                safesearch="off",
            )
            if not results:
                return

            downloaded = 0
            for i, r in enumerate(results):
                if downloaded >= count:
                    break
                url = r.get("image")
                if not url:
                    continue

                # Uzantiyi URL'den al
                ext = ".jpg"
                for e in [".jpg", ".jpeg", ".png", ".webp"]:
                    if url.lower().endswith(e):
                        ext = e
                        break

                filename = folder / f"{offset + i:06d}{ext}"
                if filename.exists():
                    continue

                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    resp = requests.get(url, timeout=DUCKDUCKGO_TIMEOUT,
                                        headers=headers, stream=True)
                    if resp.status_code == 200 and len(resp.content) > 5000:
                        filename.write_bytes(resp.content)
                        downloaded += 1
                except Exception:
                    continue

        if downloaded > 0:
            print(f"    DuckDuckGo '{query}': {downloaded} indirildi")
    except Exception as e:
        print(f"    DuckDuckGo ERROR: {e}")


ENGINE_FUNCTIONS = {
    "google":     download_google,
    "bing":       download_bing,
    "baidu":      download_baidu,
    "flickr":     download_flickr,
    "duckduckgo": download_duckduckgo,
}


# -------------------------------------------------------------------
# DOWNLOAD UNTIL TARGET
# -------------------------------------------------------------------
def download_until_target(folder: Path, queries: list, target: int, offset: int):
    """
    Her turda mevcut sayiyi kontrol eder.
    Hedefe ulasirsa hemen durur, bir sonraki sinifa gecer.
    """
    if target <= 0:
        return

    active_engines = [name for name, on in ENGINES.items() if on]
    if USE_DUCKDUCKGO and "duckduckgo" not in active_engines:
        active_engines.append("duckduckgo")

    if not active_engines:
        print("    ERROR: No active engines!")
        return

    total_combos = len(active_engines) * len(queries)
    share = max(target // total_combos, 10)

    print(f"    {len(active_engines)} engines x {len(queries)} queries = "
          f"{total_combos} combos, ~{share} each")

    idx = 0
    for query in queries:
        current = count_images(folder)
        if current >= target:
            print(f"    Target reached ({current}/{target}), stopping")
            return

        remaining = target - current
        turn_share = min(share, remaining)

        for engine_name in active_engines:
            current = count_images(folder)
            if current >= target:
                print(f"    Target reached ({current}/{target}), stopping")
                return

            remaining = target - current
            engine_share = min(turn_share, remaining)

            engine_fn = ENGINE_FUNCTIONS[engine_name]

            # Google icin tarih filtresi kullan
            if engine_name == "google":
                date_filter = pick_date_filter(idx)
                engine_fn(folder, query, engine_share,
                          offset + idx * 100000, date_filter)
            else:
                engine_fn(folder, query, engine_share,
                          offset + idx * 100000)

            idx += 1
            if engine_name == "duckduckgo":
                time.sleep(DUCKDUCKGO_WAIT)
            else:
                time.sleep(WAIT_SECONDS)

        after = count_images(folder)
        if after >= target:
            print(f"    Target reached: {after}/{target}")
            return

    final = count_images(folder)
    print(f"    Loop finished, current: {final}/{target}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Output exists: {OUTPUT_DIR.exists()}")
    print(f"Active engines: {[k for k, v in ENGINES.items() if v]}")
    print(f"DuckDuckGo   : {'ON' if USE_DUCKDUCKGO else 'OFF'}")
    print(f"Selected classes: {len(SELECTED) if SELECTED else 'ALL (30)'}")
    print(f"Target/class : {TARGET_COUNT}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_list = SELECTED if SELECTED else list(FOODS.keys())

    print(f"{'Class':<25} {'Current':>8} {'Target':>8} {'Status':>10}")
    print("-" * 60)

    plan = []

    for class_name in target_list:
        if class_name not in FOODS:
            print(f"[WARN] {class_name}: not in FOODS, skipping")
            continue

        folder = OUTPUT_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)

        current = count_images(folder)
        missing = TARGET_COUNT - current

        if missing <= 0:
            print(f"{class_name:<25} {current:>8} {TARGET_COUNT:>8} {'DONE':>10}")
            continue

        print(f"{class_name:<25} {current:>8} {TARGET_COUNT:>8} {'MISSING':>10}")
        plan.append((class_name, current))

    print("-" * 60)
    print(f"Classes to download: {len(plan)}")

    if not plan:
        print("All classes at target.")
        return

    answer = input("\nStart downloading? (y/n): ").strip().lower()
    if answer != "y":
        print("Cancelled.")
        return

    for class_name, start_count in plan:
        folder = OUTPUT_DIR / class_name
        before = count_images(folder)
        offset = max_index(folder) + 1

        print(f"\n[{class_name}] start={before}, target={TARGET_COUNT}, offset={offset}")

        download_until_target(folder, FOODS[class_name], TARGET_COUNT, offset)

        after = count_images(folder)
        print(f"  RESULT: {before} -> {after} (+{after - before})")

    print("\nDownload finished.")
    print("Next step: python Meta/cleaner.py")


if __name__ == "__main__":
    main()