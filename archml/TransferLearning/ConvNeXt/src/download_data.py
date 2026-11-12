import csv
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageOps


PEOPLE = [
    'Narendra Modi',
    'Xi Jinping',
    'Donald Trump',
    'Sanae Takaichi',
    'Vladimir Putin',
    'Javier Milei',
    'Lee Jae-myung',
    'Claudia Sheinbaum',
    'Mark Carney',
    'Mohammed bin Salman'
]

TARGET = 500
MIN_SIZE = 512

ROOT = Path(__file__).resolve().parent.parent / "data/politicians/raw"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

session = requests.Session()
session.headers["User-Agent"] = (
    "ArchML/1.0 (https://github.com/DragonTop100/ArchML)"
)


def get(url, params=None):
    response = session.get(
        url, params=params, timeout=30
    )
    return response


def get_qid(name):
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "type": "item",
        "limit": 10,
        "format": "json",
    }

    data = get(WIKIDATA_API, params).json()
    results = data.get("search", [])

    if not results:
        raise ValueError(f"Нет результатов поиска: {name}")

    matches = [
        item for item in results
        if item.get("label", "").casefold() == name.casefold()
    ]

    if not matches:
        raise ValueError(
            f"Нет точного совпадения для {name}. "
            f"Найдено: {[(x['id'], x.get('label')) for x in results]}"
        )

    person = matches[0]

    print(
        f"Найден: {person['label']} "
        f"({person['id']}) — {person.get('description', '')}"
    )

    return person["id"]


def search_images(qid):
    continuation = {}

    while True:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrsearch": f"haswbstatement:P180={qid}",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": 1024,
            **continuation,
        }

        data = get(COMMONS_API, params).json()

        if "error" in data:
            raise RuntimeError(data["error"])

        yield from data.get("query", {}).get("pages", [])

        continuation = data.get("continue")
        if not continuation:
            break


def image_hash(image):
    image = image.convert("L").resize((9, 8))
    pixels = list(image.get_flattened_data())

    result = 0

    for y in range(8):
        for x in range(8):
            result = (result << 1) | (
                pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            )

    return result


def download_politicians():
    ROOT.mkdir(parents=True, exist_ok=True)

    hashes = []

    for file in ROOT.glob("*/*.jpg"):
        try:
            with Image.open(file) as image:
                hashes.append(image_hash(image))
        except OSError:
            pass

    for name in PEOPLE:
        print(f"\n{name}")

        qid = get_qid(name)

        print(f"Wikidata ID: {qid}")

        folder = ROOT / name.lower().replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        count = len(list(folder.glob("*.jpg")))

        if count >= TARGET:
            print(f"Уже скачано: {count}/{TARGET}")
            continue

        csv_path = folder / "sources.csv"
        write_header = not csv_path.exists()

        with csv_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            if write_header:
                writer.writerow(["filename", "source", "url"])

            for page in search_images(qid):
                info = (page.get("imageinfo") or [{}])[0]

                if info.get("mime") not in {
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                }:
                    continue

                if min(
                    info.get("width", 0),
                    info.get("height", 0),
                ) < MIN_SIZE:
                    continue

                path = folder / f"{page['pageid']}.jpg"

                if path.exists():
                    continue

                url = info.get("thumburl") or info.get("url")

                if not url:
                    continue

                try:
                    content = get(url).content

                    with Image.open(BytesIO(content)) as img:
                        image = ImageOps.exif_transpose(img)
                        image = image.convert("RGB")

                    if min(image.size) < MIN_SIZE:
                        continue

                    h = image_hash(image)

                    if any(
                        (h ^ other).bit_count() <= 3
                        for other in hashes
                    ):
                        continue

                    image.save(path, "JPEG", quality=90)

                except (requests.RequestException, OSError) as error:
                    print(f"Пропущено: {error}")
                    continue

                hashes.append(h)

                writer.writerow([
                    path.name,
                    info.get("descriptionurl", ""),
                    info.get("url", ""),
                ])

                file.flush()
                count += 1

                print(f"Скачано: {count}/{TARGET}")

                if count >= TARGET:
                    break

                time.sleep(0.3)

        print(f"Итого: {count}/{TARGET}")
