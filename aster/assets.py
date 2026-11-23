                
import csv
import json
import os
import urllib.request

SOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "download_sources")
BACKGROUND_URL = "https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"


def _download(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def background_rows(n=None):
    with open(os.path.join(SOURCES, "backgrounds.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if n is not None:
        rows = rows[:n]
    return rows


def prepare_backgrounds(directory, n, workers=16, log=print):
    from concurrent.futures import ThreadPoolExecutor
    rows = background_rows(n)

    def one(r):
        url = BACKGROUND_URL.format(subset=r["subset"], image_id=r["image_id"])
        return _download(url, os.path.join(directory, r["image_id"] + ".jpg"))

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, rows), 1):
            if k % 1000 == 0:
                log(f"backgrounds {k}/{len(rows)}")
    return [(os.path.join(directory, r["image_id"] + ".jpg"), int(r["rotation"])) for r in rows]


def background_list(directory, n):
    out = []
    for r in background_rows(n):
        p = os.path.join(directory, r["image_id"] + ".jpg")
        out.append((p, int(r["rotation"])))
    return out


def font_entries():
    with open(os.path.join(SOURCES, "fonts.json")) as f:
        return json.load(f)["fonts"]


def prepare_fonts(directory):
    paths = {"latin": [], "cjk": []}
    for e in font_entries():
        p = _download(e["url"], os.path.join(directory, e["file"]))
        paths[e["script"]].append(p)
    return paths


def font_paths(directory):
    paths = {"latin": [], "cjk": []}
    for e in font_entries():
        p = os.path.join(directory, e["file"])
        paths[e["script"]].append(p)
    return paths


def prepare_pretrained(directory):
    with open(os.path.join(SOURCES, "pretrained.json")) as f:
        e = json.load(f)
    return _download(e["url"], os.path.join(directory, e["file"]))


def prepare(cfg):
    directory = cfg["assets"]["dir"]
    fonts = prepare_fonts(os.path.join(directory, "fonts"))
    print(f"fonts: {len(fonts['latin'])} latin, {len(fonts['cjk'])} cjk")
    print("pretrained:", prepare_pretrained(os.path.join(directory, "pretrained")))
    backgrounds = prepare_backgrounds(os.path.join(directory, "backgrounds"), cfg["assets"]["backgrounds"])
    print(f"backgrounds: {len(backgrounds)}")
