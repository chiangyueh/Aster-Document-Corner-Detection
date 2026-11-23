                      
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import fonts
from ..text import TextGen

INK = [(20, 20, 20), (40, 40, 45), (25, 35, 90), (10, 10, 10), (90, 20, 20)]
STYLE_KEYS = {
    "bold": ("700Bold",),
    "mono": ("SourceCodePro", "CourierPrime"),
    "hand": ("Caveat",),
    "serif": ("NotoSerif_", "Merriweather", "NotoSerifTC"),
}


class Kit:
    def __init__(self, rng, font_paths, cjk_ratio):
        self.r = rng
        self.t = TextGen(rng)
        self.latin, self.cjk = font_paths["latin"], font_paths["cjk"]
        self.cjk_ratio = cjk_ratio if self.cjk else 0.0


    def font(self, size, style="any", zh=False):
        pool = self.cjk if zh and self.cjk else self.latin
        if style != "any":
            keys = STYLE_KEYS[style]
            pick = [p for p in pool if any(k in os.path.basename(p) for k in keys)]
            if not pick and not zh:
                pick = [p for p in self.latin if any(k in os.path.basename(p) for k in keys)]
            pool = pick or pool
        else:
            pool = [p for p in pool if not any(k in os.path.basename(p) for k in STYLE_KEYS["hand"])] or pool
        return fonts.load(self.r.choice(pool), max(6, int(size)))

    def fit_font(self, d, text, max_width, size, style="any", zh=False):
        f = self.font(size, style, zh)
        path = f.path
        while size > 8 and d.textlength(text, font=f) > max_width:
            size *= 0.9
            f = fonts.load(path, int(size))
        return f

    def text_line(self, zh, chars):
        return self.t.zh_line(chars) if zh else self.t.line(chars)

    def text_block(self, d, box, zh, size, ink, style="any"):
        x0, y0, x1, y1 = box
        r = self.r
        f = self.font(size, style, zh)
        lh = int(size * r.uniform(1.25, 1.8))
        chars = max(4, int((x1 - x0) / (size * (1.0 if zh else 0.55))))
        y = y0
        while y + lh < y1:
            if r.random() < 0.08:
                y += lh
                continue
            n = chars if r.random() > 0.3 else int(chars * r.uniform(0.4, 1.0))
            d.text((x0, y), self.text_line(zh, n), font=f, fill=ink)
            y += lh
        return y


    def color(self, saturated=True):
        r = self.r
        hsv = np.uint8([[[r.randint(0, 179), r.randint(150, 255) if saturated else r.randint(0, 80), r.randint(120, 255)]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])

    def paper(self, h, w, base=None):
        r = self.r
        if base is None:
            base = [r.uniform(225, 255), r.uniform(222, 252), r.uniform(205, 250)] if r.random() < 0.85 else \
                [r.uniform(60, 255) for _ in range(3)]
        page = np.ones((h, w, 3), np.float32) * np.float32(base)
        lo = cv2.resize(np.random.randn(4, 4).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        page *= (1 + 0.03 * lo)[..., None]
        page += np.random.randn(h, w, 1).astype(np.float32) * r.uniform(1, 5)
        return np.clip(page, 0, 255).astype(np.uint8)

    @staticmethod
    def canvas(arr):
        img = Image.fromarray(arr)
        return img, ImageDraw.Draw(img)

    def paste_photo(self, img, box, bg_pool):
        x0, y0, x1, y1 = [int(v) for v in box]
        if bg_pool is None or x1 - x0 < 8 or y1 - y0 < 8:
            return
        img.paste(Image.fromarray(bg_pool.random_crop(y1 - y0, x1 - x0)[..., ::-1].copy()), (x0, y0))


    def scribble(self, d, box, ink, width):
        x0, y0, x1, y1 = box
        pts, x = [], x0
        while x < x1:
            pts.append((x, self.r.uniform(y0, y1)))
            x += self.r.uniform(3, 12)
        if len(pts) > 1:
            d.line(pts, fill=ink, width=width, joint="curve")

    def stamp(self, d, cx, cy, rad, zh):
        col = (200, 30, 30) if self.r.random() < 0.7 else (30, 40, 160)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=max(2, rad // 12))
        word = self.t.zh_word()[:4] if zh else self.t.word(1, 2).upper()
        d.text((cx - rad * 0.6, cy - rad * 0.25), word, font=self.font(rad * 0.4, "bold", zh), fill=col)

    def starburst(self, d, cx, cy, rad, color):
        n = self.r.randint(10, 18)
        pts = [(cx + (rad if k % 2 == 0 else rad * self.r.uniform(0.7, 0.85)) * np.cos(np.pi * k / n),
                cy + (rad if k % 2 == 0 else rad * self.r.uniform(0.7, 0.85)) * np.sin(np.pi * k / n))
               for k in range(2 * n)]
        d.polygon(pts, fill=color)

    def barcode(self, d, box, ink=(10, 10, 10)):
        x0, y0, x1, y1 = [int(v) for v in box]
        x = x0
        while x < x1:
            bw = self.r.choice([1, 1, 2, 3])
            if self.r.random() < 0.55:
                d.rectangle([x, y0, min(x1, x + bw - 1), y1], fill=ink)
            x += bw

    def qr(self, img, x, y, size):
        grid = (np.random.rand(21, 21) > 0.5).astype(np.uint8)
        for cx, cy in [(0, 0), (14, 0), (0, 14)]:
            grid[cy:cy + 7, cx:cx + 7] = 1
            grid[cy + 1:cy + 6, cx + 1:cx + 6] = 0
            grid[cy + 2:cy + 5, cx + 2:cx + 5] = 1
        q = cv2.resize(((1 - grid) * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        img.paste(Image.fromarray(np.dstack([q] * 3)), (int(x), int(y)))

    def dashed(self, d, p0, p1, ink, width=1, dash=8, gap=6):
        p0, p1 = np.float32(p0), np.float32(p1)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1:
            return
        u = (p1 - p0) / L
        s = 0.0
        while s < L:
            a, b = p0 + u * s, p0 + u * min(L, s + dash)
            d.line([tuple(a), tuple(b)], fill=ink, width=width)
            s += dash + gap


def rect_alpha(h, w):
    return np.full((h, w), 255, np.uint8)


def rounded_alpha(h, w, rad):
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (rad, 0), (w - 1 - rad, h - 1), 255, -1)
    cv2.rectangle(a, (0, rad), (w - 1, h - 1 - rad), 255, -1)
    for cx, cy in [(rad, rad), (w - 1 - rad, rad), (w - 1 - rad, h - 1 - rad), (rad, h - 1 - rad)]:
        cv2.circle(a, (cx, cy), rad, 255, -1, lineType=cv2.LINE_AA)
    return a


def torn_alpha(h, w, rng, top=False, bottom=True):
    a = np.full((h, w), 255, np.uint8)
    depth = max(2, int(min(h, w) * 0.03))
    for edge in [e for e, on in (("top", top), ("bottom", bottom)) if on]:
        xs = np.arange(0, w + 8, max(3, w // 40))
        ys = [rng.uniform(0, depth) for _ in xs]
        pts = [(int(x), int(y if edge == "top" else h - 1 - y)) for x, y in zip(xs, ys)]
        poly = [(0, 0 if edge == "top" else h)] + pts + [(w, 0 if edge == "top" else h)]
        cv2.fillPoly(a, [np.int32(poly)], 0, cv2.LINE_AA)
    return a
