                   
import cv2
import numpy as np

from .pages import LAYOUTS
from .pages.common import Kit


class PageGenerator:
    def __init__(self, font_paths, cjk_ratio, rng):
        self.rng = rng
        self.kit = Kit(rng, font_paths, cjk_ratio)

    def __call__(self, long_side=800, bg_pool=None):
        r = self.rng
        kinds = list(LAYOUTS)
        kind = r.choices(kinds, weights=[LAYOUTS[k][0] for k in kinds])[0]
        _, aspects, render, ornament = LAYOUTS[kind]
        aspect = r.uniform(*r.choice(aspects))
        if aspect >= 1:
            h, w = long_side, max(32, int(long_side / aspect))
        else:
            w, h = long_side, max(32, int(long_side * aspect))
        page, alpha = render(self.kit, h, w, bg_pool)
        if ornament and r.random() < 0.18:
            page = self._ornamental_border(page)
        if r.random() < 0.3:
            page = self._creases(page)
        if r.random() < 0.2:
            page = page.copy()
            if r.random() < 0.5:
                yy = int(h * r.uniform(0.3, 0.7))
                page[max(0, yy - 1):yy + 1] = (page[max(0, yy - 1):yy + 1] * 0.85).astype(np.uint8)
            else:
                xx = int(w * r.uniform(0.3, 0.7))
                page[:, max(0, xx - 1):xx + 1] = (page[:, max(0, xx - 1):xx + 1] * 0.85).astype(np.uint8)
        return np.ascontiguousarray(page[..., ::-1]), alpha, kind

    def _ornamental_border(self, page):
        r = self.rng
        h, w = page.shape[:2]
        out = page.astype(np.float32)
        tint = np.array([r.uniform(190, 225), r.uniform(170, 210), r.uniform(150, 200)], np.float32)
        m0 = int(min(h, w) * r.uniform(0.0, 0.03))
        band = int(min(h, w) * r.uniform(0.04, 0.09))
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.minimum.reduce([xx - m0, yy - m0, w - 1 - m0 - xx, h - 1 - m0 - yy])
        inband = (d >= 0) & (d < band)
        f1, f2 = r.uniform(0.15, 0.4), r.uniform(0.05, 0.2)
        wave = 0.5 + 0.5 * np.sin(xx * f1 + np.sin(yy * f2) * 3) * np.sin(yy * f1 + np.sin(xx * f2) * 3)
        a = (inband * (0.35 + 0.45 * wave))[..., None]
        out = out * (1 - a) + tint * a
        if r.random() < 0.7:
            cv2.rectangle(out, (m0 + band, m0 + band), (w - 1 - m0 - band, h - 1 - m0 - band),
                          tuple(float(c) * 0.8 for c in tint), max(1, min(h, w) // 300))
        if r.random() < 0.6:
            out = out * np.array([1.0, 0.97, 0.93], np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _creases(self, page):
        r = self.rng
        h, w = page.shape[:2]
        out = page.astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for _ in range(r.randint(1, 3)):
            ang = r.uniform(0, np.pi)
            px, py = r.uniform(0.2, 0.8) * w, r.uniform(0.2, 0.8) * h
            dist = (xx - px) * np.sin(ang) - (yy - py) * np.cos(ang)
            width = r.uniform(2, 0.03 * max(h, w))
            step = np.tanh(dist / width) * r.uniform(0.04, 0.15)
            ridge = np.exp(-(dist / max(1.0, width * 0.3)) ** 2) * r.uniform(-0.08, 0.08)
            out *= (1 + step + ridge)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
