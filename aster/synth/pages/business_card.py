         
import numpy as np

from .common import rect_alpha, rounded_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    dark = r.random() < 0.35
    bg = [r.uniform(15, 70)] * 3 if dark else None
    if r.random() < 0.25:
        bg = list(kit.color(r.random() < 0.5))
    img, d = kit.canvas(kit.paper(h, w, bg))
    ink = (240, 240, 240) if dark or (bg and sum(bg) < 360) else (25, 25, 30)
    accent = kit.color()
    m = int(unit * r.uniform(0.08, 0.14))
    layout = r.choice(["left", "center", "band"])
    if layout == "band":
        bw = int((w if r.random() < 0.5 else h) * r.uniform(0.15, 0.35))
        if r.random() < 0.5:
            d.rectangle([0, 0, bw, h], fill=accent)
            m += bw
        else:
            d.rectangle([0, h - bw, w, h], fill=accent)
    if r.random() < 0.7:
        s = int(unit * r.uniform(0.18, 0.3))
        lx, ly = (w - m - s, m) if layout != "center" else ((w - s) // 2, m)
        if r.random() < 0.5:
            d.ellipse([lx, ly, lx + s, ly + s], fill=accent)
        else:
            d.rectangle([lx, ly, lx + s, ly + s], fill=accent)
        d.text((lx + s * 0.2, ly + s * 0.2), kit.t.word(1, 1)[:2].upper(), font=kit.font(s * 0.5, "bold"), fill=(255, 255, 255))
    zh = r.random() < kit.cjk_ratio * 0.6
    name = kit.t.zh_word()[:3] if zh else kit.t.name()
    big = kit.fit_font(d, name, w - 2 * m, unit * r.uniform(0.11, 0.16), "bold", zh)
    small = kit.font(unit * r.uniform(0.055, 0.075), r.choice(["any", "serif"]))
    y = int(h * r.uniform(0.3, 0.45)) if layout != "center" else int(h * 0.45)

    def put(text, f):
        nonlocal y
        if d.textlength(text, font=f) > w - m - unit * 0.05:
            y += int(f.size * 1.35)
            return
        x = m if layout != "center" else (w - d.textlength(text, font=f)) / 2
        d.text((x, y), text, font=f, fill=ink)
        y += int(f.size * 1.35)

    put(name, big)
    put(kit.t.words(r.randint(1, 3), caps=True), small)
    y += int(small.size * 0.6)
    for line in [kit.t.phone(), kit.t.email(), kit.t.web()] + (kit.t.address() if r.random() < 0.5 else []):
        if y > h - small.size * 1.3:
            break
        put(line, small)
    alpha = rounded_alpha(h, w, int(unit * r.uniform(0.04, 0.1))) if r.random() < 0.4 else rect_alpha(h, w)
    return np.asarray(img), alpha
