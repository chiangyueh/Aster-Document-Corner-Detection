         
import numpy as np

from .common import rect_alpha, rounded_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    img, d = kit.canvas(kit.paper(h, w, list(kit.color(False)) if r.random() < 0.4 else None))
    ink = (25, 25, 30)
    accent = kit.color()
    stub = int(w * r.uniform(0.2, 0.3))
    main_w = w - stub
    if r.random() < 0.7:
        d.rectangle([0, 0, w, int(h * r.uniform(0.15, 0.3))], fill=accent)
    if r.random() < 0.3 and bg_pool is not None:
        kit.paste_photo(img, (0, int(h * 0.3), int(main_w * 0.35), h), bg_pool)
    kit.dashed(d, (main_w, 0), (main_w, h), (90, 90, 90), max(1, unit // 120), 6, 5)
    title = kit.t.words(r.randint(1, 3)).upper()
    big = kit.fit_font(d, title, main_w - unit * 0.16, unit * r.uniform(0.14, 0.22), "bold")
    d.text((int(unit * 0.08), int(h * 0.05)), title, font=big, fill=(255, 255, 255))
    small = kit.font(unit * 0.07, r.choice(["any", "mono"]))
    fields = [kit.t.date(), kit.t.time(), f"{kit.t.word(1, 1).upper()} {r.randint(1, 40)}",
              f"{kit.t.word(1, 1).upper()} {r.choice('ABCDEFGH')}{r.randint(1, 30)}", kit.t.money(5, 300)]
    x, y = int(unit * 0.08), int(h * 0.45)
    for f in fields[: r.randint(3, 5)]:
        d.text((x, y), kit.t.word(1, 1).upper(), font=kit.font(unit * 0.045), fill=(110, 110, 110))
        d.text((x, y + unit * 0.06), f, font=small, fill=ink)
        x += int(main_w / 4)
        if x > main_w - unit * 0.3:
            x, y = int(unit * 0.08), y + int(unit * 0.2)
    if r.random() < 0.6:
        kit.barcode(d, (main_w + stub * 0.15, h * 0.2, main_w + stub * 0.85, h * 0.8), ink)
    else:
        q = int(min(stub * 0.7, h * 0.6))
        kit.qr(img, main_w + (stub - q) / 2, (h - q) / 2, q)
    alpha = rounded_alpha(h, w, int(unit * 0.05)) if r.random() < 0.3 else rect_alpha(h, w)
    return np.asarray(img), alpha
