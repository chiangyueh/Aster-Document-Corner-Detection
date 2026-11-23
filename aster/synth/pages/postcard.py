          
import numpy as np

from .common import INK, rect_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    img, d = kit.canvas(kit.paper(h, w, [r.uniform(235, 252)] * 3))
    if r.random() < 0.5 and bg_pool is not None:
        b = 0 if r.random() < 0.5 else int(unit * r.uniform(0.03, 0.07))
        kit.paste_photo(img, (b, b, w - b, h - b), bg_pool)
        if r.random() < 0.6:
            f = kit.font(unit * r.uniform(0.08, 0.15), r.choice(["bold", "hand"]))
            text = kit.t.words(r.randint(1, 3), caps=True)
            d.text((b + unit * 0.05, h * r.uniform(0.05, 0.75)), text, font=f, fill=kit.color(),
                   stroke_width=max(1, int(f.size * 0.05)), stroke_fill=(255, 255, 255))
        return np.asarray(img), rect_alpha(h, w)

    ink = r.choice(INK)
    mid = int(w * r.uniform(0.5, 0.6))
    d.line([mid, int(h * 0.1), mid, int(h * 0.9)], fill=(120, 120, 120), width=max(1, unit // 250))
    s = int(unit * r.uniform(0.2, 0.28))
    sx, sy = w - s - int(unit * 0.05), int(unit * 0.05)
    kit.paste_photo(img, (sx, sy, sx + s, sy + int(s * 1.2)), bg_pool)
    d.rectangle([sx, sy, sx + s, sy + int(s * 1.2)], outline=(250, 250, 250), width=max(2, s // 15))
    if r.random() < 0.7:
        cx, cy, rad = sx - s * 0.2, sy + s * 0.8, s * 0.45
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(60, 60, 70), width=2)
        for k in range(4):
            yy = cy - rad * 0.4 + k * rad * 0.25
            d.line([cx + rad, yy, cx + rad * 2.5, yy], fill=(60, 60, 70), width=2)
    step = int(h * 0.1)
    for k in range(4):
        yy = int(h * 0.5) + k * step
        if yy < h - step * 0.5:
            d.line([mid + unit * 0.05, yy, w - unit * 0.05, yy], fill=(150, 150, 150), width=1)
    hand = kit.font(step * 0.7, "hand")
    for k, line in enumerate([kit.t.name()] + kit.t.address()):
        d.text((mid + unit * 0.07, int(h * 0.5) + k * step - step * 0.8), line, font=hand, fill=ink)
    y = int(h * 0.12)
    while y < h * 0.85:
        d.text((int(w * 0.05), y), kit.t.line(int(mid / (step * 0.3))), font=hand, fill=ink)
        y += int(step * 0.8)
    return np.asarray(img), rect_alpha(h, w)
