          
import cv2
import numpy as np

from .common import rect_alpha, rounded_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    material = r.choice(["painted", "wood", "metal", "chalk", "lightbox"])
    if material == "wood":
        base = np.ones((h, w, 3), np.float32) * np.float32([r.uniform(110, 170), r.uniform(70, 110), r.uniform(30, 60)])
        shape = (max(8, h // 6), 1) if r.random() < 0.7 else (1, max(8, w // 6))
        grain = cv2.resize(np.random.randn(*shape).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        base *= (1 + 0.12 * grain)[..., None]
    elif material == "metal":
        base = np.ones((h, w, 3), np.float32) * r.uniform(150, 210)
        base += cv2.resize(np.random.randn(max(8, h // 3), 1).astype(np.float32), (w, h))[..., None] * 8
    elif material == "chalk":
        base = np.ones((h, w, 3), np.float32) * np.float32([r.uniform(20, 45), r.uniform(40, 60), r.uniform(30, 45)])
    elif material == "lightbox":
        base = np.ones((h, w, 3), np.float32) * np.float32([240, 240, 235])
    else:
        base = np.ones((h, w, 3), np.float32) * np.float32(kit.color(r.random() < 0.8))
    base += np.random.randn(h, w, 1).astype(np.float32) * r.uniform(1, 6)
    img, d = kit.canvas(np.clip(base, 0, 255).astype(np.uint8))
    light = material in ("lightbox", "metal") or float(base.mean()) > 150
    ink = (25, 25, 25) if light else (245, 245, 240)
    if material != "chalk" and r.random() < 0.7:
        fw = int(unit * r.uniform(0.03, 0.08))
        d.rectangle([0, 0, w - 1, h - 1], outline=kit.color(False) if r.random() < 0.5 else (30, 30, 30), width=fw)
        if r.random() < 0.5:
            inset = fw + int(unit * 0.03)
            d.rectangle([inset, inset, w - 1 - inset, h - 1 - inset], outline=ink, width=max(1, fw // 3))
    style = "hand" if material == "chalk" else r.choice(["bold", "bold", "serif"])
    lines = r.randint(1, 3)
    y = h * r.uniform(0.15, 0.3)
    for k in range(lines):
        size = unit * (r.uniform(0.25, 0.45) if k == 0 else r.uniform(0.12, 0.2)) * (1.0 if w >= h else 0.6)
        text = kit.t.words(r.randint(1, 2)).upper() if k == 0 else kit.t.words(r.randint(1, 4), caps=True)
        f = kit.fit_font(d, text, w * 0.86, size, style)
        tw = d.textlength(text, font=f)
        x = (w - tw) / 2 if r.random() < 0.7 else unit * 0.1
        d.text((x, y), text, font=f, fill=ink if k else (kit.color() if material == "lightbox" else ink))
        y += f.size * 1.25
        if y > h * 0.9:
            break
    if r.random() < 0.4:
        cx, cy, s = w * r.uniform(0.15, 0.85), h * 0.8, unit * 0.12
        d.polygon([(cx - s, cy - s * 0.3), (cx + s * 0.3, cy - s * 0.3), (cx + s * 0.3, cy - s * 0.7),
                   (cx + s, cy), (cx + s * 0.3, cy + s * 0.7), (cx + s * 0.3, cy + s * 0.3), (cx - s, cy + s * 0.3)],
                  fill=ink)
    alpha = rounded_alpha(h, w, int(unit * r.uniform(0.03, 0.12))) if r.random() < 0.3 else rect_alpha(h, w)
    return np.asarray(img), alpha
