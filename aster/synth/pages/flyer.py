               
import numpy as np

from .common import rect_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    style = r.choices(["photo", "solid", "gradient", "blocks"], [0.35, 0.3, 0.2, 0.15])[0]
    if style == "photo" and bg_pool is not None:
        base = bg_pool.random_crop(h, w)[..., ::-1].astype(np.float32)
        if r.random() < 0.5:
            base = base * r.uniform(0.4, 0.8) + np.float32(kit.color()) * r.uniform(0.0, 0.3)
    elif style == "gradient":
        c1, c2 = np.float32(kit.color()), np.float32(kit.color())
        vertical = r.random() < 0.6
        t = np.linspace(0, 1, h if vertical else w, dtype=np.float32)
        g = t[:, None, None] if vertical else t[None, :, None]
        base = np.broadcast_to(c1 * (1 - g) + c2 * g, (h, w, 3)).copy()
    elif style == "solid":
        base = np.ones((h, w, 3), np.float32) * np.float32(kit.color(r.random() < 0.8))
    else:
        base = np.ones((h, w, 3), np.float32) * np.float32([r.uniform(235, 255)] * 3)
    img, d = kit.canvas(np.clip(base, 0, 255).astype(np.uint8))

    if style == "blocks" or r.random() < 0.3:
        for _ in range(r.randint(1, 3)):
            if r.random() < 0.5:
                y0 = r.randint(-unit // 10, h - 1)
                d.rectangle([-5, y0, w + 5, y0 + int(h * r.uniform(0.05, 0.3))], fill=kit.color())
            else:
                x0 = r.randint(-unit // 10, w - 1)
                d.rectangle([x0, -5, x0 + int(w * r.uniform(0.05, 0.3)), h + 5], fill=kit.color())
    if bg_pool is not None and r.random() < 0.6:
        for _ in range(r.randint(1, 3)):
            pw, ph = int(w * r.uniform(0.3, 1.0)), int(h * r.uniform(0.2, 0.55))
            x0 = 0 if r.random() < 0.4 else r.randint(0, max(0, w - pw))
            y0 = r.randint(0, max(0, h - ph))
            kit.paste_photo(img, (x0, y0, x0 + pw, y0 + ph), bg_pool)
    if r.random() < 0.3:
        k, b = unit * r.uniform(0.15, 0.35), unit * r.uniform(0.05, 0.1)
        cx, cy = r.choice([(0, 0), (w, 0), (w, h), (0, h)])
        sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
        d.polygon([(cx + sx * k, cy), (cx + sx * (k + b), cy), (cx, cy + sy * (k + b)), (cx, cy + sy * k)], fill=kit.color())

    y = int(h * r.uniform(0.03, 0.4))
    for _ in range(r.randint(1, 3)):
        size = unit * r.uniform(0.08, 0.2)
        f = kit.font(size, r.choice(["bold", "bold", "serif", "any"]))
        col = kit.color()
        stroke = int(size * 0.06) if r.random() < 0.5 else 0
        d.text((int(w * r.uniform(-0.02, 0.2)), y), kit.t.headline(), font=f, fill=col, stroke_width=stroke,
               stroke_fill=(255, 255, 255) if sum(col) < 380 else (20, 20, 20))
        y += int(size * 1.3)
    if r.random() < 0.4:
        light = style in ("photo", "solid")
        kit.text_block(d, (int(w * 0.06), y, int(w * 0.94), min(h, y + int(h * 0.25))), False, unit * 0.035,
                       (245, 245, 245) if light else (30, 30, 30))
    if r.random() < 0.6:
        rad = int(unit * r.uniform(0.08, 0.18))
        cx, cy = r.randint(rad // 2, w - rad // 2), r.randint(rad // 2, h - rad // 2)
        kit.starburst(d, cx, cy, rad, r.choice([(230, 30, 30), (250, 210, 0), (255, 120, 0)]))
        label = r.choice([kit.t.money(1, 999), f"{r.randint(1, 9) * 10}%", kit.t.word(1, 1).upper()])
        d.text((cx - rad * 0.65, cy - rad * 0.3), label, font=kit.font(rad * 0.45, "bold"), fill=(255, 255, 255))
    if r.random() < 0.3:
        x0, y0 = int(w * 0.08), int(h * r.uniform(0.6, 0.8))
        x1, y1 = int(w * 0.92), min(h - 5, y0 + int(h * 0.15))
        for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            kit.dashed(d, a, b, (30, 30, 30), 2)
        d.text((x0 + 10, y0 + 8), kit.t.words(r.randint(2, 5)).upper(), font=kit.font(unit * 0.04, "bold"), fill=(30, 30, 30))
    if r.random() < 0.4:
        q = int(unit * r.uniform(0.1, 0.18))
        kit.qr(img, r.randint(0, w - q), r.randint(int(h * 0.6), h - q), q)
    fine = kit.font(unit * 0.022)
    for k in range(r.randint(0, 3)):
        line = r.choice([kit.t.line(int(w / (unit * 0.013))), f"{kit.t.web()}  {kit.t.phone()}"])
        d.text((int(w * 0.05), int(h * (0.9 + 0.03 * k))), line, font=fine,
               fill=(250, 250, 250) if style in ("photo", "solid") else (40, 40, 40))
    return np.asarray(img), rect_alpha(h, w)
