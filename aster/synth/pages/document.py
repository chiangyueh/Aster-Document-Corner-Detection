                   
import numpy as np

from .common import INK, rect_alpha


def render_text(kit, h, w, bg_pool):
    r = kit.r
    zh = r.random() < kit.cjk_ratio
    ink = r.choice(INK)
    unit = min(h, w)
    img, d = kit.canvas(kit.paper(h, w))
    margin = int(unit * (r.uniform(0.0, 0.03) if r.random() < 0.15 else r.uniform(0.05, 0.12)))
    base = unit * r.uniform(0.025, 0.045)
    style = r.choice(["any", "any", "serif"])
    y = margin
    if r.random() < 0.35:
        bh = int(h * r.uniform(0.04, 0.12))
        full = r.random() < 0.5
        d.rectangle([0 if full else margin, 0 if full else margin, w if full else w - margin,
                     bh + (0 if full else margin)], fill=kit.color(r.random() < 0.5))
        y = bh + margin + int(base)
    if r.random() < 0.7:
        title = kit.t.zh_word() + kit.t.zh_word() if zh else kit.t.words(r.randint(1, 4), caps=True)
        d.text((margin + r.randint(0, w // 4), y), title, font=kit.font(base * 1.8, "bold", zh), fill=ink)
        y += int(base * 3)
    blocks = r.randint(1, 4)
    for b in range(blocks):
        if y > h - margin - base * 3:
            break
        y_end = min(h - margin, y + int((h - margin - y) / (blocks - b) * r.uniform(0.6, 1.0)))
        t = r.random()
        if t < 0.55:
            y = kit.text_block(d, (margin, y, w - margin, y_end), zh, base, ink, style)
        elif t < 0.8:
            _table(kit, d, (margin, y, w - margin, y_end), zh, base * 0.9, ink)
            y = y_end
        else:
            pw = int((w - 2 * margin) * r.uniform(0.4, 1.0))
            kit.paste_photo(img, (margin, y, margin + pw, y_end), bg_pool)
            y = y_end
        y += int(base)
    if r.random() < 0.25:
        rad = int(unit * r.uniform(0.06, 0.12))
        kit.stamp(d, r.randint(rad, w - rad), r.randint(rad, h - rad), rad, zh)
    if r.random() < 0.25:
        kit.scribble(d, (int(w * 0.55), h - margin - base * 3, int(w * 0.9), h - margin - base),
                     (20, 20, 80), max(1, unit // 200))
    return np.asarray(img), rect_alpha(h, w)


def _table(kit, d, box, zh, size, ink):
    r = kit.r
    x0, y0, x1, y1 = [int(v) for v in box]
    rows, cols = r.randint(3, 10), r.randint(2, 6)
    inner = r.sample(range(x0 + 10, x1 - 10), cols - 1) if x1 - x0 > 20 + cols else []
    xs = np.sort(np.r_[x0, inner, x1]).astype(int)
    ys = np.linspace(y0, y1, rows + 1).astype(int)
    lw = r.choice([1, 1, 2])
    if r.random() < 0.5:
        d.rectangle([x0, ys[0], x1, ys[1]], fill=kit.color(False))
    for y in ys:
        d.line([x0, y, x1, y], fill=ink, width=lw)
    if r.random() < 0.7:
        for x in xs:
            d.line([x, y0, x, y1], fill=ink, width=lw)
    f = kit.font(size, "any", zh)
    for i in range(rows):
        for j in range(len(xs) - 1):
            if r.random() < 0.8:
                chars = max(1, int((xs[j + 1] - xs[j]) / (size * (1.0 if zh else 0.6))))
                cell = kit.t.money() if (not zh and r.random() < 0.3) else kit.text_line(zh, chars)
                d.text((xs[j] + 3, ys[i] + 2), cell[:chars], font=f, fill=ink)


def render_note(kit, h, w, bg_pool):
    r = kit.r
    unit = min(h, w)
    img, d = kit.canvas(kit.paper(h, w))
    step = int(unit * r.uniform(0.04, 0.07))
    col = (240, int(r.uniform(150, 220)), int(r.uniform(120, 200)))[::-1]
    for y in range(step * 2, h, step):
        d.line([0, y, w, y], fill=col, width=1)
    if r.random() < 0.4:
        for x in range(step, w, step):
            d.line([x, 0, x, h], fill=col, width=1)
    hand = kit.font(step * 0.8, "hand")
    for y in range(step * 2, h - step, step):
        if r.random() < 0.6:
            if r.random() < 0.6:
                d.text((int(w * 0.08), y - step), kit.t.line(int(w / (step * 0.35))), font=hand, fill=r.choice(INK))
            else:
                kit.scribble(d, (int(w * 0.08), y - step * 0.7, int(w * r.uniform(0.4, 0.95)), y - step * 0.1),
                             r.choice(INK), max(1, unit // 250))
    return np.asarray(img), rect_alpha(h, w)


def render_blank(kit, h, w, bg_pool):
    return kit.paper(h, w), rect_alpha(h, w)
