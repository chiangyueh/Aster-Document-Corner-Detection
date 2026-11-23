         
import numpy as np

from .common import rect_alpha, torn_alpha


def render(kit, h, w, bg_pool):
    r = kit.r
    zh = r.random() < kit.cjk_ratio * 0.5
    thermal = r.random() < 0.75
    base = [r.uniform(238, 252)] * 3 if thermal else None
    img, d = kit.canvas(kit.paper(h, w, base))
    ink = (int(r.uniform(40, 110)),) * 3 if thermal else (20, 20, 20)
    size = w * r.uniform(0.045, 0.065)
    mono = kit.font(size, "mono")
    cols = max(12, int(w * 0.86 / (size * 0.6)))
    m = int(w * 0.07)
    y = int(size * r.uniform(1, 3))

    def put(text, f=mono, center=False):
        nonlocal y
        tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2 if center else m, y), text, font=f, fill=ink)
        y += int(f.size * 1.35)

    shop = kit.t.company().upper()
    put(shop, kit.fit_font(d, shop, w - 2 * m, size * 1.4, "bold"), center=True)
    for line in kit.t.address():
        put(line[:cols], center=True)
    put(kit.t.phone(), center=True)
    put("-" * cols)
    put(f"{kit.t.date()}  {kit.t.time()}  #{kit.t.code(6)}"[:cols])
    put("-" * cols)
    total = 0.0
    for _ in range(r.randint(3, 25)):
        if y > h - size * 12:
            break
        price = r.uniform(0.5, 120)
        total += price
        label = (kit.t.zh_word() if zh else kit.t.words(r.randint(1, 3))).upper()
        amount = f"{price:,.2f}"
        if r.random() < 0.3:
            put(f"  {r.randint(2, 9)} x {price / r.randint(2, 9):.2f}"[:cols])
        put(label[: cols - len(amount) - 1].ljust(cols - len(amount)) + amount)
    put("-" * cols)
    for k in ("SUBTOTAL", "TAX", "TOTAL"):
        v = total if k != "TAX" else total * 0.08
        if k == "TOTAL":
            v = total * 1.08
        label = kit.t.word(1, 2).upper() if r.random() < 0.5 else k
        put(label.ljust(cols - 12) + f"{v:12,.2f}"[-12:], kit.font(size * (1.25 if k == "TOTAL" else 1), "mono"))
    put(f"{kit.t.word(1, 2).upper()} **** {r.randint(1000, 9999)}"[:cols])
    y += int(size)
    if r.random() < 0.7 and y < h - size * 4:
        kit.barcode(d, (m, y, w - m, y + size * 2.5), ink)
        y += int(size * 3)
    if r.random() < 0.5 and y < h - size * 2:
        put(kit.t.words(r.randint(2, 4)).upper()[:cols], center=True)
    rgb = np.asarray(img).astype(np.float32)
    if thermal and r.random() < 0.4:
        fade = np.linspace(r.uniform(0.0, 0.5), 1.0, h, dtype=np.float32)[::r.choice([1, -1])][:, None, None]
        rgb = 255 - (255 - rgb) * fade
    alpha = torn_alpha(h, w, r, top=r.random() < 0.3, bottom=r.random() < 0.6) if r.random() < 0.6 else rect_alpha(h, w)
    return np.clip(rgb, 0, 255).astype(np.uint8), alpha
