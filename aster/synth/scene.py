                  
import random

import cv2
import numpy as np

from ..geometry import corners_inside, is_convex, order_corners, quad_area
from . import camera, effects
from .backgrounds import BackgroundPool
from .documents import PageGenerator

PHOTO_ASPECTS = [(4, 3), (3, 4), (16, 9), (9, 16), (1, 1)]


class SceneSynth:
    def __init__(self, backgrounds, font_paths, cfg):
        self.rng = random.Random(0)
        self.bg = BackgroundPool(backgrounds, self.rng)
        self.pages = PageGenerator(font_paths, cfg["cjk_ratio"], self.rng)
        self.out = cfg["out_size"]
        self.canvas = cfg["canvas"]
        self.modes = cfg["modes"]
        self.max_extent = cfg["max_extent"]
        self.light_surface = cfg.get("light_surface", 0.12)
        self.near_fill_light = cfg.get("near_fill_light", 0.6)
        self.light_contact_shadow = cfg.get("light_contact_shadow", 0.7)


    def _base_quad(self, W, H, page_hw, area_frac):
        r = self.rng
        ph, pw = page_hw
        a = ph / pw
        area = area_frac * W * H
        w = np.sqrt(area / a)
        h = w * a
        rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

        k = r.uniform(0.65, 1.0) if r.random() < 0.8 else 1.0
        side = r.randint(0, 3)
        if side == 0:
            rect[[0, 1], 0] *= k
        elif side == 1:
            rect[[2, 3], 0] *= k
        elif side == 2:
            rect[[0, 3], 1] *= k
        else:
            rect[[1, 2], 1] *= k
        rect += np.random.normal(0, 0.025, rect.shape) * np.array([w, h])
        u = r.random()
        if u < 0.7:
            ang = r.gauss(0, 8)
        elif u < 0.9:
            ang = r.uniform(-45, 45)
        else:
            ang = r.uniform(-180, 180)
        t = np.deg2rad(ang)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return rect @ R.T

    def _place(self, mode, W, H, page_hw):
        r = self.rng
        lo, hi = -self.max_extent, 1 + self.max_extent
        for _ in range(60):
            if mode == "normal":
                q = self._base_quad(W, H, page_hw, r.uniform(0.12, 0.8))
            elif mode == "out_of_frame":
                q = self._base_quad(W, H, page_hw, r.uniform(0.45, 1.6))
            elif mode == "near_fill":
                return self._near_fill(W, H, page_hw)
            else:
                q = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64) - [W / 2, H / 2]
                q += np.random.uniform(-0.04, 0.04, q.shape) * [W, H]
            span = q.max(0) - q.min(0)
            if mode == "normal":
                c = [r.uniform(span[0] / 2, W - span[0] / 2), r.uniform(span[1] / 2, H - span[1] / 2)] \
                    if span[0] < W and span[1] < H else None
            elif mode == "out_of_frame":
                c = [r.uniform(-0.1, 1.1) * W, r.uniform(-0.1, 1.1) * H]
            else:
                c = [W / 2, H / 2]
            if c is None:
                continue
            quad = q + c
            norm = quad / [W, H]
            if not is_convex(quad) or norm.min() < lo or norm.max() > hi:
                continue
            inside = corners_inside(norm)
            if mode == "normal" and not inside.all():
                continue
            if mode == "normal" and r.random() < 0.3:
                quad = self._push_to_edges(quad, W, H)
            if mode == "out_of_frame":
                if inside.all():
                    continue
                vis = cv2.intersectConvexConvex(quad.astype(np.float32),
                                                np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32))[0]
                if vis < 0.2 * W * H or vis < 0.25 * quad_area(quad):
                    continue
            return quad
        return None

    def _near_fill(self, W, H, page_hw):
        r = self.rng
        for _ in range(60):
            m = [r.uniform(0.005, 0.07) * (W if i % 2 == 0 else H) for i in range(4)]
            x0, y0, x1, y1 = m[0], m[1], W - m[2], H - m[3]
            box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)

            box += np.random.uniform(0, 0.05, (4, 2)) * [W, H] * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
            if r.random() < 0.5:
                t = np.deg2rad(r.gauss(0, 2.5))
                R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                c = box.mean(0)
                box = (box - c) @ R.T + c
                norm = box / [W, H]
                over = max(0.0, -norm.min(), norm.max() - 1)
                if over > 0:
                    box = c + (box - c) * (1 - 2.2 * over)
            norm = box / [W, H]
            if is_convex(box) and norm.min() > 0.002 and norm.max() < 0.998:
                return box
        return None

    def _push_to_edges(self, quad, W, H):
        m = self.rng.uniform(0.002, 0.03)
        c = quad.mean(0)
        k = np.inf
        for p in quad:
            d = p - c
            for ax, size in ((0, W), (1, H)):
                if d[ax] > 1e-6:
                    k = min(k, ((1 - m) * size - c[ax]) / d[ax])
                elif d[ax] < -1e-6:
                    k = min(k, (m * size - c[ax]) / d[ax])
        return c + (quad - c) * max(1.0, k)

    @staticmethod
    def _fill_quad(img, quad, color, texture_sigma=4.0):
        H, W = img.shape[:2]
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, np.round(quad * 4).astype(np.int32), 255, cv2.LINE_AA, shift=2)
        a = (m.astype(np.float32) / 255.0)[..., None]
        tex = np.float32(color) + np.random.randn(H, W, 1).astype(np.float32) * texture_sigma
        return img * (1 - a) + tex * a

    def _underlays(self, img, quad, W, H):
        r = self.rng
        c = quad.mean(0)
        kind = r.choices(["stack", "booklet", "cover"], [0.4, 0.4, 0.2])[0]
        paper = [r.uniform(225, 250)] * 3
        cover = [r.uniform(20, 90), r.uniform(20, 70), r.uniform(60, 140)] if r.random() < 0.7 else \
            [r.uniform(30, 200) for _ in range(3)]
        layers = []
        if kind in ("booklet", "cover"):
            side = r.randint(0, 3) if kind == "booklet" else None
            q = quad.copy()
            if side is not None:
                a, b = side, (side + 1) % 4
                opp_a, opp_b = (side + 3) % 4, (side + 2) % 4
                shift_a, shift_b = quad[a] - quad[opp_a], quad[b] - quad[opp_b]
                facing = np.array([quad[a], quad[b], quad[b] + shift_b, quad[a] + shift_a])
                q = np.vstack([quad, facing])
                layers.append(("page", facing))
            k = r.uniform(1.02, 1.07)
            hull = cv2.convexHull(q.astype(np.float32)).reshape(-1, 2)
            cc = hull.mean(0)
            layers.insert(0, ("cover", cc + (hull - cc) * k))
        for _ in range(r.randint(1, 3) if kind == "stack" else r.randint(0, 2)):
            t = np.deg2rad(r.uniform(-3, 3))
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            off = np.random.uniform(-0.02, 0.02, 2) * [W, H]
            layers.append(("sheet", (quad - c) @ R.T * r.uniform(0.99, 1.03) + c + off))
        for name, q in layers:
            if name == "cover":
                img = self._fill_quad(img, q, cover)
            elif name == "page" and r.random() < 0.6:
                page, alpha, _ = self._new_page()
                d, a, _ = self._render_page(page, alpha, q, W, H, curl=False)
                img = img * (1 - a[..., None]) + d * a[..., None]
            else:
                shade = r.uniform(0.85, 1.0)
                img = self._fill_quad(img, q, [v * shade for v in paper], 2.0)
        return img

    def _sleeve(self, img, quad, W, H):
        r = self.rng
        c = quad.mean(0)
        sq = c + (quad - c) * r.uniform(1.03, 1.15) + np.random.uniform(-0.02, 0.02, 2) * [W, H]
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, np.round(sq).astype(np.int32), 255, cv2.LINE_AA)
        a = (m.astype(np.float32) / 255.0)[..., None]
        haze = r.uniform(0.03, 0.12)
        img = img * (1 - a * haze) + 255 * a * haze
        streak = np.zeros((H, W), np.float32)
        for _ in range(r.randint(2, 8)):
            p0 = (int(r.uniform(0, W)), int(r.uniform(0, H)))
            ang = r.uniform(0, np.pi)
            L = r.uniform(0.2, 0.8) * max(W, H)
            p1 = (int(p0[0] + L * np.cos(ang)), int(p0[1] + L * np.sin(ang)))
            cv2.line(streak, p0, p1, r.uniform(0.3, 1.0), int(r.uniform(1, 0.02 * max(W, H))))
        streak = cv2.GaussianBlur(streak, (0, 0), r.uniform(1.5, 6)) * a[..., 0]
        img = img + (255 - img) * np.clip(streak, 0, 1)[..., None] * r.uniform(0.3, 0.8)
        edge = np.zeros((H, W), np.float32)
        cv2.polylines(edge, [np.round(sq).astype(np.int32)], True, 1.0, 1, cv2.LINE_AA)
        return img + (255 - img) * edge[..., None] * r.uniform(0.2, 0.6)

    def _light_surface(self, H, W):
        r = self.rng
        base = r.uniform(110, 245)
        tint = np.array([r.uniform(-8, 8), 0, r.uniform(-8, 8)], np.float32)
        img = np.full((H, W, 3), base, np.float32) + tint
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        kind = r.choices(["plain", "wood", "marble", "fabric", "speckle"], [0.3, 0.2, 0.15, 0.15, 0.2])[0]
        if kind == "wood":
            t = np.deg2rad(r.uniform(0, 180))
            u = xx * np.cos(t) + yy * np.sin(t)
            warp = cv2.resize(np.random.randn(4, 4).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            grain = np.sin(u / r.uniform(2, 8) + warp * r.uniform(3, 10))
            grain += 0.5 * np.random.randn(H, W).astype(np.float32)
            grain = cv2.GaussianBlur(grain, (0, 0), 0.8)
            img += (grain * r.uniform(2, 9))[..., None]
        elif kind == "marble":
            n = cv2.resize(np.random.randn(6, 6).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            n2 = cv2.resize(np.random.randn(20, 20).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            veins = np.abs(np.sin((xx + yy) / r.uniform(30, 90) + 4 * n + n2))
            img -= ((veins < 0.08).astype(np.float32) * r.uniform(10, 40))[..., None]
            img = cv2.GaussianBlur(img, (0, 0), 0.7)
        elif kind == "fabric":
            p = r.uniform(2.5, 5)
            weave = np.sin(xx * np.pi / p) * np.sin(yy * np.pi / p)
            img += (weave * r.uniform(2, 7))[..., None]
            img = cv2.GaussianBlur(img, (0, 0), 0.6)
        elif kind == "speckle":
            dots = (np.random.rand(H, W) < r.uniform(0.01, 0.08)).astype(np.float32)
            dots = cv2.GaussianBlur(dots, (0, 0), r.uniform(0.5, 1.2))
            img += (dots * r.uniform(-60, 30))[..., None]
        lo = cv2.resize(np.random.randn(5, 5).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        img *= (1 + r.uniform(0.02, 0.08) * lo)[..., None]
        img += np.random.randn(H, W, 1).astype(np.float32) * r.uniform(1, 6)
        return img


    def _curl_field(self, W, H):
        r = self.rng
        amp = r.uniform(0.0, 0.02) * max(W, H)
        g = 3 if r.random() < 0.5 else 4
        dx = cv2.resize(np.random.randn(g, g).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        dy = cv2.resize(np.random.randn(g, g).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        if r.random() < 0.3:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            bend = r.uniform(0.005, 0.03) * max(W, H)
            if r.random() < 0.5:
                dy += bend * np.sin(np.pi * xx / W + r.uniform(-0.5, 0.5))
            else:
                dx += bend * np.sin(np.pi * yy / H + r.uniform(-0.5, 0.5))
        return dx, dy

    @staticmethod
    def _solve_corner(c, dx, dy, iters=6):
        H, W = dx.shape
        p = np.array(c, np.float64)
        for _ in range(iters):
            x = int(np.clip(round(p[0]), 0, W - 1))
            y = int(np.clip(round(p[1]), 0, H - 1))
            p = np.array(c) - [dx[y, x], dy[y, x]]
        return p

    def _new_page(self):
        return self.pages(long_side=self.rng.choice([640, 800, 960]), bg_pool=self.bg)

    def _render_page(self, page, alpha, quad, W, H, curl):
        ph, pw = page.shape[:2]
        src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        doc = cv2.warpPerspective(page, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpPerspective(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        corners = quad.copy()
        if curl:
            dx, dy = self._curl_field(W, H)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            mx, my = xx + dx, yy + dy
            doc = cv2.remap(doc, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            a = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderValue=0)
            corners = np.array([self._solve_corner(c, dx, dy) for c in quad])
        return doc.astype(np.float32), a.astype(np.float32) / 255.0, corners


    def sample(self, seed):
        self.rng.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        r = self.rng
        modes = list(self.modes)
        mode = r.choices(modes, weights=[self.modes[m] for m in modes])[0]
        aw, ah = r.choice(PHOTO_ASPECTS)
        if aw >= ah:
            W, H = self.canvas, int(self.canvas * ah / aw)
        else:
            W, H = int(self.canvas * aw / ah), self.canvas
        p_light = self.near_fill_light if mode == "near_fill" else self.light_surface
        background = "light" if r.random() < p_light else "photo"
        if background == "light":
            img = self._light_surface(H, W)
        else:
            img = self.bg.random_crop(H, W).astype(np.float32)
        corners = np.zeros((4, 2))
        score = 0.0
        kind = ""
        light_page = False

        if mode != "negative":
            page, alpha, kind = self._new_page()
            light_page = bool(page.mean() > 200)
            quad = self._place(mode, W, H, page.shape[:2])
            if quad is None:
                mode = "negative"
        if mode != "negative":
            if mode != "near_fill" and r.random() < 0.1:
                p2, al2, _ = self._new_page()
                q2 = self._place("normal", W, H, p2.shape[:2])
                if q2 is not None:
                    d2, a2, _ = self._render_page(p2, al2, q2, W, H, curl=False)
                    img = img * (1 - a2[..., None]) + d2 * a2[..., None]
            if mode != "near_fill" and r.random() < 0.25:
                img = self._underlays(img, quad, W, H)
            doc, a, corners = self._render_page(page, alpha, quad, W, H, curl=r.random() < 0.5)

            lum = cv2.GaussianBlur(img.mean(axis=2), (0, 0), max(W, H) / 8)
            lum = (lum / max(lum.mean(), 1.0)) ** r.uniform(0.2, 0.8)
            cast = (img.reshape(-1, 3).mean(0) / max(img.mean(), 1.0)) ** r.uniform(0.0, 0.4)
            doc = doc * lum[..., None] * cast

            if r.random() < (self.light_contact_shadow if background == "light" else 0.7):
                sh = cv2.GaussianBlur(a, (0, 0), r.uniform(1.0, 6.0))
                M = np.float32([[1, 0, r.uniform(-4, 4)], [0, 1, r.uniform(-1, 6)]])
                sh = cv2.warpAffine(sh, M, (W, H))
                img = img * (1 - r.uniform(0.15, 0.5) * sh[..., None])
            img = img * (1 - a[..., None]) + doc * a[..., None]
            if r.random() < 0.1:
                img = self._sleeve(img, corners, W, H)
            if r.random() < 0.2:
                for _ in range(r.randint(1, 2)):
                    i = r.randint(0, 3)
                    t = r.random()
                    anchor = corners[i] * (1 - t) + corners[(i + 1) % 4] * t
                    img = effects.finger(img, r, anchor)
            corners = order_corners(corners / [W, H])
            score = 1.0

        if r.random() < 0.35:
            img = effects.cast_shadow(img, r)
        if r.random() < 0.4:
            img = effects.illumination_gradient(img, r)
        if r.random() < 0.12:
            img = effects.specular(img, r)
        if r.random() < 0.06:
            img = effects.pen(img, r)


        img = cv2.resize(np.clip(img, 0, 255), (self.out, self.out), interpolation=cv2.INTER_AREA)
        img = camera.degrade(img, r)
        return dict(image=img, corners=corners.astype(np.float32), score=score, mode=mode, kind=kind,
                    background=background, light_page=light_page)
