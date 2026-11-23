                    
import cv2
import numpy as np


def _soft_polygon(h, w, rng, scale):
    n = rng.randint(3, 7)
    cx, cy = rng.uniform(-0.2, 1.2) * w, rng.uniform(-0.2, 1.2) * h
    r = rng.uniform(0.2, 0.8) * max(h, w)
    ang = np.sort(np.random.uniform(0, 2 * np.pi, n))
    rad = r * np.random.uniform(0.5, 1.0, n)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [pts], 1.0)
    k = max(1, int(max(h, w) * scale)) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def cast_shadow(img, rng):
    h, w = img.shape[:2]
    m = _soft_polygon(h, w, rng, rng.uniform(0.02, 0.15))
    k = rng.uniform(0.2, 0.6)
    tint = np.array([1.0, 1.0, 1.0]) - np.array([rng.uniform(0, 0.08), 0, rng.uniform(-0.04, 0.04)])
    return img * (1 - k * m[..., None] * tint)


def specular(img, rng):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
    ax = (int(rng.uniform(0.03, 0.2) * w), int(rng.uniform(0.03, 0.2) * h))
    cv2.ellipse(m, c, ax, rng.uniform(0, 180), 0, 360, 1.0, -1)
    k = max(3, int(max(ax) * 0.8)) | 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    return img + (255 - img) * rng.uniform(0.3, 0.9) * m[..., None]


def illumination_gradient(img, rng):
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
    return img * (1 + rng.uniform(0.1, 0.5) * g)[..., None]


def finger(img, rng, anchor):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    length, width = rng.uniform(0.12, 0.3) * min(h, w), rng.uniform(0.04, 0.07) * min(h, w)
    ang = rng.uniform(0, 360)
    c = (int(anchor[0]), int(anchor[1]))
    cv2.ellipse(m, c, (int(length), int(width)), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    skin = np.array([rng.uniform(80, 170), rng.uniform(110, 190), rng.uniform(150, 235)], np.float32)
    shade = cv2.GaussianBlur(m, (0, 0), width * 0.5)
    col = skin[None, None] * (0.75 + 0.25 * shade[..., None])
    return img * (1 - m[..., None]) + col * m[..., None]


def pen(img, rng):
    h, w = img.shape[:2]
    p0 = np.array([rng.uniform(0, w), rng.uniform(0, h)])
    ang = rng.uniform(0, np.pi)
    L = rng.uniform(0.2, 0.5) * max(h, w)
    p1 = p0 + L * np.array([np.cos(ang), np.sin(ang)])
    col = [float(c) for c in np.random.randint(0, 255, 3)]
    out = img.copy()
    cv2.line(out, tuple(p0.astype(int)), tuple(p1.astype(int)), col, max(2, int(0.012 * max(h, w))), cv2.LINE_AA)
    return out
