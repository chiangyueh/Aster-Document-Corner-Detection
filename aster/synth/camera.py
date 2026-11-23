                  
import cv2
import numpy as np


def degrade(img, rng):
    if rng.random() < 0.5:
        s = rng.uniform(0.3, 1.3)
        img = cv2.GaussianBlur(img, (0, 0), s)
    if rng.random() < 0.15:
        n = rng.randint(3, 7)
        k = np.zeros((n, n), np.float32)
        k[n // 2] = 1.0 / n
        rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1.0)
        k = cv2.warpAffine(k, rot, (n, n))
        img = cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))

    img = img * rng.uniform(0.7, 1.25) + rng.uniform(-20, 20)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.75, 1.25) + mean
    img = img * np.array([rng.uniform(0.85, 1.15), 1.0, rng.uniform(0.85, 1.15)], np.float32)
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.6, 1.3)
    img = np.clip(img, 0, 255)
    g = rng.uniform(0.8, 1.25)
    img = 255.0 * (img / 255.0) ** g
    if rng.random() < 0.7:
        sigma = rng.uniform(1, 7)
        lum = np.random.randn(*img.shape[:2], 1).astype(np.float32) * sigma
        chroma = np.random.randn(*img.shape).astype(np.float32) * sigma * 0.4
        img = img + lum + chroma
    if rng.random() < 0.3:
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = img + rng.uniform(0.3, 1.0) * (img - blur)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.85:
        q = int(rng.uniform(35, 95))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img
