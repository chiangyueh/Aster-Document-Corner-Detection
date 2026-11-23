                
import cv2
import numpy as np

ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class BackgroundPool:
    def __init__(self, items, rng):
        self.items = list(items)
        self.rng = rng

    def __len__(self):
        return len(self.items)

    def _read(self):
        path, rot = self.items[self.rng.randrange(len(self.items))]
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if rot % 360 in ROTATE:
            img = cv2.rotate(img, ROTATE[rot % 360])
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return img

    def random_crop(self, h, w):
        img = self._read()
        ih, iw = img.shape[:2]
        scale = self.rng.uniform(0.35, 1.0)
        ch = min(ih, int(round(min(ih, iw * h / w) * scale)))
        cw = min(iw, max(1, int(round(ch * w / h))))
        y = self.rng.randint(0, ih - ch)
        x = self.rng.randint(0, iw - cw)
        crop = img[y:y + ch, x:x + cw]
        if self.rng.random() < 0.5:
            crop = crop[:, ::-1]
        interp = cv2.INTER_AREA if ch > h else cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(np.ascontiguousarray(crop), (w, h), interpolation=interp))
