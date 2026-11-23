                   
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .data import MEAN_STD, to_tensor
from .geometry import order_corners
from .model import EXPORT_OUTPUTS, load_model


ROTATIONS = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]


def pillow_resize(rgb, size=224):
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))


def _unrotate(c, k):
    x, y = c[:, 0], c[:, 1]
    if k == 0:
        return np.stack([x, y], 1)
    if k == 1:
        return np.stack([1 - y, x], 1)
    if k == 2:
        return np.stack([1 - x, 1 - y], 1)
    return np.stack([y, 1 - x], 1)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_views(predict, bgr, views=4):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    best, fallback = None, None
    for k in range(views):
        view = rgb if k == 0 else cv2.rotate(rgb, ROTATIONS[k])
        out = predict(pillow_resize(view))
        q = _unrotate(np.asarray(out["coords"], np.float64).reshape(4, 2), k)
        var = 0.0
        for lg in (out["lx"], out["ly"]):
            p = _softmax(np.asarray(lg, np.float64))
            mean = (p * out["centers"]).sum(-1, keepdims=True)
            var += (p * (out["centers"] - mean) ** 2).sum()
        var /= 8.0
        if fallback is None:
            fallback = (q, out["score"], var, k * 90)
        quad = order_corners(q).astype(np.float32)
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) <= 0.001:
            continue
        if best is None or var < best[2]:
            best = (q, out["score"], var, k * 90)
    if best is None:
        return order_corners(fallback[0]), fallback[1], dict(variance=fallback[2], rotation_ccw=0, invalid=True)
    return order_corners(best[0]), best[1], dict(variance=best[2], rotation_ccw=best[3], invalid=False)


class TorchRunner:
    def __init__(self, ckpt):
        self.model, cfg = load_model(ckpt)
        self.norm = cfg["data"]["norm"]
        self.centers = self.model.simcc.centers.numpy().astype(np.float64)

    @torch.no_grad()
    def __call__(self, rgb224):
        out = self.model(to_tensor(np.ascontiguousarray(rgb224[..., ::-1]), self.norm)[None])
        return dict(coords=out["coords"][0].numpy().reshape(4, 2), score=float(out["score_logit"].sigmoid()),
                    lx=out["logits_x"][0].numpy(), ly=out["logits_y"][0].numpy(), centers=self.centers)


class MNNRunner:
    def __init__(self, path):
        import MNN
        import MNN.expr as F
        self.F = F
        self.net = MNN.nn.load_module_from_file(
            path, ["image"], EXPORT_OUTPUTS)
        self.bins = 448
        self.centers = (np.arange(self.bins, dtype=np.float64) - 112 + 0.5) / 224.0

    def __call__(self, rgb224):
        mean, std = MEAN_STD["keras"]
        x = (rgb224.astype(np.float32) - mean) / std
        v = self.F.placeholder([1, 3, 224, 224], self.F.NCHW)
        v.write(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        c, s, lx, ly = self.net.forward([v])
        logit = float(np.array(s.read()).ravel()[0])
        return dict(coords=np.array(c.read(), np.float64).reshape(4, 2), score=1 / (1 + np.exp(-logit)),
                    lx=np.array(lx.read(), np.float64).reshape(4, self.bins),
                    ly=np.array(ly.read(), np.float64).reshape(4, self.bins), centers=self.centers)


def make_runner(path):
    runner = {".pt": TorchRunner, ".mnn": MNNRunner}[Path(path).suffix.lower()]
    return runner(str(path))
