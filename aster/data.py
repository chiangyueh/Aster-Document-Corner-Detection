                    
import json
import os
import random
import shutil
import time
from multiprocessing import get_context

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import assets
from .geometry import order_corners
from .repro import derive_seed


MEAN_STD = {
    "keras": (np.array([127.5] * 3, np.float32), np.array([127.5] * 3, np.float32)),
}


def to_tensor(img_bgr, norm="keras"):
    mean, std = MEAN_STD[norm]
    x = (img_bgr[..., ::-1].astype(np.float32) - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class DeterministicSampler(Sampler):
    def __init__(self, n, seed, tag, shuffle=True):
        self.n, self.seed, self.tag, self.shuffle = n, seed, tag, shuffle

    def __len__(self):
        return self.n

    def __iter__(self):
        order = np.arange(self.n)
        if self.shuffle:
            order = np.random.default_rng(derive_seed(self.seed, "order", self.tag)).permutation(self.n)
        for pos, i in enumerate(order):
            yield int(i), derive_seed(self.seed, "augment", self.tag, pos)


class SynthDataset(Dataset):
    def __init__(self, root, norm="keras", augment=True):
        self.root = root
        with open(os.path.join(root, "labels.jsonl")) as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        self.norm = norm
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        i, aug_seed = key if isinstance(key, tuple) else (key, None)
        it = self.items[i]
        img = cv2.imread(os.path.join(self.root, "images", it["file"]), cv2.IMREAD_COLOR)
        corners = np.array(it["corners"], np.float32).reshape(4, 2)
        score = float(it["score"])
        if self.augment and aug_seed is not None:
            img, corners = self._augment(img, corners, score > 0, random.Random(aug_seed))
        return dict(image=to_tensor(img, self.norm), corners=torch.from_numpy(corners.reshape(-1)),
                    score=torch.tensor(score, dtype=torch.float32))

    @staticmethod
    def _augment(img, c, has_doc, r):
        if r.random() < 0.5:
            img = img[:, ::-1]
            c = c.copy()
            c[:, 0] = 1.0 - c[:, 0]
        if r.random() < 0.1:
            img = np.rot90(img, k=-1)
            c = np.stack([1.0 - c[:, 1], c[:, 0]], 1)
        if has_doc:
            c = order_corners(c).astype(np.float32)
        img = np.ascontiguousarray(img)
        if r.random() < 0.5:
            gain, bias = r.uniform(0.85, 1.15), r.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return img, c


JPEG_QUALITY = 95


_synth = None


_ctx = None


def build_synth(cfg):
    from .synth.scene import SceneSynth
    cv2.setNumThreads(1)
    a = cfg["assets"]
    bgs = assets.background_list(os.path.join(a["dir"], "backgrounds"), a["backgrounds"])
    fonts = assets.font_paths(os.path.join(a["dir"], "fonts"))
    return SceneSynth(bgs, fonts, cfg["synth"])


def render(synth, cfg, split, i):
    s = synth.sample(derive_seed(cfg["seed"], "synth", split, i))
    _, buf = cv2.imencode(".jpg", s["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    rec = dict(file=f"{i:08d}.jpg", index=i, corners=[round(float(v), 6) for v in s["corners"].reshape(-1)],
               score=s["score"], mode=s["mode"], kind=s["kind"],
               background=s["background"], light_page=s["light_page"])
    return buf.tobytes(), rec


def _init(cfg):
    global _synth, _ctx
    _synth = build_synth(cfg)
    _ctx = cfg


def _make(job):
    split, i, out_dir = job
    data, rec = render(_synth, _ctx, split, i)
    with open(os.path.join(out_dir, "images", rec["file"]), "wb") as f:
        f.write(data)
    return rec


def generate_split(cfg, split, start, count, out_dir, workers=0, log=print):
    workers = workers or os.cpu_count()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "images"))
    t0 = time.time()
    jobs = [(split, i, out_dir) for i in range(start, start + count)]
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool, \
            open(os.path.join(out_dir, "labels.tmp"), "w") as f:
        for k, rec in enumerate(pool.imap(_make, jobs, chunksize=8), 1):
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            if k % 1000 == 0 or k == count:
                log(f"  {split}: {k}/{count}  {k / (time.time() - t0):.1f} img/s")
    os.replace(os.path.join(out_dir, "labels.tmp"), os.path.join(out_dir, "labels.jsonl"))


COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)]


def draw_sample(sample, pad=40):
    img = sample["image"]
    s = img.shape[0]
    canvas = np.full((s + 2 * pad, s + 2 * pad, 3), 60, np.uint8)
    canvas[pad:pad + s, pad:pad + s] = img
    if sample["score"] > 0:
        pts = sample["corners"] * s + pad
        cv2.polylines(canvas, [pts.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        for p, c in zip(pts, COLORS):
            cv2.circle(canvas, tuple(np.round(p).astype(int)), 4, c, -1, cv2.LINE_AA)
    cv2.putText(canvas, sample["mode"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def preview(cfg, count=48, cols=8, out="preview.jpg", split="train", start=0):
    synth = build_synth(cfg)
    tiles = [draw_sample(synth.sample(derive_seed(cfg["seed"], "synth", split, i))) for i in range(start, start + count)]
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - cols + 1, cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out)
