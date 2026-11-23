                  
import csv
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import assets
from .data import DeterministicSampler, SynthDataset, generate_split
from .evaluate import evaluate_synth
from .model import build_model, count_params
from .repro import configure_torch, derive_seed, seed_everything


def gaussian_targets(coords, centers, sigma_bins, n_inside):
    sigma = sigma_bins / n_inside
    d = coords.unsqueeze(-1) - centers.view(1, 1, -1)
    t = torch.exp(-0.5 * (d / sigma) ** 2)
    return t / t.sum(-1, keepdim=True).clamp_min(1e-12)


def corner_loss(out, corners, score, centers, cfg):
    n_inside = cfg["model"]["bins_inside"]
    lc = cfg["loss"]
    pos = score > 0.5
    bce = F.binary_cross_entropy_with_logits(out["score_logit"].squeeze(1), score)
    if pos.any():
        c = corners[pos].view(-1, 4, 2)
        tx = gaussian_targets(c[..., 0], centers, lc["sigma_bins"], n_inside)
        ty = gaussian_targets(c[..., 1], centers, lc["sigma_bins"], n_inside)
        kl = F.kl_div(out["logits_x"][pos].log_softmax(-1), tx, reduction="batchmean") / 4 \
            + F.kl_div(out["logits_y"][pos].log_softmax(-1), ty, reduction="batchmean") / 4
        l1 = (out["coords"][pos] - corners[pos]).abs().mean()
    else:
        kl = l1 = out["coords"].sum() * 0
    total = lc["w_kl"] * kl + lc["w_l1"] * l1 + lc["w_score"] * bce
    return total, dict(kl=kl.item(), l1=l1.item(), bce=bce.item())


def select_device(cfg):
    requested = cfg["train"].get("device", "cuda")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def training_plan(cfg):
    data, settings = cfg["data"], cfg["train"]
    full = data["mode"] == "full"
    size = data["train_images"] if full else data["round_images"]
    epochs = settings["epochs"] if full else settings["epochs_per_round"]
    plan = []
    for index, start in enumerate(range(0, data["train_images"], size)):
        count = min(size, data["train_images"] - start)
        for epoch in range(epochs):
            tag = f"epoch{epoch:03d}" if full else f"round{index:03d}-epoch{epoch:02d}"
            plan.append((index, start, count, tag))
    return plan


def describe(cfg):
    plan = training_plan(cfg)
    steps = sum(count // cfg["train"]["batch"] for _, _, count, _ in plan)
    disk = f"{plan[0][2]} images" + (" at a time" if cfg["data"]["mode"] == "rotating" else "")
    return (f"{cfg['data']['mode']}: {cfg['data']['train_images']} distinct training images, "
            f"{len(plan)} passes, {steps} steps, ~{math.ceil(steps * cfg['train']['batch'] / 1000)}k images seen, "
            f"on disk: {disk}")


def set_backbone_frozen(model, frozen):
    model.backbone.requires_grad_(not frozen)
    model.backbone.train(not frozen)


class Trainer:
    def __init__(self, cfg, run_dir, total_steps, pretrained_path, val_dir, log=print):
        self.cfg, self.run_dir, self.total_steps, self.log = cfg, run_dir, total_steps, log
        seed_everything(derive_seed(cfg["seed"], "model-init"))
        self.device = select_device(cfg)
        self.model = build_model(cfg, pretrained_path).to(device=self.device, memory_format=torch.channels_last)
        bb = list(self.model.backbone.parameters())
        bb_ids = {id(p) for p in bb}
        rest = [p for p in self.model.parameters() if id(p) not in bb_ids]
        t = cfg["train"]
        self.opt = torch.optim.AdamW([{"params": rest, "lr_mult": 1.0},
                                      {"params": bb, "lr_mult": t["backbone_lr_mult"]}],
                                     lr=t["lr"], weight_decay=t["weight_decay"])
        self.centers = self.model.simcc.centers
        self.step, self.passes_done, self.best = 0, 0, float("inf")
        self.log_path = os.path.join(run_dir, "log.csv")
        val_ds = SynthDataset(val_dir, cfg["data"]["norm"], augment=False)
        self.val_dl = DataLoader(val_ds, 64,
                                 sampler=DeterministicSampler(len(val_ds), 0, "val", shuffle=False),
                                 num_workers=cfg["data"]["loader_workers"], pin_memory=self.device.type == "cuda")
        log(f"model {count_params(self.model) / 1e3:.0f}K params | total steps {total_steps} | device {self.device}")


    def save(self, name):
        torch.save(dict(model=self.model.state_dict(), opt=self.opt.state_dict(), step=self.step,
                        passes_done=self.passes_done, best=self.best, cfg=self.cfg),
                   os.path.join(self.run_dir, name))

    def try_resume(self):
        p = os.path.join(self.run_dir, "Aster.resume.pt")
        if not os.path.exists(p):
            return 0
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        for state in self.opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        self.step, self.passes_done, self.best = ck["step"], ck["passes_done"], ck["best"]
        self.log(f"resumed after pass {self.passes_done} (step {self.step})")
        return self.passes_done


    def _lr(self):
        t = self.cfg["train"]
        warm = max(1, int(t["warmup_fraction"] * self.total_steps))
        if self.step < warm:
            return t["lr"] * (self.step + 1) / warm
        p = (self.step - warm) / max(1, self.total_steps - warm)
        return t["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    def train_pass(self, data_dir, tag, data_label):
        t, d = self.cfg["train"], self.cfg["data"]
        ds = SynthDataset(data_dir, d["norm"], augment=True)
        dl = DataLoader(ds, t["batch"], sampler=DeterministicSampler(len(ds), self.cfg["seed"], tag),
                        num_workers=d["loader_workers"], drop_last=True, pin_memory=self.device.type == "cuda")
        t0, agg, n = time.time(), {}, 0
        frozen_any = False
        for i, b in enumerate(dl):
            frozen = self.step < t["freeze_backbone_fraction"] * self.total_steps
            frozen_any |= frozen
            self.model.train()
            set_backbone_frozen(self.model, frozen)
            lr = self._lr()
            for g in self.opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            images = b["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)
            corners = b["corners"].to(self.device, non_blocking=True)
            score = b["score"].to(self.device, non_blocking=True)
            out = self.model(images)
            loss, parts = corner_loss(out, corners, score, self.centers, self.cfg)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            self.step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            n += 1
            if (i + 1) % t["log_every"] == 0:
                ips = (i + 1) * t["batch"] / (time.time() - t0)
                self.log(f"{tag} it {i + 1}/{len(dl)} step {self.step} lr {lr:.2e} "
                         + " ".join(f"{k} {v / n:.4f}" for k, v in agg.items()) + f" | {ips:.1f} img/s"
                         + (" [frozen]" if frozen else ""))
        self.passes_done += 1
        res = evaluate_synth(self.model, self.val_dl, self.cfg["eval"])
        a, i_, o = res["all"], res["in_frame"], res["out_of_frame"]
        row = dict(pass_index=self.passes_done, tag=tag, data=data_label, step=self.step,
                   sec=round(time.time() - t0), frozen=frozen_any,
                   **{k: round(v / max(n, 1), 5) for k, v in agg.items()},
                   usable=a["usable"], severe=a["severe"], mean_missing=a["mean_missing"], mean_extra=a["mean_extra"],
                   in_usable=i_.get("usable"), in_severe=i_.get("severe"),
                   out_usable=o.get("usable"), out_severe=o.get("severe"),
                   median_px=a["median_px"], score_pos=res["score_acc_pos"], score_neg=res["score_acc_neg"])
        self.log("VAL " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        key = -(a["usable"] - a["severe"])
        if key < self.best:
            self.best = key
            self.save("Aster.pt")
        self.save("Aster.resume.pt")
        return row


def train(cfg, resume=True):
    name = cfg["name"]
    run_dir = os.path.join(cfg["runs_dir"], name)
    data_dir = os.path.join(cfg["data"]["dir"], name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train.log"), "a", buffering=1) as logfile:
        def log(message):
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            print(line, flush=True)
            logfile.write(line + "\n")
        t = cfg["train"]
        configure_torch(t["threads"])

        log(f"run {name}: mode={cfg['data']['mode']} train_images={cfg['data']['train_images']} seed={cfg['seed']}")


        a = cfg["assets"]
        assets.prepare_fonts(os.path.join(a["dir"], "fonts"))
        assets.prepare_backgrounds(os.path.join(a["dir"], "backgrounds"), a["backgrounds"], log=log)
        pretrained = assets.prepare_pretrained(os.path.join(a["dir"], "pretrained")) if cfg["model"]["pretrained"] else None


        workers = cfg["data"]["generate_workers"]
        val_dir = os.path.join(data_dir, "val")
        restarting = not resume or not os.path.exists(os.path.join(run_dir, "Aster.resume.pt"))
        if restarting and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if not os.path.exists(os.path.join(val_dir, "labels.jsonl")):
            log("rendering validation split")
            generate_split(cfg, "val", 0, cfg["data"]["val_images"], val_dir, workers, log)

        plan = training_plan(cfg)
        total_steps = sum(count // t["batch"] for _, _, count, _ in plan)
        trainer = Trainer(cfg, run_dir, total_steps, pretrained, val_dir, log)
        done = trainer.try_resume() if resume else 0

        for index, (round_index, start, count, tag) in enumerate(plan):
            if index < done:
                continue
            folder = "train" if cfg["data"]["mode"] == "full" else f"round{round_index:03d}"
            train_dir = os.path.join(data_dir, folder)
            if not os.path.exists(os.path.join(train_dir, "labels.jsonl")):
                generate_split(cfg, "train", start, count, train_dir, workers, log)
            trainer.train_pass(train_dir, tag, folder)
            last_in_round = index + 1 == len(plan) or plan[index + 1][0] != round_index
            if cfg["data"]["mode"] == "rotating" and last_in_round and not cfg["data"]["keep_rounds"]:
                shutil.rmtree(train_dir)
                log(f"deleted {train_dir}")

        log(f"done: {run_dir}")
        return run_dir


def _benchmark_speed(model, cfg, frozen, iters):
    device = next(model.parameters()).device
    b = cfg["train"]["batch"]
    model.train()
    set_backbone_frozen(model, frozen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 1e-4)
    x = torch.randn(b, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    c, s = torch.rand(b, 8, device=device), torch.ones(b, device=device)
    for i in range(iters + 2):
        if i == 2:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        loss, _ = corner_loss(model(x), c, s, model.simcc.centers, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return b * iters / (time.time() - t0)


def benchmark(cfg, iterations=8):
    configure_torch(cfg["train"]["threads"])
    device = select_device(cfg)
    model = build_model(cfg).to(device=device, memory_format=torch.channels_last)
    fz, full = _benchmark_speed(model, cfg, True, iterations), _benchmark_speed(model, cfg, False, iterations)
    t = cfg["train"]
    seen = sum(count for _, _, count, _ in training_plan(cfg))
    f = t["freeze_backbone_fraction"]
    hours = (seen * f / fz + seen * (1 - f) / full) / 3600
    print(f"device {device} | threads {t['threads']} | frozen {fz:.1f} img/s | full {full:.1f} img/s")
    print(describe(cfg))
    print(f"estimated training time ~ {hours:.1f} h (+ rendering {cfg['data']['train_images']} images)")
