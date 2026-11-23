                   
import json
import os

import cv2
import numpy as np
import torch

from .geometry import corners_inside, order_corners
from .predict import make_runner, run_views


DEFAULT_THRESHOLDS = dict(max_missing=0.01, max_extra=0.03, severe_missing=0.05, severe_extra=0.25)


UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def _clip(poly, rect):
    area, pts = cv2.intersectConvexConvex(np.asarray(poly, np.float32), np.asarray(rect, np.float32))
    return (pts.reshape(-1, 2) if pts is not None and area > 0 else np.zeros((0, 2), np.float32)), float(area)


def _project(H, pts):
    if len(pts) == 0:
        return pts
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    if np.any(p[:, 2] <= 1e-9):
        return None
    return (p[:, :2] / p[:, 2:]).astype(np.float32)


def _area(poly):
    return float(abs(cv2.contourArea(np.asarray(poly, np.float32)))) if len(poly) >= 3 else 0.0


def page_completeness(pred, gt, w, h):
    P = np.asarray(pred, np.float64).reshape(4, 2) * [w, h]
    G = np.asarray(gt, np.float64).reshape(4, 2) * [w, h]
    R = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(G.astype(np.float32), UNIT).astype(np.float64)

    gt_vis, _ = _clip(G, R)
    vis = _project(H, gt_vis)
    a_vis = _area(vis) if vis is not None else 0.0
    if a_vis <= 1e-9:
        return 1.0, float("inf")
    if _area(P.astype(np.float32)) <= 0 or not cv2.isContourConvex(P.astype(np.float32)):
        return 1.0, float("inf")
    pred_vis, _ = _clip(P, R)
    pp = _project(H, pred_vis)
    if pp is None:
        return 1.0, float("inf")
    if len(pp) < 3:
        return 1.0, 0.0
    covered = cv2.intersectConvexConvex(pp, vis)[0]
    on_page = cv2.intersectConvexConvex(pp, UNIT)[0]
    missing = max(0.0, 1.0 - covered / a_vis)
    extra = max(0.0, (_area(pp) - on_page) / a_vis)
    return float(missing), float(extra)


def quad_iou(a, b):
    a = np.asarray(a, np.float32).reshape(4, 2)
    b = np.asarray(b, np.float32).reshape(4, 2)
    inter = cv2.intersectConvexConvex(a, b)[0]
    ua = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return float(inter / ua) if ua > 0 else 0.0


def corner_errors(pred, gt, w, h):
    d = (np.asarray(pred).reshape(4, 2) - np.asarray(gt).reshape(4, 2)) * [w, h]
    return np.linalg.norm(d, axis=1)


def score_prediction(pred, gt, w, h, thresholds=None):
    t = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    missing, extra = page_completeness(pred, gt, w, h)
    return dict(missing=missing, extra=extra,
                usable=missing <= t["max_missing"] and extra <= t["max_extra"],
                severe=missing > t["severe_missing"] or extra > t["severe_extra"],
                err_px=corner_errors(pred, gt, w, h),
                iou=quad_iou(np.asarray(pred).reshape(4, 2) * [w, h], np.asarray(gt).reshape(4, 2) * [w, h]))


def summarize(rows):
    if not rows:
        return {}
    miss = np.array([r["missing"] for r in rows])
    extra = np.array([r["extra"] for r in rows])
    finite = np.isfinite(extra)
    return {
        "n": len(rows),
        "usable": float(np.mean([r["usable"] for r in rows])),
        "severe": float(np.mean([r["severe"] for r in rows])),
        "median_missing": float(np.median(miss)),
        "mean_missing": float(miss.mean()),
        "median_extra": float(np.median(np.where(finite, extra, 1.0))),
        "mean_extra": float(np.where(finite, extra, 1.0).mean()),
        "median_px": float(np.median(np.concatenate([r["err_px"] for r in rows]))),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
    }


@torch.no_grad()
def evaluate_synth(model, loader, thresholds=None, size=224):
    model.eval()
    device = next(model.parameters()).device
    groups = {"all": [], "in_frame": [], "out_of_frame": []}
    neg_correct = neg_total = pos_correct = pos_total = 0
    for b in loader:
        out = model(b["image"].to(device, non_blocking=True))
        prob = out["score_logit"].sigmoid().squeeze(1).cpu()
        for i in range(len(prob)):
            if b["score"][i] < 0.5:
                neg_total += 1
                neg_correct += int(prob[i] < 0.5)
                continue
            pos_total += 1
            pos_correct += int(prob[i] >= 0.5)
            p = out["coords"][i].cpu().numpy().reshape(4, 2)
            g = b["corners"][i].numpy().reshape(4, 2)
            row = score_prediction(p, g, size, size, thresholds)
            groups["all"].append(row)
            groups["in_frame" if corners_inside(g).all() else "out_of_frame"].append(row)
    res = {k: summarize(v) for k, v in groups.items()}
    res["score_acc_pos"] = pos_correct / max(pos_total, 1)
    res["score_acc_neg"] = neg_correct / max(neg_total, 1)
    return res


def load_items(root):
    if os.path.exists(os.path.join(root, "labels.json")):
        with open(os.path.join(root, "labels.json")) as f:
            return json.load(f), "real"
    with open(os.path.join(root, "labels.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()], "synth"


def evaluate_items(run, root, views=4, thresholds=None, limit=None, on_image=None):
    items, kind = load_items(root)
    records = []
    for it in items[:limit] if limit else items:
        img = cv2.imread(os.path.join(root, "images", it["file"]), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        pred, prob, info = run_views(run, img, views)
        if kind == "synth":
            gt = np.array(it["corners"]).reshape(4, 2) if it["score"] > 0 else None
            tags = [f"scene:{it['mode']}"] + ([f"doc:{it['kind']}"] if it.get("kind") else [])
            if "background" in it:
                tags.append(f"bg:{it['background']}" + (":light_page" if it.get("light_page") else ""))
        else:
            gt = order_corners(np.array(it["corners"], np.float64) / [w, h]) if it.get("corners") else None
            tags = list(it.get("tags", []))
        rec = dict(file=it["file"], tags=tags, prob=float(prob), pred=np.asarray(pred).tolist(), document=gt is not None)
        if gt is not None:
            rec.update(score_prediction(pred, gt, w, h, thresholds))
            rec["frame"] = "in_frame" if corners_inside(gt).all() else "out_of_frame"
            rec["detected"] = prob >= 0.5
        if on_image:
            on_image(img, rec, gt)
        records.append(rec)
    return records


def draw(img, rec, gt):
    h, w = img.shape[:2]
    vis = img.copy()
    t = max(2, int(max(w, h) / 400))
    if gt is not None:
        cv2.polylines(vis, [(gt * [w, h]).astype(np.int32)], True, (0, 220, 0), t, cv2.LINE_AA)
    cv2.polylines(vis, [(np.array(rec["pred"]) * [w, h]).astype(np.int32)], True, (0, 0, 255), t, cv2.LINE_AA)
    if rec["document"]:
        txt = f"missing {rec['missing']:.1%}  extra {min(rec['extra'], 9.99):.1%}  " + \
              ("usable" if rec["usable"] else "SEVERE" if rec["severe"] else "not usable")
        cv2.putText(vis, txt, (10, 10 + 30 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (0, 0, 255), t, cv2.LINE_AA)
    return vis


def report(records):
    groups = {}
    for r in records:
        if r["document"]:
            for g in {"all", r["frame"], *r["tags"]}:
                groups.setdefault(g, []).append(r)
    out = {g: summarize(rs) | {"detect_rate": float(np.mean([x["detected"] for x in rs]))}
           for g, rs in sorted(groups.items())}
    neg = [r for r in records if not r["document"]]
    if neg:
        out["negatives"] = {"n": len(neg), "reject_rate": float(np.mean([r["prob"] < 0.5 for r in neg]))}
    return out


def evaluate(model_path, data_dir, thresholds=None, views=4, limit=None, vis=None, out=None):
    if vis:
        os.makedirs(vis, exist_ok=True)
    def save_vis(img, rec, gt):
        if vis:
            cv2.imwrite(os.path.join(vis, os.path.splitext(rec["file"])[0] + ".jpg"), draw(img, rec, gt))
    records = evaluate_items(make_runner(model_path), data_dir, views, thresholds, limit, save_vis)
    results = report(records)
    for group, metrics in results.items():
        print(f"{group:14s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    return results
