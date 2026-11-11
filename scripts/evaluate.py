"""Evaluate a trained YOLO11-s checkpoint on val and test.

The confidence threshold for P/R/F1 is chosen on val (max char-F1) and then frozen for test.

  python3 scripts/evaluate.py --name baseline
  python3 scripts/evaluate.py --name baseline --scale-norm      # improvement: cell-size normalised inference
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import load_image
from inference import read_page
from metrics import best_conf, centre_metrics, map_metrics, predict, reading_metrics
from yolo11 import build_yolo11s

NC = 63
TARGET_CELL_W = 22.0  # median cell width of the training pages at base width 1024


def dots(c):
    """class id -> dot string, e.g. 0 -> '1', 28 -> '1345'."""
    v = c + 1
    return "".join(str(i + 1) for i in range(6) if v >> i & 1)


def predict_scale_norm(model, img, conf=0.001, ref_conf=0.5, lo=0.6, hi=2.0):
    """Two-pass inference: measure the cell size on pass 1, rescale the page so cells are
    ~22 px wide (the training cell size), re-detect, map boxes back."""
    pb, ps, pc = predict(model, img, conf=conf)
    sel = ps > ref_conf
    if sel.sum() < 20:
        return pb, ps, pc, 1.0
    w_med = np.median(pb[sel, 2] - pb[sel, 0])
    f = float(np.clip(TARGET_CELL_W / max(w_med, 1e-3), lo, hi))
    if abs(f - 1) < 0.08:
        return pb, ps, pc, 1.0
    h, w = img.shape[:2]
    big = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    pb2, ps2, pc2 = predict(model, big, conf=conf)
    return pb2 / f, ps2, pc2, f


def infer(model, items, scale_norm, auto=False):
    out, times, factors = [], [], []
    for it in items:
        img, gb, gl = load_image(it)
        t0 = time.perf_counter()
        if auto:
            pb, ps, pc, info = read_page(model, img, conf=0.001)
            factors.append((info["scale"], info["flipped"]))
        elif scale_norm:
            pb, ps, pc, f = predict_scale_norm(model, img)
            factors.append(f)
        else:
            pb, ps, pc = predict(model, img)
        times.append(time.perf_counter() - t0)
        out.append(dict(item=it, pb=pb, ps=ps, pc=pc, gb=gb, gl=gl, img=img))
    return out, float(np.median(times[1:] if len(times) > 1 else times)), factors


def summarise(res, conf):
    mp = map_metrics(res)
    rm = reading_metrics(res, conf)
    cm = centre_metrics(res, conf)
    s = {k: float(v) for k, v in {**mp, **rm, **cm}.items() if np.isscalar(v)}
    s["conf"] = conf
    return s, mp, rm


def draw(entry, conf, path, max_w=1024):
    """Green = correct cell, red = wrong class (pred dots shown), blue = missed GT, orange = false positive."""
    from metrics import match
    img = entry["img"].copy()
    keep = entry["ps"] >= conf
    pb, ps, pc = entry["pb"][keep], entry["ps"][keep], entry["pc"][keep]
    gb, gl = entry["gb"], entry["gl"]
    m = match(pb, ps, gb)
    hit = np.zeros(len(gb), bool)
    for i, j in enumerate(m):
        x1, y1, x2, y2 = pb[i].astype(int)
        if j < 0:
            col = (255, 140, 0)
        else:
            hit[j] = True
            col = (0, 170, 0) if pc[i] == gl[j] else (220, 0, 0)
            if pc[i] != gl[j]:
                cv2.putText(img, dots(pc[i]), (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
    for j in np.where(~hit)[0]:
        x1, y1, x2, y2 = gb[j].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 90, 255), 2)
    cv2.imwrite(path, img[:, :, ::-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    p.add_argument("--weights", default="best.weights.h5")
    p.add_argument("--index", default="data/index.json")
    p.add_argument("--scale-norm", action="store_true")
    p.add_argument("--auto", action="store_true", help="Phase 1: auto scale + orientation (src/inference.py)")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    run_dir = os.path.join("results", a.name)
    tag = a.tag or ("auto" if a.auto else "scalenorm" if a.scale_norm else "plain")
    out_dir = os.path.join(run_dir, f"eval_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    model = build_yolo11s(NC)
    model.load_weights(os.path.join(run_dir, a.weights))
    index = json.load(open(a.index))

    val, t_val, _ = infer(model, index["val"], a.scale_norm, a.auto)
    conf = best_conf(val)
    test, t_test, factors = infer(model, index["test"], a.scale_norm, a.auto)

    report = {"conf_from_val": conf}
    for split, res, t in (("val", val, t_val), ("test", test, t_test)):
        s, mp, rm = summarise(res, conf)
        s["ms_per_image"] = 1000 * t
        report[split] = s
        np.save(os.path.join(out_dir, f"{split}_ap50_per_class.npy"), mp["ap50_per_class"])
        np.save(os.path.join(out_dir, f"{split}_f1_per_class.npy"), rm["f1_per_class"])
        np.save(os.path.join(out_dir, f"{split}_confusion.npy"), rm["confusion"])
        print(f"[{split}] mAP50 {s['map50']:.4f}  mAP50-95 {s['map50_95']:.4f}  "
              f"charP {s['char_P']:.4f} charR {s['char_R']:.4f} charF1 {s['char_F1']:.4f}  "
              f"ctrF1 {s['ctr_F1']:.4f}  detF1 {s['det_F1']:.4f}  macroF1 {s['macro_F1']:.4f}  cls-acc {s['cls_acc_on_matched']:.4f}  "
              f"{s['ms_per_image']:.0f} ms/img", flush=True)
    if factors and isinstance(factors[0], tuple):
        report["test_scale_rot"] = factors
        sc = [f[0] for f in factors]
        print(f"test: scale median {np.median(sc):.2f} (min {min(sc):.2f}, max {max(sc):.2f}), "
              f"lighting-flipped: {sum(f[1] for f in factors)}/{len(factors)} pages")
    elif factors:
        report["test_scale_factors"] = factors
        print("test rescale factors: median %.2f, min %.2f, max %.2f" % (np.median(factors), min(factors), max(factors)))

    # per-image test char-F1 (for picking qualitative examples)
    per_img = []
    for r in test:
        rm = reading_metrics([r], conf)
        per_img.append((rm["char_F1"], r["item"]["path"]))
    report["test_per_image_charF1"] = per_img
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=1)

    order = np.argsort([x[0] for x in per_img])
    for k, i in enumerate([order[0], order[len(order) // 2], order[-1]]):
        draw(test[i], conf, os.path.join(out_dir, f"test_example_{['worst', 'median', 'best'][k]}.jpg"))


if __name__ == "__main__":
    main()
