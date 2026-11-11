"""Try image preprocessing variants on photos without labels and score the reading.

Scores (no ground truth needed):
  cells      - number of detected cells (conf >= 0.30)
  conf       - mean confidence of those cells
  words_ok   - share of decoded words found in an English dictionary (British English list)
  letters_ok - share of decoded letters that sit in dictionary words (weights long words more)

  python3 scripts/preprocess_experiments.py OUT_DIR IMG [IMG ...]
"""
import csv
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from braille_text import page_text
from metrics import predict
from yolo11 import build_yolo11s

WIDTHS = (1024, 1536, 2000)
DICT = set(w.strip().lower() for w in open("/usr/share/dict/british-english", encoding="utf-8", errors="ignore"))


def L_op(img, fn):
    """Apply fn to the lightness channel only (keeps colours, as the model was trained on RGB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = np.clip(fn(lab[..., 0]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def gamma(g):
    lut = (255 * (np.arange(256) / 255) ** g).astype(np.uint8)
    return lambda L: cv2.LUT(L, lut)


def stretch(L):
    lo, hi = np.percentile(L, (1, 99))
    return (L.astype(np.float32) - lo) * 255 / max(hi - lo, 1)


def clahe(clip):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return lambda L: c.apply(L)


def flatfield(L):
    """Remove slow lighting gradients: divide by a heavily blurred copy (sigma ~ 2 % of width)."""
    s = 0.02 * L.shape[1]
    bg = cv2.GaussianBlur(L.astype(np.float32), (0, 0), s)
    return L.astype(np.float32) / np.maximum(bg, 1) * 160


def unsharp(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.8, blur, -0.8, 0)


VARIANTS = {
    "original": lambda x: x,
    "grayscale": lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB),
    "contrast stretch 1-99%": lambda x: L_op(x, stretch),
    "gamma 0.7 (brighter)": lambda x: L_op(x, gamma(0.7)),
    "gamma 1.5 (darker)": lambda x: L_op(x, gamma(1.5)),
    "CLAHE clip 2": lambda x: L_op(x, clahe(2.0)),
    "CLAHE clip 4": lambda x: L_op(x, clahe(4.0)),
    "flat-field (lighting)": lambda x: L_op(x, flatfield),
    "flat-field + CLAHE 2": lambda x: L_op(L_op(x, flatfield), clahe(2.0)),
    "unsharp mask": unsharp,
    "bilateral denoise": lambda x: cv2.bilateralFilter(x, 7, 30, 7),
}


def word_scores(text):
    text = text.replace("⠠", "").lower()  # English capital sign; dictionary is lowercase
    words = re.findall(r"[^\s]+", text)
    ok = tot_l = ok_l = 0
    n = 0
    for w in words:
        core = re.sub(r"^[^a-z]+|[^a-z]+$", "", w)
        if not core:
            continue
        n += 1
        tot_l += len(core)
        if core in DICT:
            ok += 1
            ok_l += len(core)
    return ok / max(n, 1), ok_l / max(tot_l, 1), n


def main():
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    model = build_yolo11s(63)
    model.load_weights("results/baseline/best.weights.h5")
    rows = []
    for p in paths:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        for W in WIDTHS:
            base = np.asarray(im.resize((W, round(im.height * W / im.width)),
                                        Image.LANCZOS if W < im.width else Image.BICUBIC))
            for name, fn in VARIANTS.items():
                x = np.ascontiguousarray(fn(base))
                pb, ps, pc = predict(model, x, conf=0.30)
                wo, lo, nw = word_scores(page_text(pb, pc, "en"))
                rows.append(dict(image=os.path.basename(p), width=W, variant=name, cells=len(pb),
                                 conf=round(float(ps.mean()) if len(ps) else 0, 3),
                                 words_ok=round(wo, 3), letters_ok=round(lo, 3), words=nw))
                if W == 1024 and os.path.basename(p) == os.path.basename(paths[0]):
                    h, w = x.shape[:2]
                    cv2.imwrite(os.path.join(out_dir, f"crop_{name.split(' ')[0]}_{list(VARIANTS).index(name)}.jpg"),
                                x[int(h * .30):int(h * .30) + 220, int(w * .15):int(w * .15) + 420][:, :, ::-1])
            print(f"{os.path.basename(p)} @ {W}: done", flush=True)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
