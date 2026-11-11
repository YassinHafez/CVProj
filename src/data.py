"""Data loading and label-safe augmentation for Angelina Braille pages.

Every image is first rescaled to a page width of 1024 px (87 % of the dataset already is),
so a Braille cell is ~22 x 33 px and a dot ~4-5 px. Training samples are random 640 x 640
crops of that native-resolution page ("640 tiles" from Milestone 1), never a downscale of
the full page.
"""
import json
import math

import cv2
import numpy as np

BASE_W = 1024
PAD_VALUE = 114


def load_index(path):
    with open(path) as f:
        return json.load(f)


def load_image(item, base_w=BASE_W):
    img = cv2.imread(item["path"])[:, :, ::-1]  # RGB
    s = base_w / img.shape[1]
    img = cv2.resize(img, (base_w, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    b = np.array(item["boxes"], np.float32).reshape(-1, 4) * [w, h, w, h]
    return np.ascontiguousarray(img), b, np.array(item["labels"], np.int32)


def letterbox_full(img, mult=32):
    """Pad bottom/right so H, W are multiples of 32 (full-page inference)."""
    h, w = img.shape[:2]
    H, W = math.ceil(h / mult) * mult, math.ceil(w / mult) * mult
    out = np.full((H, W, 3), PAD_VALUE, np.uint8)
    out[:h, :w] = img
    return out


def hsv_jitter(img, rng, hgain=0.015, sgain=0.7, vgain=0.4):
    r = rng.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class CropSampler:
    """Infinite generator of augmented (crop, boxes, labels, mask) training samples."""

    def __init__(self, items, size=640, max_boxes=400, scale=(0.75, 1.25), degrees=3.0,
                 blur_p=0.15, min_visible=0.6, seed=0, sample_weights=None):
        self.data = [load_image(it) for it in items]
        self.size, self.max_boxes = size, max_boxes
        self.scale, self.degrees, self.blur_p, self.min_visible = scale, degrees, blur_p, min_visible
        self.rng = np.random.default_rng(seed)
        self.p = None if sample_weights is None else np.asarray(sample_weights) / np.sum(sample_weights)
        self.truncated = 0

    def sample(self, seed=None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        img, boxes, labels = self.data[rng.choice(len(self.data), p=self.p)]
        h, w = img.shape[:2]
        S = self.size
        s = rng.uniform(*self.scale)
        a = math.radians(rng.uniform(-self.degrees, self.degrees))
        # random crop centre on the page (in source coords)
        half = S / (2 * s)  # half crop size in source pixels; keep the crop mostly on the page
        cx = rng.uniform(half, w - half) if w > 2 * half else w / 2
        cy = rng.uniform(half, h - half) if h > 2 * half else h / 2
        ca, sa = math.cos(a) * s, math.sin(a) * s
        # dst = R*s*(src - c) + S/2
        M = np.array([[ca, -sa, S / 2 - (ca * cx - sa * cy)],
                      [sa, ca, S / 2 - (sa * cx + ca * cy)]], np.float32)
        crop = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=(PAD_VALUE,) * 3)

        out_b = np.zeros((self.max_boxes, 4), np.float32)
        out_l = np.zeros((self.max_boxes,), np.int32)
        out_m = np.zeros((self.max_boxes,), np.float32)
        if len(boxes):
            x1, y1, x2, y2 = boxes.T
            corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                                np.stack([x1, y2], 1), np.stack([x2, y2], 1)], 1)  # N,4,2
            tc = corners @ M[:, :2].T + M[:, 2]
            nb = np.concatenate([tc.min(1), tc.max(1)], 1)
            area = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
            cb = np.clip(nb, 0, S)
            carea = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])
            keep = (carea > self.min_visible * area) & (cb[:, 2] - cb[:, 0] > 2) & (cb[:, 3] - cb[:, 1] > 2)
            cb, cl = cb[keep], labels[keep]
            if len(cb) > self.max_boxes:
                self.truncated += 1
                cb, cl = cb[:self.max_boxes], cl[:self.max_boxes]
            n = len(cb)
            out_b[:n], out_l[:n], out_m[:n] = cb, cl, 1.0

        crop = hsv_jitter(crop, rng)
        if rng.random() < self.blur_p:
            k = rng.choice([3, 5])
            crop = cv2.GaussianBlur(crop, (k, k), 0)
        return crop, out_b, out_l, out_m

    def __call__(self):
        while True:
            yield self.sample()
