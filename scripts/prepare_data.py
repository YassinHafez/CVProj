"""Build a single JSON index of the Angelina Braille dataset (v1.1).

For every image in the official train / val / test lists we store:
  - its path
  - its original size
  - all boxes in normalised [0,1] xyxy coordinates
  - class ids 0..62 (= Angelina dot-code 1..63 minus 1)

Usage:  python3 scripts/prepare_data.py data/angelina data/index.json
"""
import csv
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

SPLITS = {
    "train": [("books", "train.txt"), ("handwritten", "train.txt"), ("not_braille", "train.txt")],
    "val": [("books", "val.txt"), ("handwritten", "val.txt")],
    "test": [("uploaded", "test.txt")],
}


def read_csv(path):
    boxes, labels = [], []
    if not os.path.exists(path):
        return boxes, labels
    with open(path) as f:
        for row in csv.reader(f, delimiter=";"):
            if not row:
                continue
            l, t, r, b, lab = row
            boxes.append([float(l), float(t), float(r), float(b)])
            labels.append(int(lab) - 1)
    return boxes, labels


def main(root, out):
    index = {}
    for split, lists in SPLITS.items():
        items = []
        for group, lst in lists:
            with open(os.path.join(root, group, lst)) as f:
                rels = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]
            for rel in rels:
                img_path = os.path.join(root, group, rel)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                boxes, labels = read_csv(os.path.splitext(img_path)[0] + ".csv")
                items.append(dict(path=img_path, group=group, w=w, h=h, boxes=boxes, labels=labels))
        index[split] = items

    with open(out, "w") as f:
        json.dump(index, f)

    # ---- statistics ----
    for split, items in index.items():
        n_cells = sum(len(it["labels"]) for it in items)
        cls = Counter(l for it in items for l in it["labels"])
        print(f"{split:5s}: {len(items):3d} images, {n_cells:6d} cells, {len(cls)} classes present")
    all_items = [it for s in index.values() for it in s]
    widths = Counter(it["w"] for it in all_items)
    print("most common widths:", widths.most_common(5))
    # cell size after resizing every image to width 1024
    ws, hs, per_img = [], [], []
    for it in all_items:
        s = 1024 / it["w"]
        b = np.array(it["boxes"]).reshape(-1, 4)
        if len(b):
            ws += list((b[:, 2] - b[:, 0]) * it["w"] * s)
            hs += list((b[:, 3] - b[:, 1]) * it["h"] * s)
            per_img.append(len(b))
    print(f"cell size at width 1024: median {np.median(ws):.1f} x {np.median(hs):.1f} px, "
          f"p5 {np.percentile(ws,5):.1f}x{np.percentile(hs,5):.1f}, p95 {np.percentile(ws,95):.1f}x{np.percentile(hs,95):.1f}")
    print(f"cells per labelled image: median {np.median(per_img):.0f}, max {max(per_img)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
