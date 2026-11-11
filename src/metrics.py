"""Inference on full pages + detection / reading metrics."""
import numpy as np
import tensorflow as tf

from data import letterbox_full, load_image
from yolo11 import decode, make_anchors

IOU_THRS = np.linspace(0.5, 0.95, 10)


@tf.function(reduce_retracing=True)
def _forward(model, x):
    raw = model(x, training=False)
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    anchors, strides = make_anchors(h, w)
    boxes, logits, _ = decode(raw, anchors, strides)
    return boxes[0], tf.sigmoid(logits[0])


def predict(model, img, conf=0.001, iou=0.5, max_det=1500):
    """img: RGB uint8 at base width. Returns boxes (N,4) px, scores (N,), classes (N,).

    Class-agnostic NMS: Braille cells never overlap, so two boxes on the same cell are
    duplicates whatever their class.
    """
    h, w = img.shape[:2]
    x = letterbox_full(img)[None]
    boxes, probs = _forward(model, tf.constant(x))
    scores = tf.reduce_max(probs, -1)
    cls = tf.argmax(probs, -1, output_type=tf.int32)
    keep = tf.where(scores > conf)[:, 0]
    boxes, scores, cls = tf.gather(boxes, keep), tf.gather(scores, keep), tf.gather(cls, keep)
    yx = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], 1)
    sel = tf.image.non_max_suppression(yx, scores, max_det, iou_threshold=iou)
    b = tf.gather(boxes, sel).numpy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b, tf.gather(scores, sel).numpy(), tf.gather(cls, sel).numpy()


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def run_inference(model, items, conf=0.001):
    """Returns per-image dicts with predictions and ground truth (base-width pixels)."""
    out = []
    for it in items:
        img, gb, gl = load_image(it)
        pb, ps, pc = predict(model, img, conf=conf)
        out.append(dict(item=it, pb=pb, ps=ps, pc=pc, gb=gb, gl=gl, shape=img.shape[:2]))
    return out


def match(pb, ps, gb, iou_thr=0.5):
    """Greedy class-agnostic matching by descending score. Returns gt index per pred (-1 = none)."""
    m = -np.ones(len(pb), int)
    if len(pb) == 0 or len(gb) == 0:
        return m
    ious = box_iou(pb, gb)
    taken = np.zeros(len(gb), bool)
    for i in np.argsort(-ps):
        cand = np.where(~taken & (ious[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            m[i], taken[j] = j, True
    return m


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x)


def map_metrics(results, nc=63):
    """COCO-style mAP@0.5 and mAP@0.5:0.95 (101-pt), class-aware matching, classes present in GT."""
    tps, confs, pcls, gcls = [], [], [], []
    for r in results:
        pb, ps, pc, gb, gl = r["pb"], r["ps"], r["pc"], r["gb"], r["gl"]
        tp = np.zeros((len(pb), len(IOU_THRS)), bool)
        if len(pb) and len(gb):
            ious = box_iou(pb, gb) * (pc[:, None] == gl[None, :])
            order = np.argsort(-ps)
            for t, thr in enumerate(IOU_THRS):
                taken = np.zeros(len(gb), bool)
                for i in order:
                    cand = np.where(~taken & (ious[i] >= thr))[0]
                    if len(cand):
                        taken[cand[np.argmax(ious[i, cand])]] = True
                        tp[i, t] = True
        tps.append(tp); confs.append(ps); pcls.append(pc); gcls.append(gl)
    tp, conf, pc, gc = np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(gcls)
    order = np.argsort(-conf)
    tp, pc = tp[order], pc[order]
    ap = np.full((nc, len(IOU_THRS)), np.nan)
    for c in np.unique(gc):
        sel = pc == c
        n_gt = (gc == c).sum()
        if sel.sum() == 0:
            ap[c] = 0
            continue
        ctp = np.cumsum(tp[sel], 0)
        cfp = np.cumsum(~tp[sel], 0)
        rec = ctp / n_gt
        prec = ctp / (ctp + cfp)
        for t in range(len(IOU_THRS)):
            ap[c, t] = compute_ap(rec[:, t], prec[:, t])
    return dict(map50=np.nanmean(ap[:, 0]), map50_95=np.nanmean(ap), ap50_per_class=ap[:, 0])


def reading_metrics(results, conf, nc=63):
    """Ovodov's char P/R/F1 (IoU >= 0.5 and correct class) + detection-only F1 + macro-F1 + confusion."""
    n_pred = n_gt = n_det = n_char = 0
    tp_c, fp_c, fn_c = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    conf_mat = np.zeros((nc + 1, nc + 1), int)  # last row/col = background
    for r in results:
        keep = r["ps"] >= conf
        pb, ps, pc = r["pb"][keep], r["ps"][keep], r["pc"][keep]
        gb, gl = r["gb"], r["gl"]
        m = match(pb, ps, gb)
        n_pred += len(pb); n_gt += len(gb)
        matched_gt = np.zeros(len(gb), bool)
        for i, j in enumerate(m):
            if j < 0:
                fp_c[pc[i]] += 1
                conf_mat[nc, pc[i]] += 1
                continue
            matched_gt[j] = True
            n_det += 1
            conf_mat[gl[j], pc[i]] += 1
            if pc[i] == gl[j]:
                n_char += 1; tp_c[pc[i]] += 1
            else:
                fp_c[pc[i]] += 1; fn_c[gl[j]] += 1
        for j in np.where(~matched_gt)[0]:
            fn_c[gl[j]] += 1
            conf_mat[gl[j], nc] += 1
    P = n_char / max(n_pred, 1); R = n_char / max(n_gt, 1)
    dP = n_det / max(n_pred, 1); dR = n_det / max(n_gt, 1)
    present = (tp_c + fn_c) > 0
    f1_c = 2 * tp_c / np.maximum(2 * tp_c + fp_c + fn_c, 1)
    return dict(char_P=P, char_R=R, char_F1=2 * P * R / max(P + R, 1e-9),
                det_F1=2 * dP * dR / max(dP + dR, 1e-9),
                cls_acc_on_matched=n_char / max(n_det, 1),
                macro_F1=f1_c[present].mean(), f1_per_class=np.where(present, f1_c, np.nan),
                n_pred=n_pred, n_gt=n_gt, confusion=conf_mat)


def best_conf(results, grid=np.arange(0.05, 0.95, 0.025)):
    f1s = [reading_metrics(results, c)["char_F1"] for c in grid]
    return float(grid[int(np.argmax(f1s))])


def centre_metrics(results, conf):
    """Char P/R/F1 where a prediction matches a GT cell if its centre lies inside the GT box
    (one-to-one, by descending score). Insensitive to how tightly boxes are drawn, which differs
    between Angelina's train/val labels and its test labels."""
    n_pred = n_gt = n_ok = 0
    for r in results:
        keep = r["ps"] >= conf
        pb, ps, pc = r["pb"][keep], r["ps"][keep], r["pc"][keep]
        gb, gl = r["gb"], r["gl"]
        n_pred += len(pb); n_gt += len(gb)
        if not len(pb) or not len(gb):
            continue
        cx, cy = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2
        inside = ((cx[:, None] >= gb[None, :, 0]) & (cx[:, None] <= gb[None, :, 2]) &
                  (cy[:, None] >= gb[None, :, 1]) & (cy[:, None] <= gb[None, :, 3]))
        used = np.zeros(len(gb), bool)
        for i in np.argsort(-ps):
            js = np.where(inside[i] & ~used)[0]
            if len(js):
                used[js[0]] = True
                n_ok += int(pc[i] == gl[js[0]])
    P, R = n_ok / max(n_pred, 1), n_ok / max(n_gt, 1)
    return dict(ctr_P=P, ctr_R=R, ctr_F1=2 * P * R / max(P + R, 1e-9))
