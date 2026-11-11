"""YOLOv8/YOLO11 detection loss in TensorFlow.

Task-Aligned Assigner (topk=10, alpha=0.5, beta=6) + BCE classification + CIoU box loss
+ Distribution Focal Loss, with the ultralytics gains box=7.5, cls=0.5, dfl=1.5.
Everything is computed with dense masks (static shapes) instead of boolean indexing.
"""
import math

import tensorflow as tf

from yolo11 import REG_MAX, decode

EPS = 1e-9


def pairwise_ciou(a, b):
    """CIoU between broadcastable xyxy boxes a (...,4) and b (...,4)."""
    ax1, ay1, ax2, ay2 = tf.unstack(a, axis=-1)
    bx1, by1, bx2, by2 = tf.unstack(b, axis=-1)
    w1, h1 = ax2 - ax1, ay2 - ay1 + 1e-7
    w2, h2 = bx2 - bx1, by2 - by1 + 1e-7
    iw = tf.maximum(tf.minimum(ax2, bx2) - tf.maximum(ax1, bx1), 0)
    ih = tf.maximum(tf.minimum(ay2, by2) - tf.maximum(ay1, by1), 0)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter + 1e-7
    iou = inter / union
    cw = tf.maximum(ax2, bx2) - tf.minimum(ax1, bx1)
    ch = tf.maximum(ay2, by2) - tf.minimum(ay1, by1)
    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = ((bx1 + bx2 - ax1 - ax2) ** 2 + (by1 + by2 - ay1 - ay2) ** 2) / 4
    v = (4 / math.pi ** 2) * (tf.atan(w2 / h2) - tf.atan(w1 / h1)) ** 2
    alpha = tf.stop_gradient(v / (v - iou + (1 + 1e-7)))
    return iou - (rho2 / c2 + v * alpha)


def task_aligned_assign(scores, pboxes, anchors, gt_labels, gt_boxes, gt_mask, nc, topk=10, alpha=0.5, beta=6.0):
    """scores (B,A,C) sigmoid, pboxes (B,A,4) px, anchors (A,2) px, gt_* (B,G,...).

    Returns target_boxes (B,A,4) px, target_scores (B,A,C), fg_mask (B,A) float.
    """
    scores, pboxes = tf.stop_gradient(scores), tf.stop_gradient(pboxes)
    B, G = tf.shape(gt_labels)[0], tf.shape(gt_labels)[1]
    # anchor centre strictly inside the GT box -> (B,G,A)
    x, y = anchors[None, None, :, 0], anchors[None, None, :, 1]
    gx1, gy1, gx2, gy2 = [gt_boxes[..., i:i + 1] for i in range(4)]
    d = tf.stack([x - gx1, y - gy1, gx2 - x, gy2 - y], -1)
    in_gts = tf.cast(tf.reduce_min(d, -1) > EPS, tf.float32) * gt_mask[..., None]

    overlaps = tf.maximum(pairwise_ciou(gt_boxes[:, :, None, :], pboxes[:, None, :, :]), 0.0) * in_gts
    cls_scores = tf.gather(tf.transpose(scores, (0, 2, 1)), gt_labels, batch_dims=1)  # (B,G,A)
    align = tf.pow(cls_scores, alpha) * tf.pow(overlaps, beta) * in_gts

    # top-k anchors per GT (threshold at the k-th largest metric)
    kth = tf.math.top_k(align, k=topk).values[..., -1:]
    mask_pos = tf.cast((align >= kth) & (align > 0), tf.float32) * in_gts

    # an anchor assigned to several GTs keeps the one with the highest IoU
    fg_count = tf.reduce_sum(mask_pos, 1)  # (B,A)
    best = tf.one_hot(tf.argmax(overlaps, 1), G, axis=1)  # (B,G,A)
    mask_pos = tf.where(fg_count[:, None, :] > 1, best, mask_pos)
    fg_mask = tf.cast(tf.reduce_sum(mask_pos, 1) > 0, tf.float32)
    gt_idx = tf.argmax(mask_pos, 1, output_type=tf.int32)  # (B,A)

    t_labels = tf.gather(gt_labels, gt_idx, batch_dims=1)
    t_boxes = tf.gather(gt_boxes, gt_idx, batch_dims=1)
    t_scores = tf.one_hot(t_labels, nc) * fg_mask[..., None]

    # soft targets: normalised alignment metric (as in ultralytics)
    align = align * mask_pos
    pos_align = tf.reduce_max(align, -1, keepdims=True)
    pos_ovl = tf.reduce_max(overlaps * mask_pos, -1, keepdims=True)
    norm = tf.reduce_max(align * pos_ovl / (pos_align + EPS), 1)  # (B,A)
    return t_boxes, t_scores * norm[..., None], fg_mask


def dfl_loss(dist_logits, target_ltrb):
    """dist_logits (B,A,4,16), target_ltrb (B,A,4) in grid units, clipped to [0, 15 - 0.01]."""
    tl = tf.floor(target_ltrb)
    wl = tl + 1 - target_ltrb
    wr = 1 - wl
    logp = tf.nn.log_softmax(dist_logits, -1)
    il, ir = tf.cast(tl, tf.int32), tf.minimum(tf.cast(tl, tf.int32) + 1, REG_MAX - 1)
    ce_l = -tf.gather(logp, il[..., None], batch_dims=3)[..., 0]
    ce_r = -tf.gather(logp, ir[..., None], batch_dims=3)[..., 0]
    return tf.reduce_mean(ce_l * wl + ce_r * wr, -1)  # (B,A)


def yolo_loss(raw, anchors, strides, gt_labels, gt_boxes, gt_mask, nc, gains=(7.5, 0.5, 1.5)):
    """Returns total loss (scaled by batch size like ultralytics) and its 3 components."""
    pboxes, cls_logits, dist = decode(raw, anchors, strides)
    scores = tf.sigmoid(cls_logits)
    t_boxes, t_scores, fg = task_aligned_assign(scores, pboxes, anchors, gt_labels, gt_boxes, gt_mask, nc)
    tss = tf.maximum(tf.reduce_sum(t_scores), 1.0)

    l_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=t_scores, logits=cls_logits)) / tss

    weight = tf.reduce_sum(t_scores, -1)  # (B,A), zero on background
    s = strides  # (A,1)
    ciou = pairwise_ciou(pboxes / s, t_boxes / s)
    l_box = tf.reduce_sum((1.0 - ciou) * weight * fg) / tss

    a = anchors / s
    tb = t_boxes / s
    ltrb = tf.concat([a - tb[..., :2], tb[..., 2:] - a], -1)
    ltrb = tf.clip_by_value(ltrb, 0.0, REG_MAX - 1 - 0.01)
    l_dfl = tf.reduce_sum(dfl_loss(dist, ltrb) * weight * fg) / tss

    bs = tf.cast(tf.shape(raw)[0], tf.float32)
    parts = tf.stack([l_box * gains[0], l_cls * gains[1], l_dfl * gains[2]])
    return tf.reduce_sum(parts) * bs, parts
