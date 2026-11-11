"""Phase-1 improvement: scale- and lighting-aware page reading (no retraining).

What we measured on the baseline (see results/ and the Milestone-2 notes):
  * Cell size: the detector handles cells of roughly 16.5-27.5 px (training scale augmentation
    0.75-1.25 around 22 px at page width 1024). English book photos come out at ~13.5 px when the
    page is resized to 1024 wide and are read badly; Angelina test uploads are ~25-30 px and are
    read fine as they are (shrinking them made results worse).
  * Lighting direction: some uploads are lit from below, so raised dots look like dents, which the
    model learned to ignore (unlabelled back sides of two-sided pages look the same). Flipping the
    image vertically makes them look raised again; each read cell must then have its dot rows
    swapped back (1<->3, 4<->6) and its box flipped back.
  * Two-sided pages: a flipped view of a normal page also shows plausible "text" (the reverse side),
    with up to ~1.4x the confident cells on the validation set. So we only switch when the flipped
    view finds at least 2x the confident cells - a fixed safety margin, not tuned on the test set.
  * Rotating 180 deg never improved a validation or test page, so it is not used.
"""
import cv2
import numpy as np

from metrics import predict

TARGET_CELL_W = 22.0        # median training cell width at page width 1024
MIN_OK, MAX_OK = 16.5, 34.0  # cell widths the model reads well; rescale only outside this band
MAX_W = 3200                # largest processing width (GPU memory)
FLIP_MARGIN = 2.0           # flipped view must find >= 2x confident cells to be used

# dot i (0-based: dots 1..6) -> dot after a vertical flip of the cell: 1<->3, 4<->6, 2 and 5 stay
_VFLIP = {0: 2, 1: 1, 2: 0, 3: 5, 4: 4, 5: 3}
VFLIP_CLASS = np.array([sum(1 << _VFLIP[i] for i in range(6) if (c + 1) >> i & 1) - 1 for c in range(63)])


def _resize(img, w):
    h = max(32, round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC)


def _score(ps):
    return float(ps[ps >= 0.5].sum())


def _cell_width(pb, ps):
    sel = ps >= 0.5
    return float(np.median(pb[sel, 2] - pb[sel, 0])) if sel.sum() >= 20 else None


def read_page(model, img, conf=0.30, auto_scale=True, auto_light=True):
    """img: RGB uint8 of any size. Returns boxes (img pixel coords), scores, classes, info."""
    H, W = img.shape[:2]
    info = dict(scale=1.0, flipped=False, proc_width=1024, cell_px=None)

    # 1) probe at page width 1024 (the training normalisation)
    x = _resize(img, 1024)
    pb, ps, pc = predict(model, x, conf=conf)
    cw = _cell_width(pb, ps)
    if cw is None and W > 1024:  # nothing confident at 1024: cells may be tiny, probe larger
        x2 = _resize(img, min(2048, MAX_W))
        pb2, ps2, _ = predict(model, x2, conf=conf)
        cw2 = _cell_width(pb2, ps2)
        cw = None if cw2 is None else cw2 * 1024 / x2.shape[1]
    info["cell_px"] = None if cw is None else round(cw, 1)

    # 2) rescale only if cells are outside the band the model reads well
    if auto_scale and cw is not None and not (MIN_OK <= cw <= MAX_OK):
        proc_w = int(min(MAX_W, round(1024 * TARGET_CELL_W / cw)))
        x = _resize(img, proc_w)
        pb, ps, pc = predict(model, x, conf=conf)
        info.update(scale=round(proc_w / 1024, 2), proc_width=proc_w)

    # 3) lighting from below: vertically flipped view, switch only with a clear win
    if auto_light:
        h = x.shape[0]
        fb, fs, fc = predict(model, np.ascontiguousarray(x[::-1]), conf=conf)
        if _score(fs) >= FLIP_MARGIN * max(_score(ps), 1e-6) and len(fb):
            pb = np.stack([fb[:, 0], h - fb[:, 3], fb[:, 2], h - fb[:, 1]], 1)
            ps, pc = fs, VFLIP_CLASS[fc]
            info["flipped"] = True

    f = x.shape[1] / W
    return pb / f, ps, pc, info
