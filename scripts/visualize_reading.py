"""Human-checkable reading report: page + boxes, zoomed crop with decoded letters,
model text vs ground-truth text (decoded from the human labels with the same decoder), and a diff.

  python3 scripts/visualize_reading.py --name baseline --out results/reading_check/reading_check.html
"""
import argparse
import base64
import difflib
import html
import io
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from braille_text import RU, SYM, NUM, dots, group_lines, levenshtein, page_text, unicode_cell
from data import load_image
from metrics import box_iou, match, predict
from yolo11 import build_yolo11s

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COL = dict(ok=(0, 160, 60), wrong=(220, 30, 30), missed=(30, 100, 255), extra=(255, 140, 0))


def centre_match(pb, pc, gb, gl):
    """Pred counts as correct if its centre is inside a GT box with the same class (one-to-one)."""
    cx, cy = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2
    used, n = np.zeros(len(gb), bool), 0
    for i in range(len(pb)):
        inside = np.where(~used & (cx[i] >= gb[:, 0]) & (cx[i] <= gb[:, 2]) & (cy[i] >= gb[:, 1]) & (cy[i] <= gb[:, 3]))[0]
        if len(inside):
            used[inside[0]] = True
            n += int(pc[i] == gl[inside[0]])
    p, r = n / max(len(pb), 1), n / max(len(gb), 1)
    return 2 * p * r / max(p + r, 1e-9)


def iou_char_f1(pb, ps, pc, gb, gl):
    m = match(pb, ps, gb)
    n = sum(1 for i, j in enumerate(m) if j >= 0 and pc[i] == gl[j])
    p, r = n / max(len(pb), 1), n / max(len(gb), 1)
    return 2 * p * r / max(p + r, 1e-9)


def status(pb, ps, pc, gb, gl):
    m = match(pb, ps, gb, iou_thr=0.3)
    st = ["extra"] * len(pb)
    hit = np.zeros(len(gb), bool)
    for i, j in enumerate(m):
        if j >= 0:
            hit[j] = True
            st[i] = "ok" if pc[i] == gl[j] else "wrong"
    return st, np.where(~hit)[0]


def char_of(c):
    d = dots(c)
    return RU.get(d) or SYM.get(d) or ("#" if d == "3456" else unicode_cell(c))


def to_b64(im, q=82):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def render(img, pb, pc, gb, st, missed, scale_w=620):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for b, s in zip(pb, st):
        d.rectangle(b.tolist(), outline=COL[s], width=2)
    for j in missed:
        d.rectangle(gb[j].tolist(), outline=COL["missed"], width=3)
    s = scale_w / im.width
    return im.resize((scale_w, round(im.height * s)), Image.LANCZOS)


def zoom(img, pb, pc, gb, st, missed, n_lines=4, width=760):
    """Full-resolution crop of a few lines with the decoded letter above every predicted cell."""
    if len(pb) == 0:
        return None
    lines = group_lines(pb)
    k = max(0, len(lines) // 3)
    sel = [i for l in lines[k:k + n_lines] for i in l]
    x0 = max(0, int(pb[sel, 0].min()) - 10)
    y0 = max(0, int(pb[sel, 1].min()) - 22)
    y1 = min(img.shape[0], int(pb[sel, 3].max()) + 8)
    x1 = min(img.shape[1], x0 + 520)
    crop = Image.fromarray(img[y0:y1, x0:x1]).convert("RGB")
    f = width / crop.width
    crop = crop.resize((width, round(crop.height * f)), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    font = ImageFont.truetype(FONT, 17)
    for i in range(len(pb)):
        b = (pb[i] - [x0, y0, x0, y0]) * f
        if b[2] < 0 or b[0] > width or b[3] < 0 or b[1] > crop.height:
            continue
        d.rectangle(b.tolist(), outline=COL[st[i]], width=2)
        d.text((b[0] + 1, b[1] - 20), char_of(pc[i]), fill=COL[st[i]], font=font)
    for j in missed:
        b = (gb[j] - [x0, y0, x0, y0]) * f
        if b[2] < 0 or b[0] > width or b[3] < 0 or b[1] > crop.height:
            continue
        d.rectangle(b.tolist(), outline=COL["missed"], width=3)
    return crop


def diff_html(gt, pr):
    sm = difflib.SequenceMatcher(None, gt, pr, autojunk=False)
    out = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        g, p = html.escape(gt[a0:a1]), html.escape(pr[b0:b1])
        if op == "equal":
            out.append(g)
        else:
            if g:
                out.append(f"<del>{g}</del>")
            if p:
                out.append(f"<ins>{p}</ins>")
    return "".join(out)


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--del-bg:#fde2e1;--del:#b42318;--ins-bg:#dcf5e7;--ins:#05603a;--chip:#f2f1ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--del-bg:#4a1d1b;--del:#ffb4ab;--ins-bg:#13392a;--ins:#9be3bd;--chip:#2e2d2b}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--del-bg:#4a1d1b;--del:#ffb4ab;--ins-bg:#13392a;--ins:#9be3bd;--chip:#2e2d2b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0}
.sub{color:var(--ink2);margin:0 0 20px}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-bottom:24px}
.legend span::before{content:"";display:inline-block;width:12px;height:12px;border:3px solid var(--c);margin-right:6px;vertical-align:-2px}
.page{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:28px}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 18px;margin-bottom:12px}.why{color:var(--ink2);font-size:14px;margin:0 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:13px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:minmax(0,620px) minmax(0,1fr);gap:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
img{max-width:100%;height:auto;border-radius:8px;display:block}.zoom{margin-top:14px}
.texts{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:600px){.texts{grid-template-columns:1fr}}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--chip);padding:10px;border-radius:8px;max-height:420px;overflow:auto}
.lbl{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:0 0 4px}
del{background:var(--del-bg);color:var(--del);text-decoration:line-through}ins{background:var(--ins-bg);color:var(--ins);text-decoration:none}
.gloss{margin-top:12px;font-size:14px;color:var(--ink2);border-left:3px solid var(--line);padding-left:10px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--out", default="results/reading_check/reading_check.html")
    ap.add_argument("--pages", default="results/reading_check/reading_pages.json")
    a = ap.parse_args()
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    conf = json.load(open(f"results/{a.name}/eval_plain/metrics.json"))["conf_from_val"]
    model = build_yolo11s(63)
    model.load_weights(f"results/{a.name}/{a.weights}")
    idx = json.load(open("data/index.json"))
    by_name = {it["path"].split("/")[-1]: (sp, it) for sp, items in idx.items() for it in items}
    pages = json.load(open(a.pages))

    parts = []
    for pg in pages:
        sp, it = by_name[pg["file"]]
        img, gb, gl = load_image(it)
        pb, ps, pc = predict(model, img, conf=conf)
        st, missed = status(pb, ps, pc, gb, gl)
        gt_txt, pr_txt = page_text(gb, gl), page_text(pb, pc)
        cer = levenshtein(gt_txt, pr_txt) / max(len(gt_txt), 1)
        f_iou, f_ctr = iou_char_f1(pb, ps, pc, gb, gl), centre_match(pb, pc, gb, gl)
        full = to_b64(render(img, pb, pc, gb, st, missed))
        z = zoom(img, pb, pc, gb, st, missed)
        ztag = f'<img class="zoom" alt="zoomed crop with decoded letters" src="data:image/jpeg;base64,{to_b64(z)}">' if z else ""
        gloss = f'<div class="gloss"><b>English gloss of the first lines</b> (written by Claude from the ground-truth text, for reference only, not model output): {html.escape(pg["gloss"])}</div>' if pg.get("gloss") else ""
        parts.append(f"""
<section class="page">
  <div class="head"><h2>{html.escape(pg['title'])}</h2>
    <div class="chips"><span class="chip">{sp} · {html.escape(pg['file'][:28])}</span>
      <span class="chip">cells: {len(gb)} labelled / {len(pb)} predicted</span>
      <span class="chip">char-F1 (IoU≥0.5): {f_iou:.3f}</span><span class="chip">char-F1 (centre): {f_ctr:.3f}</span>
      <span class="chip">CER of decoded text: {100 * cer:.1f}%</span></div></div>
  <p class="why">{html.escape(pg['why'])}</p>
  <div class="grid">
    <div><img alt="page with predicted cells" src="data:image/jpeg;base64,{full}">{ztag}</div>
    <div>
      <div class="texts">
        <div><p class="lbl">Model reading</p><pre>{html.escape(pr_txt)}</pre></div>
        <div><p class="lbl">Ground truth (human labels, same decoder)</p><pre>{html.escape(gt_txt)}</pre></div>
      </div>
      <p class="lbl" style="margin-top:12px">Difference (<del>ground truth only</del> <ins>model only</ins>)</p>
      <pre>{diff_html(gt_txt, pr_txt)}</pre>
      {gloss}
    </div>
  </div>
</section>""")
        print(f"{pg['file'][:30]:30s} F1(IoU) {f_iou:.3f} F1(centre) {f_ctr:.3f} CER {100 * cer:.1f}%")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Braille Reading Check</title><style>{CSS}</style></head><body><main>
<h1>Braille reading check: YOLO11-s baseline</h1>
<p class="sub">Model: YOLO11-s (TensorFlow, trained from scratch, 300 epochs, best-val checkpoint). Confidence threshold {conf:.2f}, chosen on val.
Text is decoded with grade-1 Russian Braille tables (Angelina dataset), the same decoder for model and ground truth.</p>
<div class="legend"><span style="--c:rgb{COL['ok']}">correct cell</span><span style="--c:rgb{COL['wrong']}">wrong dot pattern</span>
<span style="--c:rgb{COL['extra']}">extra detection (no label)</span><span style="--c:rgb{COL['missed']}">missed labelled cell</span></div>
{''.join(parts)}
</main></body></html>"""
    with open(a.out, "w") as f:
        f.write(doc)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
