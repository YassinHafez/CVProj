"""Local Braille reader web app.

  python3 app/server.py [--port 8000] [--weights results/baseline/best.weights.h5]

GET  /                      -> upload page
POST /predict?lang=ru&rotate=0&conf=0.30&auto=1   body = raw image bytes (JPEG/PNG/WebP; HEIC is not supported)
     -> JSON {text, braille, image (annotated JPEG, base64), cells, ms, mean_conf, size, cell_px, scale, flipped}
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps

from braille_text import EN, NUM, RU, SYM, dots, page_braille, page_text, unicode_cell
from data import BASE_W
from inference import read_page
from metrics import predict
from yolo11 import build_yolo11s

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MAX_BYTES = 25 * 1024 * 1024
MAX_W = 3200
MODEL = None
try:
    with open(os.path.join(ROOT, "data", "index.json")) as _f:
        SAMPLES = [it["path"] if os.path.isabs(it["path"]) else os.path.join(ROOT, it["path"])
                   for it in json.load(_f)["test"]]
except OSError:
    SAMPLES = []


def load_model(weights):
    global MODEL
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    MODEL = build_yolo11s(63)
    MODEL.load_weights(weights)
    predict(MODEL, np.full((1376, BASE_W, 3), 200, np.uint8))  # warm-up / trace
    print("model ready on", MODEL.trainable_variables[0].value.device, flush=True)


def char_of(c, lang):
    d = dots(c)
    alpha = RU if lang == "ru" else EN
    return alpha.get(d) or SYM.get(d) or ("#" if d == "3456" else unicode_cell(c))


def run(raw, lang="ru", rotate=0, conf=0.30, auto=True):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if rotate:
        im = im.rotate(rotate, expand=True)
    if im.width > MAX_W:  # keep huge photos manageable; read_page rescales from here
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    img = np.asarray(im)
    t0 = time.perf_counter()
    # auto: measure cell size and rescale if outside what the model reads well, and try the
    # vertically flipped view for pages lit from below (src/inference.py). Off: old behaviour.
    pb, ps, pc, info = read_page(MODEL, img, conf=conf, auto_scale=auto, auto_light=auto)
    ms = 1000 * (time.perf_counter() - t0)

    # draw on a display copy ~1400 px wide
    k = min(1.0, 1400 / im.width)
    disp = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS) if k < 1 else im.copy()
    draw = ImageDraw.Draw(disp)
    db = pb * k
    fsize = int(np.clip(np.median(db[:, 3] - db[:, 1]) * 0.45 if len(db) else 13, 9, 22))
    font = ImageFont.truetype(FONT_PATH, fsize)
    for b, c, p in zip(db, pc, ps):
        col = (0, 150, 60) if p >= 0.6 else (230, 140, 0)  # orange = low confidence
        draw.rectangle(b.tolist(), outline=col, width=2)
        draw.text((b[0] + 1, b[1] - fsize - 2), char_of(c, lang), fill=col, font=font)
    buf = io.BytesIO()
    disp.save(buf, "JPEG", quality=85)
    return dict(
        text=page_text(pb, pc, lang) if len(pb) else "",
        braille=page_braille(pb, pc) if len(pb) else "",
        image=base64.b64encode(buf.getvalue()).decode(),
        cells=int(len(pb)), low_conf=int((ps < 0.6).sum()),
        mean_conf=float(ps.mean()) if len(ps) else 0.0,
        ms=round(ms), size=[im.width, im.height],
        cell_px=info["cell_px"], scale=info["scale"], flipped=info["flipped"],
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "app", "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/sample" and SAMPLES:
            # a random photo from the Angelina test set (user uploads, never used in training)
            p = SAMPLES[np.random.randint(len(SAMPLES))]
            with open(p, "rb") as f:
                self._send(200, f.read(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/predict":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES:
            return self._send(400, json.dumps({"error": "send an image up to 25 MB"}).encode(), "application/json")
        q = parse_qs(u.query)
        try:
            res = run(self.rfile.read(n),
                      lang=q.get("lang", ["ru"])[0],
                      rotate=int(q.get("rotate", ["0"])[0]) % 360,
                      conf=float(q.get("conf", ["0.30"])[0]),
                      auto=q.get("auto", ["1"])[0] == "1")
            self._send(200, json.dumps(res, ensure_ascii=False).encode(), "application/json")
        except Exception as e:  # unreadable image etc.
            self._send(400, json.dumps({"error": f"could not process image: {e}"}).encode(), "application/json")

    def log_message(self, fmt, *args):
        print("%s %s" % (self.command, self.path), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--weights", default=os.path.join(ROOT, "results", "baseline", "best.weights.h5"))
    a = ap.parse_args()
    load_model(a.weights)
    print(f"Braille reader on http://{a.host}:{a.port}", flush=True)
    HTTPServer((a.host, a.port), Handler).serve_forever()
