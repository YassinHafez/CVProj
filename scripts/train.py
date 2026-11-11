"""Train YOLO11-s (Keras/TensorFlow) on Angelina Braille.

Example:
  python3 scripts/train.py --name baseline --epochs 150
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import CropSampler, load_index
from loss import yolo_loss
from metrics import best_conf, map_metrics, reading_metrics, run_inference
from yolo11 import build_yolo11s, make_anchors

NC = 63


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0]
    except Exception:
        return "?"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    p.add_argument("--index", default="data/index.json")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--crops-per-image", type=int, default=4, help="crops per train image per epoch")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-final", type=float, default=0.01, help="final lr as fraction of lr")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=3)
    p.add_argument("--clipnorm", type=float, default=10.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--scale", type=float, nargs=2, default=(0.75, 1.25))
    p.add_argument("--degrees", type=float, default=3.0)
    p.add_argument("--blur-p", type=float, default=0.15)
    p.add_argument("--max-boxes", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--rare-oversample", type=float, default=0.0,
                   help="improvement: repeat-factor sampling threshold (0 = off)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-xla", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def repeat_factor_weights(items, thr):
    """LVIS repeat-factor sampling: r_c = max(1, sqrt(thr / f_c)), image weight = max_c r_c."""
    n = len(items)
    img_freq = np.zeros(NC)
    for it in items:
        for c in set(it["labels"]):
            img_freq[c] += 1
    f = img_freq / n
    r = np.where(f > 0, np.maximum(1.0, np.sqrt(thr / np.maximum(f, 1e-12))), 1.0)
    return [max([r[c] for c in set(it["labels"])], default=1.0) for it in items]


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, total, warmup, final_frac):
        self.lr, self.total, self.warmup, self.final = lr, float(total), float(warmup), final_frac

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.lr * step / max(self.warmup, 1.0)
        prog = tf.clip_by_value((step - self.warmup) / max(self.total - self.warmup, 1.0), 0.0, 1.0)
        cos = self.lr * (self.final + (1 - self.final) * 0.5 * (1 + tf.cos(math.pi * prog)))
        return tf.where(step < self.warmup, warm, cos)


def main():
    a = parse()
    out_dir = os.path.join("results", a.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.utils.set_random_seed(a.seed)
    if not a.no_amp:
        keras.mixed_precision.set_global_policy("mixed_float16")

    index = load_index(a.index)
    train_items, val_items = index["train"], index["val"]
    weights = repeat_factor_weights(train_items, a.rare_oversample) if a.rare_oversample > 0 else None
    sampler = CropSampler(train_items, a.imgsz, a.max_boxes, tuple(a.scale), a.degrees, a.blur_p,
                          seed=a.seed, sample_weights=weights)
    steps = math.ceil(a.crops_per_image * len(train_items) / a.batch)
    total_steps = steps * a.epochs
    print(f"train images {len(train_items)}, {steps} steps/epoch, {total_steps} total steps", flush=True)

    def gen_np(i):
        return sampler.sample(seed=a.seed * 10_000_000 + int(i))

    def tf_sample(i):
        img, b, l, m = tf.numpy_function(gen_np, [i], [tf.uint8, tf.float32, tf.int32, tf.float32])
        img.set_shape((a.imgsz, a.imgsz, 3)); b.set_shape((a.max_boxes, 4))
        l.set_shape((a.max_boxes,)); m.set_shape((a.max_boxes,))
        return img, b, l, m

    ds = (tf.data.Dataset.range(total_steps * a.batch)
          .map(tf_sample, num_parallel_calls=6, deterministic=True)
          .batch(a.batch, drop_remainder=True).prefetch(4))

    model = build_yolo11s(NC)
    ema = build_yolo11s(NC)
    ema.set_weights(model.get_weights())
    print(f"params {model.count_params() / 1e6:.2f} M", flush=True)

    sched = WarmupCosine(a.lr, total_steps, a.warmup_epochs * steps, a.lr_final)
    opt = keras.optimizers.AdamW(sched, weight_decay=a.weight_decay, global_clipnorm=a.clipnorm)
    opt.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])
    if not a.no_amp:
        opt = keras.optimizers.LossScaleOptimizer(opt)
    opt.build(model.trainable_variables)

    anchors, strides = make_anchors(a.imgsz, a.imgsz)
    step_var = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function(jit_compile=not a.no_xla)
    def train_step(img, boxes, labels, mask):
        with tf.GradientTape() as tape:
            raw = model(img, training=True)
            loss, parts = yolo_loss(raw, anchors, strides, labels, boxes, mask, NC)
            scaled = opt.scale_loss(loss) if not a.no_amp else loss
        grads = tape.gradient(scaled, model.trainable_variables)
        opt.apply(grads, model.trainable_variables)
        # EMA over all weights (incl. BN statistics), ramped like ultralytics ModelEMA (tau=2000)
        step_var.assign_add(1)
        d = a.ema_decay * (1 - tf.exp(-tf.cast(step_var, tf.float32) / 2000.0))
        for ve, vm in zip(ema.weights, model.weights):
            ve.assign(d * ve + (1 - d) * tf.cast(vm, ve.dtype))
        return parts

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "box", "cls", "dfl", "lr", "sec", "val_map50", "val_map50_95",
                                "val_char_F1", "val_macro_F1", "val_conf"])
    best = -1.0
    it = iter(ds)
    print(f"device: {model.trainable_variables[0].value.device}", flush=True)
    print("compiling the XLA training graph (~1.5 min, the GPU is idle during this)...", flush=True)
    t0 = time.time()
    first = train_step(*next(it)).numpy()
    print(f"compiled in {time.time() - t0:.0f}s, training starts", flush=True)
    bar = tqdm(total=total_steps, initial=1, unit="step", dynamic_ncols=True, smoothing=0.05,
               bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    gpu = gpu_util()
    for ep in range(1, a.epochs + 1):
        bar.set_description(f"epoch {ep}/{a.epochs}")
        t0 = time.time()
        n_ep = steps - 1 if ep == 1 else steps  # step 1 already ran during compilation
        acc_t = tf.constant(first) if ep == 1 else tf.zeros(3)
        for k in range(n_ep):
            parts = train_step(*next(it))
            acc_t += parts
            bar.update(1)
            if k % 20 == 0:  # sync with the GPU only every 20 steps
                p = parts.numpy()
                bar.set_postfix_str(f"box {p[0]:.2f} cls {p[1]:.2f} dfl {p[2]:.2f} | gpu {gpu}% | best mAP50 {max(best, 0):.3f}")
        acc = acc_t.numpy() / steps
        gpu = gpu_util()
        lr = float(sched(opt.iterations if a.no_amp else opt.inner_optimizer.iterations))
        dt = time.time() - t0
        row = [ep, *np.round(acc, 4), f"{lr:.2e}", round(dt, 1)]
        msg = f"ep {ep:3d}/{a.epochs} box {acc[0]:.3f} cls {acc[1]:.3f} dfl {acc[2]:.3f} lr {lr:.2e} {dt:.0f}s"
        if ep % a.eval_every == 0 or ep == a.epochs:
            bar.set_postfix_str("evaluating on val...")
            res = run_inference(ema, val_items)
            mp = map_metrics(res)
            c = best_conf(res)
            rm = reading_metrics(res, c)
            row += [round(mp["map50"], 4), round(mp["map50_95"], 4), round(rm["char_F1"], 4),
                    round(rm["macro_F1"], 4), c]
            msg += (f" | val mAP50 {mp['map50']:.4f} mAP50-95 {mp['map50_95']:.4f} "
                    f"charF1 {rm['char_F1']:.4f} macroF1 {rm['macro_F1']:.4f} @conf {c:.3f}")
            if mp["map50"] > best:
                best = mp["map50"]
                ema.save_weights(os.path.join(out_dir, "best.weights.h5"))
                msg += " *"
        bar.write(msg)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        el = bar.format_dict["elapsed"]
        done = bar.n / total_steps
        with open(os.path.join(out_dir, "progress.txt"), "w") as f:
            f.write(f"epoch {ep}/{a.epochs} ({100 * done:.1f}%), elapsed {el / 60:.1f} min, "
                    f"remaining ~{el * (1 - done) / max(done, 1e-9) / 60:.1f} min, gpu {gpu}%, best val mAP50 {best:.4f}\n")
    bar.close()
    ema.save_weights(os.path.join(out_dir, "last.weights.h5"))
    if sampler.truncated:
        print(f"note: {sampler.truncated} crops had > {a.max_boxes} boxes and were truncated")


if __name__ == "__main__":
    main()
