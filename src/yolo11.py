"""YOLO11-s re-implemented in Keras 3 / TensorFlow.

Mirrors ultralytics/cfg/models/11/yolo11.yaml at scale 's' (depth 0.50, width 0.50,
max_channels 1024): Conv-BN-SiLU, C3k2, SPPF, C2PSA backbone; PAN neck; decoupled
anchor-free Detect head with Distribution Focal Loss (reg_max = 16) on strides 8/16/32.
"""
import math

import keras
import tensorflow as tf
from keras import layers, ops

BN_EPS, BN_MOM = 1e-3, 0.97  # = PyTorch eps 1e-3, momentum 0.03
REG_MAX = 16
STRIDES = (8, 16, 32)


def conv(x, c, k=1, s=1, act=True, name=None):
    pad = "same" if s == 1 else "valid"
    if s > 1:
        x = layers.ZeroPadding2D(k // 2, name=None if name is None else name + "_pad")(x)
    x = layers.Conv2D(c, k, s, padding=pad, use_bias=False, name=name)(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOM)(x)
    return layers.Activation("silu")(x) if act else x


def dwconv(x, k=3, act=True):
    x = layers.DepthwiseConv2D(k, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOM)(x)
    return layers.Activation("silu")(x) if act else x


def bottleneck(x, c2, shortcut=True, e=0.5, k=3):
    c1 = x.shape[-1]
    y = conv(x, int(c2 * e), k)
    y = conv(y, c2, k)
    return layers.Add()([x, y]) if shortcut and c1 == c2 else y


def c3k(x, c2, n=2, shortcut=True, e=0.5):
    c_ = int(c2 * e)
    a = conv(x, c_, 1)
    for _ in range(n):
        a = bottleneck(a, c_, shortcut, e=1.0)
    b = conv(x, c_, 1)
    return conv(layers.Concatenate()([a, b]), c2, 1)


def c3k2(x, c2, n=1, use_c3k=False, e=0.5, shortcut=True):
    c = int(c2 * e)
    y = conv(x, 2 * c, 1)
    y1, y2 = layers.Lambda(lambda t: tf.split(t, 2, axis=-1))(y)
    outs = [y1, y2]
    cur = y2
    for _ in range(n):
        cur = c3k(cur, c, 2, shortcut) if use_c3k else bottleneck(cur, c, shortcut, e=0.5)
        outs.append(cur)
    return conv(layers.Concatenate()(outs), c2, 1)


def sppf(x, c2, k=5):
    c_ = x.shape[-1] // 2
    x = conv(x, c_, 1)
    y1 = layers.MaxPooling2D(k, 1, padding="same")(x)
    y2 = layers.MaxPooling2D(k, 1, padding="same")(y1)
    y3 = layers.MaxPooling2D(k, 1, padding="same")(y2)
    return conv(layers.Concatenate()([x, y1, y2, y3]), c2, 1)


class Attention(layers.Layer):
    """Multi-head self-attention of YOLO11 C2PSA, with a depthwise positional encoding on V."""

    def __init__(self, dim, num_heads, attn_ratio=0.5, **kw):
        super().__init__(**kw)
        self.dim, self.nh = dim, num_heads
        self.hd = dim // num_heads
        self.kd = int(self.hd * attn_ratio)
        self.scale = self.kd ** -0.5
        h = dim + 2 * self.kd * num_heads
        self.qkv = keras.Sequential([layers.Conv2D(h, 1, use_bias=False),
                                     layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOM)])
        self.proj = keras.Sequential([layers.Conv2D(dim, 1, use_bias=False),
                                      layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOM)])
        self.pe = keras.Sequential([layers.DepthwiseConv2D(3, padding="same", use_bias=False),
                                    layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOM)])

    def call(self, x, training=None):
        B, H, W = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        N = H * W
        qkv = ops.reshape(self.qkv(x, training=training), (B, N, self.nh, 2 * self.kd + self.hd))
        q, k, v = qkv[..., :self.kd], qkv[..., self.kd:2 * self.kd], qkv[..., 2 * self.kd:]
        q, k, v = (ops.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))  # B, nh, N, d
        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1)
        out = ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3))  # B, N, nh, hd
        out = ops.reshape(out, (B, H, W, self.dim))
        v_img = ops.reshape(ops.transpose(v, (0, 2, 1, 3)), (B, H, W, self.dim))
        return self.proj(out + self.pe(v_img, training=training), training=training)


def c2psa(x, c2, n=1, e=0.5):
    c = int(c2 * e)
    y = conv(x, 2 * c, 1)
    a, b = layers.Lambda(lambda t: tf.split(t, 2, axis=-1))(y)
    for _ in range(n):
        b = layers.Add()([b, Attention(c, max(c // 64, 1))(b)])
        f = conv(b, 2 * c, 1)
        f = conv(f, c, 1, act=False)
        b = layers.Add()([b, f])
    return conv(layers.Concatenate()([a, b]), c2, 1)


def detect_head(feats, nc):
    c2 = max(16, feats[0].shape[-1] // 4, REG_MAX * 4)
    c3 = max(feats[0].shape[-1], min(nc, 100))
    outs = []
    for x, s in zip(feats, STRIDES):
        bx = conv(x, c2, 3)
        bx = conv(bx, c2, 3)
        bx = layers.Conv2D(4 * REG_MAX, 1, bias_initializer=keras.initializers.Constant(1.0))(bx)
        cl = conv(dwconv(x, 3), c3, 1)
        cl = conv(dwconv(cl, 3), c3, 1)
        # prior: ~5 objects per 640x640 image, as in ultralytics Detect.bias_init
        prior = math.log(5 / nc / (640 / s) ** 2)
        cl = layers.Conv2D(nc, 1, bias_initializer=keras.initializers.Constant(prior))(cl)
        o = layers.Concatenate()([bx, cl])
        outs.append(layers.Reshape((-1, 4 * REG_MAX + nc))(o))
    out = layers.Concatenate(axis=1)(outs)
    return layers.Activation("linear", dtype="float32", name="raw")(out)


def build_yolo11s(nc=63, input_shape=(None, None, 3)):
    """Returns a Keras model: uint8/float image (B,H,W,3) in [0,255] -> (B, A, 64 + nc) raw logits."""
    w = lambda c: min(c, 1024) // 2  # width multiple 0.5
    inp = keras.Input(input_shape)
    x = layers.Rescaling(1 / 255.0)(inp)
    x = conv(x, w(64), 3, 2)                       # 0  P1/2
    x = conv(x, w(128), 3, 2)                      # 1  P2/4
    x = c3k2(x, w(256), 1, False, e=0.25)          # 2
    x = conv(x, w(256), 3, 2)                      # 3  P3/8
    p3 = c3k2(x, w(512), 1, False, e=0.25)         # 4
    x = conv(p3, w(512), 3, 2)                     # 5  P4/16
    p4 = c3k2(x, w(512), 1, True)                  # 6
    x = conv(p4, w(1024), 3, 2)                    # 7  P5/32
    x = c3k2(x, w(1024), 1, True)                  # 8
    x = sppf(x, w(1024), 5)                        # 9
    p5 = c2psa(x, w(1024), 1)                      # 10
    # PAN neck
    x = layers.UpSampling2D(2)(p5)                 # 11
    x = layers.Concatenate()([x, p4])              # 12
    h4 = c3k2(x, w(512), 1, False)                 # 13
    x = layers.UpSampling2D(2)(h4)                 # 14
    x = layers.Concatenate()([x, p3])              # 15
    o3 = c3k2(x, w(256), 1, False)                 # 16 (P3/8 out)
    x = conv(o3, w(256), 3, 2)                     # 17
    x = layers.Concatenate()([x, h4])              # 18
    o4 = c3k2(x, w(512), 1, False)                 # 19 (P4/16 out)
    x = conv(o4, w(512), 3, 2)                     # 20
    x = layers.Concatenate()([x, p5])              # 21
    o5 = c3k2(x, w(1024), 1, True)                 # 22 (P5/32 out)
    out = detect_head([o3, o4, o5], nc)            # 23
    return keras.Model(inp, out, name="yolo11s")


def make_anchors(h, w):
    """Anchor centres (A,2) in pixels and strides (A,1) for an input of size h x w."""
    pts, strd = [], []
    for s in STRIDES:
        gh, gw = h // s, w // s
        sy, sx = tf.meshgrid(tf.range(gh, dtype=tf.float32) + 0.5, tf.range(gw, dtype=tf.float32) + 0.5, indexing="ij")
        pts.append(tf.reshape(tf.stack([sx, sy], -1), (-1, 2)) * s)
        strd.append(tf.fill((gh * gw, 1), float(s)))
    return tf.concat(pts, 0), tf.concat(strd, 0)


def decode(raw, anchors, strides):
    """raw (B,A,64+nc) -> boxes xyxy pixels (B,A,4), class logits (B,A,nc), dist probs (B,A,4,16)."""
    dist = tf.reshape(raw[..., :4 * REG_MAX], (tf.shape(raw)[0], -1, 4, REG_MAX))
    prob = tf.nn.softmax(dist, -1)
    ltrb = tf.reduce_sum(prob * tf.range(REG_MAX, dtype=tf.float32), -1) * strides  # pixels
    boxes = tf.concat([anchors - ltrb[..., :2], anchors + ltrb[..., 2:]], -1)
    return boxes, raw[..., 4 * REG_MAX:], dist


if __name__ == "__main__":
    m = build_yolo11s()
    print(f"params: {m.count_params() / 1e6:.2f} M")
    y = m(tf.zeros((1, 640, 640, 3)))
    print(y.shape)
