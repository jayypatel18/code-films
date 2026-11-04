"""Compositing of premultiplied RGBA layers (float32, display space, 0..1)."""

import numpy as np


def empty(h=1080, w=1920):
    return np.zeros((h, w, 4), dtype=np.float32)


def place(dst, tile, x0, y0):
    """Composite a premultiplied tile over dst with its top-left at (x0, y0)."""
    H, W = dst.shape[:2]
    h, w = tile.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return dst
    t = tile[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    d = dst[dy0:dy1, dx0:dx1]
    d *= (1.0 - t[..., 3:4])
    d += t
    return dst


def over(dst, src):
    """dst = src over dst, in place. Both premultiplied RGBA."""
    dst *= (1.0 - src[..., 3:4])
    dst += src
    return dst


def add(dst_rgb, glow_rgb, amount=1.0):
    dst_rgb += glow_rgb * amount
    return dst_rgb


def screen(dst_rgb, glow_rgb, amount=1.0):
    g = np.clip(glow_rgb * amount, 0.0, 1.0)
    dst_rgb[...] = 1.0 - (1.0 - np.clip(dst_rgb, 0.0, 1.0)) * (1.0 - g)
    return dst_rgb


def flatten(layer, bg=(0.0, 0.0, 0.0)):
    """Premultiplied RGBA over a solid colour -> RGB."""
    return layer[..., :3] + (1.0 - layer[..., 3:4]) * np.asarray(bg, dtype=np.float32)
