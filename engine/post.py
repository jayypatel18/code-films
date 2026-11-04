"""Finishing, like an anime compositing department: glow, grade, bloom,
vignette, film grain, dithering to 8 bit.

All functions take and return float32 RGB (H, W, 3) in display space.
"""

import math

import numpy as np
from scipy import ndimage

from . import vnoise


def radial_glow(h, w, x, y, radius, rgb, strength=1.0, falloff=2.0):
    """Soft additive light around a point (lantern, sun): a sum of gaussians."""
    out = np.zeros((h, w, 3), dtype=np.float32)
    r = max(radius, 1.0)
    x0, x1 = int(max(0, x - r * 6)), int(min(w, x + r * 6))
    y0, y1 = int(max(0, y - r * 6)), int(min(h, y + r * 6))
    if x1 <= x0 or y1 <= y0:
        return out
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = ((xs - x) ** 2 + (ys - y) ** 2) / (r * r)
    g = 0.55 * np.exp(-d2 * 1.6) + 0.30 * np.exp(-d2 * 0.30) + 0.15 * np.exp(-d2 * 0.06)
    out[y0:y1, x0:x1] = (g * strength)[..., None] * np.asarray(rgb, dtype=np.float32)
    return out


def bloom(rgb, threshold=0.72, strength=0.35, sigmas=(6, 18, 48), tint=(1.0, 0.92, 0.82)):
    """Bright areas bleed soft light, done at reduced resolution for speed."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    k = np.clip((lum - threshold) / max(1e-3, 1 - threshold), 0, 1)[..., None]
    src = rgb * k
    small = src[::4, ::4]
    acc = np.zeros_like(small)
    for s in sigmas:
        acc += ndimage.gaussian_filter(small, (s / 4.0, s / 4.0, 0))
    acc /= len(sigmas)
    up = np.repeat(np.repeat(acc, 4, axis=0), 4, axis=1)[: rgb.shape[0], : rgb.shape[1]]
    up = ndimage.uniform_filter(up, (4, 4, 1))
    return rgb + up * strength * np.asarray(tint, dtype=np.float32)


def grade(rgb, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1), sat=1.0, contrast=1.0, pivot=0.45):
    """Lift, gamma, gain per channel, then saturation and contrast around a pivot."""
    x = np.clip(rgb, 0, None)
    lift = np.asarray(lift, np.float32)
    gain = np.asarray(gain, np.float32)
    gamma = np.asarray(gamma, np.float32)
    x = x * gain + lift * (1 - x)
    x = np.clip(x, 0, None) ** (1.0 / gamma)
    if contrast != 1.0:
        x = (x - pivot) * contrast + pivot
    if sat != 1.0:
        lum = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        x = lum + (x - lum) * sat
    return x.astype(np.float32)


def vignette(rgb, amount=0.22, power=2.2, tint=(0.06, 0.05, 0.10)):
    h, w = rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - w / 2) / (w / 2)
    ny = (ys - h / 2) / (h / 2)
    d = np.sqrt(nx * nx * 0.85 + ny * ny * 1.05)
    v = np.clip(d / 1.25, 0, 1) ** power * amount
    t = np.asarray(tint, np.float32)
    return rgb * (1 - v[..., None]) + t * v[..., None] * 0.5


def grain(rgb, frame, amount=0.018, key="grain"):
    """Fine animated film grain, a different pattern every frame, luminance weighted."""
    h, w = rgb.shape[:2]
    r = vnoise.rng(key, frame)
    g = r.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)[:h, :w]
    g = ndimage.uniform_filter(g, 2) * 1.6
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    wgt = 0.6 + 0.8 * lum * (1 - lum) * 2
    return rgb + (g * wgt * amount)[..., None]


def to8(rgb, frame=0, key="dither"):
    """Triangular dither then quantise, so gradients never band."""
    h, w = rgb.shape[:2]
    r = vnoise.rng(key, frame)
    d = (r.random((h, w, 1), dtype=np.float32) + r.random((h, w, 1), dtype=np.float32) - 1.0) / 255.0
    return (np.clip(rgb + d, 0, 1) * 255.0 + 0.5).astype(np.uint8)
