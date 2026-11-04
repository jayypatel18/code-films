"""Finishing: grade, bloom, vignette, grain, dither, downsample, save."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from . import raster as R


def bloom(img, threshold=0.75, sigma=40, strength=0.35):
    lum = img.mean(axis=2)
    hi = np.clip((lum - threshold) / (1 - threshold), 0, 1)[..., None] * img
    b = np.stack([gaussian_filter(hi[..., c], sigma) for c in range(3)], axis=2)
    return img + b * strength


def vignette(img, strength=0.35, power=2.2):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - W / 2) / (W / 2)
    dy = (yy - H / 2) / (H / 2)
    d = np.sqrt(dx * dx + dy * dy) / 1.2
    v = 1 - strength * np.clip(d, 0, 1) ** power
    return img * v[..., None]


def grade(img, lift=(0, 0, 0), gain=(1, 1, 1), gamma=1.0, sat=1.0):
    out = img * np.asarray(gain, np.float32) + np.asarray(lift, np.float32)
    out = np.clip(out, 0, 1) ** gamma
    if sat != 1.0:
        l = out.mean(axis=2, keepdims=True)
        out = l + (out - l) * sat
    return np.clip(out, 0, 1)


def finish(img2x, seed, grain=0.035, vig=0.3, do_bloom=True, bloom_kw=None):
    """2x float canvas to 1x uint8 frame."""
    img = img2x
    if do_bloom:
        img = bloom(img, **(bloom_kw or {}))
    img = vignette(img, vig)
    img = R.downsample(np.ascontiguousarray(img, np.float32), 2)
    img = np.clip(img, 0, 1).astype(np.float32)
    return R.grain_dither(img, seed, grain)


def save(arr, path):
    Image.fromarray(arr).save(path)
