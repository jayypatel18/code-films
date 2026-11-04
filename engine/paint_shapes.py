"""Shape helpers for composing painted references: hand wobbled outlines,
antialiased masks, and simple lighting on primitive forms."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from engine.noise import rng, value_noise


def resample(pts, spacing, closed=False):
    """Resample a polyline to roughly even spacing."""
    pts = np.asarray(pts, np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return pts.copy()
    n = max(int(total / spacing), 2)
    s = np.linspace(0, total, n + 1)
    if closed:
        s = s[:-1]
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], 1)


def wobble(pts, amp, wavelength, key, closed=False, spacing=None):
    """Displace a polyline along its normals with smooth noise, like a hand
    drawn line. amp in px, wavelength in px along the line."""
    p = resample(pts, spacing or max(wavelength / 6.0, 1.0), closed)
    n = len(p)
    if closed:
        d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        d = np.gradient(p, axis=0)
    ln = np.sqrt((d ** 2).sum(1)) + 1e-9
    nx, ny = -d[:, 1] / ln, d[:, 0] / ln
    seg = np.sqrt(((np.diff(p, axis=0)) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    r = rng("wobble", key)
    k = max(int(s[-1] / wavelength) + 4, 4)
    knots = r.standard_normal(k)
    knots2 = r.standard_normal(k * 3)
    off = np.interp(s / wavelength, np.arange(k), knots) + 0.35 * np.interp(s / wavelength * 3, np.arange(k * 3), knots2)
    if closed and n > 2:
        # blend the seam so a closed outline joins without a step
        t = np.clip(s / max(s[-1], 1e-6), 0, 1)
        off = off * (1 - t) + off[0] * t * (t > 0.85) + off * t * (t <= 0.85)
    out = p.copy()
    out[:, 0] += nx * off * amp
    out[:, 1] += ny * off * amp
    return out


def poly_mask(h, w, polys, ss=4, value=1.0):
    """Antialiased filled polygon(s) as float32 (h, w)."""
    if isinstance(polys, np.ndarray) and polys.ndim == 2:
        polys = [polys]
    im = Image.new("L", (w * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    for p in polys:
        p = np.asarray(p, np.float64)
        if len(p) < 3:
            continue
        dr.polygon([(float(x * ss), float(y * ss)) for x, y in p], fill=255)
    im = im.resize((w, h), Image.BOX)
    return (np.asarray(im, np.float32) / 255.0 * value).astype(np.float32)


def ellipse_mask(h, w, cx, cy, rx, ry, ss=4):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).ellipse([(cx - rx) * ss, (cy - ry) * ss, (cx + rx) * ss, (cy + ry) * ss], fill=255)
    return (np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255.0).astype(np.float32)


def line_mask(h, w, pts, width, ss=4):
    im = Image.new("L", (w * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    p = [(float(x * ss), float(y * ss)) for x, y in np.asarray(pts)]
    dr.line(p, fill=255, width=max(int(width * ss), 1), joint="curve")
    return (np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255.0).astype(np.float32)


def soft(mask, sigma):
    return ndimage.gaussian_filter(mask, sigma).astype(np.float32)


def rough_edge(mask, amp, cell, key):
    """Roughen a mask's edge by warping it with noise (amp px)."""
    h, w = mask.shape
    dx = (value_noise(h, w, cell, ("re_x", key)) - 0.5) * 2 * amp
    dy = (value_noise(h, w, cell, ("re_y", key)) - 0.5) * 2 * amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ndimage.map_coordinates(mask, [yy + dy, xx + dx], order=1, mode="nearest").astype(np.float32)
