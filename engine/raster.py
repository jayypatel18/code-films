"""Rasterising: polygon masks and pressure strokes on a supersampled grid.

Fills are rasterised hard at the supersampled resolution and become
anti-aliased when the canvas is box-filtered down. Strokes are rasterised as
swept discs with analytic coverage, so lines stay smooth at any width.
"""

import math

import numpy as np
from numba import njit
from PIL import Image, ImageDraw


def poly_mask(polys, h, w, ox=0.0, oy=0.0):
    """Rasterise one or more polygons (canvas coords) into a bool (h, w) mask.

    ox, oy: the canvas coordinate of mask pixel (0, 0). Pixel centres are at
    integer + 0.5, PIL samples at integer corners, so we shift by 0.5.
    """
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    if isinstance(polys, np.ndarray) and polys.ndim == 2:
        polys = [polys]
    for P in polys:
        P = np.asarray(P, dtype=np.float64)
        if len(P) < 3:
            continue
        pts = [(float(x - ox - 0.5), float(y - oy - 0.5)) for x, y in P]
        dr.polygon(pts, fill=255)
    return np.asarray(img) > 127


@njit(cache=True)
def _stroke_kernel(buf, pts, rad, ox, oy, op):
    h, w = buf.shape
    n = pts.shape[0]
    for i in range(n - 1):
        ax = pts[i, 0] - ox
        ay = pts[i, 1] - oy
        bx = pts[i + 1, 0] - ox
        by = pts[i + 1, 1] - oy
        ar = rad[i]
        br = rad[i + 1]
        if ar <= 0.02 and br <= 0.02:
            continue
        mr = max(ar, br) + 1.5
        x0 = int(math.floor(min(ax, bx) - mr))
        x1 = int(math.ceil(max(ax, bx) + mr))
        y0 = int(math.floor(min(ay, by) - mr))
        y1 = int(math.ceil(max(ay, by) + mr))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h
        vx = bx - ax
        vy = by - ay
        vv = vx * vx + vy * vy
        for y in range(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                px = x + 0.5
                if vv > 1e-12:
                    t = ((px - ax) * vx + (py - ay) * vy) / vv
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                else:
                    t = 0.0
                qx = ax + vx * t - px
                qy = ay + vy * t - py
                d = math.sqrt(qx * qx + qy * qy)
                r = ar + (br - ar) * t
                c = r - d + 0.5
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                if r < 0.5:
                    # thin line: keep the ink density proportional to width
                    c = c * (r / 0.5)
                if op == 0:
                    if c > buf[y, x]:
                        buf[y, x] = c
                else:
                    buf[y, x] = buf[y, x] + c - buf[y, x] * c


def stroke(buf, pts, rad, ox=0.0, oy=0.0, accumulate=False):
    """Draw a pressure stroke into coverage buffer buf (float32, h x w).

    pts: (n, 2) canvas coords, rad: (n,) radius in canvas pixels.
    accumulate=False unions by max (one ink pass); True builds up like
    overlapping wet brush passes.
    """
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    rad = np.ascontiguousarray(rad, dtype=np.float64)
    if len(pts) < 2:
        return
    _stroke_kernel(buf, pts, rad, float(ox), float(oy), 1 if accumulate else 0)


def downsample(a, ss):
    """Box filter by an integer factor along the first two axes."""
    if ss == 1:
        return a
    h, w = a.shape[0] // ss, a.shape[1] // ss
    a = a[: h * ss, : w * ss]
    if a.ndim == 2:
        return a.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return a.reshape(h, ss, w, ss, a.shape[2]).mean(axis=(1, 3))


def shift(mask, dx, dy, fill=False):
    """Shift a 2D array by integer pixels (positive dx moves content right)."""
    out = np.full_like(mask, fill)
    h, w = mask.shape
    dx, dy = int(round(dx)), int(round(dy))
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs0 >= xs1 or ys0 >= ys1:
        return out
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = mask[ys0:ys1, xs0:xs1]
    return out
