"""Cel layer: flat colour shapes, one hard shadow tone, pressure inked outlines."""

import numpy as np
from . import raster as R
from .raster import vnoise
from .geom import catmull, wobble, resample, stroke_outline, pressure, polyline_length

INK = (0.19, 0.10, 0.07)


class Cel:
    """An RGBA drawing surface in canvas pixels (already at working scale)."""

    def __init__(self, w, h, scale=1.0):
        self.w = w
        self.h = h
        self.img = np.zeros((h, w, 3), np.float32)
        self.alpha = np.zeros((h, w), np.float32)
        self.scale = scale  # pixels per unit for the drawing helpers

    # ---- shapes -------------------------------------------------------
    def fill(self, poly, color, a=1.0):
        return R.fill_poly(self.img, poly, color, a, self.alpha)

    def fill_mask(self, m, color, a=1.0):
        R.fill_mask(self.img, m, color, a, self.alpha)

    def mask(self, poly):
        return R.poly_mask(poly, self.w, self.h)

    def shade(self, m, color, dx, dy, a=1.0, inset=0.0):
        """Hard edged crescent shadow on the side away from (dx, dy)."""
        if m is None:
            return None
        sub, x0, y0 = m
        sh = shift_sub(sub, dx, dy)
        cres = sub * (1.0 - sh)
        if inset > 0:
            cres = cres * shift_sub(sub, -dx * inset, -dy * inset)
        cm = (cres, x0, y0)
        self.fill_mask(cm, color, a)
        return cm

    def rim(self, m, color, dx, dy, a=1.0):
        """Thin light crescent on the side toward (dx, dy)."""
        return self.shade(m, color, -dx, -dy, a)

    def paint_in(self, m, poly, color, a=1.0):
        """Fill poly clipped by mask m (sub mask)."""
        m2 = self.mask(poly)
        if m2 is None or m is None:
            return
        sub, x0, y0 = m
        sub2, x2, y2 = m2
        # intersect in sub2's frame
        full = np.zeros((self.h, self.w), np.float32)
        full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
        inter = sub2 * full[y2:y2 + sub2.shape[0], x2:x2 + sub2.shape[1]]
        self.fill_mask((inter, x2, y2), color, a)

    def mult_in(self, m, poly, tint, a=0.8):
        """Multiply tint inside mask m intersected with poly: a cast or form shadow shape."""
        m2 = self.mask(poly)
        if m2 is None or m is None:
            return
        sub, x0, y0 = m
        sub2, x2, y2 = m2
        full = np.zeros((self.h, self.w), np.float32)
        full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
        inter = np.clip(sub2 * full[y2:y2 + sub2.shape[0], x2:x2 + sub2.shape[1]] * 1.6, 0, 1) * a
        reg = self.img[y2:y2 + sub2.shape[0], x2:x2 + sub2.shape[1]]
        t = np.asarray(tint, np.float32)
        reg[:] = reg * (1 - inter[..., None]) + reg * t * inter[..., None]

    # ---- ink ----------------------------------------------------------
    def ink(self, pts, w, color=INK, seed=0, taper_in=0.3, taper_out=0.3, var=0.3,
            wob=0.0, a=1.0, min_frac=0.12):
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        step = max(w * 0.6, 1.5)
        pts = resample(pts, step)
        if wob > 0:
            pts = wobble(pts, wob, seed)
        widths = pressure(len(pts), w, taper_in, taper_out, seed, var, min_frac)
        poly = stroke_outline(pts, widths)
        if poly is not None:
            R.fill_poly(self.img, poly, color, a, self.alpha)

    def ink_closed(self, poly, w, light=(-0.6, -0.8), color=INK, seed=0, gap=True,
                   wob=0.0, shadow_boost=0.7, lit_thin=0.5, a=1.0):
        """Outline of a closed shape: thicker on the shadow side, thin and broken on the lit edge."""
        pts = np.asarray(poly, np.float64)
        pts = np.concatenate([pts, pts[:1]], axis=0)
        step = max(w * 0.6, 1.5)
        pts = resample(pts, step)
        if wob > 0:
            pts = wobble(pts, wob, seed)
        n = len(pts)
        if n < 4:
            return
        d = np.gradient(pts, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        L[L < 1e-9] = 1e-9
        nx = -d[:, 1] / L
        ny = d[:, 0] / L
        # outward normals: polygon orientation check
        area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        if area > 0:
            nx, ny = -nx, -ny
        lx, ly = light
        ln = np.hypot(lx, ly) + 1e-9
        lit = (nx * lx + ny * ly) / ln  # +1 facing the light
        base = w * (1.0 + shadow_boost * np.clip(-lit, 0, 1)) * (1.0 - (1 - lit_thin) * np.clip(lit, 0, 1) ** 2)
        noise = np.array([R.vnoise(i * 0.06, 0.5, seed + 31) for i in range(n)])
        base *= 1 + 0.35 * (noise - 0.5) * 2
        if gap:
            # rotate so the stroke starts and ends at the most lit point: one hand lifted gap
            k = int(np.argmax(lit))
            pts = np.concatenate([pts[k:-1], pts[:k + 1]], axis=0)
            base = np.concatenate([base[k:-1], base[:k + 1]])
            n = len(pts)
            t = np.linspace(0, 1, n)
            tap = np.clip(t / 0.06, 0, 1) ** 0.8 * np.clip((1 - t) / 0.06, 0, 1) ** 0.8
            base = base * (0.1 + 0.9 * tap)
        po = stroke_outline(pts, base)
        if po is not None:
            R.fill_poly(self.img, po, color, a, self.alpha)

    # ---- composite ----------------------------------------------------
    def over(self, dst, ox=0, oy=0, tint=None, a=1.0):
        """Composite this cel onto dst (H,W,3) at integer offset."""
        H, W = dst.shape[:2]
        x0 = max(0, ox)
        y0 = max(0, oy)
        x1 = min(W, ox + self.w)
        y1 = min(H, oy + self.h)
        if x1 <= x0 or y1 <= y0:
            return
        src = self.img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        al = self.alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox][..., None] * a
        if tint is not None:
            src = src * np.asarray(tint, np.float32)
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - al) + src * al


def shift_sub(sub, dx, dy):
    out = np.zeros_like(sub)
    h, w = sub.shape
    dx = int(round(dx))
    dy = int(round(dy))
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)
    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = sub[sy0:sy1, sx0:sx1]
    return out


def tufted(points, n=10, tufts=(), seed=0, amp=0.0):
    """Closed catmull outline through points with optional fur tufts.
    tufts: list of (index_fraction, length, width) spikes pushed outward."""
    base = catmull(points, closed=True, n=n)
    if amp > 0:
        base = wobble(base, amp, seed, closed=True)
    if not tufts:
        return base
    m = len(base)
    cx, cy = base.mean(axis=0)
    out = []
    marks = {}
    for (f, ln, wd) in tufts:
        marks[int(f * m) % m] = (ln, wd)
    for i in range(m):
        out.append(base[i])
        if i in marks:
            ln, wd = marks[i]
            p = base[i]
            q = base[(i + max(1, int(wd))) % m]
            mid = (p + q) * 0.5
            dx, dy = mid[0] - cx, mid[1] - cy
            L = np.hypot(dx, dy) + 1e-9
            tip = mid + np.array([dx / L, dy / L]) * ln
            out.append(tip)
    return np.array(out)


def fluffy(points, seed=0, amp=0.06, step=0.05, spikes=1.0, n=8, wob=0.015):
    """Closed catmull outline through points, then fur: irregular outward bumps all round."""
    base = catmull(points, closed=True, n=n)
    base = resample(np.concatenate([base, base[:1]]), step)[:-1]
    m = len(base)
    nxt = np.roll(base, -1, axis=0)
    prv = np.roll(base, 1, axis=0)
    d = nxt - prv
    L = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = d[:, 1] / L, -d[:, 0] / L
    area = 0.5 * np.sum(base[:, 0] * nxt[:, 1] - nxt[:, 0] * base[:, 1])
    if area > 0:
        nx, ny = -nx, -ny
    out = base.copy()
    for i in range(m):
        t = i * 0.55
        sp = vnoise(t, seed * 0.13 + 2.0, seed + 300)
        big = vnoise(i * 0.09, 3.3, seed + 302)  # which stretches carry tufts at all
        sp = max(0.0, sp - 0.58) * 2.6 * max(0.0, big - 0.35) * 1.8
        spike = amp * spikes * min(sp, 1.6) ** 1.1
        w = (vnoise(i * 0.12, 7.7, seed + 301) - 0.5) * 2 * wob
        out[i, 0] += nx[i] * (spike + w)
        out[i, 1] += ny[i] * (spike + w)
    return out
