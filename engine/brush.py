"""Ink brush on paper: pressure, pooling where the brush lands, dry brush where it
lifts, bleeding into the fibres, and strokes that write themselves on.

A stroke is a centre line with a radius per point. stroke() rasterises it into
two buffers: coverage (how much ink) and density (how dark: pooled ink is
darker). Only the part of the stroke up to `reveal` (0..1 of its length) is
drawn, so letters and the ink bird can write on frame by frame.
"""

import math

import numpy as np
from numba import njit
from scipy import ndimage

from . import geom


@njit(cache=True)
def _hash(i, j, seed):
    h = (i * 374761393 + j * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True)
def _lane(v, s, seed, lanes):
    """Bristle lanes across the brush: v in [-1, 1], s along the stroke (px)."""
    x = (v + 1.0) * 0.5 * lanes
    i = int(math.floor(x))
    f = x - i
    a = _hash(i, int(s * 0.02), seed)
    b = _hash(i + 1, int(s * 0.02), seed)
    w = f * f * (3 - 2 * f)
    return a + (b - a) * w


@njit(cache=True)
def _kernel(cov, den, pts, rad, sarc, smax, total, ox, oy, seed, dry0, dryk, pool, lanes):
    h, w = cov.shape
    n = pts.shape[0]
    for k in range(n - 1):
        if sarc[k] > smax:
            break
        ax = pts[k, 0] - ox
        ay = pts[k, 1] - oy
        bx = pts[k + 1, 0] - ox
        by = pts[k + 1, 1] - oy
        ra = rad[k]
        rb = rad[k + 1]
        sa = sarc[k]
        sb = sarc[k + 1]
        mr = max(ra, rb) + 2.0
        x0 = max(0, int(math.floor(min(ax, bx) - mr)))
        x1 = min(w, int(math.ceil(max(ax, bx) + mr)))
        y0 = max(0, int(math.floor(min(ay, by) - mr)))
        y1 = min(h, int(math.ceil(max(ay, by) + mr)))
        vx = bx - ax
        vy = by - ay
        vv = vx * vx + vy * vy
        for y in range(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                px = x + 0.5
                t = 0.0
                if vv > 1e-12:
                    t = ((px - ax) * vx + (py - ay) * vy) / vv
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                s = sa + (sb - sa) * t
                if s > smax:
                    continue
                qx = ax + vx * t - px
                qy = ay + vy * t - py
                d = math.sqrt(qx * qx + qy * qy)
                r = ra + (rb - ra) * t
                c = r - d + 0.7
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                u = s / total
                # signed position across the brush
                side = (vx * qy - vy * qx) / (math.sqrt(vv) + 1e-9)
                v = side / (r + 1e-6)
                # dry brush: toward the end the bristles separate into streaks
                if u > dry0:
                    dk = (u - dry0) / max(1e-6, 1.0 - dry0) * dryk
                    ln = _lane(v, s, seed, lanes)
                    if ln < dk:
                        c *= max(0.0, 1.0 - (dk - ln) * 6.0)
                # pooling: dense where the brush lands, a little at the wet front
                dd = 0.72 + pool * math.exp(-s / (r * 2.5 + 1e-6)) + 0.12 * _hash(int(s * 0.3), int(v * 4 + 9), seed)
                if c > cov[y, x]:
                    cov[y, x] = c
                if dd * c > den[y, x]:
                    den[y, x] = dd * c


def stroke(cov, den, pts, rad, reveal=1.0, ox=0.0, oy=0.0, seed=0, dry0=0.72, dryk=0.9, pool=0.35, lanes=9.0,
           spacing=1.0):
    """Draw one brush stroke into coverage and density buffers (float32, same shape)."""
    P = np.asarray(pts, dtype=np.float64)
    R = np.asarray(rad, dtype=np.float64)
    if len(P) < 2 or reveal <= 0:
        return
    P, R = geom.resample(P, spacing, extra=R)
    s = geom.arclen(P)
    total = max(s[-1], 1e-6)
    smax = reveal * total
    _kernel(cov, den, np.ascontiguousarray(P), np.ascontiguousarray(R), np.ascontiguousarray(s),
            float(smax), float(total), float(ox), float(oy), int(seed) & 0x7FFFFFFF, float(dry0), float(dryk),
            float(pool), float(lanes))


def bleed(cov, fibre=None, amount=0.6, sigma=0.7):
    """Ink soaks a little into the paper: soften the edge and let fibres catch it."""
    c = ndimage.gaussian_filter(cov, sigma) if sigma > 0 else cov.copy()
    if fibre is not None:
        edge = np.clip(c * (1 - c) * 4.0, 0, 1)
        c = np.clip(c + amount * 0.35 * (fibre - 0.5) * edge, 0, 1)
    return np.maximum(c, cov * 0.92)


def pressure(u, land=0.12, lift=0.30, base=1.0, swell=0.12):
    """A brush pressure profile along a stroke: lands, swells, lifts."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0, 1)
    a = np.clip(u / land, 0, 1) ** 0.6
    b = np.clip((1 - u) / lift, 0, 1) ** 0.8
    return base * (0.25 + 0.75 * np.minimum(a, b)) * (1 + swell * np.sin(math.pi * u))
