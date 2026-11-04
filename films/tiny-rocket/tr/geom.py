"""Curves and stroke geometry for TINY ROCKET."""

import numpy as np
from .raster import vnoise


def catmull(points, closed=False, n=12, tension=0.5):
    """Catmull-Rom spline through points. Returns (M, 2) dense polyline."""
    P = np.asarray(points, np.float64)
    k = P.shape[0]
    if k < 2:
        return P.copy()
    if closed:
        idx = lambda i: P[i % k]
        segs = k
    else:
        idx = lambda i: P[min(max(i, 0), k - 1)]
        segs = k - 1
    out = []
    ts = np.linspace(0, 1, n, endpoint=False)
    for i in range(segs):
        p0, p1, p2, p3 = idx(i - 1), idx(i), idx(i + 1), idx(i + 2)
        m1 = tension * (p2 - p0)
        m2 = tension * (p3 - p1)
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out.append(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)
    if not closed:
        out.append(P[-1])
    return np.array(out)


def ellipse(cx, cy, rx, ry, n=48, rot=0.0, squash=0.0):
    """Closed polygon of a (slightly egg shaped) ellipse."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = rx * np.cos(t)
    y = ry * np.sin(t) * (1 + squash * np.cos(t))
    c, s = np.cos(rot), np.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], axis=1)


def wobble(pts, amp, seed, freq=0.02, closed=False):
    """Hand tremor: displace points by low frequency noise along the path."""
    pts = np.asarray(pts, np.float64)
    if amp <= 0 or len(pts) < 2:
        return pts.copy()
    d = np.diff(pts, axis=0)
    s = np.concatenate([[0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    out = pts.copy()
    for i in range(len(pts)):
        if closed:
            # make the noise periodic along a closed loop by sampling on a circle
            a = 2 * np.pi * s[i] / max(s[-1] + np.hypot(*(pts[0] - pts[-1])), 1e-6)
            nx = (vnoise(3 + 2 * np.cos(a), 3 + 2 * np.sin(a), seed) - 0.5)
            ny = (vnoise(7 + 2 * np.cos(a), 7 + 2 * np.sin(a), seed + 5) - 0.5)
        else:
            nx = vnoise(s[i] * freq, 0.37, seed) - 0.5
            ny = vnoise(s[i] * freq, 9.13, seed + 5) - 0.5
        out[i, 0] += nx * 2 * amp
        out[i, 1] += ny * 2 * amp
    return out


def resample(pts, step):
    """Resample a polyline at roughly uniform arc length."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts.copy()
    d = np.diff(pts, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return pts[:1].copy()
    n = max(int(total / step) + 1, 2)
    ss = np.linspace(0, total, n)
    x = np.interp(ss, s, pts[:, 0])
    y = np.interp(ss, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def stroke_outline(pts, widths):
    """Polygon for a polyline with per vertex width (round caps approximated)."""
    pts = np.asarray(pts, np.float64)
    widths = np.asarray(widths, np.float64)
    n = len(pts)
    if n < 2:
        return None
    d = np.gradient(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    L[L < 1e-9] = 1e-9
    nx = -d[:, 1] / L
    ny = d[:, 0] / L
    hw = widths * 0.5
    left = pts + np.stack([nx * hw, ny * hw], axis=1)
    right = pts - np.stack([nx * hw, ny * hw], axis=1)
    # caps
    def cap(center, tangent, radius, sign):
        a0 = np.arctan2(tangent[1], tangent[0])
        angs = a0 + sign * np.linspace(-np.pi / 2, np.pi / 2, 6)
        return np.stack([center[0] + radius * np.cos(angs), center[1] + radius * np.sin(angs)], axis=1)
    t_end = d[-1] / L[-1]
    t_start = -d[0] / L[0]
    end_cap = cap(pts[-1], t_end, hw[-1], 1)
    start_cap = cap(pts[0], t_start, hw[0], 1)
    poly = np.concatenate([left, end_cap, right[::-1], start_cap], axis=0)
    return poly


def pressure(n, w, taper_in=0.25, taper_out=0.3, seed=0, var=0.25, min_frac=0.15):
    """Width profile for an ink line: taper at ends, soft pressure variation."""
    t = np.linspace(0, 1, n)
    prof = np.ones(n)
    if taper_in > 0:
        prof *= np.clip(t / taper_in, 0, 1) ** 0.7 * (1 - min_frac) + min_frac
    if taper_out > 0:
        prof *= np.clip((1 - t) / taper_out, 0, 1) ** 0.7 * (1 - min_frac) + min_frac
    noise = np.array([vnoise(i * 0.08, 0.5, seed + 99) for i in range(n)])
    prof *= 1 + var * (noise - 0.5) * 2
    return prof * w


def polyline_length(pts):
    d = np.diff(np.asarray(pts, np.float64), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def lerp(a, b, t):
    return a + (b - a) * t


def rot(pts, cx, cy, ang):
    pts = np.asarray(pts, np.float64)
    c, s = np.cos(ang), np.sin(ang)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], axis=1)
