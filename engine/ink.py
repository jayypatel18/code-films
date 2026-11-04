"""Hand-inked lines: pressure, taper, light-aware weight, wobble and boil.

An outline is never drawn as one perfect closed curve. It is split into
several strokes the way an inker draws it: each stroke lands thin, swells,
lifts thin; lines are heavier on the side away from the light and at the
bottoms of forms, thinner and sometimes broken on the lit side; spike tips
(fur tufts) end the stroke so the line comes to a point.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import geom, raster, vnoise


@dataclass
class Pen:
    width: float = 6.0            # nominal line width in canvas pixels
    light: tuple = (0.55, -0.83)  # screen direction toward the key light
    shadow_gain: float = 0.8      # extra weight on the side away from the light
    lit_thin: float = 0.45        # weight multiplier on the fully lit side
    bottom_gain: float = 0.25     # extra weight where the normal points down
    min_len: float = 60.0         # stroke length range in canvas pixels
    max_len: float = 260.0
    taper: float = 0.22           # fraction of each stroke used to land and lift
    end_width: float = 0.12       # width at the very ends, fraction of nominal
    wobble: float = 0.5           # wobble amplitude in canvas pixels
    wobble_freq: float = 0.012    # wobble frequency per canvas pixel
    break_lit: float = 0.0        # chance of a gap on strongly lit stretches
    overshoot: float = 0.0        # extend stroke ends past their run (canvas px)
    seed: int = 0


def _light_weight(normals, pen):
    l = np.asarray(pen.light, dtype=np.float64)
    l = l / max(np.linalg.norm(l), 1e-9)
    ndl = normals @ l
    shade = np.clip(-ndl, 0.0, 1.0)
    lit = np.clip(ndl, 0.0, 1.0)
    w = 1.0 + pen.shadow_gain * shade - (1.0 - pen.lit_thin) * lit
    w += pen.bottom_gain * np.clip(normals[:, 1], 0.0, 1.0)
    return np.maximum(w, 0.15), lit


def _profile(s, L, pen, tin=True, tout=True):
    """Pressure along one stroke: land thin, swell, lift thin."""
    u = s / max(L, 1e-9)
    t = max(pen.taper, 1e-3)
    p = np.ones_like(u)
    if tin:
        a = np.clip(u / t, 0, 1)
        p = np.minimum(p, pen.end_width + (1 - pen.end_width) * np.sin(a * math.pi / 2) ** 0.8)
    if tout:
        b = np.clip((1 - u) / (t * 1.2), 0, 1)
        p = np.minimum(p, pen.end_width + (1 - pen.end_width) * np.sin(b * math.pi / 2) ** 0.9)
    return p


def _draw_run(buf, P, W, pen, ox, oy, seed, tin=True, tout=True):
    """Draw one stroke with nominal widths W (already light weighted)."""
    if len(P) < 2:
        return
    s = geom.arclen(P)
    L = s[-1]
    if L < 2.0:
        return
    if pen.overshoot > 0 and len(P) >= 3:
        d0 = geom.normalize(P[0] - P[1])
        d1 = geom.normalize(P[-1] - P[-2])
        P = np.vstack([P[0] + d0 * pen.overshoot, P, P[-1] + d1 * pen.overshoot])
        W = np.concatenate([[W[0]], W, [W[-1]]])
        s = geom.arclen(P)
        L = s[-1]
    prof = _profile(s, L, pen, tin, tout)
    N = geom.normals2d(P)
    if pen.wobble > 0:
        wob = vnoise.value1(s * pen.wobble_freq + seed * 7.13, seed) * pen.wobble
        P = P + N * wob[:, None]
    rad = 0.5 * W * prof
    raster.stroke(buf, P, rad, ox, oy)


def ink_polyline(buf, P, pen, ox=0.0, oy=0.0, closed=False, visible=None, tips=None,
                 widths=None, seed=None):
    """Ink a polyline (canvas coords) into coverage buffer buf.

    visible: optional bool per point; hidden points cut the line (the ends taper).
    tips: optional indices where a stroke must end in a point (fur spike tips).
    widths: optional per-point multiplier on top of the light weighting.
    """
    seed = pen.seed if seed is None else seed
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n < 2:
        return
    rnd = vnoise.rng("ink", seed, n)
    if closed:
        N = geom.outward_normals(P)
    else:
        N = geom.normals2d(P)
    lw, lit = _light_weight(N, pen)
    W = pen.width * lw
    if widths is not None:
        W = W * np.asarray(widths, dtype=np.float64)
    vis = np.ones(n, dtype=bool) if visible is None else np.asarray(visible, dtype=bool).copy()
    # Lit edges: occasional gaps where the light hits hardest.
    if pen.break_lit > 0:
        sarc = geom.arclen(P, closed=False)
        g = vnoise.value1(sarc * 0.02 + seed * 3.1, seed + 17)
        vis &= ~((lit > 0.75) & (g > 1.0 - 2.0 * pen.break_lit))
    tipmask = np.zeros(n, dtype=bool)
    if tips is not None and len(tips):
        tipmask[np.asarray(tips, dtype=int) % n] = True

    # Rotate a fully visible closed loop so it starts at a tip or a random point.
    if closed:
        if vis.all():
            start = int(np.flatnonzero(tipmask)[0]) if tipmask.any() else int(rnd.integers(0, n))
        else:
            start = int(np.flatnonzero(~vis)[0])
        order = np.r_[start:n, 0:start]
        P, W, vis, tipmask = P[order], W[order], vis[order], tipmask[order]
        P = np.vstack([P, P[:1]])
        W = np.concatenate([W, W[:1]])
        vis = np.concatenate([vis, vis[:1]])
        tipmask = np.concatenate([tipmask, tipmask[:1]])

    # Runs of visible points.
    runs = []
    i = 0
    m = len(P)
    while i < m:
        if not vis[i]:
            i += 1
            continue
        j = i
        while j < m and vis[j]:
            j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = j

    for (a, b) in runs:
        # split the run into strokes at tips and at length limits
        cuts = [a]
        s = geom.arclen(P[a:b])
        target = rnd.uniform(pen.min_len, pen.max_len)
        last = 0.0
        for k in range(a + 1, b - 1):
            sk = s[k - a]
            if tipmask[k] and sk - last > 4.0:
                cuts.append(k)
                last = sk
                target = rnd.uniform(pen.min_len, pen.max_len)
            elif sk - last > target:
                cuts.append(k)
                last = sk
                target = rnd.uniform(pen.min_len, pen.max_len)
        cuts.append(b - 1)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            # strokes overlap slightly at length cuts, like a real inker's joins
            lap = 0 if tipmask[c0] else 2
            k0 = max(a, c0 - lap)
            k1 = min(b, c1 + 1)
            seg = P[k0:k1]
            ws = W[k0:k1]
            _draw_run(buf, seg, ws, pen, ox, oy, seed + c0 * 31)


def ink_designed(buf, P, width, profile=None, ox=0.0, oy=0.0, wobble=0.0, seed=0):
    """A single designed stroke (eyelid, mouth, whisker) with an explicit profile.

    profile: per-point multiplier (n,) or a callable u -> multiplier.
    """
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return
    s = geom.arclen(P)
    u = s / max(s[-1], 1e-9)
    if profile is None:
        prof = np.sin(np.clip(u, 0, 1) * math.pi) ** 0.6
    elif callable(profile):
        prof = profile(u)
    else:
        prof = np.asarray(profile, dtype=np.float64)
    if wobble > 0:
        N = geom.normals2d(P)
        P = P + N * (vnoise.value1(s * 0.015 + seed, seed) * wobble)[:, None]
    raster.stroke(buf, P, 0.5 * width * prof, ox, oy)
