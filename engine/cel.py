"""Cel helpers shared by drawn characters: fur edges, tone assignment, masks."""

import math
from dataclasses import dataclass

import numpy as np

from . import geom, light, raster, vnoise

# tone slots per material
LIT, SHADE, WARM, WARM_SHADE, RIM = 0, 1, 2, 3, 4
NTONE = 5


class Palette:
    """Materials x tones colour table for one scene light."""

    def __init__(self, albedos, L, name, overrides=None, designed=None):
        """designed: optional {material: {slot: '#hex'}} hand-picked colours that win
        over the formula (a colour designer's script for this light)."""
        self.names = list(albedos.keys())
        self.index = {k: i for i, k in enumerate(self.names)}
        self.table = np.zeros((len(self.names) * NTONE, 3), dtype=np.float32)
        overrides = overrides or {}
        designed = designed or {}
        for i, k in enumerate(self.names):
            kw = overrides.get(k, {})
            t = light.tones(albedos[k], L, name, **kw)
            for slot, hx in designed.get(k, {}).items():
                t[slot] = light.hexrgb(hx) if isinstance(hx, str) else np.asarray(hx, np.float32)
            for j, slot in enumerate(("lit", "shade", "warm", "warm_shade", "rim")):
                self.table[i * NTONE + j] = t[slot]

    def __getitem__(self, key):
        return self.index[key]

    def colour(self, mat, tone=LIT):
        return self.table[self.index[mat] * NTONE + tone]

    def set(self, mat, tone, rgb):
        self.table[self.index[mat] * NTONE + tone] = np.asarray(rgb, dtype=np.float32)


@dataclass
class LightCtx:
    L: dict
    name: str
    key3: np.ndarray            # unit vector toward the key light (screen space)
    rim_dir: np.ndarray         # 2D screen direction toward the rim light
    lantern: tuple = None       # (x, y, z) canvas coords of the warm point light
    lantern_reach: float = 0.0  # canvas px where its light ends
    pal: Palette = None
    hard_warm: float = 0.0      # > 0: lantern side also gets a hard warm cel tone above this facing


def lantern_terms(n, pos, ctx):
    """(facing, falloff) of the warm point light per pixel, or (None, None)."""
    if ctx.lantern is None or ctx.lantern_reach <= 0:
        return None, None
    lx = ctx.lantern[0] - pos[0]
    ly = ctx.lantern[1] - pos[1]
    lz = ctx.lantern[2] - pos[2]
    dist = np.sqrt(lx * lx + ly * ly + lz * lz) + 1e-6
    ndw = (n[0] * lx + n[1] * ly + n[2] * lz) / dist
    fall = np.clip(1.0 - dist / ctx.lantern_reach, 0.0, 1.0)
    return ndw, fall


def tone_index(n, pos, ctx, thr=0.0):
    """Per pixel tone slot from normals n (3 arrays) and canvas positions pos (3 arrays).

    The key light gives the one hard shadow shape. The lantern normally arrives
    as a soft compositing light (see the actor's warm pass); only when
    ctx.hard_warm is set does it also cut a hard warm tone.
    """
    k = ctx.key3
    ndl = n[0] * k[0] + n[1] * k[1] + n[2] * k[2]
    lit = ndl > thr
    tone = np.where(lit, LIT, SHADE).astype(np.int16)
    if ctx.hard_warm > 0:
        ndw, fall = lantern_terms(n, pos, ctx)
        if ndw is not None:
            warm = (ndw > ctx.hard_warm) & (fall > 0.1)
            tone = np.where(warm, np.where(lit, WARM, WARM_SHADE), tone).astype(np.int16)
    return tone


def fur_edge(P, t, N, ranges, seed, unit, closed=True):
    """Replace stretches of a closed outline with irregular fur tufts.

    P (n, 2) outline, t (n,) increasing params in [0, 2 pi), N (n, 2) outward
    normals. ranges: list of dicts with t0, t1 (param range), length and width
    (in units of `unit` canvas px and param radians), lean (screen direction the
    tufts comb toward, (2,)), jitter. Returns (polygon, tip_indices).
    """
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    tt = np.asarray(t, dtype=np.float64)
    rnd = vnoise.rng("fur", seed)

    def at(tq):
        tq = tq % (2 * math.pi)
        x = np.interp(tq, tt, P[:, 0], period=2 * math.pi)
        y = np.interp(tq, tt, P[:, 1], period=2 * math.pi)
        nx = np.interp(tq, tt, N[:, 0], period=2 * math.pi)
        ny = np.interp(tq, tt, N[:, 1], period=2 * math.pi)
        nn = math.hypot(nx, ny) + 1e-12
        return np.array([x, y]), np.array([nx / nn, ny / nn])

    spikes = []  # (ta, tb, tip_point)
    for r in ranges:
        t0, t1 = r["t0"], r["t1"]
        cur = t0
        k = 0
        while cur < t1 - 1e-6:
            w = r["width"] * rnd.uniform(0.65, 1.35)
            if cur + w > t1:
                w = t1 - cur
            if w < r["width"] * 0.3:
                break
            ta, tb = cur, cur + w
            # profile along the range: tufts grow toward the middle of a range
            u = ((ta + tb) * 0.5 - t0) / max(t1 - t0, 1e-9)
            env = r.get("env", lambda u: math.sin(math.pi * min(max(u, 0), 1)) ** 0.5)(u)
            ln = r["length"] * unit * env * rnd.uniform(0.55, 1.25)
            tm = ta + w * rnd.uniform(0.55, 0.8)
            pm, nm = at(tm)
            lean = r.get("lean", (0.0, 1.0))
            lean = np.asarray(lean(tm) if callable(lean) else lean, dtype=np.float64)
            lean = lean / (np.linalg.norm(lean) + 1e-12)
            d = geom.normalize(nm + lean * r.get("comb", 0.8))
            tip = pm + d * ln
            spikes.append((ta, tb, tip, ln))
            cur = tb
            k += 1
    spikes.sort(key=lambda s: s[0])
    out = []
    tips = []

    def emit(sp):
        ta, tb, tip, ln = sp
        b0, n0 = at(ta)
        b1, n1 = at(tb)
        # concave tuft edges: quadratic curves pulled back toward the body
        for (pa, pb, nb, last) in ((b0, tip, n0, False), (tip, b1, n1, True)):
            mid = 0.5 * (pa + pb)
            ctrl = mid - nb * ln * 0.14
            ss = np.linspace(0, 1, 6) if last else np.linspace(0, 1, 6)[:-1]
            for s in ss:
                if last and s == 0:
                    tips.append(len(out))
                out.append((1 - s) ** 2 * pa + 2 * (1 - s) * s * ctrl + s * s * pb)

    si = 0
    for i in range(n):
        ti = tt[i]
        while si < len(spikes) and spikes[si][0] <= ti:
            emit(spikes[si])
            si += 1
        if any(sp[0] <= ti <= sp[1] for sp in spikes):
            continue
        out.append(P[i])
    while si < len(spikes):
        emit(spikes[si])
        si += 1
    return np.array(out), tips


def rim_mask(union, rim_dir, width):
    """Pixels of a union mask within `width` of its silhouette on the rim side."""
    d = np.asarray(rim_dir, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    sh = raster.shift(union, -d[0] * width, -d[1] * width, fill=False)
    return union & ~sh


def cast_shadow(caster, dx, dy):
    """Shifted caster mask: where its shadow falls on what lies behind it."""
    return raster.shift(caster, dx, dy, fill=False)
