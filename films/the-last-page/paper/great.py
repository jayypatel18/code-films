"""The great bird of pages (cut 8).

One fixed design, posed per frame, so it stays on model through the gather,
the wingbeat and the arc. Everything is whole book pages:

- wings: a bent wing surface (planform in x forward, y outward; the span
  bends about the body axis with a root, mid and tip elevation, the hand
  sweeps about the wrist) carrying rows of pages like feathers: primaries and
  secondaries are chains of overlapping pages, coverts and marginals are rows
  of single pages laid like shingles, a sprinkle of loose small pages on top.
- body: rings of pages laid like scales on an ellipsoid, front rings on top.
- neck: rings of pages on a curved tube. head: a small cap of pages, a long
  origami beak (two folded pages), a swept crest of long pages, an ink eye.
- tail: long streamers, each a chain of pages flowing with a travelling wave.

Each page is six vertices (root, middle, tip cross sections; the middle
bulges so the paper curls) and four triangles, textured with a whole page of
the atlas, cel toned by the sun (lit gold, lilac shadow, a warm translucent
glow where the sun is behind the paper) and inked with a closed pressure
outline. The whole bird is one object with one depth buffer, so the pages
overlap exactly as they are stacked.
"""

import math

import numpy as np

from engine import light as elight
from . import bird as BD
from . import pages as P_
from .drawbird import INK

D = math.radians


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- planform

WRIST = np.array([0.30, 0.62])
YTIP = 1.74


def le_x(y):
    y = np.asarray(y, np.float64)
    a = 0.19 + 0.12 * np.sin(0.5 * math.pi * np.clip(y / 0.62, 0, 1))
    b = 0.31 - 0.55 * np.clip((y - 0.62) / 1.12, 0, None) ** 1.35
    return np.where(y <= 0.62, a, b)


# ---------------------------------------------------------------- design

class Design:
    """The static layout of every page of the great bird."""

    def __init__(self, key="great-v1"):
        r = P_.rng(key)
        self.r = r
        parts = []        # dicts of arrays per group, concatenated at the end

        def add(**kw):
            parts.append({k: float(v) for k, v in kw.items()})

        # --- wing pages (planform coords); kind 0, side filled later
        def wing_rows(side):
            # primaries: 10 chains of pages fanning from the hand
            for k in range(10):
                u = k / 9.0
                root = WRIST + (np.array([0.02, 1.22]) - WRIST) * (1 - u) * 0.95
                ang = D(10 + 74 * u)              # from outward (+y) toward back (-x)
                Lf = 0.66 + 0.10 * math.sin(math.pi * min(1, (1 - u) * 1.3)) - 0.16 * u
                npg = 6
                pl = Lf / (npg * 0.74)
                for j in range(npg):
                    s0 = j * pl * 0.74
                    add(kind=0, side=side, x=root[0], y=root[1], ang=ang, s0=s0, L=pl, w=pl / 1.38,
                        h=0.004 * k + 0.008 * (npg - j), bulge=0.05 + 0.03 * (j / npg), grp=0, chain=k, cj=j)
            # secondaries: 15 chains along the forearm
            for k in range(15):
                u = k / 14.0
                root = np.array([0.10 + 0.10 * u, 0.04 + 0.56 * u])
                ang = D(92 - 6 * u)
                Lf = 0.54 + 0.05 * math.sin(math.pi * u)
                npg = 5
                pl = Lf / (npg * 0.74)
                for j in range(npg):
                    add(kind=0, side=side, x=root[0], y=root[1], ang=ang, s0=j * pl * 0.74, L=pl, w=pl / 1.38,
                        h=0.0035 * (14 - k) + 0.008 * (npg - j), bulge=0.05, grp=1, chain=100 + k, cj=j)
            # covert rows: (distance behind the leading edge, page length, count, y range, layer)
            rows = [(0.19, 0.115, 24, (0.03, 1.05), 0.075),
                    (0.135, 0.10, 28, (0.02, 1.15), 0.095),
                    (0.09, 0.09, 34, (0.02, 1.28), 0.115),
                    (0.05, 0.078, 40, (0.0, 1.42), 0.135),
                    (0.018, 0.066, 46, (0.0, 1.56), 0.155),
                    (-0.008, 0.056, 52, (0.0, 1.66), 0.175)]
            for ri, (back, pl, n, (y0, y1), h) in enumerate(rows):
                ys = np.linspace(y0, y1, n) + r.normal(0, 0.008, n)
                for yy in ys:
                    xx = le_x(yy) - back
                    # direction: back, turning outward over the hand like the primaries
                    hand = smooth(0.6, 1.3, yy)
                    ang = D(90 - 55 * hand) + r.normal(0, 0.06)
                    add(kind=0, side=side, x=xx, y=yy, ang=ang, s0=0.0, L=pl * r.uniform(0.9, 1.1),
                        w=pl / 1.38 * r.uniform(0.9, 1.1), h=h + r.uniform(0, 0.004), bulge=0.06,
                        grp=2 + min(ri, 3), chain=-1, cj=0)
            # second page on the greater coverts (they are long)
            ys = np.linspace(0.05, 1.0, 18)
            for yy in ys:
                hand = smooth(0.6, 1.3, yy)
                add(kind=0, side=side, x=le_x(yy) - 0.17, y=yy, ang=D(90 - 55 * hand), s0=0.085, L=0.12,
                    w=0.12 / 1.38, h=0.068, bulge=0.05, grp=2, chain=-1, cj=1)
            # loose small pages over the coverts: the mosaic of many pages
            n = 70
            yy = r.uniform(0.05, 1.4, n)
            xx = le_x(yy) - r.uniform(0.0, 0.2, n)
            for i in range(n):
                add(kind=0, side=side, x=xx[i], y=yy[i], ang=r.uniform(0, 6.28), s0=0.0, L=r.uniform(0.045, 0.075),
                    w=r.uniform(0.035, 0.055), h=0.19 + r.uniform(0, 0.01), bulge=0.04, grp=6, chain=-1, cj=0)

        wing_rows(+1)
        wing_rows(-1)

        # --- body: rings of pages on an ellipsoid (kind 1, local 3D fixed)
        C = np.array([-0.08, 0.0, 0.0])
        RA = np.array([0.50, 0.19, 0.17])
        rings = 12
        for i in range(rings):
            u = i / (rings - 1)
            xr = 0.40 - 0.95 * u
            t = (xr - C[0]) / RA[0]
            rad = np.sqrt(max(1e-4, 1 - t * t))
            ry, rz = RA[1] * rad, RA[2] * rad
            circ = math.pi * (ry + rz)
            n = max(7, int(circ / 0.06))
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2)) / n
                p = np.array([xr, ry * math.cos(ph), rz * math.sin(ph)])
                nrm = np.array([t / RA[0] * rad * 0.6, math.cos(ph) / RA[1], math.sin(ph) / RA[2]])
                nrm /= np.linalg.norm(nrm)
                add(kind=1, side=0, px=p[0], py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2],
                    dx=-1.0, dy=0.0, dz=0.0, L=0.15, w=0.105, h=0.004 * (rings - i), bulge=0.03, grp=7,
                    chain=-1, cj=0)
        # --- neck: rings on a curved tube
        A0, A1, A2 = np.array([0.34, 0, 0.04]), np.array([0.68, 0, 0.05]), np.array([0.86, 0, 0.42])
        nr = 12
        for i in range(nr):
            s = 0.08 + 0.92 * i / (nr - 1)
            c = (1 - s) ** 2 * A0 + 2 * (1 - s) * s * A1 + s * s * A2
            tg = 2 * (1 - s) * (A1 - A0) + 2 * s * (A2 - A1)
            tg /= np.linalg.norm(tg)
            rad = 0.10 - 0.048 * s
            a = np.cross(tg, [0, 1.0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(a, tg)
            n = max(7, int(2 * math.pi * rad / 0.05))
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2)) / n
                nrm = a * math.cos(ph) + b * math.sin(ph)
                p = c + nrm * rad
                add(kind=1, side=0, px=p[0], py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2],
                    dx=-tg[0], dy=-tg[1], dz=-tg[2], L=0.13, w=0.068, h=0.004 * i + 0.05, bulge=0.03, grp=8,
                    chain=-1, cj=0)
        # --- head cap
        HC = np.array([0.90, 0.0, 0.45])
        HR = np.array([0.13, 0.08, 0.085])
        for i in range(4):
            u = i / 3.0
            xr = HC[0] + HR[0] * (0.7 - 1.5 * u)
            t = (xr - HC[0]) / HR[0]
            rad = np.sqrt(max(1e-3, 1 - t * t))
            n = 8
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2)) / n
                p = HC + np.array([xr - HC[0], HR[1] * rad * math.cos(ph), HR[2] * rad * math.sin(ph)])
                nrm = np.array([t * 0.6, math.cos(ph) / HR[1], math.sin(ph) / HR[2]])
                nrm /= np.linalg.norm(nrm)
                add(kind=1, side=0, px=p[0], py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2],
                    dx=-1.0, dy=0.0, dz=0.1, L=0.075, w=0.055, h=0.004 * (4 - i) + 0.09, bulge=0.02, grp=9,
                    chain=-1, cj=0)
        # --- beak: two folded pages (upper mandible halves) and a lower mandible
        for sgn in (+1, -1):
            tilt = D(38) * sgn
            nrm = np.array([0.0, math.sin(tilt), math.cos(tilt)])
            add(kind=2, side=0, px=0.99, py=0.012 * sgn, pz=0.475, nx=nrm[0], ny=nrm[1], nz=nrm[2],
                dx=0.94, dy=0.0, dz=-0.30, L=0.46, w=0.085, h=0.1, bulge=0.0, grp=10, chain=-1, cj=0)
            nrm2 = np.array([0.0, math.sin(tilt) * 1.2, -math.cos(tilt)])
            nrm2 /= np.linalg.norm(nrm2)
            add(kind=2, side=0, px=0.985, py=0.01 * sgn, pz=0.43, nx=nrm2[0], ny=nrm2[1], nz=nrm2[2],
                dx=0.95, dy=0.0, dz=-0.24, L=0.38, w=0.05, h=0.1, bulge=0.0, grp=10, chain=-1, cj=0)
        # --- crest: long pages swept back from the head
        for k in range(5):
            sgn = (-1) ** k
            yy = 0.012 * sgn * (k // 2 + 1) if k else 0.0
            d = np.array([-0.82, 0.08 * sgn * (k // 2) * 0.5, 0.56 - 0.07 * k])
            d /= np.linalg.norm(d)
            for j in range(3):
                add(kind=1, side=0, px=0.85 + d[0] * 0.13 * j, py=yy + d[1] * 0.13 * j, pz=0.53 + d[2] * 0.13 * j,
                    nx=0.0, ny=sgn * 0.3, nz=1.0, dx=d[0], dy=d[1], dz=d[2], L=0.16 - 0.015 * k, w=0.06 - 0.006 * k,
                    h=0.11 + 0.003 * (3 - j), bulge=0.04, grp=11, chain=200 + k, cj=j)
        # --- tail streamers (kind 3): streamer index, arc start
        ns = 9
        for i in range(ns):
            a = -1.0 + 2.0 * i / (ns - 1)
            Ls = 1.05 + 1.15 * (1 - abs(a)) ** 1.2
            npg = int(round(Ls / 0.085))
            pl0 = Ls / (npg * 0.62)
            s0 = 0.0
            for j in range(npg):
                fw = 1.0 - 0.5 * (j / npg)
                pl = pl0 * fw
                add(kind=3, side=0, si=i, fan=D(34) * a, s0=s0, L=pl * 1.02, w=pl / 1.38,
                    h=0.003 * (npg - j) + 0.002 * (4 - abs(i - 4)), bulge=0.05, grp=12, chain=300 + i, cj=j,
                    slen=Ls)
                s0 += pl * 0.62

        # one array per field (missing fields are zero)
        keys = set()
        for p in parts:
            keys |= set(p.keys())
        self.a = {k: np.array([p.get(k, 0.0) for p in parts]) for k in keys}
        self.a["kind"] = self.a["kind"].astype(np.int64)
        n = len(parts)
        self.n = n
        self.page = r.integers(1, P_.NPAGES, n)
        self.page_back = (self.page + r.integers(7, P_.NPAGES - 7, n)) % P_.NPAGES
        self.page_back[self.page_back == P_.BLANK] = 3
        self.flipu = r.random(n) < 0.5
        self.flipv = r.random(n) < 0.5
        self.tint = r.choice(len(BD.TINTS), n, p=[0.30, 0.26, 0.10, 0.10, 0.14, 0.06, 0.02, 0.015, 0.005])
        self.tintj = r.normal(0, 0.012, (n, 3)).astype(np.float32)
        self.toff = r.normal(0, 0.05, n)           # shadow edges step along whole pages
        self.ink = r.uniform(0.6, 1.0, n)
        self.flut = r.uniform(0, 1000, n)
        self.rot = r.normal(0, D(5), n)            # each page sits a little askew


DESIGN = None


def design():
    global DESIGN
    if DESIGN is None:
        DESIGN = Design()
    return DESIGN


# ---------------------------------------------------------------- pose

def wing_map(x, y, h, wp):
    """Planform points (x forward, y outward) plus height h above the surface to
    wing local coords (x, y outward, z up) for one wing pose wp."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    # hand sweep about the wrist, arm sweep about the shoulder (in plane)
    hs = wp["sweep_hand"] * smooth(0.45, 0.8, y)
    dx, dy = x - WRIST[0], y - WRIST[1]
    c, s = np.cos(hs), np.sin(hs)
    x2 = np.where(y > 0.3, WRIST[0] + dx * c - dy * s, x)
    y2 = np.where(y > 0.3, WRIST[1] + dx * s + dy * c, y)
    a = wp["sweep_arm"]
    c, s = math.cos(a), math.sin(a)
    x3 = x2 * c - y2 * s
    y3 = x2 * s + y2 * c
    # span bending: integrate the elevation profile along the (swept) span
    ys = np.linspace(0, 2.2, 89)
    th = (wp["root"] + (wp["mid"] - wp["root"]) * smooth(0.12, 0.75, ys)
          + (wp["tip"] - wp["mid"]) * smooth(0.75, 1.75, ys))
    dyy = ys[1] - ys[0]
    Y = np.concatenate([[0], np.cumsum(np.cos(th[:-1]) * dyy)])
    Z = np.concatenate([[0], np.cumsum(np.sin(th[:-1]) * dyy)])
    yc = np.clip(y3, 0, 2.2)
    Yi = np.interp(yc, ys, Y) + np.minimum(y3, 0)
    Zi = np.interp(yc, ys, Z)
    thi = np.interp(yc, ys, th)
    # chordwise camber: the leading edge bends down a touch (paper under load)
    camber = wp.get("camber", 0.0) * np.clip(x3 - le_x(np.clip(y, 0, YTIP)) + 0.25, 0, 0.3)
    hz = h + camber
    out = np.stack([x3, Yi - hz * np.sin(thi), Zi + hz * np.cos(thi)], -1)
    return out


def pose_at(t):
    """Wing and body pose of the great bird at time t (seconds)."""
    t0, tf, tb = 44.0, 46.3, 46.5
    p = {}
    # elevation keys (degrees) for the wing: (time, root, mid, tip, sweep_hand, sweep_arm)
    keys = [
        (44.0, 6, 10, 16, -4, 0),
        (45.4, 22, 30, 38, 4, 0),
        (46.15, 36, 48, 58, 10, 2),      # raised: anticipation
        (46.42, 40, 54, 66, 12, 3),      # top of the hold, tips still rising
        (46.52, 26, 38, 56, 6, 1),       # the beat starts: root leads
        (46.62, -2, 10, 34, 0, 0),       # fastest: tips lag (smear)
        (46.72, -34, -26, 6, -2, -2),    # bottom: tips flex up from the load
        (46.85, -40, -40, -22, -6, -3),  # overshoot: tips snap down
        (47.10, -30, -28, -20, -8, -3),  # settle
        (47.70, 2, 6, 6, -14, -2),       # recovery
        (48.30, 10, 12, 14, -6, 0),      # glide into the arc
        (50.0, 8, 12, 18, -10, -4),
        (52.0, 10, 14, 20, -16, -6),
    ]
    ks = np.array([k[0] for k in keys])
    vals = np.array([k[1:] for k in keys], np.float64)
    i = int(np.clip(np.searchsorted(ks, t) - 1, 0, len(ks) - 2))
    u = (t - ks[i]) / (ks[i + 1] - ks[i])
    u = float(np.clip(u, 0, 1))
    # the beat itself is sharp: linear into the bottom, eased elsewhere
    if not (46.5 <= t <= 46.75):
        u = u * u * (3 - 2 * u)
    v = vals[i] * (1 - u) + vals[i + 1] * u
    wp = dict(root=D(v[0]), mid=D(v[1]), tip=D(v[2]), sweep_hand=D(v[3]), sweep_arm=D(v[4]), camber=0.0)
    p["wing"] = wp
    p["smear"] = 46.55 <= t <= 46.70
    p["beat"] = float(smooth(46.5, 46.72, t)) * float(1 - smooth(47.0, 47.6, t))
    p["neck"] = 0.0
    p["tail_amp"] = 0.10 + 0.10 * p["beat"]
    p["t"] = t
    return p


# ---------------------------------------------------------------- per frame geometry

def page_frames(dz, pose, drawing):
    """Root, direction (length axis), width axis, normal for every page in bird local coords."""
    a = dz.a
    n = dz.n
    R = np.zeros((n, 3))
    Dl = np.zeros((n, 3))
    Wd = np.zeros((n, 3))
    Nn = np.zeros((n, 3))
    wp = pose["wing"]
    # flutter: each page lifts a little at its tip, redrawn on twos
    fl = np.sin(dz.flut + drawing * 1.7) * 0.5 + np.sin(dz.flut * 1.3 + drawing * 0.9) * 0.5
    kind = a["kind"]
    # --- wings
    for side in (+1, -1):
        m = (kind == 0) & (a["side"] == side)
        if not m.any():
            continue
        ang = a["ang"][m] + dz.rot[m] * 0.6
        dirp = np.stack([-np.sin(ang), np.cos(ang)], -1)            # planform direction
        x0 = a["x"][m] + dirp[:, 0] * a["s0"][m]
        y0 = a["y"][m] + dirp[:, 1] * a["s0"][m]
        L = a["L"][m]
        x1 = x0 + dirp[:, 0] * L
        y1 = y0 + dirp[:, 1] * L
        perp = np.stack([dirp[:, 1], -dirp[:, 0]], -1)
        h = a["h"][m]
        h = h * 0.3
        P0 = wing_map(x0, y0, h, wp)
        P1 = wing_map(x1, y1, h + 0.002, wp)
        Pw = wing_map(x0 + perp[:, 0] * 0.02, y0 + perp[:, 1] * 0.02, h, wp)
        d = P1 - P0
        wv = Pw - P0
        nrm = np.cross(wv, d)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        if side < 0:
            pass
        # mirror the right wing (y -> -y)
        if side < 0:
            P0 = P0 * np.array([1, -1, 1])
            d = d * np.array([1, -1, 1])
            wv = wv * np.array([1, -1, 1])
            nrm = np.cross(wv, d)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        # make the normal point up off the wing's top (the side the coverts lie on)
        up = wing_map(x0, y0, h + 0.05, wp) - wing_map(x0, y0, h, wp)
        if side < 0:
            up = up * np.array([1, -1, 1])
        nrm *= np.sign((nrm * up).sum(-1, keepdims=True) + 1e-12)
        # shoulder offset: wings attach at the body side
        P0 = P0 + np.array([0.0, 0.13 * side, 0.06])
        R[m] = P0
        Ln = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        Dl[m] = d / Ln
        w_ = np.cross(nrm, d / Ln)
        Wd[m] = w_ / (np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12)
        Nn[m] = nrm
    # --- body, neck, head, beak, crest (fixed local frames)
    m = (kind == 1) | (kind == 2)
    if m.any():
        p = np.stack([a["px"][m], a["py"][m], a["pz"][m]], -1)
        nrm = np.stack([a["nx"][m], a["ny"][m], a["nz"][m]], -1)
        d = np.stack([a["dx"][m], a["dy"][m], a["dz"][m]], -1)
        d = d - nrm * (d * nrm).sum(-1, keepdims=True) * (kind[m] == 1)[:, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        w_ = np.cross(nrm, d)
        w_ /= np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12
        # askew: rotate the length axis about the normal a little
        rr = dz.rot[m][:, None] * (kind[m] == 1)[:, None]
        d2 = d * np.cos(rr) + w_ * np.sin(rr)
        w2 = np.cross(nrm, d2)
        R[m] = p + nrm * a["h"][m][:, None] * 0.25
        Dl[m] = d2
        Wd[m] = w2
        Nn[m] = nrm
    # --- tail streamers
    m = kind == 3
    if m.any():
        t = pose["t"]
        si = a["si"][m]
        fan = a["fan"][m]
        s0 = a["s0"][m]
        L = a["L"][m]
        amp = pose["tail_amp"]
        base = np.array([-0.56, 0.0, 0.02])

        def curve(s):
            dirx = -np.cos(fan)
            diry = np.sin(fan)
            ph = 2 * math.pi * (t * 0.55) - 2.0 * s + si * 0.7
            z = amp * s ** 1.4 * np.sin(ph) - 0.06 * s * s + 0.05 * s
            yw = 0.35 * amp * s ** 1.4 * np.cos(ph * 0.8 + 1.0)
            return np.stack([base[0] + dirx * s, base[1] + diry * s * (1 + 0.15 * s) + yw, base[2] + z], -1)

        P0 = curve(s0)
        P1 = curve(s0 + L)
        d = P1 - P0
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        side = np.stack([np.sin(fan), np.cos(fan), np.zeros_like(fan)], -1)
        nrm = np.cross(d, side)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        nrm *= np.sign(nrm[:, 2:3] + 1e-9)
        # tone the streamers by a smoothed normal so the wave does not stripe them
        nrm = nrm * 0.35 + np.array([0.0, 0.0, 1.0]) * 0.65
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        w_ = np.cross(nrm, d)
        R[m] = P0 + nrm * a["h"][m][:, None]
        Dl[m] = d
        Wd[m] = w_ / (np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12)
        Nn[m] = nrm
    return R, Dl, Wd, Nn, fl


def page_verts(dz, pose, drawing, smear_from=None):
    """Six vertices per page (n, 6, 3) in bird local coords."""
    a = dz.a
    R, Dl, Wd, Nn, fl = page_frames(dz, pose, drawing)
    L = a["L"][:, None]
    w = a["w"][:, None]
    beak = (a["kind"] == 2)[:, None]
    tipw = np.where(beak, 0.06, 0.92)
    bul = a["bulge"][:, None] * L
    lift = (0.05 + 0.05 * fl[:, None]) * L        # paper lifts a little at its free end
    M = R + Dl * L * 0.5 + Nn * bul
    Tp = R + Dl * L + Nn * lift
    V = np.stack([R - Wd * w / 2, R + Wd * w / 2,
                  M - Wd * w * (0.5 * (1 + tipw) / 2 * 2) / 2, M + Wd * w * (0.5 * (1 + tipw) / 2 * 2) / 2,
                  Tp - Wd * w * tipw / 2, Tp + Wd * w * tipw / 2], 1)
    return V, Nn


TRIS6 = np.array([[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]])
LOOP6 = np.array([0, 2, 4, 5, 3, 1, 0])


# ---------------------------------------------------------------- look

class GreatLook:
    def __init__(self, L, name, sun_world, haze_col, haze_z, ink_mix=0.5):
        self.L = L
        sw = np.asarray(sun_world, np.float64)
        self.sun = sw / np.linalg.norm(sw)
        self.haze_col = np.asarray(haze_col, np.float32)
        self.haze_z = haze_z
        tab = np.zeros((len(BD.TINTS), 4, 3), np.float32)
        key = np.asarray(L["key"], np.float32)
        for i, alb in enumerate(BD.TINTS):
            t = elight.tones(alb, L, name, shade_depth=0.66)
            tab[i, 0] = np.clip(t["lit"] * 1.02, 0, 1)
            tab[i, 1] = np.clip(t["shade"] * 0.8 + alb * np.array([0.30, 0.22, 0.24]), 0, 1)
            # translucent: the sun through the paper
            tab[i, 2] = np.clip(alb * np.array([1.0, 0.87, 0.70]) * 0.93 + np.array([0.03, 0.01, 0.0]), 0, 1)
            tab[i, 3] = t["rim"]
        self.tones = tab
        self.ink_mix = ink_mix


def bird_draw(dl, dz, pose, look, cam, Rw, pos, scale, drawing, ss=3, line_px=0.75, alpha=1.0,
              order=0, smear_dir=None, smear_amt=0.0, extra=None):
    """Add the whole great bird as one object to the DrawList."""
    V, Nn = page_verts(dz, pose, drawing)
    n = dz.n
    Xw = pos[None, None] + scale * np.einsum("ij,pkj->pki", Rw, V)
    if smear_dir is not None and smear_amt > 0:
        # smear: stretch the wing pages back along the stroke, more at the tips
        wingm = (dz.a["kind"] == 0)
        k = (np.abs(dz.a["y"]) / YTIP) ** 1.5 * smear_amt * wingm
        tipside = np.array([0, 0, 0, 0, 1, 1], np.float64)[None, :, None]
        Xw = Xw - smear_dir[None, None] * (k[:, None, None] * (0.4 + 0.6 * tipside)) * scale
    S = cam.project(Xw)
    rb = np.random.default_rng(int(drawing) * 7 + 3)
    S[..., :2] += rb.normal(0, 0.15, S[..., :2].shape)
    nw = np.einsum("ij,pj->pi", Rw, Nn)
    tv = S[:, TRIS6]                       # (n, 4, 3, 3)
    # visible side and tone per page (pages are small: one tone per page, plus
    # a second tone on the bulged tip half where it turns)
    cen = Xw.mean(1)
    vd = cam.view_dir(cen)
    front = (nw * vd).sum(-1) > 0
    nv = np.where(front[:, None], nw, -nw)
    # thin parts (wings, tail, beak, crest): the sunward side is lit, the other
    # side glows with the sun coming through. Thick parts (body, neck, head):
    # the mass turns away from the sun into the lilac shade.
    kind = dz.a["kind"]
    thin = (kind == 0) | (kind == 2) | (kind == 3) | (dz.a["grp"] == 11)
    ndl = (nv * look.sun).sum(-1) + dz.toff
    mass = (nw * look.sun).sum(-1) + dz.toff
    tone = np.where(thin, np.where(ndl > 0.0, 0, 2), np.where(mass > 0.02, np.where(front, 0, 2), 1))
    tone_tip = np.where(thin, tone, np.where(mass + 0.08 > 0.02, np.where(front, 0, 2), 1))
    tab = look.tones[dz.tint]
    col = np.take_along_axis(tab, tone[:, None, None].repeat(3, -1), 1)[:, 0]
    colt = np.take_along_axis(tab, tone_tip[:, None, None].repeat(3, -1), 1)[:, 0]
    col = col + dz.tintj
    colt = colt + dz.tintj
    yabs = np.abs(dz.a["y"])
    occ = np.where(kind == 0, 0.16 * (1 - smooth(0.0, 0.55, yabs)) + 0.07 * (dz.a["grp"] <= 1), 0.0)
    occ = occ + np.where((kind == 3), 0.10 * (1 - smooth(0.0, 0.6, dz.a["s0"])), 0.0)
    rose = np.array([0.93, 0.80, 0.86], np.float32)
    col = col * (1 - occ[:, None]) + col * rose * occ[:, None] * 0.75
    colt = colt * (1 - occ[:, None]) + colt * rose * occ[:, None] * 0.75
    # aerial perspective by depth
    z = S[..., 2].mean(1)
    z0, z1, amax = look.haze_z
    hz = np.clip((z - z0) / max(1e-6, z1 - z0), 0, 1) * amax
    col = col * (1 - hz[:, None]) + look.haze_col * hz[:, None]
    colt = colt * (1 - hz[:, None]) + look.haze_col * hz[:, None]
    tcol = np.zeros((n, 4, 4), np.float32)
    tcol[:, :2, :3] = col[:, None]
    tcol[:, 2:, :3] = colt[:, None]
    tcol[..., 3] = alpha
    k = dz.ink * np.where(tone == 2, 0.8, np.where(tone == 1, 0.75, 0.95)) * (1 - hz)
    tk = np.repeat(k[:, None], 4, 1)
    page = np.where(front, dz.page, dz.page_back)
    tpage = np.repeat(page[:, None], 4, 1)
    # texel coords: the whole page, length along the page height
    PW, PH = P_.PW - 1, P_.PH - 1
    uv6 = np.array([[0, 0], [PW, 0], [0, PH / 2], [PW, PH / 2], [0, PH], [PW, PH]], np.float64)
    uv = np.broadcast_to(uv6, (n, 6, 2)).copy()
    uv[..., 0] = np.where(dz.flipu[:, None], PW - uv[..., 0], uv[..., 0])
    uv[..., 1] = np.where(dz.flipv[:, None], PH - uv[..., 1], uv[..., 1])
    tuv = uv[:, TRIS6].reshape(n, 4, 6)
    # outlines: closed loop per page
    loop = S[:, LOOP6]                     # (n, 7, 3)
    Nl = 15
    tt = np.linspace(0, 6, Nl)
    i0 = np.floor(tt).astype(int).clip(0, 5)
    fr = (tt - i0)[None, :, None]
    P = loop[:, i0] * (1 - fr) + loop[:, i0 + 1] * fr
    shade = (tone == 1) | (tone == 2)
    # line weight: heavier on shade pages and on the edge away from the sun,
    # lighter on lit pages; a little pressure variation per page
    pr = np.random.default_rng(int(drawing) * 13 + 5)
    press = 0.8 + 0.4 * pr.random((n, 1))
    wt = np.where(shade, 1.15, 0.8)[:, None] * press
    prof = 0.55 + 0.45 * np.sin(np.linspace(0, 6 * math.pi, Nl))[None] ** 2
    rad = line_px * wt * prof * (1 - 0.6 * hz[:, None])
    pts = np.concatenate([P, rad[..., None]], -1).astype(np.float32)
    lc = INK[None] * (1 - look.ink_mix) + col * 0.55 * look.ink_mix
    lc = lc * (1 - hz[:, None]) + look.haze_col * hz[:, None] * 0.8
    scol = np.concatenate([lc, np.full((n, 1), 0.92 * alpha)], -1).astype(np.float32)
    bias = scale * 0.0004
    if extra is not None:
        ep, ec = extra
        ep = [np.asarray(p, np.float32) for p in ep]
        Nl2 = pts.shape[1]
        ep = [p if len(p) == Nl2 else p[np.linspace(0, len(p) - 1, Nl2).round().astype(int)] for p in ep]
        pts = np.concatenate([pts, np.stack(ep)], 0)
        scol = np.concatenate([scol, np.stack(ec).astype(np.float32)], 0)
        n_s = len(pts)
    else:
        n_s = n
    dl.add(tri_xyz=tv.reshape(1, n * 4, 3, 3).astype(np.float32), tri_uv=tuv.reshape(1, n * 4, 6).astype(np.float32),
           tri_col=tcol.reshape(1, n * 4, 4), tri_k=tk.reshape(1, n * 4).astype(np.float32),
           tri_page=tpage.reshape(1, n * 4).astype(np.int64),
           str_pts=pts.reshape(1, n_s, Nl, 4), str_col=scol.reshape(1, n_s, 4), ss=ss,
           depth=float(z.mean()), alpha=1.0, bias=bias, order=order)
    return S, Xw


def eye_strokes(dz, pose, cam, Rw, pos, scale, line_px):
    """The ink eye on each side of the head: a ring, a dark pupil, a glint."""
    out_p, out_c = [], []
    for sgn in (+1, -1):
        c = np.array([0.975, 0.071 * sgn, 0.47])
        t = np.linspace(0, 2 * math.pi, 15)
        ring = c[None] + np.stack([np.cos(t) * 0.022, np.zeros_like(t), np.sin(t) * 0.018], -1)
        Xw = pos[None] + scale * (ring @ Rw.T)
        S = cam.project(Xw)
        pts = np.concatenate([S, np.full((15, 1), line_px * 1.1)], -1)
        out_p.append(pts)
        out_c.append(np.array([*INK, 1.0]))
        cw = pos + scale * (Rw @ c)
        Sc = cam.project(cw[None])[0]
        dot = np.repeat(np.concatenate([Sc, [line_px * 2.6]])[None], 15, 0)
        dot[:, 0] += np.linspace(-0.3, 0.3, 15)
        out_p.append(dot)
        out_c.append(np.array([*(INK * 0.8), 1.0]))
    return out_p, out_c
