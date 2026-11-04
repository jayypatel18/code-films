"""The great bird of pages, second design (cut 8).

The big shape comes first: a solid base form (wing surfaces, body, an S curved
neck, the head) drawn as cel shaded paper underlayers, so the bird reads as
one creature before any detail. Over it, feathers made of whole pages:

- primaries: ten big pages at each wingtip, each folded lengthwise like a
  quill and splayed in a fan with sky between them,
- secondaries: two overlapping rows along the arm, tertials near the body,
- coverts: rows that shrink toward the leading edge (greater, median, lesser,
  marginal), primary coverts over the hand, two rows of underwing coverts,
- body and neck: pages laid along the form, overlapping like scales, edges
  lifting; sizes fall from the wingtips (0.6) to the head (0.04),
- head: a clear skull, a long folded page beak, an ink eye, a swept crest,
- tail: a fan of folded tail pages and two long streamers.

Each page has nine vertices (root, middle, tip; left edge, crease, right edge)
so the two halves of the fold take different cel tones and the lit half
glows. Pages flutter on twos. One object, one depth buffer.
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


# ---------------------------------------------------------------- wing planform

WRIST = np.array([0.30, 0.66])
YTIP = 1.95
SHOULDER = np.array([0.0, 0.12, 0.05])


def le_x(y):
    y = np.asarray(y, np.float64)
    a = 0.20 + 0.11 * np.sin(0.5 * math.pi * np.clip(y / 0.66, 0, 1))
    b = 0.31 - 0.42 * np.clip((y - 0.66) / 1.3, 0, None) ** 1.4
    return np.where(y <= 0.66, a, b)


def te_x(y):
    """Trailing edge of the arm (the secondaries' tips), and of the hand base."""
    y = np.asarray(y, np.float64)
    return le_x(y) - (0.56 - 0.16 * np.clip(y / 1.15, 0, 1))


def wing_map(x, y, h, wp):
    """Planform (x fwd, y out) plus height h over the surface to wing local coords."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    h = np.asarray(h, np.float64)
    hs = wp["sweep_hand"] * smooth(0.45, 0.85, y)
    dx, dy = x - WRIST[0], y - WRIST[1]
    c, s = np.cos(hs), np.sin(hs)
    x2 = np.where(y > 0.35, WRIST[0] + dx * c - dy * s, x)
    y2 = np.where(y > 0.35, WRIST[1] + dx * s + dy * c, y)
    a = wp["sweep_arm"]
    c, s = math.cos(a), math.sin(a)
    x3 = x2 * c - y2 * s
    y3 = x2 * s + y2 * c
    ys = np.linspace(0, 2.6, 105)
    th = (wp["root"] + (wp["mid"] - wp["root"]) * smooth(0.12, 0.8, ys)
          + (wp["tip"] - wp["mid"]) * smooth(0.8, 1.9, ys))
    dyy = ys[1] - ys[0]
    Y = np.concatenate([[0], np.cumsum(np.cos(th[:-1]) * dyy)])
    Z = np.concatenate([[0], np.cumsum(np.sin(th[:-1]) * dyy)])
    yc = np.clip(y3, 0, 2.6)
    Yi = np.interp(yc, ys, Y) + np.minimum(y3, 0)
    Zi = np.interp(yc, ys, Z)
    thi = np.interp(yc, ys, th)
    # the wing is thicker at the arm: a gentle airfoil hump behind the leading edge
    chord_u = np.clip((le_x(np.clip(y, 0, YTIP)) - x) / 0.6, 0, 1)
    hump = 0.035 * np.sin(np.pi * chord_u) * (1 - smooth(0.6, 1.4, y))
    hz = h + hump
    return np.stack([x3, Yi - hz * np.sin(thi), Zi + hz * np.cos(thi)], -1)


def wing_point(x, y, h, wp, side):
    P = wing_map(x, y, h, wp) + SHOULDER
    if side < 0:
        P = P * np.array([1.0, -1.0, 1.0])
    return P


# ---------------------------------------------------------------- body form

NECK = np.array([[0.30, 0.0, 0.05], [0.58, 0.0, 0.06], [0.76, 0.0, 0.26], [0.84, 0.0, 0.50]])
HEAD_C = np.array([0.93, 0.0, 0.56])
HEAD_R = np.array([0.115, 0.068, 0.075])
BODY_C = np.array([-0.10, 0.0, 0.0])
BODY_R = np.array([0.52, 0.20, 0.17])


def neck_curve(s):
    s = np.asarray(s, np.float64)[..., None]
    P = NECK
    return ((1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s * s * P[2] + s ** 3 * P[3])


def neck_tangent(s):
    s = np.asarray(s, np.float64)[..., None]
    P = NECK
    d = 3 * (1 - s) ** 2 * (P[1] - P[0]) + 6 * (1 - s) * s * (P[2] - P[1]) + 3 * s * s * (P[3] - P[2])
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)


def neck_radius(s):
    return 0.115 - 0.06 * np.asarray(s, np.float64) ** 0.8


# ---------------------------------------------------------------- design

class Design:
    def __init__(self, key="great-v2"):
        r = P_.rng(key)
        recs = []

        def add(**kw):
            recs.append({k: float(v) for k, v in kw.items()})

        # kinds: 0 wing page (planform frame), 1 fixed local frame, 3 tail streamer
        for side in (+1, -1):
            # primaries: seven long pages splayed outward from the hand like fingers,
            # each folded lengthwise and tapering, sky between their tips
            for k in range(7):
                u = k / 6.0
                root = np.array([0.20 - 0.26 * u, 1.18 - 0.30 * u])
                ang = D(-2 + 64 * u)               # outward, fanning back
                L = 0.86 - 0.30 * u
                add(kind=0, side=side, x=root[0], y=root[1], ang=ang, s0=0.0, L=L, w=L / 2.3,
                    h=-0.004 * k, fold=D(36), grp=0, lift=0.07, flex=1.0, taper=0.42)
            # inner primaries under them, shorter, filling the hand
            for k in range(4):
                u = k / 3.0
                add(kind=0, side=side, x=-0.10 - 0.08 * u, y=0.92 - 0.16 * u, ang=D(66 + 14 * u), s0=0.0,
                    L=0.52 - 0.06 * u, w=0.52 / 1.9, h=-0.03, fold=D(30), grp=0, lift=0.06, flex=0.8, taper=0.6)
            # secondaries: two overlapping rows, tertials
            for row, (n, Lr, h, off) in enumerate(((15, 0.42, -0.06, 0.0), (15, 0.36, -0.03, 0.5))):
                for k in range(n):
                    u = (k + off) / (n - 0.5)
                    y = 0.06 + 0.70 * u
                    x = te_x(y) + Lr * 0.92
                    add(kind=0, side=side, x=x, y=y, ang=D(93 - 8 * u), s0=0.0, L=Lr * r.uniform(0.95, 1.05),
                        w=Lr / 1.38, h=h + 0.002 * k, fold=D(28), grp=1, lift=0.06, flex=0.6)
            for k in range(4):
                add(kind=0, side=side, x=0.02 - 0.03 * k, y=0.03 + 0.05 * k, ang=D(118 - 6 * k), s0=0.0, L=0.32,
                    w=0.32 / 1.38, h=-0.02 + 0.004 * k, fold=D(24), grp=1, lift=0.06, flex=0.5)
            # covert rows on top: (back from leading edge, page length, count, y0, y1, h)
            rows = [(0.36, 0.24, 15, 0.04, 0.95, 0.010), (0.25, 0.17, 19, 0.03, 1.08, 0.022),
                    (0.16, 0.12, 24, 0.02, 1.22, 0.032), (0.09, 0.09, 28, 0.0, 1.34, 0.040),
                    (0.035, 0.065, 34, 0.0, 1.5, 0.048)]
            for (back, Lr, n, y0, y1, h) in rows:
                ys = np.linspace(y0, y1, n) + r.normal(0, 0.006, n)
                for yy in ys:
                    hand = smooth(0.62, 1.3, yy)
                    add(kind=0, side=side, x=le_x(yy) - back + Lr * 0.85, y=yy, ang=D(92 - 52 * hand) + r.normal(0, 0.05),
                        s0=0.0, L=Lr * r.uniform(0.9, 1.1), w=Lr / 1.38 * r.uniform(0.9, 1.1), h=h,
                        fold=D(14), grp=2, lift=0.08, flex=0.4)
            # primary coverts over the hand
            for k in range(9):
                u = k / 8.0
                yy = 0.78 + 0.5 * (1 - u)
                add(kind=0, side=side, x=le_x(yy) - 0.03, y=yy, ang=D(14 + 70 * u), s0=0.0, L=0.24 - 0.05 * u,
                    w=0.17, h=0.015, fold=D(30), grp=2, lift=0.07, flex=0.5)
            # underwing coverts (seen from below)
            for (back, Lr, n, h) in ((0.12, 0.13, 24, -0.025), (0.25, 0.17, 20, -0.035)):
                ys = np.linspace(0.04, 1.15, n)
                for yy in ys:
                    hand = smooth(0.62, 1.3, yy)
                    add(kind=0, side=side, x=le_x(yy) - back + Lr * 0.8, y=yy, ang=D(92 - 52 * hand), s0=0.0, L=Lr,
                        w=Lr / 1.38, h=h, fold=D(-10), grp=3, lift=-0.05, flex=0.3)
        # body: pages laid along the form from chest to rump, overlapping backward
        nr = 13
        for i in range(nr):
            u = i / (nr - 1)
            xr = 0.36 - 0.88 * u
            t = (xr - BODY_C[0]) / BODY_R[0]
            rad = math.sqrt(max(1e-3, 1 - t * t))
            ry, rz = BODY_R[1] * rad, BODY_R[2] * rad
            n = max(7, int(math.pi * (ry + rz) / 0.075))
            Lp = 0.27 - 0.05 * abs(u - 0.4)
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2) + r.uniform(-0.3, 0.3)) / n
                p = BODY_C + np.array([xr - BODY_C[0], ry * math.cos(ph), rz * math.sin(ph)])
                nrm = np.array([t / BODY_R[0] * rad * 0.5, math.cos(ph) / BODY_R[1], math.sin(ph) / BODY_R[2]])
                nrm /= np.linalg.norm(nrm)
                Lj = Lp * r.uniform(0.8, 1.25)
                add(kind=1, side=0, px=p[0] + r.normal(0, 0.012), py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2],
                    dx=-1.0, dy=r.normal(0, 0.12), dz=-0.05, L=Lj, w=Lj / r.uniform(1.6, 2.1), h=0.01 + 0.003 * (nr - i),
                    fold=D(r.uniform(4, 16)), grp=4, lift=r.uniform(0.15, 0.35), flex=0.8, rotj=r.normal(0, 0.12),
                    taper=r.uniform(0.25, 0.5))
        # neck: flowing toward the body, shrinking toward the head
        nn = 16
        for i in range(nn):
            s = 0.06 + 0.94 * i / (nn - 1)
            c = neck_curve(s)
            tg = neck_tangent(s)
            rad = neck_radius(s)
            a = np.cross(tg, [0.0, 1.0, 0.0])
            a /= np.linalg.norm(a)
            b = np.cross(a, tg)
            n = max(7, int(2 * math.pi * rad / 0.055))
            Lp = 0.19 - 0.09 * s
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2) + r.uniform(-0.3, 0.3)) / n
                nrm = a * math.cos(ph) + b * math.sin(ph)
                p = c + nrm * rad
                Lj = Lp * r.uniform(0.8, 1.3)
                add(kind=1, side=0, px=p[0], py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2], dx=-tg[0], dy=-tg[1],
                    dz=-tg[2], L=Lj, w=Lj / r.uniform(1.6, 2.1), h=0.03 + 0.003 * i, fold=D(r.uniform(4, 14)), grp=5,
                    lift=r.uniform(0.15, 0.35), flex=0.7, rotj=r.normal(0, 0.12), taper=r.uniform(0.25, 0.5))
        # head cap
        for i in range(5):
            u = i / 4.0
            xr = HEAD_C[0] + HEAD_R[0] * (0.75 - 1.6 * u)
            t = (xr - HEAD_C[0]) / HEAD_R[0]
            rad = math.sqrt(max(1e-3, 1 - t * t))
            n = 9
            for k in range(n):
                ph = 2 * math.pi * (k + 0.5 * (i % 2)) / n
                p = HEAD_C + np.array([xr - HEAD_C[0], HEAD_R[1] * rad * math.cos(ph), HEAD_R[2] * rad * math.sin(ph)])
                nrm = np.array([t * 0.6, math.cos(ph) / HEAD_R[1], math.sin(ph) / HEAD_R[2]])
                nrm /= np.linalg.norm(nrm)
                add(kind=1, side=0, px=p[0], py=p[1], pz=p[2], nx=nrm[0], ny=nrm[1], nz=nrm[2], dx=-1.0, dy=0.0,
                    dz=0.05, L=0.05, w=0.04, h=0.08 + 0.003 * (5 - i), fold=0.0, grp=6, lift=0.08, flex=0.2)
        # beak: two long folded pages (upper), one shorter (lower)
        for (pz, dz, L, w, nz) in ((0.578, -0.22, 0.36, 0.11, 1.0), (0.532, -0.16, 0.29, 0.07, -1.0)):
            add(kind=1, side=0, px=1.02, py=0.0, pz=pz, nx=0.0, ny=0.0, nz=nz, dx=0.97, dy=0.0, dz=dz, L=L, w=w,
                h=0.0, fold=D(46), grp=7, lift=0.0, flex=0.0, beak=1, taper=0.06)
        # crest: long thin pages swept back
        for k in range(4):
            d = np.array([-0.84, 0.0, 0.50 - 0.09 * k])
            d /= np.linalg.norm(d)
            add(kind=1, side=0, px=0.88, py=0.012 * ((-1) ** k) * (k // 2), pz=0.62, nx=0.0, ny=0.0, nz=1.0,
                dx=d[0], dy=d[1], dz=d[2], L=0.30 - 0.04 * k, w=0.07 - 0.008 * k, h=0.0, fold=D(40), grp=8,
                lift=0.12, flex=0.6)
        # tail fan
        for k in range(9):
            a = -1.0 + 2.0 * k / 8
            L = 0.52 - 0.10 * abs(a)
            fan = D(30) * a
            d = np.array([-math.cos(fan), math.sin(fan), 0.02])
            add(kind=1, side=0, px=-0.56, py=0.03 * a, pz=0.02 + 0.004 * (4 - abs(k - 4)), nx=0.0, ny=0.0, nz=1.0,
                dx=d[0], dy=d[1], dz=d[2], L=L, w=L / 1.38, h=0.002 * (4 - abs(k - 4)), fold=D(30), grp=9, lift=0.10,
                flex=0.7)
        # two long streamers from the middle of the tail
        for si, a in enumerate((-0.35, 0.35)):
            npg = 7
            s0 = 0.30
            for j in range(npg):
                Lp = 0.16 * (1 - 0.45 * j / npg)
                add(kind=3, side=0, si=si, fan=D(10) * a, s0=s0, L=Lp, w=Lp / 1.38, h=0.004 * (npg - j), fold=D(20),
                    grp=10, lift=0.06, flex=0.8)
                s0 += Lp * 0.7
        keys = set()
        for p in recs:
            keys |= set(p.keys())
        self.a = {k: np.array([p.get(k, 0.0) for p in recs]) for k in keys}
        self.a["kind"] = self.a["kind"].astype(np.int64)
        self.a["grp"] = self.a["grp"].astype(np.int64)
        n = len(recs)
        self.n = n
        self.page = r.integers(1, P_.NPAGES, n)
        self.page_back = (self.page + r.integers(7, P_.NPAGES - 7, n)) % P_.NPAGES
        self.page_back[self.page_back == P_.BLANK] = 3
        self.flipu = r.random(n) < 0.5
        self.flipv = r.random(n) < 0.5
        self.tint = r.choice(len(BD.TINTS), n, p=[0.30, 0.26, 0.10, 0.10, 0.14, 0.06, 0.02, 0.015, 0.005])
        bodyish = (self.a["grp"] >= 4) & (self.a["grp"] <= 6)
        self.tint = np.where(bodyish, r.choice([0, 1, 4, 5], n, p=[0.35, 0.35, 0.2, 0.1]), self.tint)
        self.tintj = r.normal(0, 0.012, (n, 3)).astype(np.float32)
        self.ink = r.uniform(0.6, 1.0, n)
        self.flut = r.uniform(0, 1000, n)
        self.rot = r.normal(0, D(4), n)
        self.toff = r.normal(0, 0.04, n) * np.where(bodyish, 0.3, 1.0)
        # assembly order: distance from the body centre in the formed pose
        Vf, _ = page_verts(self, pose_at(46.3), 0)
        c = Vf.mean(1)
        d = np.linalg.norm((c - np.array([-0.05, 0.0, 0.05])) * np.array([1.0, 0.85, 1.0]), axis=1)
        self.dmax = float(d.max())
        u = (d / self.dmax) ** 0.85
        # arrivals spread smoothly from 44.5 s (no clamped cluster that pops in at once)
        self.arrive = 44.5 + 1.55 * u + r.uniform(0.0, 0.18, n)
        self.arrive = np.minimum(self.arrive, 46.22)
        self.centre_formed = c
        self.from_dir = r.normal(0, 1, (n, 3))
        self.from_dir /= np.linalg.norm(self.from_dir, axis=1, keepdims=True)


DESIGN = None


def design():
    global DESIGN
    if DESIGN is None:
        DESIGN = Design()
    return DESIGN


# ---------------------------------------------------------------- pose (same keys as v1)

def pose_at(t):
    keys = [
        (44.0, 8, 12, 18, -4, 0),
        (45.4, 22, 30, 38, 4, 0),
        (46.15, 34, 44, 52, 8, 2),
        (46.42, 38, 50, 60, 10, 3),
        (46.52, 26, 36, 52, 5, 1),
        (46.62, -2, 8, 30, 0, 0),
        (46.72, -32, -24, 4, -2, -2),
        (46.85, -38, -38, -20, -6, -3),
        (47.10, -28, -26, -18, -8, -3),
        (47.70, 2, 6, 6, -12, -2),
        (48.30, 10, 12, 14, -6, 0),
        (50.0, 8, 12, 18, -10, -4),
        (52.0, 10, 14, 20, -14, -6),
    ]
    ks = np.array([k[0] for k in keys])
    vals = np.array([k[1:] for k in keys], np.float64)
    i = int(np.clip(np.searchsorted(ks, t) - 1, 0, len(ks) - 2))
    u = float(np.clip((t - ks[i]) / (ks[i + 1] - ks[i]), 0, 1))
    if not (46.5 <= t <= 46.75):
        u = u * u * (3 - 2 * u)
    v = vals[i] * (1 - u) + vals[i + 1] * u
    # a slow glide flap in the arc so the wings live
    if t > 48.3:
        g = math.sin((t - 48.3) * 2 * math.pi / 1.6) * smooth(48.3, 48.9, t)
        v = v + np.array([10, 12, 14, 0, 0]) * g
    p = {"wing": dict(root=D(v[0]), mid=D(v[1]), tip=D(v[2]), sweep_hand=D(v[3]), sweep_arm=D(v[4])),
         "smear": 46.55 <= t <= 46.70, "t": t,
         "beat": float(smooth(46.5, 46.72, t)) * float(1 - smooth(47.0, 47.6, t))}
    return p


# ---------------------------------------------------------------- per frame geometry

def page_frames(dz, pose, drawing):
    a = dz.a
    n = dz.n
    R = np.zeros((n, 3))
    Dl = np.zeros((n, 3))
    Wd = np.zeros((n, 3))
    Nn = np.zeros((n, 3))
    wp = pose["wing"]
    kind = a["kind"]
    for side in (+1, -1):
        m = (kind == 0) & (a["side"] == side)
        if not m.any():
            continue
        ang = a["ang"][m] + dz.rot[m] * 0.5
        dirp = np.stack([-np.sin(ang), np.cos(ang)], -1)
        x0 = a["x"][m]
        y0 = a["y"][m]
        L = a["L"][m]
        h = a["h"][m]
        P0 = wing_point(x0, y0, h, wp, side)
        P1 = wing_point(x0 + dirp[:, 0] * L, y0 + dirp[:, 1] * L, h + 0.002, wp, side)
        perp = np.stack([dirp[:, 1], -dirp[:, 0]], -1)
        Pw = wing_point(x0 + perp[:, 0] * 0.02, y0 + perp[:, 1] * 0.02, h, wp, side)
        up = wing_point(x0, y0, h + 0.05, wp, side) - wing_point(x0, y0, h, wp, side)
        d = P1 - P0
        Ln = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        d = d / Ln
        nrm = np.cross(Pw - P0, d)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        nrm *= np.sign((nrm * up).sum(-1, keepdims=True) + 1e-12)
        w_ = np.cross(nrm, d)
        R[m] = P0
        Dl[m] = d
        Wd[m] = w_ / (np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12)
        Nn[m] = nrm
    m = kind == 1
    if m.any():
        p = np.stack([a["px"][m], a["py"][m], a["pz"][m]], -1)
        nrm = np.stack([a["nx"][m], a["ny"][m], a["nz"][m]], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        d = np.stack([a["dx"][m], a["dy"][m], a["dz"][m]], -1)
        beak = a.get("beak", np.zeros(n))[m] > 0
        d = np.where(beak[:, None], d, d - nrm * (d * nrm).sum(-1, keepdims=True))
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        w_ = np.cross(nrm, d)
        w_ /= np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12
        rr = (dz.rot[m] + a.get("rotj", np.zeros(n))[m])[:, None] * (~beak)[:, None]
        d2 = d * np.cos(rr) + w_ * np.sin(rr)
        w2 = np.cross(nrm, d2)
        R[m] = p + nrm * a["h"][m][:, None] * 0.2
        Dl[m] = d2
        Wd[m] = w2
        Nn[m] = nrm
    m = kind == 3
    if m.any():
        t = pose["t"]
        si = a["si"][m]
        fan = a["fan"][m]
        s0 = a["s0"][m]
        L = a["L"][m]
        base = np.array([-0.56, 0.0, 0.03])

        def curve(s):
            ph = 2 * math.pi * (t * 0.5) - 2.2 * s + si * 1.3
            z = 0.10 * s ** 1.3 * np.sin(ph) - 0.04 * s
            yw = 0.06 * s ** 1.3 * np.cos(ph * 0.8 + 1.0)
            return np.stack([base[0] - np.cos(fan) * s, base[1] + np.sin(fan) * s + yw, base[2] + z], -1)

        P0 = curve(s0)
        P1 = curve(s0 + L)
        d = P1 - P0
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
        side = np.stack([np.sin(fan), np.cos(fan), np.zeros_like(fan)], -1)
        nrm = np.cross(d, side)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        nrm *= np.sign(nrm[:, 2:3] + 1e-9)
        w_ = np.cross(nrm, d)
        R[m] = P0 + nrm * a["h"][m][:, None]
        Dl[m] = d
        Wd[m] = w_ / (np.linalg.norm(w_, axis=-1, keepdims=True) + 1e-12)
        Nn[m] = nrm
    fl = np.sin(dz.flut + drawing * 1.7) * 0.5 + np.sin(dz.flut * 1.3 + drawing * 0.9) * 0.5
    return R, Dl, Wd, Nn, fl


# 9 vertices: (s along, c across): rows s = 0, 0.5, 1; columns c = -1 (left edge), 0 (crease), +1
VS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
VC = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
TRIS9 = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
HALF9 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
LOOP9 = np.array([0, 3, 6, 7, 8, 5, 2, 1, 0])


def page_verts(dz, pose, drawing):
    a = dz.a
    R, Dl, Wd, Nn, fl = page_frames(dz, pose, drawing)
    L = a["L"][:, None]
    w = a["w"][:, None]
    tap = np.where(a.get("taper", np.zeros(dz.n)) > 0, a.get("taper", np.zeros(dz.n)), 1.0)[:, None]
    fold = a["fold"][:, None]
    lift = (a["lift"][:, None] * (1.0 + 0.6 * fl[:, None] * a["flex"][:, None])) * L
    V = np.zeros((dz.n, 9, 3))
    for k in range(9):
        s = VS[k] / 2.0
        c = VC[k]
        taper = 1.0 - (1.0 - tap) * s ** 1.5
        half = w * 0.5 * taper * np.cos(fold) ** 0.5
        drop = -np.abs(c) * w * 0.5 * np.sin(fold) * taper
        bul = 0.04 * L * math.sin(math.pi * s)
        lz = lift * s * s
        V[:, k] = R + Dl * (L * s) + Wd * (half * c) + Nn * (drop + bul + lz)
    return V, Nn


# ---------------------------------------------------------------- base form meshes

def _ellipsoid_mesh(C, Rr, nu=14, nv=10, x_cut=None):
    us = np.linspace(0, 1, nu + 1)
    vs = np.linspace(0, 1, nv + 1)
    U, Vv = np.meshgrid(us, vs, indexing="ij")
    th = math.pi * (1 - U)
    ph = 2 * math.pi * Vv
    P = np.stack([C[0] + Rr[0] * np.cos(th), C[1] + Rr[1] * np.sin(th) * np.cos(ph),
                  C[2] + Rr[2] * np.sin(th) * np.sin(ph)], -1)
    N = np.stack([np.cos(th) / Rr[0], np.sin(th) * np.cos(ph) / Rr[1], np.sin(th) * np.sin(ph) / Rr[2]], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    return _grid_tris(P, N)


def _grid_tris(P, N):
    a, b = P.shape[:2]
    tris, nrm = [], []
    for i in range(a - 1):
        for j in range(b - 1):
            q = [P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]
            qn = (N[i, j] + N[i + 1, j] + N[i + 1, j + 1] + N[i, j + 1])
            qn = qn / (np.linalg.norm(qn) + 1e-12)
            tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            nrm += [qn, qn]
    return np.array(tris), np.array(nrm)


def base_form(pose):
    """Solid underlayer triangles (n, 3, 3), normals (n, 3), and a thin flag (n,)."""
    wp = pose["wing"]
    out_t, out_n, out_thin = [], [], []
    # wings: a grid over the arm and hand, inset a little under the pages
    for side in (+1, -1):
        ys = np.linspace(0.0, 1.18, 15)
        us = np.linspace(0, 1, 7)
        Yg, Ug = np.meshgrid(ys, us, indexing="ij")
        lx = le_x(Yg)
        tx = te_x(Yg) + 0.10
        Xg = lx + (tx - lx) * Ug
        Pt = wing_point(Xg, Yg, np.full_like(Xg, -0.012), wp, side)
        # normals from the grid
        dY = np.gradient(Pt, axis=0)
        dU = np.gradient(Pt, axis=1)
        Nt = np.cross(dU, dY) * side
        Nt /= np.linalg.norm(Nt, axis=-1, keepdims=True) + 1e-12
        tt, nn = _grid_tris(Pt, Nt)
        out_t.append(tt)
        out_n.append(nn)
        out_thin.append(np.ones(len(tt), bool))
    for (C, Rr) in ((BODY_C, BODY_R * 0.96), (HEAD_C, HEAD_R * 0.97)):
        tt, nn = _ellipsoid_mesh(C, Rr)
        out_t.append(tt)
        out_n.append(nn)
        out_thin.append(np.zeros(len(tt), bool))
    # neck tube
    ss = np.linspace(0.0, 1.0, 14)
    ph = np.linspace(0, 2 * math.pi, 11)
    Pn = np.zeros((len(ss), len(ph), 3))
    Nn = np.zeros_like(Pn)
    for i, s in enumerate(ss):
        c = neck_curve(s)
        tg = neck_tangent(s)
        a = np.cross(tg, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(a, tg)
        rad = neck_radius(s) * 0.95
        for j, p in enumerate(ph):
            nv = a * math.cos(p) + b * math.sin(p)
            Pn[i, j] = c + nv * rad
            Nn[i, j] = nv
    tt, nn = _grid_tris(Pn, Nn)
    out_t.append(tt)
    out_n.append(nn)
    out_thin.append(np.zeros(len(tt), bool))
    return np.concatenate(out_t), np.concatenate(out_n), np.concatenate(out_thin)


# ---------------------------------------------------------------- look

class GreatLook:
    def __init__(self, L, name, sun_world, haze_col, haze_z, ink_mix=0.5):
        self.L = L
        sw = np.asarray(sun_world, np.float64)
        self.sun = sw / np.linalg.norm(sw)
        self.haze_col = np.asarray(haze_col, np.float32)
        self.haze_z = haze_z
        tab = np.zeros((len(BD.TINTS), 4, 3), np.float32)
        for i, alb in enumerate(BD.TINTS):
            t = elight.tones(alb, L, name, shade_depth=0.66)
            tab[i, 0] = np.clip(t["lit"] * 1.02, 0, 1)
            tab[i, 1] = np.clip(t["shade"] * 0.8 + alb * np.array([0.30, 0.22, 0.24]), 0, 1)
            tab[i, 2] = np.clip(alb * np.array([1.0, 0.87, 0.70]) * 0.93 + np.array([0.03, 0.01, 0.0]), 0, 1)
            tab[i, 3] = t["rim"]
        self.tones = tab
        self.ink_mix = ink_mix
        self.base = {
            "lit": np.array([0.93, 0.80, 0.62], np.float32),
            "glow": np.array([0.94, 0.74, 0.52], np.float32),
            "shade": np.array([0.62, 0.50, 0.60], np.float32),
            "body_shade": np.array([0.76, 0.54, 0.48], np.float32),
        }
        # warm shade for the body, neck and head: peach and amber next to the gold wings
        self.body_tones = tab.copy()
        for i, alb in enumerate(BD.TINTS):
            self.body_tones[i, 1] = np.clip(alb * np.array([0.80, 0.57, 0.50]), 0, 1)
            self.body_tones[i, 2] = np.clip(alb * np.array([0.98, 0.84, 0.66]), 0, 1)


def _tone(nv_dot_sun, thin, front):
    """0 lit, 1 shade, 2 glow (sun through the paper)."""
    return np.where(thin, np.where(nv_dot_sun > 0.0, 0, 2),
                    np.where(nv_dot_sun > 0.02, np.where(front, 0, 2), 1))


def bird_draw(dl, dz, pose, look, cam, Rw, pos, scale, drawing, ss=3, line_px=0.8, order=0, smear_dir=None,
              smear_amt=0.0, extra=None, far_haze=0.32, backlit=0.0, assemble=None):
    n = dz.n
    V, Nn = page_verts(dz, pose, drawing)
    shown = np.ones(n)
    base_a = 1.0
    if assemble is not None:
        # the flock pulls together: each page lands in its place at its arrival
        # time, sweeping in the last few frames from the direction its bird came
        tq = assemble
        k = np.clip((tq - (dz.arrive - 0.16)) / 0.16, 0, 1)
        shown = (k > 0).astype(float)
        c = V.mean(1, keepdims=True)
        sc = (0.35 + 0.65 * k ** 0.7)[:, None, None]
        off = dz.from_dir[:, None] * (1 - k)[:, None, None] * 0.6
        V = c + (V - c) * sc + off
        base_a = float(smooth(44.9, 46.1, tq))
    Xw = pos[None, None] + scale * np.einsum("ij,pkj->pki", Rw, V)
    a = dz.a
    kind = a["kind"]
    if smear_dir is not None and smear_amt > 0:
        wingm = kind == 0
        k = (np.clip(a["y"], 0, None) / YTIP) ** 1.4 * smear_amt * wingm
        tip = (VS / 2.0)[None, :, None]
        Xw = Xw - smear_dir[None, None] * (k[:, None, None] * (0.4 + 0.6 * tip)) * scale
    S = cam.project(Xw)
    rb = np.random.default_rng(int(drawing) * 7 + 3)
    S[..., :2] += rb.normal(0, 0.15, S[..., :2].shape)
    zc = cam.project(pos[None])[0, 2]
    # ---- base form
    bt, bn, bthin = base_form(pose)
    btw = pos[None, None] + scale * np.einsum("ij,pkj->pki", Rw, bt)
    bnw = np.einsum("ij,pj->pi", Rw, bn)
    bS = cam.project(btw)
    bvd = cam.view_dir(btw.mean(1))
    bfront = (bnw * bvd).sum(-1) > 0
    bnv = np.where(bfront[:, None], bnw, -bnw)
    bdot = (bnv * look.sun).sum(-1)
    btone = _tone(bdot, bthin, bfront)
    bsh = np.where(bthin[:, None], look.base["shade"], look.base["body_shade"])
    bcol = np.where((btone == 0)[:, None], look.base["lit"], np.where((btone == 2)[:, None], look.base["glow"], bsh))
    bb = np.clip((-bnv[:, 1] - 0.15) / 0.5, 0, 1) * 0.5 * (btone == 1)
    bcol = bcol * (1 - bb[:, None]) + (bcol * 0.5 + np.array([1.0, 0.78, 0.58]) * 0.5) * bb[:, None]
    bz = bS[..., 2].mean(1)
    span = scale * 1.8
    bhz = np.clip((bz - zc) / span, 0, 1) * far_haze
    z0, z1, amax = look.haze_z
    bhz = np.clip(bhz + np.clip((bz - z0) / (z1 - z0), 0, 1) * amax, 0, 0.9)
    bcol = bcol * (1 - bhz[:, None]) + look.haze_col * bhz[:, None]
    SIL = np.array([0.50, 0.33, 0.34], np.float32)
    if backlit > 0:
        bcol = bcol * (1 - 0.78 * backlit) + SIL * 0.78 * backlit
    nb = len(bt)
    if assemble is not None:
        # each piece of the base form appears (opaque) when the pages around it land
        bc = bt.mean(1)
        bd = np.linalg.norm((bc - np.array([-0.05, 0.0, 0.05])) * np.array([1.0, 0.85, 1.0]), axis=1)
        b_arr = 44.5 + 1.55 * np.clip(bd / dz.dmax, 0, 1) ** 0.85 + 0.2
        b_alpha = (assemble >= b_arr).astype(np.float32)
    else:
        b_alpha = np.ones(nb, np.float32)
    bcol4 = np.concatenate([bcol, b_alpha[:, None]], -1).astype(np.float32)
    # ---- pages
    nw = np.einsum("ij,pj->pi", Rw, Nn)
    tv = S[:, TRIS9]                                  # (n, 8, 3, 3)
    tw = Xw[:, TRIS9]
    tn = np.cross(tw[:, :, 1] - tw[:, :, 0], tw[:, :, 2] - tw[:, :, 0])
    tn /= np.linalg.norm(tn, axis=-1, keepdims=True) + 1e-12
    # orient each half's normal to the page's outward side
    tn *= np.sign((tn * nw[:, None]).sum(-1, keepdims=True) + 1e-9)
    # one normal per half (average of its four triangles... two per row)
    hn = np.zeros((n, 2, 3))
    for h_ in (0, 1):
        hn[:, h_] = tn[:, HALF9 == h_].mean(1)
    hn /= np.linalg.norm(hn, axis=-1, keepdims=True) + 1e-12
    cen = Xw.mean(1)
    vd = cam.view_dir(cen)
    thin = (kind == 0) | (kind == 3) | (a["grp"] == 7) | (a["grp"] == 8) | (a["grp"] == 9)
    tone_h = np.zeros((n, 2), int)
    for h_ in (0, 1):
        front = (hn[:, h_] * vd).sum(-1) > 0
        nv = np.where(front[:, None], hn[:, h_], -hn[:, h_])
        dot = (nv * look.sun).sum(-1) + dz.toff
        mass = (nw * look.sun).sum(-1) * 0.85 + dz.toff
        tone_h[:, h_] = np.where(thin, np.where(dot > 0.0, 0, 2), np.where(mass > 0.02, np.where(front, 0, 2), 1))
    front_page = (nw * vd).sum(-1) > 0
    tab = look.tones[dz.tint]
    vis_n = np.where(front_page[:, None], nw, -nw)
    bounce = np.clip((-vis_n[:, 1] - 0.15) / 0.5, 0, 1) * 0.5
    bodyish = (a["grp"] >= 4) & (a["grp"] <= 6)
    tab = np.where(bodyish[:, None, None], look.body_tones[dz.tint], tab)
    col_h = np.stack([np.take_along_axis(tab, tone_h[:, h_, None, None].repeat(3, -1), 1)[:, 0] for h_ in (0, 1)], 1)
    col_h = col_h + dz.tintj[:, None]
    warm = np.array([1.0, 0.78, 0.58], np.float32)
    shade_m = (tone_h == 1)[..., None]
    col_h = np.where(shade_m, col_h * (1 - bounce[:, None, None]) + (col_h * 0.5 + warm * 0.5) * bounce[:, None, None], col_h)
    z = S[..., 2].mean(1)
    hz = np.clip((z - zc) / span, 0, 1) * far_haze + np.clip((z - z0) / (z1 - z0), 0, 1) * amax
    # depth inside the wing: pages under others and near the body sit in soft shade
    occ = np.where(kind == 0, 0.12 * (1 - smooth(0.0, 0.5, np.abs(a["y"]))), 0.0)
    rose = np.array([0.93, 0.80, 0.86], np.float32)
    col_h = col_h * (1 - occ[:, None, None]) + col_h * rose * occ[:, None, None] * 0.75
    col_h = col_h * (1 - hz[:, None, None]) + look.haze_col * hz[:, None, None]
    if backlit > 0:
        # crossing the sun: the bird goes to a warm silhouette, the paper glowing at its thin edges
        col_h = col_h * (1 - 0.72 * backlit) + SIL * 0.72 * backlit
    tcol = np.zeros((n, 8, 4), np.float32)
    tcol[..., :3] = col_h[:, HALF9]
    tcol[..., 3] = shown[:, None]
    k = dz.ink * (1 - hz) * 0.9
    tk = np.repeat(k[:, None], 8, 1)
    page = np.where(front_page, dz.page, dz.page_back)
    tpage = np.repeat(page[:, None], 8, 1)
    PW, PH = P_.PW - 1, P_.PH - 1
    uv9 = np.stack([(VC + 1) / 2.0 * PW, VS / 2.0 * PH], -1).astype(np.float64)
    uv = np.broadcast_to(uv9, (n, 9, 2)).copy()
    uv[..., 0] = np.where(dz.flipu[:, None], PW - uv[..., 0], uv[..., 0])
    uv[..., 1] = np.where(dz.flipv[:, None], PH - uv[..., 1], uv[..., 1])
    tuv = uv[:, TRIS9].reshape(n, 8, 6)
    # ---- lines: outline loop and the crease
    Nl = 17
    loop = S[:, LOOP9]
    tt = np.linspace(0, 8, Nl)
    i0 = np.floor(tt).astype(int).clip(0, 7)
    fr = (tt - i0)[None, :, None]
    P = loop[:, i0] * (1 - fr) + loop[:, i0 + 1] * fr
    shade = (tone_h == 1).any(1) | (tone_h == 2).all(1)
    pr = np.random.default_rng(int(drawing) * 13 + 5)
    press = 0.8 + 0.4 * pr.random((n, 1))
    ext = np.ptp(S[..., 0], axis=1) + np.ptp(S[..., 1], axis=1)
    wsz = np.clip(ext / 60.0, 0.5, 1.6)[:, None]
    wt = np.where(shade, 1.1, 0.8)[:, None] * press * wsz
    small = (a["grp"] >= 4) & (a["grp"] <= 6)
    prof = 0.55 + 0.45 * np.sin(np.linspace(0, 8 * math.pi / 2, Nl))[None] ** 2
    rad = line_px * wt * prof * (1 - 0.7 * hz[:, None])
    pts = np.concatenate([P, rad[..., None]], -1).astype(np.float32)
    colm = col_h.mean(1)
    lc = INK[None] * (1 - look.ink_mix) + colm * 0.55 * look.ink_mix
    lc = lc * (1 - hz[:, None]) + look.haze_col * hz[:, None] * 0.8
    lc = np.where(small[:, None], lc * 0.35 + colm * 0.65 * 0.72, lc)
    lc = np.where(((kind == 0) & (a["grp"] >= 1))[:, None], lc * 0.7 + colm * 0.3 * 0.7, lc)
    sop = np.where(small, 0.5, np.where(a["grp"] == 2, 0.55, np.where(a["grp"] == 1, 0.7, 0.92)))
    scol = np.concatenate([lc, (sop * shown)[:, None]], -1).astype(np.float32)
    # crease (rachis) on folded pages: a softer, thinner line from root to 85 %
    folded = np.abs(a["fold"]) > D(15)
    cr = S[:, [1, 4, 7]]
    ct = np.linspace(0, 1.7, Nl)
    c0 = np.floor(ct).astype(int).clip(0, 1)
    cf = (ct - c0)[None, :, None]
    CP = cr[:, c0] * (1 - cf) + cr[:, c0 + 1] * cf
    crad = rad * 0.38 * folded[:, None]
    cpts = np.concatenate([CP, crad[..., None]], -1).astype(np.float32)
    ccol = scol.copy()
    ccol[:, :3] = colm * 0.72 + INK[None] * 0.28
    ccol[:, 3] = 0.38 * shown
    all_pts = np.concatenate([pts, cpts], 0)
    all_col = np.concatenate([scol, ccol], 0)
    if extra is not None:
        ep, ec = extra
        ep = [np.asarray(p_, np.float32) for p_ in ep]
        ep = [p_ if len(p_) == Nl else p_[np.linspace(0, len(p_) - 1, Nl).round().astype(int)] for p_ in ep]
        all_pts = np.concatenate([all_pts, np.stack(ep)], 0)
        all_col = np.concatenate([all_col, np.stack(ec).astype(np.float32)], 0)
    ns = len(all_pts)
    # assemble: base triangles first (no texture), then page triangles
    tri_xyz = np.concatenate([bS.reshape(nb, 3, 3), tv.reshape(n * 8, 3, 3)], 0)
    tri_uv = np.concatenate([np.zeros((nb, 6)), tuv.reshape(n * 8, 6)], 0)
    tri_col = np.concatenate([bcol4, tcol.reshape(n * 8, 4)], 0)
    tri_k = np.concatenate([np.zeros(nb), tk.reshape(n * 8)], 0)
    tri_page = np.concatenate([np.full(nb, -1), tpage.reshape(n * 8)], 0)
    T = len(tri_xyz)
    dl.add(tri_xyz=tri_xyz.reshape(1, T, 3, 3).astype(np.float32), tri_uv=tri_uv.reshape(1, T, 6).astype(np.float32),
           tri_col=tri_col.reshape(1, T, 4).astype(np.float32), tri_k=tri_k.reshape(1, T).astype(np.float32),
           tri_page=tri_page.reshape(1, T).astype(np.int64),
           str_pts=all_pts.reshape(1, ns, Nl, 4), str_col=all_col.reshape(1, ns, 4), ss=ss,
           depth=float(z.mean()), alpha=1.0, bias=scale * 0.0005, order=order)
    return S, Xw


def eye_strokes(cam, Rw, pos, scale, line_px):
    """Ink eye on each side of the head: ring, pupil, a glint left as paper."""
    out_p, out_c = [], []
    for sgn in (+1, -1):
        c = HEAD_C + np.array([0.045, HEAD_R[1] * 0.93 * sgn, 0.012])
        t = np.linspace(0, 2 * math.pi, 17)
        ring = c[None] + np.stack([np.cos(t) * 0.032, np.full_like(t, 0.004 * sgn), np.sin(t) * 0.027], -1)
        S = cam.project(pos[None] + scale * (ring @ Rw.T))
        out_p.append(np.concatenate([S, np.full((17, 1), line_px * 1.2)], -1))
        out_c.append(np.array([*INK, 1.0]))
        Sc = cam.project((pos + scale * (Rw @ (c + np.array([0.004, 0.005 * sgn, 0.0]))))[None])[0]
        dot = np.repeat(np.concatenate([Sc, [line_px * 4.6]])[None], 17, 0)
        dot[:, 0] += np.linspace(-0.4, 0.4, 17)
        out_p.append(dot)
        out_c.append(np.array([*(INK * 0.7), 1.0]))
        Sg = cam.project((pos + scale * (Rw @ (c + np.array([0.014, 0.008 * sgn, 0.010]))))[None])[0]
        g = np.repeat(np.concatenate([Sg, [line_px * 1.4]])[None], 17, 0)
        g[:, 0] += np.linspace(-0.15, 0.15, 17)
        out_p.append(g)
        out_c.append(np.array([1.0, 0.97, 0.88, 1.0]))
        # brow line over the eye
        bt = np.linspace(-0.6, 0.9, 17)
        brow = c[None] + np.stack([np.cos(bt) * 0.04 - 0.005, np.full_like(bt, 0.006 * sgn), np.sin(bt) * 0.006 + 0.03], -1)
        Sb = cam.project(pos[None] + scale * (brow @ Rw.T))
        w = line_px * 0.9 * np.sin(np.linspace(0, math.pi, 17)) ** 0.6
        out_p.append(np.concatenate([Sb, w[:, None]], -1))
        out_c.append(np.array([*INK, 0.85]))
    return out_p, out_c
