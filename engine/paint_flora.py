"""Plants for painted backgrounds: windswept pines, climbing ivy, pampas
grass and grass blades. Each generator returns geometry (paths, pads,
leaves) that a scene turns into reference shapes and hand placed strokes, so
every tree and plume is individually different and nothing is a stamp."""

import math

import numpy as np

from engine.noise import rng, noise1


def pine(base, height, lean, key, spread=1.0, wind=-1.0):
    """An old windswept pine in the Japanese black pine manner: a contorted
    trunk and flat foliage pads streaming downwind.

    base: (x, y) where the trunk meets the ground. height in px. lean: trunk
    top offset per unit height (negative leans left). wind: -1 streams pads
    to the left. Returns dict(trunk=[(x, y, w)], branches=[[(x, y, w)]],
    pads=[(cx, cy, rx, ry)])."""
    r = rng("pine", key)
    bx, by = base
    n = 28
    trunk = []
    s1 = (r.random() - 0.5) * 0.28
    s2 = (r.random() - 0.5) * 0.16
    w0 = height * 0.06 * (0.85 + 0.3 * r.random())
    for i in range(n):
        t = i / (n - 1)
        x = bx + lean * height * t ** 1.15 + height * (s1 * math.sin(math.pi * t) + s2 * math.sin(2.3 * math.pi * t))
        y = by - height * t * (1.0 - 0.15 * abs(lean) * t)
        w = w0 * (1.0 - 0.72 * t) * (1.15 if t < 0.06 else 1.0)
        trunk.append((x, y, w))
    branches = []
    pads = []
    nb = int(7 + r.integers(0, 3))
    for j in range(nb):
        t = 0.28 + 0.72 * (j + r.random() * 0.6) / nb
        i = min(int(t * (n - 1)), n - 1)
        tx, ty, tw = trunk[i]
        side = wind if r.random() < 0.72 else -wind
        L = height * spread * (0.22 + 0.32 * r.random()) * (1.15 - 0.55 * t)
        if j == nb - 1:
            L *= 0.6
        path = []
        droop = (r.random() * 0.25 + 0.05) * L
        up = (0.1 + 0.25 * r.random()) * L
        for k in range(10):
            u = k / 9
            x = tx + side * L * u
            y = ty + droop * math.sin(math.pi * u * 0.8) - up * u ** 2 + (r.random() - 0.5) * 2
            path.append((x, y, tw * (0.55 - 0.4 * u)))
        branches.append(path)
        # pads along the outer part of the branch
        npads = 2 + int(r.integers(0, 3))
        for q in range(npads):
            u = 1.0 - q * (0.28 + 0.1 * r.random())
            k = min(int(u * 9), 9)
            px, py, _ = path[k]
            prx = L * (0.30 + 0.22 * r.random()) * (1.0 - 0.25 * q)
            pry = prx * (0.26 + 0.12 * r.random())
            pads.append((px + side * prx * 0.15, py - pry * 0.55, prx, pry))
    # crown pad on top of the trunk
    tx, ty, _ = trunk[-1]
    cr = height * spread * 0.16
    pads.append((tx + wind * cr * 0.3, ty - cr * 0.15, cr, cr * 0.38))
    return dict(trunk=trunk, branches=branches, pads=pads)


def pad_lobes(pad, key, count=None):
    """Break a foliage pad into lumpy lobes: (cx, cy, rx, ry) list, upper edge
    bumpy, lower edge flatter."""
    cx, cy, rx, ry = pad
    r = rng("padlobes", key)
    n = count or int(5 + rx / 9)
    out = []
    for i in range(n):
        u = (i + 0.5) / n * 2 - 1 + (r.random() - 0.5) * 0.25
        prof = math.sqrt(max(1 - u * u, 0.05))
        lrx = rx * (0.22 + 0.18 * r.random()) * (0.7 + 0.5 * prof)
        lry = ry * (0.55 + 0.35 * r.random()) * (0.6 + 0.5 * prof)
        out.append((cx + u * rx * 0.85, cy - ry * 0.25 * prof + (r.random() - 0.5) * ry * 0.3, lrx, lry))
    out.append((cx, cy + ry * 0.1, rx * 0.8, ry * 0.55))
    return out


def ivy_vines(origin_pts, key, steps=40, step=6.0, up=0.75, branch=0.08, wander=0.6, bounds=None):
    """Random walk vines climbing from origin points. Returns a list of
    polylines [(x, y), ...]. bounds: callable(x, y) -> bool keeps vines on a surface."""
    r = rng("ivy", key)
    vines = []
    stack = [(float(x), float(y), -math.pi / 2 + (r.random() - 0.5) * 0.8, steps) for x, y in origin_pts]
    while stack and len(vines) < 400:
        x, y, a, n = stack.pop()
        path = [(x, y)]
        for i in range(n):
            a += (r.random() - 0.5) * wander
            a = a * (1 - up * 0.15) + (-math.pi / 2) * up * 0.15
            nx, ny = x + math.cos(a) * step, y + math.sin(a) * step
            if bounds is not None and not bounds(nx, ny):
                a += math.pi * 0.5 * (1 if r.random() < 0.5 else -1)
                continue
            x, y = nx, ny
            path.append((x, y))
            if r.random() < branch and n - i > 6:
                stack.append((x, y, a + (r.random() - 0.5) * 2.2, int((n - i) * (0.4 + 0.5 * r.random()))))
        if len(path) > 2:
            vines.append(path)
    return vines


def pampas_stalk(base, height, lean, key):
    """Geometry of one pampas grass stalk in its rest pose: a stem polyline
    and a plume description (start fraction, plume length, width, droop)."""
    r = rng("pampas", key)
    return dict(base=base, height=height * (0.85 + 0.3 * r.random()), lean=lean + (r.random() - 0.5) * 0.15,
                plume_len=height * (0.36 + 0.14 * r.random()), plume_w=height * (0.075 + 0.035 * r.random()),
                droop=0.35 + 0.35 * r.random(), phase=r.random() * 6.283, freq=0.8 + 0.4 * r.random(),
                seed=int(r.integers(1, 1 << 30)), hairs=int(200 + 120 * r.random()))
