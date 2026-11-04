"""The homemade rocket: rolled paper, tin band, teakettle nose, bamboo legs, one sailcloth fin."""

import numpy as np
from .cel import Cel, INK
from .geom import catmull, ellipse, wobble, rot, stroke_outline
from .panda import part, mult_shade, _ink_w, Transform, SHADOW_TINT

PAPER = (0.93, 0.86, 0.70)
PAPER_D = (0.80, 0.66, 0.50)
TAPE = (0.78, 0.66, 0.42)
TIN = (0.62, 0.66, 0.66)
TIN_D = (0.40, 0.44, 0.50)
KETTLE = (0.58, 0.60, 0.62)
BAMBOO = (0.72, 0.68, 0.40)
BAMBOO_D = (0.50, 0.46, 0.26)
SAIL = (0.86, 0.82, 0.70)
PATCH = (0.55, 0.45, 0.55)
PAINT = (0.30, 0.42, 0.62)
GLASS = (0.55, 0.70, 0.78)
BRASS = (0.72, 0.56, 0.28)


def draw_rocket(cel, T, bx, by, h, light, seed=0, shudder=0.0, fin_flap=0.0):
    """Base centre at unit (bx, by), total height h units (unit = whatever T scales)."""
    w = 0.155 * h
    top = by - 0.84 * h
    bot = by - 0.2 * h
    sx = bx + shudder
    # ---- bamboo legs (behind) ----
    for k, (dx, far) in enumerate([(-0.28, True), (0.3, False), (0.02, True)]):
        x0 = sx + dx * w * 1.2
        y0 = bot - 0.06 * h
        x1 = bx + dx * w * 3.6 + (0.12 * w if dx == 0 else 0)
        y1 = by
        if far and dx == 0:
            x1 = bx + 0.3 * w
        leg = [(x0, y0), (x1, y1)]
        poly = stroke_outline(T(leg), T.px(np.array([0.045 * h, 0.04 * h])))
        m = cel.fill(poly, BAMBOO_D if far else BAMBOO)
        if m is not None:
            mult_shade(cel, m, SHADOW_TINT, -light[0] * T.px(0.012 * h), -light[1] * T.px(0.012 * h), 0.7)
            cel.ink_closed(poly, _ink_w(T, 0.012 * h), light=light, seed=seed + 10 + k, gap=True)
        # nodes
        for t in (0.35, 0.7):
            nx = x0 + (x1 - x0) * t
            ny = y0 + (y1 - y0) * t
            d = np.array([x1 - x0, y1 - y0])
            d = d / (np.hypot(*d) + 1e-9)
            n = np.array([-d[1], d[0]]) * 0.026 * h
            cel.ink(T([(nx - n[0], ny - n[1]), (nx + n[0], ny + n[1])]), _ink_w(T, 0.009 * h), seed=seed + 20 + k)
    # ---- exhaust can ----
    can = [(sx - 0.55 * w, bot - 0.02 * h), (sx + 0.55 * w, bot - 0.02 * h), (sx + 0.62 * w, bot + 0.06 * h), (sx - 0.62 * w, bot + 0.06 * h)]
    part(cel, T, wobble(np.array(can), 0.004 * h, seed + 1, closed=True), TIN_D, SHADOW_TINT, light, 0.012 * h, seed + 1)
    # ---- sailcloth fin (behind body, right) ----
    fx = sx + 0.9 * w
    fin = [(fx - 0.1 * w, top + 0.44 * h), (fx + 0.35 * w + fin_flap * 0.5, top + 0.5 * h), (fx + 1.25 * w + fin_flap, bot + 0.02 * h), (fx + 0.85 * w, bot + 0.06 * h), (fx - 0.05 * w, bot)]
    fp = wobble(np.array(fin), 0.006 * h, seed + 2, closed=True)
    fm = part(cel, T, fp, SAIL, SHADOW_TINT, light, 0.012 * h, seed + 2)
    patch = rot(np.array([(fx + 0.3 * w, bot - 0.16 * h), (fx + 0.62 * w, bot - 0.16 * h), (fx + 0.62 * w, bot - 0.05 * h), (fx + 0.3 * w, bot - 0.05 * h)]), fx + 0.45 * w, bot - 0.1 * h, 0.15)
    cel.paint_in(fm, T(wobble(patch, 0.004 * h, seed + 3, closed=True)), PATCH)
    for k in range(5):
        px = fx + 0.3 * w + k * 0.08 * w
        cel.ink(T([(px, bot - 0.17 * h), (px + 0.02 * w, bot - 0.15 * h)]), _ink_w(T, 0.006 * h), seed=seed + 30 + k, a=0.7)
    # seam line along the fin
    cel.ink(T([(fx + 0.05 * w, top + 0.62 * h), (fx + 0.7 * w, bot - 0.06 * h)]), _ink_w(T, 0.006 * h), seed=seed + 4, a=0.6)
    # ---- paper body ----
    body = [(sx - w * 0.62, top + 0.02 * h), (sx + w * 0.62, top + 0.02 * h), (sx + w * 0.9, top + 0.3 * h), (sx + w * 1.0, bot - 0.2 * h), (sx + w * 1.02, bot),
            (sx - w * 1.02, bot), (sx - w * 1.0, bot - 0.2 * h), (sx - w * 0.9, top + 0.3 * h)]
    bp = catmull(body, closed=True, n=10)
    bp = wobble(bp, 0.004 * h, seed + 5, closed=True)
    bm = part(cel, T, bp, PAPER, SHADOW_TINT, light, 0.014 * h, seed + 5, shade=False, ink=False)
    # paper seams (rolled sheets): slanted lines and a wrapped join
    for k, yy in enumerate(np.linspace(top + 0.14 * h, bot - 0.1 * h, 4)):
        seam = [(sx - w * 0.95, yy + 0.02 * h), (sx, yy - 0.005 * h), (sx + w * 0.95, yy + 0.015 * h)]
        cel.ink(T(catmull(seam, n=6)), _ink_w(T, 0.009 * h), color=PAPER_D, seed=seed + 40 + k, a=0.9)
    # a patch of darker paper glued over a tear
    pq = rot(np.array([(sx - 0.55 * w, top + 0.64 * h), (sx - 0.15 * w, top + 0.63 * h), (sx - 0.12 * w, top + 0.74 * h), (sx - 0.58 * w, top + 0.75 * h)]), sx - 0.35 * w, top + 0.69 * h, -0.1)
    cel.paint_in(bm, T(wobble(pq, 0.004 * h, seed + 90, closed=True)), (0.84, 0.74, 0.56))
    cel.ink_closed(T(pq), _ink_w(T, 0.007 * h), light=light, seed=seed + 91, gap=False, a=0.7)
    for k in range(6):
        px_ = sx - 0.55 * w + k * 0.085 * w
        cel.ink(T([(px_, top + 0.64 * h), (px_ + 0.02 * w, top + 0.625 * h)]), _ink_w(T, 0.005 * h), seed=seed + 92 + k, a=0.6)
    # tape strips
    for k, (yy, ang) in enumerate([(top + 0.2 * h, 0.1), (bot - 0.22 * h, -0.15), (top + 0.42 * h, 0.4)]):
        tp = rot(np.array([(sx - 0.18 * w, yy - 0.025 * h), (sx + 0.18 * w, yy - 0.025 * h), (sx + 0.18 * w, yy + 0.025 * h), (sx - 0.18 * w, yy + 0.025 * h)]) + np.array([0.35 * w * ((k % 2) * 2 - 1), 0]), sx, yy, ang)
        cel.paint_in(bm, T(tp), TAPE)
        cel.ink_closed(T(tp), _ink_w(T, 0.007 * h), light=light, seed=seed + 50 + k, gap=False, a=0.7)
    # painted stars (by paw: wobbly)
    g = np.random.default_rng(seed + 60)
    for k in range(6):
        cx = sx + g.uniform(-0.65, 0.65) * w
        cy = g.uniform(top + 0.3 * h, bot - 0.12 * h)
        r = 0.11 * w * g.uniform(0.7, 1.3)
        pts = []
        for i in range(10):
            a = -np.pi / 2 + i * np.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
        st = wobble(np.array(pts), 0.01 * w, seed + 70 + k, closed=True)
        cel.paint_in(bm, T(st), PAINT)
    # a paint drip
    drip = [(sx + 0.3 * w, top + 0.3 * h), (sx + 0.31 * w, top + 0.38 * h), (sx + 0.29 * w, top + 0.46 * h)]
    cel.ink(T(catmull(drip, n=6)), _ink_w(T, 0.01 * h), color=PAINT, seed=seed + 6, taper_in=0.1, taper_out=0.6)
    # tin band with rivets
    band = [(sx - w * 0.98, top + 0.52 * h), (sx + w * 0.98, top + 0.52 * h), (sx + w * 1.0, top + 0.6 * h), (sx - w * 1.0, top + 0.6 * h)]
    tm = part(cel, T, wobble(np.array(band), 0.003 * h, seed + 7, closed=True), TIN, SHADOW_TINT, light, 0.01 * h, seed + 7)
    for k in range(5):
        rx = sx - 0.8 * w + k * 0.4 * w
        cel.fill(T(ellipse(rx, top + 0.56 * h, 0.012 * h, 0.012 * h, n=10)), TIN_D)
    # dent
    dent = [(sx - 0.7 * w, bot - 0.2 * h), (sx - 0.5 * w, bot - 0.26 * h), (sx - 0.35 * w, bot - 0.19 * h)]
    cel.ink(T(catmull(dent, n=6)), _ink_w(T, 0.007 * h), color=PAPER_D, seed=seed + 8, a=0.8)
    # one hard shadow over the whole body, then outline
    if bm is not None:
        mult_shade(cel, bm, SHADOW_TINT, -light[0] * T.px(0.06 * h), -light[1] * T.px(0.02 * h), 0.8)
        cel.ink_closed(T(bp), _ink_w(T, 0.014 * h), light=light, seed=seed + 5, gap=True)
    # ---- round window with brass ring ----
    wx, wy = sx - 0.1 * w, top + 0.3 * h
    ring = ellipse(wx, wy, 0.42 * w, 0.42 * w, n=24)
    part(cel, T, wobble(ring, 0.003 * h, seed + 9, closed=True), BRASS, SHADOW_TINT, light, 0.01 * h, seed + 9)
    glass = ellipse(wx, wy, 0.32 * w, 0.32 * w, n=22)
    gm = cel.fill(T(glass), GLASS, 0.5)
    cel.paint_in(gm, T(ellipse(wx - 0.1 * w, wy - 0.12 * w, 0.12 * w, 0.07 * w, n=10, rot=-0.6)), (0.92, 0.96, 0.98), 0.85)
    cel.paint_in(gm, T(ellipse(wx + 0.08 * w, wy + 0.12 * w, 0.2 * w, 0.1 * w, n=10)), (0.40, 0.55, 0.65), 0.6)
    cel.ink_closed(T(glass), _ink_w(T, 0.008 * h), light=light, seed=seed + 11, gap=False)
    # teacup handle poking out of the left side
    hnd = [(sx - w * 0.98, wy - 0.02 * h), (sx - w * 1.3, wy - 0.05 * h), (sx - w * 1.35, wy + 0.05 * h), (sx - w * 0.98, wy + 0.07 * h)]
    hp = catmull(hnd, n=6)
    hpoly = stroke_outline(T(hp), T.px(np.full(len(hp), 0.04 * w)))
    hm = cel.fill(hpoly, (0.95, 0.94, 0.90))
    if hm is not None:
        cel.ink_closed(hpoly, _ink_w(T, 0.008 * h), light=light, seed=seed + 12, gap=True)
    # ---- teakettle nose cone ----
    # kettle handle: a wire arch over the lid (drawn before the dome)
    arch = catmull([(sx - w * 0.45, top - 0.1 * h), (sx - w * 0.42, top - 0.27 * h), (sx, top - 0.33 * h), (sx + w * 0.42, top - 0.27 * h), (sx + w * 0.45, top - 0.1 * h)], n=8)
    apoly = stroke_outline(T(arch), T.px(np.full(len(arch), 0.03 * w)))
    am = cel.fill(apoly, BRASS)
    if am is not None:
        cel.ink_closed(apoly, _ink_w(T, 0.008 * h), light=light, seed=seed + 16, gap=True)
    kettle = [(sx - w * 0.68, top + 0.03 * h), (sx - w * 0.62, top - 0.06 * h), (sx - w * 0.38, top - 0.17 * h), (sx, top - 0.21 * h),
              (sx + w * 0.38, top - 0.17 * h), (sx + w * 0.62, top - 0.06 * h), (sx + w * 0.68, top + 0.03 * h)]
    kp = catmull(kettle, closed=True, n=8)
    kp = wobble(kp, 0.003 * h, seed + 13, closed=True)
    km = part(cel, T, kp, KETTLE, SHADOW_TINT, light, 0.012 * h, seed + 13)
    # highlight stripe
    cel.paint_in(km, T(ellipse(sx - 0.45 * w, top - 0.07 * h, 0.12 * w, 0.035 * h, n=10, rot=0.5)), (0.90, 0.92, 0.94), 0.8)
    # lid knob
    knob = ellipse(sx, top - 0.2 * h, 0.08 * w, 0.025 * h, n=12)
    part(cel, T, knob, BRASS, SHADOW_TINT, light, 0.008 * h, seed + 14)
    cel.fill(T(ellipse(sx, top - 0.2 * h - 0.012 * h, 0.05 * w, 0.015 * h, n=10)), BRASS)
    # spout to the right
    spout = [(sx + w * 0.5, top - 0.07 * h), (sx + w * 1.0, top - 0.2 * h), (sx + w * 1.25, top - 0.36 * h)]
    sp = catmull(spout, n=6)
    spoly = stroke_outline(T(sp), T.px(np.linspace(0.09 * w, 0.05 * w, len(sp))))
    sm = cel.fill(spoly, KETTLE)
    if sm is not None:
        mult_shade(cel, sm, SHADOW_TINT, -light[0] * T.px(0.01 * h), -light[1] * T.px(0.01 * h), 0.7)
        cel.ink_closed(spoly, _ink_w(T, 0.009 * h), light=light, seed=seed + 15, gap=True)
    # string tied round a leg (the knot from shot 1)
    kx, ky = bx + 0.26 * w * 2.6 * 0.6, by - 0.4 * (by - (bot - 0.06 * h))
    for k in range(3):
        loop = ellipse(kx, ky + k * 0.012 * h, 0.03 * h, 0.012 * h, n=10, rot=0.3)
        cel.ink(T(np.concatenate([loop, loop[:1]])), _ink_w(T, 0.006 * h), color=(0.85, 0.78, 0.60), seed=seed + 80 + k, taper_in=0.02, taper_out=0.02)
