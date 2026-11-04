"""Cumulus designs for the sunrise cuts (7, 8, 9a, 9b), built with
engine/paint_cumulus and repainted with the stroke painter.

For a new film, copy this pattern:
    banks = far_strata(key, y_horizon, y_horizon + 200, x0, x1, 46)   # flat rows to the horizon
    banks += [bank(cx, base_y, width, height, rmax, rmin, depth, towers=[...]), ...]  # a few hero banks
    sky_at = sky_fn(horizon_y, sun_xy, low_rgb, gold_rgb, top_rgb)
    layer = paint_banks(cut, banks, sun_xy, PAL_SUN, sky_at, key, floor=(y, near_rgb, far_rgb, wave))
    common.over(sky_canvas, layer)
bank() takes plate coordinates (it adds the margin). Palettes: PAL_SUN (sunrise),
PAL_MORNING (soft day light), PAL_NIGHT (moonlit; it did not suit cut 1's painted
band sea, so cut 1 does not use it). Wisps exist in paint_banks but read as
scratches when laid across banks; Film A passes wisps=None.
"""

import math

import numpy as np

from engine import paint as P
from engine import paint_cumulus as PCU
from engine.noise import fbm, rng, seed_of

import common as C
from common import T, MARGIN

M = MARGIN

PAL_SUN = dict(
    lit=np.array([1.00, 0.80, 0.52], np.float32),
    high=np.array([1.00, 0.95, 0.80], np.float32),
    mid=np.array([0.93, 0.66, 0.62], np.float32),
    shadow=np.array([0.72, 0.52, 0.64], np.float32),
    deep=np.array([0.46, 0.34, 0.52], np.float32),
    top=np.array([0.55, 0.55, 0.85], np.float32),
    rim=np.array([1.00, 0.93, 0.74], np.float32),
)
PAL_MORNING = dict(
    lit=np.array([1.00, 0.93, 0.78], np.float32),
    high=np.array([1.00, 0.98, 0.92], np.float32),
    mid=np.array([0.90, 0.80, 0.80], np.float32),
    shadow=np.array([0.66, 0.60, 0.74], np.float32),
    deep=np.array([0.50, 0.46, 0.62], np.float32),
    top=np.array([0.60, 0.70, 0.95], np.float32),
    rim=np.array([1.00, 0.96, 0.86], np.float32),
)


def sky_fn(horizon, sun, low, gold, top):
    low, gold, top = (np.asarray(c, np.float32) for c in (low, gold, top))

    def f(X, Y):
        t = np.clip((horizon - Y) / max(horizon, 1.0), 0, 1)
        base = low * (1 - t[..., None]) + top * t[..., None]
        d = np.sqrt(((X - sun[0]) / 900.0) ** 2 + ((Y - sun[1]) / 500.0) ** 2)
        g = np.exp(-d ** 2)
        return base * (1 - g[..., None] * 0.6) + gold * (g[..., None] * 0.6)
    return f


def bank(cx, base, width, height, rmax, rmin, depth, flat=1.0, towers=None, back_reach=900.0):
    return dict(cx=cx + M, base=base + M, width=width, height=height, rmax=rmax, rmin=rmin, depth=depth,
                flat=flat, towers=towers, back_reach=back_reach)


def far_strata(key, y0, y1, x0, x1, n, depth0=0.75, scale=1.0):
    """Rows of low, flat heaps between the horizon and the middle distance:
    they get bigger, taller and clearer as they come nearer."""
    r = rng("strata", key)
    out = []
    for i in range(n):
        t = ((i + r.random()) / n) ** 1.6
        y = y0 + (y1 - y0) * t
        sc = (0.35 + 1.4 * t) * scale
        w = (260 + 520 * r.random()) * sc
        cx = x0 + (x1 - x0) * r.random()
        hgt = (16 + 40 * r.random()) * sc
        out.append(bank(cx, y, w, hgt, max(8.0, hgt * 0.75), max(2.0, 2.5 * sc), depth0 * (1 - 0.6 * t) + 0.05,
                        flat=0.55 + 0.25 * t, towers=[(r.random() - 0.5, 1.0, 0.45 + 0.3 * r.random())]))
    return out


def paint_banks(cut, banks, sun, pal, sky_at, key, wisps=None, rim_col=None, stroke_scale=1.0, floor=None):
    """Composite the banks into a reference and paint it. Returns an RGBA
    alpha layer (premultiplied) the size of the plate."""
    pw, ph = T.plate_size(cut)
    aw, ah = pw + 2 * M, ph + 2 * M
    rgb, alpha, rim, idx = PCU.compose(banks, (ah, aw), (sun[0] + M, sun[1] + M), pal, sky_at, key, floor=floor)
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    m = (P.blur(alpha, 1.5) > 0.25).astype(np.float32)
    ref = rgb.copy()
    ang = P.orientation(ref, sigma=2.0, rho=6.0, base=0.0, base_weight=0.35, edge_gain=1.4)
    s = stroke_scale
    P.paint(canvas, ref, ang, [
        dict(radius=16 * s, spacing=0.9, steps=(3, 9), opacity=0.95, dry=0.3, cthresh=0.16, wob=0.35,
             jitter=(0.03, 0.02, 0.05), streak=0.07, ragged=0.25, pickup=0.25, curve=0.55, mask=m),
        dict(radius=7 * s, spacing=0.95, steps=(2, 8), opacity=0.92, dry=0.4, cthresh=0.12, wob=0.35,
             jitter=(0.03, 0.02, 0.05), streak=0.07, pickup=0.2, mask=m, use_err=True, err_thresh=0.035),
        dict(radius=3.2 * s, spacing=1.0, steps=(2, 7), opacity=0.9, dry=0.45, cthresh=0.09, wob=0.35,
             jitter=(0.03, 0.02, 0.05), streak=0.06, pickup=0.15, mask=m, use_err=True, err_thresh=0.05),
        dict(radius=1.6 * s, spacing=1.1, steps=(2, 5), opacity=0.88, dry=0.5, cthresh=0.07, wob=0.35,
             jitter=(0.03, 0.02, 0.04), streak=0.05, pickup=0.1, mask=m, use_err=True, err_thresh=0.07),
    ], paper, (key, "banks"), clip=alpha)
    # the linings: crisp short strokes along the brightest rims, following the edge
    rc = np.asarray(rim_col if rim_col is not None else pal["rim"], np.float32)
    ys, xs = np.nonzero((rim > 0.45) & (alpha > 0.3))
    r = rng(key, "linings")
    st = P.Strokes(max_pts=4)
    if len(xs):
        pick = r.choice(len(xs), min(len(xs), 2600), replace=False)
        for i in pick:
            x, y = float(xs[i]), float(ys[i])
            a = float(ang[ys[i], xs[i]])
            ln = 2.5 + 5 * r.random()
            ca, sa = math.cos(a), math.sin(a)
            st.add([(x - ca * ln, y - sa * ln), (x, y), (x + ca * ln, y + sa * ln)], 0.7 + 0.9 * r.random(),
                   np.clip(rc * (0.92 + 0.1 * r.random()), 0, 1), seed=int(r.integers(1 << 30)),
                   opacity=0.55 + 0.35 * float(rim[ys[i], xs[i]]), dry=0.45, ragged=0.3, bristles=2, pool=0,
                   streak=0.05, taper0=0.3, taper1=0.5, pickup=0.2)
        P.draw(canvas, st, paper, clip=np.clip(alpha * 1.2, 0, 1))
    # wisps and streaks lying in front of the banks
    if wisps:
        ws = P.Strokes(max_pts=16)
        for i, (x0, y0, ln, thick, col, op) in enumerate(wisps):
            rr = rng(key, "wisp", i)
            pts = [(x0 + M + ln * q / 14, y0 + M + 5 * math.sin(q * 0.6 + i) + (rr.random() - 0.5) * 4) for q in range(15)]
            ws.add(pts, thick, np.asarray(col, np.float32), seed=int(rr.integers(1 << 30)), opacity=op, dry=0.8,
                   ragged=0.4, bristles=10, pool=0, streak=0.1, taper0=0.35, taper1=0.5, pickup=0.35)
            ws.add([(x + 3, y - thick * 0.35) for x, y in pts[2:-2]], max(0.8, thick * 0.25),
                   np.clip(np.asarray(col) * 1.08 + 0.04, 0, 1), seed=int(rr.integers(1 << 30)), opacity=op * 0.8,
                   dry=0.7, ragged=0.35, bristles=3, pool=0, streak=0, taper0=0.3, taper1=0.5)
        P.draw(canvas, ws, paper)
    P.finish(canvas, paper, grain=0.035)
    return canvas


# ---------------------------------------------------------------- cut 7

def banks_7():
    """Cut 7: the sun rises at x 1560 through a low gap; a great towering bank
    to the right catches it on its flank, a lower mass rises behind the
    ledge on the left, a broken floor of big heaps fills the foreground, and
    the far layers lie flat and gold at the horizon."""
    r = rng("c7-banks")
    B = []
    B += far_strata("c7", 800, 1000, -200, 2500, 46, depth0=0.85)
    # middle distance: heaps of very different sizes, kept low around the sun gap
    mids = [(150, 905, 520, 120), (640, 885, 380, 80), (980, 900, 300, 150), (1250, 878, 260, 60),
            (1880, 890, 340, 70), (2350, 905, 420, 110), (1460, 870, 160, 40), (1700, 875, 220, 55)]
    for i, (cx, base, w, hgt) in enumerate(mids):
        B.append(bank(cx, base, w, hgt, hgt * 0.55 + 10, 3.0, 0.55 + 0.15 * r.random(),
                      towers=[(r.random() - 0.5, 1.0, 0.5 + 0.3 * r.random())]))
    # the great towering bank, right: its sunward flank faces the gap
    B.append(bank(2140, 1170, 1000, 780, 120, 15.0, 0.18,
                  towers=[(-0.15, 1.0, 0.45), (0.35, 0.72, 0.4), (-0.6, 0.38, 0.35)], back_reach=700.0))
    # a lower mass rising behind the ledge, left of the sun
    B.append(bank(620, 1080, 1050, 470, 95, 13.0, 0.30,
                  towers=[(0.25, 1.0, 0.4), (-0.3, 0.7, 0.45), (0.7, 0.45, 0.3)], back_reach=700.0))
    # the near floor: a few big heaps, broken, overlapping
    for (cx, base, w, hgt, dep) in [(1350, 1400, 1000, 300, 0.08), (1980, 1430, 900, 250, 0.06),
                                     (2450, 1400, 800, 330, 0.05), (850, 1440, 900, 240, 0.07),
                                     (1620, 1290, 520, 130, 0.15)]:
        B.append(bank(cx, base, w, hgt, hgt * 0.36, 13.0, dep,
                      towers=[(r.random() - 0.5, 1.0, 0.35), (r.random() - 0.5, 0.65, 0.3),
                              (r.random() - 0.5, 0.45, 0.4)], back_reach=600.0))
    return B


def wisps_7():
    W = []
    r = rng("c7-wisps")
    for i in range(4):
        y = 700 + 80 * r.random()
        x = 300 + 1500 * r.random()
        W.append((x, y, 300 + 500 * r.random(), 7 + 8 * r.random(),
                  np.array([1.0, 0.86, 0.66]) * (0.9 + 0.1 * r.random()), 0.16 + 0.1 * r.random()))
    return W


def floor_7():
    return (800 + M, np.array([0.62, 0.46, 0.58], np.float32), np.array([1.0, 0.82, 0.60], np.float32), 10.0)


# ---------------------------------------------------------------- cut 8

def banks_8():
    """Cut 8, the widest image: a towering bank on the left frames the sky,
    another stands against the sun on the right (its edges blazing), a great
    distant anvil sits low in the middle under the bird, rows of heaps run to
    the horizon and a broken floor of big heaps fills the bottom."""
    r = rng("c8-banks")
    B = []
    B += far_strata("c8", 1075, 1290, -300, 2900, 60, depth0=0.85)
    for (cx, base, w, hgt) in [(700, 1240, 520, 150), (1050, 1215, 300, 90), (1650, 1225, 440, 120),
                               (1950, 1205, 300, 70), (2600, 1250, 500, 160), (150, 1230, 380, 110)]:
        B.append(bank(cx, base, w, hgt, hgt * 0.5 + 10, 4.0, 0.5 + 0.15 * r.random(),
                      towers=[(r.random() - 0.5, 1.0, 0.5 + 0.3 * r.random())]))
    B.append(bank(1350, 1160, 1400, 430, 90, 10.0, 0.62,
                  towers=[(-0.1, 1.0, 0.5), (0.4, 0.7, 0.35), (-0.55, 0.55, 0.3)], back_reach=500.0))
    B.append(bank(380, 1420, 1150, 1050, 140, 16.0, 0.22,
                  towers=[(0.05, 1.0, 0.42), (-0.45, 0.75, 0.4), (0.6, 0.55, 0.3)], back_reach=500.0))
    B.append(bank(2300, 1330, 900, 640, 115, 14.0, 0.32,
                  towers=[(-0.2, 1.0, 0.4), (0.4, 0.7, 0.4)], back_reach=650.0))
    for (cx, base, w, hgt, dep) in [(1200, 1700, 1100, 330, 0.07), (1900, 1680, 950, 300, 0.06),
                                     (2600, 1690, 800, 350, 0.05), (700, 1720, 900, 260, 0.08)]:
        B.append(bank(cx, base, w, hgt, hgt * 0.36, 13.0, dep,
                      towers=[(r.random() - 0.5, 1.0, 0.35), (r.random() - 0.5, 0.65, 0.3),
                              (r.random() - 0.5, 0.45, 0.4)], back_reach=600.0))
    return B


def wisps_8():
    W = []
    r = rng("c8-wisps")
    for i in range(5):
        y = 980 + 90 * r.random()
        x = 500 + 1600 * r.random()
        W.append((x, y, 400 + 600 * r.random(), 8 + 9 * r.random(),
                  np.array([1.0, 0.88, 0.70]) * (0.9 + 0.1 * r.random()), 0.15 + 0.1 * r.random()))
    return W


def floor_8():
    return (1080 + M, np.array([0.66, 0.50, 0.60], np.float32), np.array([1.0, 0.84, 0.62], np.float32), 12.0)


# ---------------------------------------------------------------- 9a / 9b

def banks_far(horizon, key, scale=1.0, width=2400):
    """The sea seen beyond the balustrade: rows of low heaps from the horizon
    down, a few rounder heaps nearer."""
    r = rng(key, "far")
    B = far_strata(key, horizon + 2, horizon + 230 * scale, -300, width + 300, 46, depth0=0.75, scale=scale)
    for i in range(6):
        cx = -100 + (width + 200) * r.random()
        hgt = (60 + 90 * r.random()) * scale
        B.append(bank(cx, horizon + (140 + 90 * r.random()) * scale, (300 + 300 * r.random()) * scale, hgt,
                      hgt * 0.5, max(4.0, 6 * scale), 0.35 + 0.2 * r.random(),
                      towers=[(r.random() - 0.5, 1.0, 0.5)]))
    return B


# ---------------------------------------------------------------- cut 1 (night, moonlit)

PAL_NIGHT = dict(
    lit=np.array([0.30, 0.30, 0.56], np.float32),
    high=np.array([0.62, 0.64, 0.92], np.float32),
    mid=np.array([0.17, 0.16, 0.36], np.float32),
    shadow=np.array([0.10, 0.09, 0.24], np.float32),
    deep=np.array([0.06, 0.05, 0.16], np.float32),
    top=np.array([0.20, 0.20, 0.45], np.float32),
    rim=np.array([0.80, 0.84, 1.00], np.float32),
)


def banks_1_near():
    """Cut 1 foreground: big heaps in moon shadow across the bottom right,
    only their upper outlines catching the moon, framing the lit middle."""
    r = rng("c1-near")
    B = []
    for (cx, base, w, hgt) in [(1250, 1470, 700, 200), (1750, 1460, 800, 300), (2300, 1440, 800, 420)]:
        B.append(bank(cx, base, w, hgt, hgt * 0.36, 12.0, 0.0,
                      towers=[(r.random() - 0.5, 1.0, 0.38), (r.random() - 0.5, 0.65, 0.32),
                              (r.random() - 0.5, 0.45, 0.4)], back_reach=300.0))
    return B
