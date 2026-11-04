"""Cut 2 and cut 5d, the Reading Hall.

A vast hall: three storeys of shelves with galleries and railings, the high
arched window at the top right that the moonlight shaft comes through, a
nearer bookcase on the left, a stone arch on the right opening onto a deeper
room, a stair climbing to the first gallery, a book cart and a ladder. Worn
irregular flagstones, a soft pool of moonlight, blue air in the depth.

Layers (plates carry common.MARGIN):
  hall_bg     opaque. Everything above.
  hall_pillar alpha.  A near fluted column at the left edge (fg, parallax 1.18).
  hall_shaft  rgb.    The moonlight shaft, the window's bloom, the pool's glow.
Animated in render(): dust motes drifting inside the shaft only (on twos).
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise, rng, noise1, seed_of, ridged  # noqa: E402
from engine.paint_shapes import poly_mask, wobble, ellipse_mask, soft  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402
import books as BK  # noqa: E402

M = MARGIN
CUT = "2"
PW, PH = T.plate_size(CUT)
AW, AH = PW + 2 * M, PH + 2 * M
A = T.ANCHORS["2"]

EYE_Y = 560.0
VP_X = 1150.0
FOCAL = 1600.0
WALL_BASE = 820.0
FLOOR_Y = float(A["floor_y"])

MOON = np.array(T.LIGHTS["hall"]["key"], np.float32)
AMB = np.array(T.LIGHTS["hall"]["ambient"], np.float32)
SHAFT_TOP = np.array(A["moon_shaft_top"], np.float32)
SHAFT_FLOOR = np.array(A["moon_shaft_floor"], np.float32)
SHAFT_W = float(A["moon_shaft_width_at_floor"])
PILLAR_PARALLAX = 1.18

DARK = np.array([0.030, 0.034, 0.075], np.float32)
WOOD = np.array([0.300, 0.200, 0.150], np.float32)
STONE = np.array([0.420, 0.405, 0.470], np.float32)
AIR = np.array([0.060, 0.080, 0.170], np.float32)     # the blue air of the hall

# --- layout (plate coordinates)
PILASTERS = [722.0, 1242.0, 1800.0]
PIL_W = 64.0
STOREYS = [  # (rows as (board y, room), gallery slab y or None)
    ([(812.0, 84.0), (724.0, 84.0), (636.0, 84.0)], 548.0),
    ([(540.0, 84.0), (452.0, 84.0), (364.0, 84.0)], 268.0),
    ([(260.0, 84.0), (172.0, 84.0), (84.0, 84.0)], None),
]
WINDOW = dict(x0=1588.0, x1=1736.0, top=-40.0, sill=150.0)
NEAR = dict(x0=110.0, x1=640.0, base=925.0, rows=[(915.0 - 120 * k, 112.0) for k in range(8)])
ARCH = dict(x0=1885.0, x1=2235.0, spring=470.0, jamb=56.0)
STAIR = dict(bottom=(1430.0, WALL_BASE), top=(1105.0, 548.0), steps=15)
CART = (1585.0, 884.0)
LADDER = (1745.0, WALL_BASE, 1785.0, 556.0)


def _grid():
    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float32)
    return yy - M, xx - M


# ---------------------------------------------------------------- light

def illumination():
    """Light reaching the walls: lifted blue ambient so the structure reads,
    cool bounce from the pool, scatter around the shaft, the window's spill."""
    yy, xx = _grid()
    cx, cy = SHAFT_FLOOR
    dpool = np.sqrt(((xx - cx) / 660.0) ** 2 + ((yy - cy) / 430.0) ** 2)
    bounce = np.exp(-dpool ** 2 * 2.0)
    ax, ay = SHAFT_TOP
    t = np.clip(((xx - ax) * (cx - ax) + (yy - ay) * (cy - ay)) / ((cx - ax) ** 2 + (cy - ay) ** 2), 0, 1)
    px, py = ax + t * (cx - ax), ay + t * (cy - ay)
    dsh = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / (70 + SHAFT_W * 0.5 * t + 40)
    scatter = np.exp(-dsh ** 2 * 0.5) * (0.35 + 0.65 * t)
    wx, wy = (WINDOW["x0"] + WINDOW["x1"]) / 2, (WINDOW["top"] + WINDOW["sill"]) / 2
    win = np.exp(-(((xx - wx) / 420.0) ** 2 + ((yy - wy) / 300.0) ** 2))
    climb = P.smoothstep(-60, 900, yy)
    L = AMB[None, None] * (0.42 + 0.55 * climb)[..., None]
    L = L + MOON * (bounce * 0.62 + scatter * 0.30 + win * 0.16)[..., None]
    return L.astype(np.float32), bounce, scatter, win


# ---------------------------------------------------------------- floor

def floor_stones():
    """Worn flagstones of different sizes in perspective, no grid: rows of
    varied depth, each split into stones of varied width, some merged into
    long slabs, joints broken and faded, corners chipped. Returns colour and
    a joint mask for the floor region (plate y >= WALL_BASE)."""
    yy, xx = _grid()
    r = rng("flagstones")
    dy = np.maximum(yy - EYE_Y, 1.0)
    d = FOCAL / dy                       # distance (camera height = 1)
    X = (xx - VP_X) * d / FOCAL          # lateral position on the floor
    # row boundaries in depth, near to far
    bounds = [1.6]
    while bounds[-1] < 7.2:
        bounds.append(bounds[-1] + 0.45 + 0.45 * r.random())
    bounds = np.array(bounds)
    row = np.clip(np.searchsorted(bounds, d) - 1, 0, len(bounds) - 2)
    col = np.zeros((AH, AW, 3), np.float32)
    joint = np.zeros((AH, AW), np.float32)
    stone_id = np.zeros((AH, AW), np.int64)
    ybound = EYE_Y + FOCAL / bounds
    for ri in range(len(bounds) - 1):
        m = row == ri
        if not m.any():
            continue
        # lateral splits for this row
        xs = [-9.0]
        while xs[-1] < 9.0:
            wdt = 0.45 + 0.75 * r.random()
            if r.random() < 0.12:
                wdt *= 1.8          # a long slab
            xs.append(xs[-1] + wdt)
        xs = np.array(xs) + r.random() * 0.4
        Xm = X[m]
        k = np.clip(np.searchsorted(xs, Xm) - 1, 0, len(xs) - 2)
        xl, xr = xs[k], xs[k + 1]
        dm = d[m]
        lat_px = np.minimum(Xm - xl, xr - Xm) * FOCAL / dm
        ym = yy[m]
        dep_px = np.minimum(np.abs(ym - ybound[ri]), np.abs(ym - ybound[ri + 1]))
        sid = ri * 1000 + k
        stone_id[m] = sid
        near = np.clip(3.0 / dm, 0.4, 1.6)
        wj = 0.5 + 1.1 * near
        # broken, faded joints: each joint fades in and out along its length
        fade_l = value_noise(1, 4096, 9, ("jl", ri))[0]
        fade_d = value_noise(1, 4096, 7, ("jd", ri))[0]
        fl = fade_l[np.clip(((ym - EYE_Y) * 3).astype(int), 0, 4095)]
        fd = fade_d[np.clip(((Xm + 9) * 200).astype(int), 0, 4095)]
        jl = P.smoothstep(wj + 0.9, wj * 0.6, lat_px) * (0.35 + 0.65 * fl)
        jd = P.smoothstep(wj + 0.7, wj * 0.5, dep_px) * (0.35 + 0.65 * fd)
        # some stones run long into the next course: drop their far joint
        hs = ((sid * 2246822519) % 4294967296) / 4294967296.0
        farside = np.abs(ym - ybound[ri + 1]) < np.abs(ym - ybound[ri])
        jd = np.where((hs < 0.28) & farside, jd * 0.15, jd)
        joint[m] = np.maximum(jl, jd)
    # stone colours: value and hue vary per stone, worn mottling inside
    h = (stone_id * 2654435761) % 4294967296 / 4294967296.0
    h2 = (stone_id * 40503) % 65536 / 65536.0
    base = STONE * np.array([0.78, 0.78, 0.84], np.float32)
    col = base * (0.70 + 0.42 * h)[..., None]
    col = col + ((h2 > 0.82)[..., None] * np.array([0.035, 0.018, -0.02], np.float32))
    col = col + (((h2 > 0.05) & (h2 < 0.14))[..., None] * np.array([-0.02, 0.02, -0.005], np.float32))
    mott = fbm(AH, AW, 14, "floor-mott", octaves=4)
    col = col * (0.80 + 0.32 * mott)[..., None]
    # foot worn path down the middle of the hall: smoother and a touch lighter
    worn = np.exp(-((X - 0.3) / 1.6) ** 2)
    col = col * (1 + 0.08 * worn)[..., None]
    # cracks on some stones
    cr = P.smoothstep(0.93, 0.985, ridged(AH, AW, 60, "fl-crack", octaves=3)) * (h2 > 0.6)
    col = col * (1 - 0.4 * cr)[..., None]
    # joints: dark, a little moss in some
    col = col * (1 - 0.5 * joint)[..., None]
    moss = joint * P.smoothstep(0.62, 0.8, fbm(AH, AW, 50, "floor-moss", octaves=3))
    col = P.mix(col, np.array([0.12, 0.20, 0.15], np.float32), moss * 0.5)
    # dust gathers near the wall
    col = col * (0.72 + 0.28 * P.smoothstep(WALL_BASE, WALL_BASE + 180, yy))[..., None]
    return col.astype(np.float32), joint, stone_id


# ---------------------------------------------------------------- pool

def pool_shape():
    """Soft, slightly stretched shape of the window's light on the floor with
    soft bar shadows inside. Returns (light 0..1, bars 0..1)."""
    cx, cy = SHAFT_FLOOR
    yy, xx = _grid()
    rx, ry = SHAFT_W * 0.66, SHAFT_W * 0.31
    u = (xx - cx + (yy - cy) * 0.55) / rx
    v = (yy - cy) / ry
    vv = v
    d = np.sqrt(u ** 2 + vv ** 2)
    wob = (value_noise(AH, AW, 30, "pool-edge") - 0.5) * 0.08
    light = P.smoothstep(1.12 + wob, 0.70 + wob, d)
    bars = np.maximum(P.smoothstep(0.10, 0.03, np.abs(u - 0.02 + v * 0.12)),
                      P.smoothstep(0.12, 0.04, np.abs(v + 0.08)))
    bars = np.maximum(bars, 0.7 * P.smoothstep(0.08, 0.02, np.abs(u + 0.5 + v * 0.12)))
    bars = np.maximum(bars, 0.7 * P.smoothstep(0.08, 0.02, np.abs(u - 0.54 + v * 0.12)))
    bars = P.blur(bars.astype(np.float32), 1.5)
    return light.astype(np.float32), (bars * light).astype(np.float32)


def soft_light(rgb, k):
    """Brighten by k (rgb multiplier array) and roll highlights off so the
    texture of the stone always shows through (never clips)."""
    v = rgb * k
    lo = 0.46
    over = np.maximum(v - lo, 0)
    return np.where(v > lo, lo + over / (1 + over * 2.2), v).astype(np.float32)


# ---------------------------------------------------------------- structure reference

def structure_reference():
    """Everything that is not a book, a stroke prop or the floor, lit."""
    yy, xx = _grid()
    Lgt, bounce, scatter, win = illumination()
    ref = np.zeros((AH, AW, 3), np.float32)
    ref[:] = DARK * 1.5 + np.array([0.015, 0.008, 0.006], np.float32)   # shelf interiors
    wood = WOOD * (0.8 + 0.4 * fbm(AH, AW, 20, "wood-var", octaves=3))[..., None]
    # boards for every storey across the back wall, uprights between
    shelf = np.zeros((AH, AW), np.float32)
    for rows, slab in STOREYS:
        for (by, rh) in rows:
            shelf = np.maximum(shelf, ((yy >= by) & (yy < by + 7)).astype(np.float32))
        if slab is not None:
            shelf = np.maximum(shelf, ((yy >= slab) & (yy < slab + 26)).astype(np.float32) * 0.0)
    for ux in [900.0, 1070.0, 1420.0, 1610.0, 2330.0]:
        shelf = np.maximum(shelf, ((np.abs(xx - ux) < 5.0) & (yy < WALL_BASE)).astype(np.float32))
    shelf *= (yy < WALL_BASE)
    ref = P.mix(ref, wood, shelf)
    # skirting
    ref = P.mix(ref, WOOD * 0.85, ((yy >= WALL_BASE - 16) & (yy < WALL_BASE)).astype(np.float32))
    # pilasters
    for px in PILASTERS:
        d = (xx - px) / (PIL_W / 2)
        m = (np.abs(d) < 1.0) & (yy < WALL_BASE)
        course = np.floor((yy + 7) / 34.0)
        rj = (np.sin(course * 12.9898 + px) * 43758.5453) % 1.0
        sc = STONE * (0.82 + 0.3 * rj)[..., None] * (1 - 0.35 * (np.mod(yy + 7, 34.0) < 2.0))[..., None]
        sc = sc * 0.72 * (np.clip(0.75 + 0.25 * d, 0.4, 1.0) * (1 - 0.45 * P.smoothstep(0.8, 1.0, np.abs(d))))[..., None]
        ref = np.where(m[..., None], sc, ref)
    # the high window: night sky glowing through leaded panes
    wx0, wx1, wt, ws = WINDOW["x0"], WINDOW["x1"], WINDOW["top"], WINDOW["sill"]
    wcx, wr = (wx0 + wx1) / 2, (wx1 - wx0) / 2
    spring = wt + wr
    inwin = (np.abs(xx - wcx) < wr) & (yy < ws) & ((yy > spring) | ((xx - wcx) ** 2 + (yy - spring) ** 2 < wr ** 2))
    frame = (np.abs(xx - wcx) < wr + 16) & (yy < ws + 14) & ((yy > spring) | ((xx - wcx) ** 2 + (yy - spring) ** 2 < (wr + 16) ** 2)) & ~inwin
    sky = np.array([0.40, 0.48, 0.74], np.float32) * (0.85 + 0.25 * P.smoothstep(ws, wt, yy))[..., None]
    sky = sky * (0.9 + 0.2 * fbm(AH, AW, 12, "panes", octaves=2))[..., None]
    lead = (np.abs(xx - wcx) < 3.0) | (np.abs(np.mod(yy - spring, 46.0)) < 2.4) | (np.abs(np.abs(xx - wcx) - wr * 0.52) < 1.6)
    wincol = np.where(lead[..., None], np.array([0.05, 0.05, 0.09], np.float32), sky)
    ref = np.where(inwin[..., None], wincol, ref)
    ref = np.where(frame[..., None], STONE * 0.75, ref)
    # deep room through the arch: smaller, bluer, hazier
    ax0, ax1, asp = ARCH["x0"], ARCH["x1"], ARCH["spring"]
    acx, ar = (ax0 + ax1) / 2, (ax1 - ax0) / 2
    opening = (np.abs(xx - acx) < ar) & (yy < WALL_BASE) & ((yy > asp) | ((xx - acx) ** 2 + (yy - asp) ** 2 < ar ** 2))
    deep = AIR * 0.9 + np.array([0.0, 0.004, 0.012], np.float32)
    deepfloor = (yy > 760) & opening
    ref = np.where(opening[..., None], deep, ref)
    ref = np.where(deepfloor[..., None], STONE * 0.28, ref)
    jamb = ARCH["jamb"]
    ring = ((np.abs(xx - acx) < ar + jamb) & (yy < WALL_BASE) &
            ((yy > asp) | ((xx - acx) ** 2 + (yy - asp) ** 2 < (ar + jamb) ** 2)) & ~opening)
    vang = np.arctan2(yy - asp, xx - acx)
    vou = np.abs(np.mod(vang / 0.16, 1.0) - 0.5) < 0.06
    course = np.abs(np.mod(yy, 38.0)) < 2.0
    stonec = STONE * (0.85 + 0.25 * fbm(AH, AW, 9, "arch-st", octaves=2))[..., None]
    stonec = np.where(((vou & (yy < asp)) | (course & (yy >= asp)))[..., None], stonec * 0.6, stonec)
    ref = np.where(ring[..., None], stonec, ref)
    # near bookcase on the left: side panel facing the shaft, front frame
    side = poly_mask(AH, AW, np.array([(640, -96), (704, -18), (704, 880), (640, 925)]) + M) > 0.5
    pan = WOOD * 0.9 * (0.85 + 0.25 * fbm(AH, AW, 16, "panel", octaves=2))[..., None]
    inset = poly_mask(AH, AW, np.array([(652, -60), (692, -6), (692, 860), (652, 890)]) + M) > 0.5
    pan = np.where((side & ~inset)[..., None], pan * 1.15, pan * 0.9)
    ref = np.where(side[..., None], pan, ref)
    nearm = (xx >= NEAR["x0"]) & (xx < NEAR["x1"]) & (yy < NEAR["base"])
    ref = np.where(nearm[..., None], DARK * 1.5 + 0.01, ref)
    for (by, rh) in NEAR["rows"]:
        b = nearm & (yy >= by) & (yy < by + 10)
        ref = np.where(b[..., None], wood * 1.05, ref)
    for ux in [NEAR["x0"] + 6, 375.0, NEAR["x1"] - 6]:
        b = (np.abs(xx - ux) < 7) & (yy < NEAR["base"]) & (xx < NEAR["x1"] + 1)
        ref = np.where(b[..., None], wood, ref)
    plinth = (xx >= NEAR["x0"] - 6) & (xx < NEAR["x1"] + 2) & (yy >= NEAR["base"] - 18) & (yy < NEAR["base"])
    ref = np.where(plinth[..., None], WOOD * 0.8, ref)
    # light it (the window is its own light)
    lit = ref * Lgt * 2.3
    lit = np.where(inwin[..., None], wincol, lit)
    info = dict(Lgt=Lgt, bounce=bounce, scatter=scatter, win=win, inwin=inwin, opening=opening,
                nearm=nearm | side, side=side)
    return np.clip(lit, 0, 1).astype(np.float32), info


# ---------------------------------------------------------------- shelves and books

def back_segments():
    """Shelf runs on the back wall (between the near bookcase and the arch,
    and right of the arch), skipping pilasters, the window and the arch."""
    segs = []
    runs = [(704.0, 1880.0), (2290.0, PW + M)]
    cuts = [(p - PIL_W / 2 - 3, p + PIL_W / 2 + 3) for p in PILASTERS] + [(895.0, 905.0), (1065.0, 1075.0),
                                                                          (1415.0, 1425.0), (1605.0, 1615.0),
                                                                          (2325.0, 2335.0)]
    for si, (rows, slab) in enumerate(STOREYS):
        for ri, (by, rh) in enumerate(rows):
            for (a, b) in runs:
                pieces = [(a, b)]
                cl = list(cuts)
                if si == 2:
                    cl.append((WINDOW["x0"] - 26, WINDOW["x1"] + 26))
                for (c0, c1) in cl:
                    nxt = []
                    for (p0, p1) in pieces:
                        if c1 <= p0 or c0 >= p1:
                            nxt.append((p0, p1))
                        else:
                            if c0 > p0:
                                nxt.append((p0, c0))
                            if c1 < p1:
                                nxt.append((c1, p1))
                    pieces = nxt
                for (p0, p1) in pieces:
                    if p1 - p0 > 20:
                        segs.append((p0 + 4, p1 - 4, by, rh, (si, ri, int(p0))))
    return segs


def all_back_books():
    out = []
    for (x0, x1, by, rh, key) in back_segments():
        out.extend(BK.books_for_segment(x0, x1, by, rh - 9, ("back",) + key))
    return out


def near_books():
    out = []
    for k, (by, rh) in enumerate(NEAR["rows"]):
        for (a, b) in [(NEAR["x0"] + 14, 368.0), (382.0, NEAR["x1"] - 14)]:
            out.extend(BK.books_for_segment(a, b, by, rh - 12, ("near", k, int(a)), scale=1.35))
    return out


def deep_books():
    out = []
    ax0, ax1 = ARCH["x0"], ARCH["x1"]
    for k in range(9):
        by = 752.0 - 54 * k
        out.extend(BK.books_for_segment(ax0 + 20, ax1 - 20, by, 46, ("deep", k), scale=0.62))
    return out


# ---------------------------------------------------------------- stroke props

def gallery_strokes(Lgt):
    """Gallery slabs, brackets and railings with faint moon rims."""
    st = P.Strokes(max_pts=12)
    r = rng("gallery")

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * 2.3
    for rows, slab in STOREYS:
        if slab is None:
            continue
        for (a, b) in [(704.0, 1880.0), (2290.0, PW + M)]:
            xs = np.linspace(a, b, 12)
            c = WOOD * L((a + b) / 2, slab)
            st.add([(x, slab + 12) for x in xs], 13.0, np.clip(c * 0.8, 0, 1), seed=seed_of("slab", slab, a),
                   opacity=0.97, dry=0.25, ragged=0.08, bristles=5, pool=0.1, streak=0.2, taper0=0, taper1=0, cap=1.0)
            st.add([(x, slab + 1) for x in xs], 2.0, np.clip(c * 1.5 + MOON * 0.04, 0, 1), seed=seed_of("slabh", slab, a),
                   opacity=0.8, dry=0.5, ragged=0.2, bristles=3, pool=0, streak=0.1, taper0=0, taper1=0, cap=1.0)
            # brackets under the slab
            for bx in np.arange(a + 30, b - 10, 66.0):
                c2 = WOOD * L(bx, slab + 30) * 0.8
                st.add([(bx, slab + 25), (bx - 2, slab + 44)], 4.0, np.clip(c2, 0, 1), seed=int(r.integers(1 << 30)),
                       opacity=0.95, dry=0.3, ragged=0.1, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0.6)
            # railing: top rail, bottom rail, turned balusters with a rim on the window side
            top = slab - 64
            st.add([(x, top) for x in xs], 3.2, np.clip(WOOD * L((a + b) / 2, top) * 1.1, 0, 1),
                   seed=seed_of("rail", slab, a), opacity=0.97, dry=0.3, ragged=0.08, bristles=3, pool=0.1,
                   streak=0.15, taper0=0, taper1=0, cap=1.0)
            st.add([(x, top - 2.4) for x in xs], 0.9, np.clip(WOOD * L((a + b) / 2, top) * 1.8 + MOON * 0.07, 0, 1),
                   seed=seed_of("railh", slab, a), opacity=0.75, dry=0.6, ragged=0.25, bristles=2, pool=0, streak=0,
                   taper0=0.05, taper1=0.05, cap=1.0)
            st.add([(x, slab - 6) for x in xs], 2.2, np.clip(WOOD * L((a + b) / 2, slab) * 0.9, 0, 1),
                   seed=seed_of("railb", slab, a), opacity=0.95, dry=0.3, ragged=0.08, bristles=2, pool=0.1,
                   streak=0.1, taper0=0, taper1=0, cap=1.0)
            for bx in np.arange(a + 8, b - 4, 22.0):
                c3 = WOOD * L(bx, top + 30) * 0.95
                st.add([(bx, top + 3), (bx + 0.3, top + 20), (bx, top + 40), (bx, slab - 8)], 2.4, np.clip(c3, 0, 1),
                       seed=int(r.integers(1 << 30)), opacity=0.95, dry=0.25, ragged=0.08, bristles=2, pool=0.12,
                       streak=0.1, taper0=0, taper1=0, cap=1.0)
                st.add([(bx + 1.3, top + 6), (bx + 1.3, slab - 10)], 0.55, np.clip(c3 * 1.7 + MOON * 0.05, 0, 1),
                       seed=int(r.integers(1 << 30)), opacity=0.55, dry=0.6, ragged=0.3, bristles=2, pool=0,
                       streak=0, taper0=0.2, taper1=0.2)
    return st.arrays()


def stair_strokes(Lgt):
    """A wooden stair climbing along the back wall to the first gallery."""
    st = P.Strokes(max_pts=8)
    (x0, y0), (x1, y1) = STAIR["bottom"], STAIR["top"]
    n = STAIR["steps"]
    r = rng("stair")

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * 2.3
    # stringer under the steps
    st.add([(x0 + 8, y0 + 2), (x1 + 6, y1 + 18)], 9.0, np.clip(WOOD * 0.8 * L((x0 + x1) / 2, (y0 + y1) / 2), 0, 1),
           seed=seed_of("stringer"), opacity=0.97, dry=0.3, ragged=0.08, bristles=4, pool=0.1, streak=0.2,
           taper0=0, taper1=0, cap=1.0)
    for k in range(n):
        t = (k + 0.5) / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        c = WOOD * L(x, y)
        # riser (dark) and tread (lit edge)
        st.add([(x - 22, y + 6), (x + 22, y + 6)], 5.5, np.clip(c * 0.6, 0, 1), seed=int(r.integers(1 << 30)),
               opacity=0.97, dry=0.2, ragged=0.06, bristles=3, pool=0.1, streak=0.1, taper0=0, taper1=0, cap=1.0)
        st.add([(x - 24, y), (x + 23, y)], 1.6, np.clip(c * 1.6 + MOON * 0.05, 0, 1), seed=int(r.integers(1 << 30)),
               opacity=0.9, dry=0.4, ragged=0.15, bristles=2, pool=0, streak=0.1, taper0=0, taper1=0, cap=1.0)
    # banister and balusters on the open side
    hx, hy = 0.0, -62.0
    st.add([(x0 + 22 + hx, y0 + hy + 4), (x1 + 22 + hx, y1 + hy + 10)], 2.6,
           np.clip(WOOD * 1.1 * L((x0 + x1) / 2, (y0 + y1) / 2 - 60), 0, 1), seed=seed_of("banister"), opacity=0.97,
           dry=0.3, ragged=0.08, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0, cap=1.0)
    st.add([(x0 + 22 + hx, y0 + hy + 1.6), (x1 + 22 + hx, y1 + hy + 7.6)], 0.8,
           np.clip(WOOD * 1.8 * L((x0 + x1) / 2, (y0 + y1) / 2 - 60) + MOON * 0.07, 0, 1), seed=seed_of("banh"),
           opacity=0.75, dry=0.6, ragged=0.25, bristles=2, pool=0, streak=0, taper0=0.05, taper1=0.05)
    for k in range(0, n, 1):
        t = (k + 0.3) / n
        x = x0 + (x1 - x0) * t + 18
        y = y0 + (y1 - y0) * t
        st.add([(x, y - 2), (x + 0.3, y + hy * 0.5), (x, y + hy + 6)], 1.7, np.clip(WOOD * L(x, y - 30), 0, 1),
               seed=int(r.integers(1 << 30)), opacity=0.95, dry=0.25, ragged=0.08, bristles=2, pool=0.1, streak=0.1,
               taper0=0, taper1=0, cap=1.0)
    # newel post at the bottom
    st.add([(x0 + 24, y0 + 2), (x0 + 24, y0 + hy - 10)], 5.0, np.clip(WOOD * 1.05 * L(x0, y0 - 40), 0, 1),
           seed=seed_of("newel"), opacity=0.97, dry=0.25, ragged=0.06, bristles=3, pool=0.1, streak=0.15,
           taper0=0, taper1=0, cap=1.0)
    st.add([(x0 + 24, y0 + hy - 12), (x0 + 24, y0 + hy - 16)], 7.0, np.clip(WOOD * 1.2 * L(x0, y0 - 60), 0, 1),
           seed=seed_of("newelcap"), opacity=0.97, dry=0.2, ragged=0.06, bristles=3, pool=0.1, streak=0.1)
    return st.arrays()


def cart_strokes(Lgt):
    """A small wooden book cart left on the floor."""
    st = P.Strokes(max_pts=6)
    cx, cy = CART
    r = rng("cart")

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * 2.3
    c = WOOD * L(cx, cy)
    for sx in (-56, 56):
        st.add([(cx + sx, cy - 64), (cx + sx, cy + 6)], 3.0, np.clip(c, 0, 1), seed=seed_of("cartpost", sx),
               opacity=0.97, dry=0.3, ragged=0.06, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0, cap=1.0)
    for yy_ in (cy - 62, cy - 30, cy + 2):
        st.add([(cx - 60, yy_), (cx + 60, yy_)], 3.2, np.clip(c * 1.1, 0, 1), seed=seed_of("cartshelf", yy_),
               opacity=0.97, dry=0.3, ragged=0.06, bristles=3, pool=0.1, streak=0.15, taper0=0, taper1=0, cap=1.0)
        st.add([(cx - 60, yy_ - 2.6), (cx + 60, yy_ - 2.6)], 0.8, np.clip(c * 1.8 + MOON * 0.05, 0, 1),
               seed=seed_of("cartsh", yy_), opacity=0.7, dry=0.6, ragged=0.2, bristles=2, pool=0, streak=0, cap=1.0)
    for sx in (-44, 44):
        st.add([(cx + sx, cy + 12), (cx + sx + 0.4, cy + 12.4)], 6.0, np.clip(c * 0.6, 0, 1), seed=seed_of("wheel", sx),
               opacity=0.97, dry=0.2, ragged=0.08, bristles=2, pool=0.15, streak=0.05)
    cbooks = BK.books_for_segment(cx - 54, cx + 54, cy - 34, 30, ("cart", 0), scale=0.9, flat_rate=0.15) + \
        BK.books_for_segment(cx - 54, cx + 54, cy - 2, 28, ("cart", 1), scale=0.9, flat_rate=0.3)
    return st.arrays(), cbooks


def ladder_strokes(Lgt):
    st = P.Strokes(max_pts=6)
    xb, yb, xt, yt = LADDER

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * 2.3
    for side in (-1, 1):
        st.add([(xb + side * 20, yb + 4), (xt + side * 16, yt)], 2.8, np.clip(WOOD * 1.15 * L(xb, (yb + yt) / 2), 0, 1),
               seed=seed_of("lad", side), opacity=0.97, dry=0.3, ragged=0.08, bristles=3, pool=0.1, streak=0.15,
               taper0=0.02, taper1=0.03, cap=1.0)
    n = 13
    for k in range(n):
        t = (k + 0.5) / n
        y = yb + (yt - yb) * t
        xc = xb + (xt - xb) * t
        hw = 20 + (16 - 20) * t
        st.add([(xc - hw, y), (xc + hw, y + 0.5)], 1.6, np.clip(WOOD * 1.3 * L(xc, y), 0, 1), seed=seed_of("rung", k),
               opacity=0.95, dry=0.3, ragged=0.1, bristles=2, pool=0.1, streak=0.1, taper0=0.05, taper1=0.05,
               cap=1.0)
    return st.arrays()


def board_edges(Lgt):
    """Front edges of every shelf board, restated over the books' feet."""
    st = P.Strokes(max_pts=12)
    r = rng("board-edges")

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * 2.4
    for (x0, x1, by, rh, key) in back_segments():
        xs = np.linspace(x0 - 4, x1 + 4, 8)
        sag = 2.5 * r.random()
        pts = [(x, by + 1.5 + sag * math.sin(math.pi * (x - x0) / max(x1 - x0, 1))) for x in xs]
        c = np.clip(WOOD * 1.5 * L((x0 + x1) / 2, by), 0, 1)
        st.add(pts, 3.2, c, seed=int(r.integers(1 << 30)), opacity=0.95, dry=0.35, ragged=0.12, bristles=4,
               pool=0.1, streak=0.15, taper0=0.0, taper1=0.0, pickup=0.05, cap=1.0)
        st.add([(x, y - 2.6) for x, y in pts], 0.8, np.clip(c * 1.5 + MOON * 0.04, 0, 1), seed=int(r.integers(1 << 30)),
               opacity=0.55, dry=0.6, ragged=0.2, bristles=2, pool=0, streak=0, pickup=0, cap=1.0)
    for (by, rh) in NEAR["rows"]:
        xs = np.linspace(NEAR["x0"], NEAR["x1"], 8)
        c = np.clip(WOOD * 1.4 * L(400, by), 0, 1)
        st.add([(x, by + 2) for x in xs], 5.0, c, seed=int(r.integers(1 << 30)), opacity=0.96, dry=0.35,
               ragged=0.1, bristles=4, pool=0.1, streak=0.15, taper0=0, taper1=0, cap=1.0)
        st.add([(x, by - 1.5) for x in xs], 1.0, np.clip(c * 1.6 + MOON * 0.05, 0, 1), seed=int(r.integers(1 << 30)),
               opacity=0.6, dry=0.6, ragged=0.2, bristles=2, pool=0, streak=0, cap=1.0)
    # moon rim down the near bookcase's edge facing the shaft
    st.add([(703, -10), (703, 878)], 1.2, np.clip(WOOD * 2.0 * L(700, 600) + MOON * 0.10, 0, 1),
           seed=seed_of("nearrim"), opacity=0.7, dry=0.55, ragged=0.25, bristles=2, pool=0, streak=0, cap=1.0)
    return st.arrays()


def floor_piles():
    r = rng("piles")
    out = []
    for (px, n) in [(1200.0, 4), (1990.0, 6), (2120.0, 3)]:
        y = WALL_BASE + 26
        for j in range(n):
            th = 7 + r.random() * 6
            bw = 48 + r.random() * 30
            c = BK.BOOK_COLS[int(r.integers(0, len(BK.BOOK_COLS)))] * (0.85 + 0.3 * r.random())
            out.append(dict(kind="flat", x=px + (r.random() - 0.5) * 10, y=y - th / 2, w=bw, h=th, angle=0.0,
                            col=c, faded=False, mossy=False, seed=int(r.integers(1 << 30))))
            y -= th
    return out


# ---------------------------------------------------------------- shaft (glow)

def shaft_plate():
    yy, xx = _grid()
    ax, ay = SHAFT_TOP[0], SHAFT_TOP[1] - 40
    bx, by = SHAFT_FLOOR
    dx, dy = bx - ax, by - ay
    ln = math.hypot(dx, dy)
    ux, uy = dx / ln, dy / ln
    t = ((xx - ax) * ux + (yy - ay) * uy) / ln
    s = ((xx - ax) * (-uy) + (yy - ay) * ux)
    half = 66 + (SHAFT_W * 0.5 - 66) * np.clip(t, 0, 1)
    sn = s / half
    edge = P.smoothstep(1.15, 0.55, np.abs(sn))
    along = P.smoothstep(0.02, 0.16, t) * P.smoothstep(1.06, 1.0, t)
    q = sn * 9.0
    rays = np.zeros_like(q)
    for k, (fr, amp) in enumerate([(1.0, 0.22), (2.3, 0.13), (5.1, 0.07)]):
        rays += amp * np.sin(q * fr + k * 1.7 + np.sin(t * 3 + k) * 0.6)
    bars = 1 - 0.35 * (P.smoothstep(0.1, 0.0, np.abs(sn - 0.36)) + P.smoothstep(0.1, 0.0, np.abs(sn + 0.36)))
    inten = edge * along * (0.75 + rays) * bars * (0.45 + 0.75 * np.clip(t, 0, 1) ** 1.3)
    inten *= 0.85 + 0.3 * fbm(AH, AW, 140, "shaft-dust", octaves=3)
    glow = inten[..., None] * MOON * 0.26
    light, bars2 = pool_shape()
    pl = light * (1 - 0.7 * bars2)
    glow = glow + (P.blur(pl, 22) * 0.10 + P.blur(pl, 80) * 0.07)[..., None] * MOON
    # the window blooms
    wx0, wx1, wt, ws = WINDOW["x0"], WINDOW["x1"], WINDOW["top"], WINDOW["sill"]
    wm = ((xx > wx0) & (xx < wx1) & (yy > wt) & (yy < ws)).astype(np.float32)
    glow = glow + (P.blur(wm, 18) * 0.16 + P.blur(wm, 70) * 0.10)[..., None] * MOON
    return glow.astype(np.float32)


def shaft_motes(f, cam):
    """Dust motes drifting inside the shaft only (on twos)."""
    f2 = f - (f % 2)
    t = f2 / T.FPS
    r = rng("motes")
    n = 260
    tt, ss = r.random(n), r.random(n) * 2 - 1
    ph = r.random(n) * 6.283
    sp = 0.004 + 0.01 * r.random(n)
    ax, ay = SHAFT_TOP[0], SHAFT_TOP[1] - 40
    bx, by = SHAFT_FLOOR
    ln = math.hypot(bx - ax, by - ay)
    canvas = np.zeros((T.H, T.W, 4), np.float32)
    st = P.Strokes(max_pts=2)
    for i in range(n):
        ti = 0.1 + 0.9 * ((tt[i] + sp[i] * t * 0.6) % 1.0)
        si = ss[i] + 0.08 * math.sin(t * 0.5 + ph[i])
        half = 66 + (SHAFT_W * 0.5 - 66) * ti
        px = ax + (bx - ax) * ti - (by - ay) / ln * si * half * 0.9 + 6 * math.sin(t * 0.7 + ph[i] * 3)
        py = ay + (by - ay) * ti + (bx - ax) / ln * si * half * 0.9 + 4 * math.cos(t * 0.6 + ph[i] * 2)
        edge = max(0.0, 1.0 - abs(si) ** 3)
        b = edge * (0.3 + 0.7 * ti) * (0.4 + 0.6 * ((i * 7919) % 100) / 100.0)
        b *= 0.6 + 0.4 * math.sin(t * 2.0 + ph[i] * 5) ** 2
        sx, sy = C.to_screen(cam, px, py)
        if not (0 <= sx < T.W and 0 <= sy < T.H) or b < 0.05:
            continue
        w = (0.7 + 0.9 * ((i * 31) % 10) / 10.0) * cam[0]
        st.add([(sx, sy), (sx + 0.3, sy + 0.2)], w, MOON * min(1.0, b * 1.4), seed=i, opacity=min(1.0, 0.5 + b),
               dry=0, ragged=0.05, bristles=2, pool=0, streak=0, paper_k=0, pickup=0)
    P.draw(canvas, st, P.paper_for(T.H, T.W, "screen-paper"))
    return canvas[..., :3] * 1.1


# ---------------------------------------------------------------- fg pillar

def pillar_reference():
    yy, xx = _grid()
    cx, R = -30.0, 112.0
    d = (xx - cx) / R
    m = (np.abs(d) < 1.0).astype(np.float32)
    base = (yy > 1180) & (np.abs(d) < 1.12)
    m = np.maximum(m, base.astype(np.float32))
    nz = np.sqrt(np.clip(1 - d * d, 0, 1))
    ndl = np.clip(0.8 * d + 0.3 * nz, 0, 1)
    col = DARK * 1.6 + np.array([0.06, 0.06, 0.12], np.float32) * ndl[..., None]
    rim = P.smoothstep(0.82, 0.97, d) * (1 - P.smoothstep(0.985, 1.0, d))
    col = col + rim[..., None] * MOON * 0.20
    th = np.arcsin(np.clip(d, -0.999, 0.999))
    flute = np.abs(np.mod(th / 0.2 + 0.5, 1.0) - 0.5)
    groove = P.smoothstep(0.18, 0.05, flute)
    col = col * (1 - 0.45 * groove * (0.4 + 0.6 * nz))[..., None]
    ridge = P.smoothstep(0.42, 0.5, flute) * P.smoothstep(0.3, 0.9, d)
    col = col + ridge[..., None] * MOON * 0.06
    for yb, hb in [(1186.0, 10.0), (1210.0, 14.0), (1240.0, 30.0)]:
        band = (np.abs(yy - yb) < hb / 2) & (np.abs(d) < 1.12)
        col = np.where(band[..., None], col * 1.25, col)
    col = col * (1 - 0.2 * (np.mod(yy, 46.0) < 2.0))[..., None]
    col = col * (0.85 + 0.3 * fbm(AH, AW, 30, "pillar", octaves=3))[..., None]
    return col.astype(np.float32), soft(m, 0.8)


# ---------------------------------------------------------------- paint

def light_fn(Lgt, k=2.25):
    def f(x, y):
        return Lgt[int(min(max(y + M, 0), AH - 1)), int(min(max(x + M, 0), AW - 1))] * k
    return f


def paint_hall(paper, variant="closed"):
    """Paint the hall plate. Returns (canvas, info)."""
    yy, xx = _grid()
    log("hall: reference")
    ref, info = structure_reference()
    Lgt = info["Lgt"]
    flc, joints, sid = floor_stones()
    Lf = AMB[None, None] * 0.95 + MOON * (info["bounce"] ** 1.6 * 0.30 + info["scatter"] * 0.10)[..., None]
    floor = yy >= WALL_BASE
    ref = np.where((floor & ~info["nearm"])[..., None], np.clip(flc * Lf * 2.3, 0, 1), ref)
    canvas = P.new_canvas(AH, AW)
    ang = P.orientation(ref, sigma=1.5, rho=5, base=np.where(yy > WALL_BASE, 0.0, math.pi / 2), base_weight=0.6)
    log("hall: structure passes")
    P.paint(canvas, ref, ang, [
        dict(radius=16, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=7, spacing=1.0, steps=(2, 8), opacity=0.93, dry=0.4, cthresh=0.08, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.15, use_err=True, err_thresh=0.02),
        dict(radius=3.2, spacing=1.05, steps=(2, 7), opacity=0.9, dry=0.45, cthresh=0.06, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.07, pickup=0.1, use_err=True, err_thresh=0.03),
        dict(radius=1.6, spacing=1.15, steps=(2, 6), opacity=0.88, dry=0.5, cthresh=0.05, wob=0.25,
             jitter=(0.04, 0.02, 0.04), streak=0.06, pickup=0.05, use_err=True, err_thresh=0.045),
    ], paper, "hall-struct")
    # books: deep room (sunk in blue air), back wall, piles
    log("hall: books")
    lf = light_fn(Lgt)
    deep = deep_books()
    b1, a1 = BK.book_strokes(deep, lambda x, y: lf(x, y) * 0.6, haze=(AIR * 1.1, lambda x, y: 0.55),
                             offset=(M, M))
    P.draw(canvas, b1, paper, clip=info["opening"].astype(np.float32))
    P.draw(canvas, a1, paper, clip=info["opening"].astype(np.float32))
    back = all_back_books() + floor_piles()
    amt = {"closed": 0.0, "half": 0.5, "open": 1.0}[variant]
    if amt == 0.0:
        b2, a2 = BK.book_strokes(back, lf, offset=(M, M))
        P.draw(canvas, b2, paper)
        P.draw(canvas, a2, paper)
    else:
        P.draw(canvas, BK.open_book_strokes(back, lf, amount=amt, offset=(M, M)), paper)
    # shade under each board falls on the tops of the books below it
    shade = np.zeros((AH, AW), np.float32)
    for rows, slab in STOREYS:
        for (by, rh) in rows:
            under = by - rh + 7
            shade = np.maximum(shade, P.smoothstep(under + 14, under, yy) * (yy >= under) * 0.5)
    shade *= (yy < WALL_BASE) & (xx > 704) & ~info["opening"]
    canvas[..., :3] *= (1 - shade)[..., None]
    P.draw(canvas, board_edges(Lgt) if False else _offset(board_edges(Lgt)), paper)
    P.draw(canvas, _offset(gallery_strokes(Lgt)), paper)
    P.draw(canvas, _offset(ladder_strokes(Lgt)), paper)
    P.draw(canvas, _offset(stair_strokes(Lgt)), paper)
    cst, cbooks = cart_strokes(Lgt)
    P.draw(canvas, _offset(cst), paper)
    b3, a3 = BK.book_strokes(cbooks, lf, offset=(M, M))
    P.draw(canvas, b3, paper)
    P.draw(canvas, a3, paper)
    # the near bookcase: its books, bigger and closer
    nb = near_books()
    if amt == 0.0:
        b4, a4 = BK.book_strokes(nb, lambda x, y: lf(x, y) * 1.05, offset=(M, M), light_side=1.0)
        P.draw(canvas, b4, paper)
        P.draw(canvas, a4, paper)
    else:
        P.draw(canvas, BK.open_book_strokes(nb, lambda x, y: lf(x, y) * 1.05, amount=amt, offset=(M, M)), paper)
    nshade = np.zeros((AH, AW), np.float32)
    for (by, rh) in NEAR["rows"]:
        under = by - rh + 10
        nshade = np.maximum(nshade, P.smoothstep(under + 18, under, yy) * (yy >= under) * 0.5)
    nshade *= ((xx >= NEAR["x0"]) & (xx < NEAR["x1"]) & (yy < NEAR["base"]))
    canvas[..., :3] *= (1 - nshade)[..., None]
    P.draw(canvas, _offset(board_edges(Lgt)), paper)
    # the pool: cool light laid over the painted floor, never clipped
    log("hall: pool and air")
    light, bars = pool_shape()
    pl = light * (1 - 0.72 * bars)
    gain = 1.0 + pl[..., None] * (MOON / MOON.max()) ** 1.5 * 2.9
    canvas[..., :3] = soft_light(canvas[..., :3], gain) + (pl[..., None] * MOON * 0.035)
    # a faint sheen of moonlight across the stone toward the camera
    sheen = np.exp(-(((xx - SHAFT_FLOOR[0] + (yy - SHAFT_FLOOR[1]) * 0.35) / 380.0) ** 2)) * \
        P.smoothstep(WALL_BASE + 40, SHAFT_FLOOR[1] + 40, yy) * P.smoothstep(PH + 60, SHAFT_FLOOR[1], yy)
    gloss = (0.5 + 0.5 * fbm(AH, AW, 40, "gloss", octaves=2)) * (1 - joints * 0.8)
    canvas[..., :3] += (sheen * gloss * floor * 0.045)[..., None] * MOON
    # moonlit air hangs low around the pool; the near floor falls into shadow
    cxp, cyp = SHAFT_FLOOR
    airg = np.exp(-(((xx - cxp) / 560.0) ** 2 + ((yy - cyp + 140) / 300.0) ** 2))
    canvas[..., :3] += (airg * 0.04)[..., None] * MOON
    canvas[..., :3] *= (1 - 0.40 * P.smoothstep(1120, PH + 60, yy))[..., None]
    # blue atmosphere in the depth and up into the vault: lift, never black
    veil = (1 - P.smoothstep(-200, 760, yy)) * (0.7 + 0.3 * fbm(AH, AW, 220, "veil", octaves=3))
    canvas[..., :3] = canvas[..., :3] * (1 - veil[..., None] * 0.55) + AIR * veil[..., None] * 0.55
    dep = info["opening"] * 0.35
    canvas[..., :3] = canvas[..., :3] * (1 - dep[..., None]) + AIR * 1.2 * dep[..., None]
    canvas[..., :3] = np.maximum(canvas[..., :3], DARK * 0.9)
    P.finish(canvas, paper, grain=0.05, tooth=0.02)
    return canvas, info


def _offset(a):
    a = dict(a)
    a["xs"] = a["xs"] + M
    a["ys"] = a["ys"] + M
    return a


def paint_all():
    paper = P.paper_for(AH, AW, "hall")
    canvas, info = paint_hall(paper)
    C.save_plate("hall_bg", canvas)
    log("hall: pillar")
    pref, pmask = pillar_reference()
    pc = P.new_canvas(AH, AW)
    pang = P.orientation(pref, 2, 6, base=math.pi / 2, base_weight=1.0)
    pm = (pmask > 0.5).astype(np.float32)
    P.paint(pc, pref, pang, [
        dict(radius=14, spacing=0.9, steps=(3, 10), opacity=0.96, dry=0.35, cthresh=0.08, wob=0.15,
             jitter=(0.06, 0.02, 0.05), streak=0.08, mask=pm),
        dict(radius=5, spacing=1.0, steps=(2, 8), opacity=0.92, dry=0.5, cthresh=0.06, wob=0.15,
             jitter=(0.06, 0.02, 0.05), streak=0.08, mask=pm, use_err=True, err_thresh=0.02),
    ], paper, "hall-pillar", clip=pmask)
    P.finish(pc, paper, grain=0.04)
    C.save_plate("hall_pillar", pc)
    log("hall: shaft")
    glow = shaft_plate()
    np.save(C.plate_path("hall_shaft"), glow.astype(np.float16))


# ---------------------------------------------------------------- render

def render_cut(cut_id, f, bg_name="hall_bg"):
    bgp = C.load_plate(bg_name)
    pil = C.load_plate("hall_pillar")
    sh = C.load_plate("hall_shaft")
    cam = T.camera(cut_id, f)
    bg = C.sample(bgp, cam)
    bg[..., 3] = 1.0
    fg = C.sample(pil, C.parallax_cam(cut_id, f, PILLAR_PARALLAX))
    glow = C.sample(sh, cam)
    fm = f if cut_id != "5d" else T.cut_range("5d")[0]
    glow = glow + shaft_motes(fm, cam)
    return dict(bg=bg, fg=fg, glow=glow)


def render(f):
    return render_cut(CUT, f)


if __name__ == "__main__":
    import time
    t0 = time.time()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "paint"
    if cmd == "paint":
        paint_all()
    log(f"done {time.time() - t0:.1f}s")
