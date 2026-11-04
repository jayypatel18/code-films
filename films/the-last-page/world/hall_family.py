"""The hall family: cuts 3, 4a, 4b, 5a, 5b, 5c and 6a. All reuse the hall's
books, light and painter.

  3   close-up background: a dark, out of focus library, warm lantern spill
      from bottom right (ANCHORS["3"]["lantern_glow_from_screen"]).
  4a  the enormous open book on a carved lectern in the hall, moonlit.
  4b  the page itself filling the frame: aged paper, fibres, foxing, a ruled edge.
  5a  low angle on a high shelf: one book trembles on hard held key poses.
  5b  the whole row trembles, dust sifts down in thin streams.
  5c  floor and shelves behind the medium shot, lantern spill from lower left.
  6a  every book flies open: closed hall, one snapped half open drawing, then open.

Run `python3 world/hall_family.py paint [cut ...]` through tools/heavy.py.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise, rng, noise1, seed_of, ridged  # noqa: E402
from engine.paint_shapes import poly_mask, wobble, soft  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402
import books as BK  # noqa: E402
import cut2_hall as H  # noqa: E402

M = MARGIN
MOON = H.MOON
AMB = H.AMB
WARM = np.array(T.LIGHTS["hall"]["warm"], np.float32)
WOOD = H.WOOD
DARK = H.DARK
AIR = H.AIR


def dims(cut):
    pw, ph = T.plate_size(cut)
    return pw, ph, pw + 2 * M, ph + 2 * M


def grid(cut):
    pw, ph, aw, ah = dims(cut)
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    return yy - M, xx - M


def screen_to_plate0(cut, sx, sy):
    f0, _ = T.cut_range(cut)
    return T.screen_to_plate(cut, f0, sx, sy)


def offset(a, dx=M, dy=M):
    a = dict(a)
    a["xs"] = a["xs"] + dx
    a["ys"] = a["ys"] + dy
    return a


# ---------------------------------------------------------------- close shelves (3 and 5c)

def close_shelves(cut, lantern_plate, blur_sigma, floor_y, key, rh=190.0, scale=2.4, warm_k=1.0):
    """A nearer wall of shelves, lit by the lantern on the floor and cool
    moon ambient, then softened as a background behind the fox."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    lx, ly = lantern_plate
    dl = np.sqrt(((xx - lx) / 900.0) ** 2 + ((yy - ly) / 700.0) ** 2)
    warm = np.exp(-dl ** 2 * 1.6) * warm_k
    climb = P.smoothstep(-100, ph, yy)
    Lgt = AMB[None, None] * (0.55 + 0.5 * climb)[..., None] + MOON[None, None] * 0.10 + WARM * (warm * 0.85)[..., None]
    ref = np.zeros((ah, aw, 3), np.float32)
    ref[:] = DARK * 1.5 + 0.01
    wood = WOOD * (0.8 + 0.4 * fbm(ah, aw, 30, (key, "wood"), octaves=3))[..., None]
    rows = []
    y = floor_y - 14
    while y > -M - rh:
        rows.append(y)
        y -= rh
    for by in rows:
        ref = np.where(((yy >= by) & (yy < by + 16))[..., None], wood * 1.1, ref)
    ups = list(np.arange(-M + 60, aw, 470.0) + rng(key, "ups").random() * 120)
    for ux in ups:
        ref = np.where(((np.abs(xx - ux) < 13) & (yy < floor_y))[..., None], wood, ref)
    flc = np.zeros_like(ref)
    floor = yy >= floor_y
    fl = H.STONE * 0.6 * (0.8 + 0.35 * fbm(ah, aw, 30, (key, "floor"), octaves=4))[..., None]
    ref = np.where(floor[..., None], fl, ref)
    lit = np.clip(ref * Lgt * 2.3, 0, 1)
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    ang = P.orientation(lit, 2, 6, base=np.where(floor, 0.0, math.pi / 2), base_weight=0.6)
    P.paint(canvas, lit, ang, [
        dict(radius=18, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=7, spacing=1.0, steps=(2, 8), opacity=0.92, dry=0.4, cthresh=0.08, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.02),
    ], paper, (key, "struct"))
    books = []
    for i, by in enumerate(rows):
        for j in range(len(ups) + 1):
            a = (ups[j - 1] + 16) if j > 0 else -M
            b = (ups[j] - 16) if j < len(ups) else aw - M
            if b - a > 40:
                books.extend(BK.books_for_segment(a, b, by, rh - 22, (key, i, j), scale=scale))

    def lf(x, y):
        return Lgt[int(min(max(y + M, 0), ah - 1)), int(min(max(x + M, 0), aw - 1))] * 2.25
    side = 1.0 if lx > pw / 2 else -1.0
    b1, a1 = BK.book_strokes(books, lf, offset=(M, M), light_side=side)
    P.draw(canvas, b1, paper)
    P.draw(canvas, a1, paper)
    shade = np.zeros((ah, aw), np.float32)
    for by in rows:
        under = by - rh + 16
        shade = np.maximum(shade, P.smoothstep(under + 30, under, yy) * (yy >= under) * 0.5)
    canvas[..., :3] *= (1 - shade * (yy < floor_y))[..., None]
    # warm sheen on the floor near the lantern
    canvas[..., :3] += (warm * floor * 0.10)[..., None] * WARM
    P.finish(canvas, paper, grain=0.05)
    if blur_sigma > 0:
        for c in range(3):
            canvas[..., c] = P.blur(canvas[..., c], blur_sigma)
        # a whisper of grain back on top so the softened plate still reads as paint
        canvas[..., :3] *= (1 + 0.04 * (paper - 0.5))[..., None]
    canvas[..., 3] = 1.0
    return canvas


def defocus(cut, sharp, lantern_plate, key, sigma=10.0, dark=0.42, floor_y=1e9, n_bokeh=48):
    """Turn a sharp painted shelf plate into a deep, defocused background the
    way a lens does it: a disc shaped blur (so narrow bright spines become soft
    round blurs, never cones or flames), the whole thing sunk into deep blue, a
    few soft round bokeh discs of varied warm and cool colour, and the warm
    spill of the lantern on the floor."""
    from scipy.signal import fftconvolve
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    rgb = sharp[..., :3].copy()
    lx, ly = lantern_plate
    dl = np.sqrt(((xx - lx) / 1000.0) ** 2 + ((yy - ly) / 760.0) ** 2)
    warm = np.exp(-dl ** 2 * 1.4)
    # lens blur: a soft edged disc kernel, then a light gaussian to melt the disc edges
    R = int(sigma * 1.6)
    ky, kx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    ker = np.clip(R - np.sqrt(kx * kx + ky * ky) + 0.5, 0, 1)
    ker /= ker.sum()
    for c in range(3):
        rgb[..., c] = fftconvolve(rgb[..., c], ker, mode="same")
        rgb[..., c] = P.blur(rgb[..., c], sigma * 0.35)
    # tame the brightest warm spines so nothing glows like a flame
    lum = P.lum(rgb)
    hi = P.smoothstep(0.25, 0.6, lum)
    rgb = rgb * (1 - 0.35 * hi)[..., None] + lum[..., None] * (0.35 * hi)[..., None] * 0.8
    deep = np.array([0.030, 0.040, 0.105], np.float32)
    rgb = rgb * dark * (1 - 0.35 * P.smoothstep(ph * 0.9, -100, yy))[..., None]
    rgb = rgb * (1 - 0.4 * (1 - warm))[..., None] + deep * (1 - warm)[..., None] * 0.6
    # bokeh: a few soft round discs, warm near the lantern, cool blue away from it,
    # each with a faintly brighter rim like a real lens, none overlapping much
    r = rng(key, "bokeh2")
    bok = np.zeros((ah, aw, 3), np.float32)
    placed = []
    tries = 0
    while len(placed) < n_bokeh and tries < 4000:
        tries += 1
        x0 = r.random() * aw
        y0 = M + r.random() * (min(floor_y, ph) - 60)
        rad = 14 + 22 * r.random()
        if any((x0 - px) ** 2 + (y0 - py) ** 2 < (rad + pr) ** 2 * 0.6 for (px, py, pr) in placed):
            continue
        w0 = float(warm[int(min(y0, ah - 1)), int(min(x0, aw - 1))])
        if r.random() > 0.25 + 0.75 * w0 and r.random() < 0.6:
            continue
        placed.append((x0, y0, rad))
        if r.random() < 0.4 + 0.5 * w0:
            c = np.array([1.0, 0.70, 0.40]) * (0.55 + 0.35 * r.random())
        elif r.random() < 0.5:
            c = np.array([0.45, 0.55, 0.95]) * (0.35 + 0.25 * r.random())
        else:
            c = np.array([0.80, 0.60, 0.75]) * (0.35 + 0.25 * r.random())
        x1, x2 = max(int(x0 - rad - 3), 0), min(int(x0 + rad + 4), aw)
        y1, y2 = max(int(y0 - rad - 3), 0), min(int(y0 + rad + 4), ah)
        Y, X = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        d = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) / rad
        disc = np.clip((1.0 - d) * rad / 2.5, 0, 1) * (0.75 + 0.25 * np.clip(d, 0, 1) ** 3)
        bok[y1:y2, x1:x2] += disc[..., None] * c[None, None] * (0.10 + 0.14 * w0)
    rgb = rgb + bok
    rgb = rgb + (warm ** 1.5 * 0.16)[..., None] * WARM
    paper = P.paper_for(ah, aw, key)
    rgb = rgb * (1 + 0.05 * (paper - 0.5))[..., None]
    out = np.empty((ah, aw, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0, 1)
    out[..., 3] = 1.0
    return out


def lantern_glow_plate(cut, lantern_plate, strength=0.35):
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    lx, ly = lantern_plate
    d = np.sqrt(((xx - lx) / 520.0) ** 2 + ((yy - ly) / 420.0) ** 2)
    g = np.exp(-d ** 2 * 1.4) * strength
    return (g[..., None] * WARM).astype(np.float32)


# ---------------------------------------------------------------- 4a lectern

A4 = T.ANCHORS["4a"]


def lectern_layer():
    """The carved lectern and the enormous open book, as an alpha layer
    (strokes over a reference). Returns (canvas, page_mask)."""
    cut = "4a"
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    cx = float(A4["lectern_centre_x"])
    floor_y = float(A4["floor_y"])
    bx0, bx1 = A4["book_span_x"]
    gy = float(A4["book_top_y"])
    lan = np.array(A4["lantern"], np.float32)
    # light: moon from upper right (the shaft falls on the book), lantern warm from lower left
    dl = np.sqrt(((xx - lan[0]) / 700.0) ** 2 + ((yy - lan[1]) / 520.0) ** 2)
    warm = np.exp(-dl ** 2 * 1.5)
    beam = np.exp(-(((xx - (cx - 40)) / 420.0) ** 2 + ((yy - gy) / 260.0) ** 2))
    Lgt = AMB[None, None] * 1.0 + MOON * (beam * 0.34)[..., None] + WARM * (warm * 0.65)[..., None]
    ref = np.zeros((ah, aw, 3), np.float32)
    alpha = np.zeros((ah, aw), np.float32)

    def put(poly, col, a=1.0):
        nonlocal ref, alpha
        m = poly_mask(ah, aw, np.asarray(poly) + M) * a
        ref = ref * (1 - m[..., None]) + np.asarray(col, np.float32) * m[..., None]
        alpha = np.maximum(alpha, m)
        return m
    wood = WOOD * 1.0
    # base: stepped plinth
    put([(cx - 190, floor_y), (cx + 190, floor_y), (cx + 170, floor_y - 40), (cx - 170, floor_y - 40)], wood * 0.8)
    put([(cx - 140, floor_y - 40), (cx + 140, floor_y - 40), (cx + 120, floor_y - 74), (cx - 120, floor_y - 74)], wood * 0.9)
    # carved column with bulges
    prof = []
    for k in range(21):
        t = k / 20
        y = floor_y - 74 - t * (floor_y - 74 - (gy + 70))
        wdt = 46 + 14 * math.sin(t * math.pi * 3) ** 2 + (20 if t < 0.1 else 0) + (24 if t > 0.92 else 0)
        prof.append((y, wdt))
    left = [(cx - w, y) for y, w in prof]
    right = [(cx + w, y) for y, w in prof[::-1]]
    put(left + right, wood)
    # the reading board, tilted toward us: a trapezoid, the far edge narrower
    TOP = gy - 175
    put([(bx0 - 34, gy + 46), (bx1 + 34, gy + 46), (bx1 + 30, gy + 78), (bx0 - 30, gy + 78)], wood * 0.6)
    put([(bx0 - 34, gy + 46), (bx1 + 34, gy + 46), (bx1 + 10, TOP - 8), (bx0 - 10, TOP - 8)], wood * 0.8)
    # bracket from the column up under the board
    put([(cx - 70, gy + 78), (cx + 70, gy + 78), (cx + 40, gy + 120), (cx - 40, gy + 120)], wood * 0.7)
    # book: leather covers showing as a border, then the page block edge at the near side
    cover = np.array([0.34, 0.12, 0.10], np.float32)
    put([(bx0 - 16, gy + 40), (bx1 + 16, gy + 40), (bx1 - 2, TOP - 2), (bx0 + 2, TOP - 2)], cover)
    put([(bx0 - 4, gy + 36), (cx, gy + 44), (bx1 + 4, gy + 36), (bx1 + 4, gy + 18), (cx, gy + 26), (bx0 - 4, gy + 18)],
        BK.PAGE * 0.7)

    def page(side):
        # outer edge x, the near edge bows down at the gutter, the far edge bows up
        x_out = bx0 if side < 0 else bx1
        near = []
        far = []
        for t in np.linspace(0, 1, 21):
            x = cx + (x_out - cx) * t
            near.append((x, gy + 20 + 26 * (1 - t) ** 3 - 16 * math.sin(math.pi * t ** 0.8) * 0.7))
        for t in np.linspace(1, 0, 21):
            x = cx + (x_out + (-1 if side < 0 else 1) * -14 - cx) * t
            far.append((x, TOP + 10 + 22 * (1 - t) ** 3 - 14 * math.sin(math.pi * t ** 0.8) * 0.7))
        return near + far
    pm_l = put(page(-1), BK.PAGE)
    pm_r = put(page(1), BK.PAGE * 1.02)
    page_mask = np.clip(pm_l + pm_r, 0, 1)
    # paper: yellowed, each page bowing (lighter across the middle of its curve,
    # shadowed into the gutter and slightly at the outer edge), foxing, a tide mark
    tpage = np.clip(np.abs(xx - cx) / ((bx1 - bx0) / 2), 0, 1)
    bow = np.sin(np.pi * np.clip(tpage * 0.9 + 0.08, 0, 1)) ** 0.6
    gut = np.exp(-((xx - cx) / 26.0) ** 2)
    pv = fbm(ah, aw, 40, "4a-paper", octaves=4)
    pcol = np.array([0.90, 0.78, 0.50], np.float32) * (0.88 + 0.16 * pv)[..., None]
    pcol = pcol * (0.70 + 0.30 * bow)[..., None] * (1 - 0.55 * gut)[..., None]
    fox = P.smoothstep(0.80, 0.92, fbm(ah, aw, 14, "4a-fox", octaves=3))
    pcol = P.mix(pcol, np.array([0.58, 0.42, 0.26], np.float32), fox * 0.35)
    tide = P.smoothstep(0.03, 0.0, np.abs(fbm(ah, aw, 120, "4a-tide", octaves=2) - 0.6))
    pcol = pcol * (1 - 0.08 * tide)[..., None]
    ref = np.where(page_mask[..., None] > 0.5, pcol, ref)
    # model the wood: moon on the right flank, lantern warmth on the left, darker toward the floor
    side = np.clip((xx - cx) / 90.0, -1, 1)
    wood_m = (alpha > 0.5) & (page_mask < 0.5)
    shade = (0.62 + 0.42 * side) * (0.75 + 0.35 * P.smoothstep(floor_y, gy, yy))
    ref = np.where(wood_m[..., None], ref * shade[..., None] + (np.clip(-side, 0, 1) * 0.06)[..., None] * WARM, ref)
    ref = np.where(wood_m[..., None], ref * (0.85 + 0.3 * fbm(ah, aw, 6, "4a-grain", octaves=2))[..., None], ref)
    lit = np.clip(ref * Lgt * 1.8, 0, 1)
    # paint
    paper = P.paper_for(ah, aw, "4a")
    canvas = P.new_canvas(ah, aw)
    m = (alpha > 0.5).astype(np.float32)
    ang = P.orientation(lit, 1.5, 5, base=math.pi / 2, base_weight=0.5)
    P.paint(canvas, lit, ang, [
        dict(radius=9, spacing=0.9, steps=(2, 6), opacity=0.96, dry=0.3, cthresh=0.12, wob=0.25,
             jitter=(0.04, 0.02, 0.05), streak=0.08, pickup=0.15, mask=m),
        dict(radius=4, spacing=1.0, steps=(2, 6), opacity=0.93, dry=0.4, cthresh=0.08, wob=0.25,
             jitter=(0.04, 0.02, 0.05), streak=0.08, mask=m, use_err=True, err_thresh=0.03),
        dict(radius=1.8, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.45, cthresh=0.06, wob=0.25,
             jitter=(0.03, 0.02, 0.04), streak=0.06, mask=m, use_err=True, err_thresh=0.05),
    ], paper, "4a-lectern", clip=alpha)
    # hand placed: carving lines on the column, gilt corners, page edge lines, ribbon
    st = P.Strokes(max_pts=12)

    def L(x, y):
        return Lgt[int(min(max(y + M, 0), ah - 1)), int(min(max(x + M, 0), aw - 1))] * 2.0
    for k in range(1, 20, 2):
        y, w = prof[k]
        st.add([(cx - w + 4, y), (cx + w - 4, y)], 1.2, np.clip(WOOD * 0.5 * L(cx, y), 0, 1), seed=seed_of("carve", k),
               opacity=0.8, dry=0.4, ragged=0.2, bristles=2, pool=0, streak=0, taper0=0.2, taper1=0.2)
        st.add([(cx + w * 0.3, y - 3), (cx + w - 6, y - 3)], 0.9, np.clip(WOOD * 1.8 * L(cx + 30, y) + MOON * 0.05, 0, 1),
               seed=seed_of("carveh", k), opacity=0.6, dry=0.6, ragged=0.3, bristles=2, pool=0, streak=0)
    for k in range(6):
        yk = gy + 24 + k * 2.4
        st.add([(bx0 + 2, yk), (cx, yk + 8), (bx1 - 2, yk)], 0.6, np.clip(BK.PAGE * 0.5 * L(cx, yk), 0, 1),
               seed=seed_of("pedge", k), opacity=0.6, dry=0.5, ragged=0.2, bristles=2, pool=0, streak=0)
    # a few pages curling up at the outer edges, drawn as thin lit lines
    for side in (-1, 1):
        x_out = bx0 if side < 0 else bx1
        for k in range(4):
            xo = x_out - side * (6 + k * 5)
            st.add([(xo, gy + 18), (xo - side * 3, gy - 70), (xo - side * 1, TOP + 14)], 0.7,
                   np.clip(np.array([0.95, 0.88, 0.70]) * L(xo, gy - 60) * 0.9, 0, 1), seed=seed_of("curl", side, k),
                   opacity=0.5, dry=0.6, ragged=0.25, bristles=2, pool=0, streak=0, taper0=0.2, taper1=0.3)
    # the page block's thickness down the outer edges: stacked page lines
    for side in (-1, 1):
        x_out = bx0 if side < 0 else bx1
        for k in range(7):
            xo = x_out + side * (2 + k * 1.6)
            st.add([(xo, gy + 34), (xo + side * 1.5, gy - 60), (xo, TOP + 8)], 0.6,
                   np.clip(BK.PAGE * (0.45 + 0.1 * (k % 2)) * L(xo, gy - 60), 0, 1), seed=seed_of("block", side, k),
                   opacity=0.7, dry=0.45, ragged=0.2, bristles=2, pool=0, streak=0, taper0=0.05, taper1=0.05)
    # worn leather: scuffs and lighter worn corners on the boards
    for (wx, wy) in [(bx0 - 10, gy + 30), (bx1 + 10, gy + 30), (bx0 - 4, TOP + 6), (bx1 + 4, TOP + 6), (cx, gy + 40)]:
        st.add([(wx - 8, wy), (wx + 8, wy + 1)], 3.0, np.clip(np.array([0.55, 0.30, 0.22]) * L(wx, wy) * 1.2, 0, 1),
               seed=seed_of("scuff", wx, wy), opacity=0.6, dry=0.6, ragged=0.4, bristles=3, pool=0, streak=0.1)
    # the gutter crease
    st.add([(cx, TOP + 12), (cx + 1, gy - 60), (cx, gy + 28)], 1.4, np.clip(np.array([0.30, 0.22, 0.14]) * L(cx, gy - 60), 0, 1),
           seed=seed_of("gutter"), opacity=0.7, dry=0.4, ragged=0.2, bristles=2, pool=0, streak=0, taper0=0.2, taper1=0.2)
    ribbon = [(cx + 4, gy + 40), (cx + 10, gy + 80), (cx + 5, gy + 118), (cx + 13, gy + 160)]
    st.add(ribbon, 3.2, np.clip(np.array([0.55, 0.10, 0.10]) * L(cx, gy + 100), 0, 1), seed=seed_of("ribbon"),
           opacity=0.95, dry=0.2, ragged=0.1, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0.2)
    for (gx_, gy_) in [(bx0 - 8, gy + 38), (bx1 + 8, gy + 38), (bx0 + 2, TOP), (bx1 - 2, TOP)]:
        st.add([(gx_ - 7, gy_), (gx_ + 7, gy_)], 3.5, np.clip(BK.GILT * L(gx_, gy_) * 1.4, 0, 1),
               seed=seed_of("gilt", gx_, gy_), opacity=0.85, dry=0.3, ragged=0.2, bristles=2, pool=0.1, streak=0)
    P.draw(canvas, offset(st.arrays()), paper)
    P.finish(canvas, paper, grain=0.04)
    return canvas


def backdrop_from_hall(cut, region, darken=0.8, blur=1.2):
    """Make a backdrop plate for another hall cut by resampling a region of
    the painted hall (x0, y0, x1, y1 in hall plate coords)."""
    pw, ph, aw, ah = dims(cut)
    hb = C.load_plate("hall_bg")
    x0, y0, x1, y1 = region
    out = np.empty((ah, aw, 4), np.float32)
    for c in range(4):
        im = Image.fromarray(np.ascontiguousarray(hb[..., c]), "F")
        out[..., c] = np.asarray(im.resize((aw, ah), Image.BICUBIC, box=(x0 + M, y0 + M, x1 + M, y1 + M)), np.float32)
    if blur > 0:
        for c in range(3):
            out[..., c] = P.blur(out[..., c], blur)
    out[..., :3] *= darken
    out[..., 3] = 1.0
    return np.clip(out, 0, 1)


def beam_4a():
    cut = "4a"
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    cx = float(A4["lectern_centre_x"])
    gy = float(A4["book_top_y"])
    ax, ay = cx + 520, -150.0
    bx, by = cx - 60, gy - 10
    dx, dy = bx - ax, by - ay
    ln = math.hypot(dx, dy)
    ux, uy = dx / ln, dy / ln
    t = ((xx - ax) * ux + (yy - ay) * uy) / ln
    s = ((xx - ax) * (-uy) + (yy - ay) * ux)
    half = 90 + 200 * np.clip(t, 0, 1.2)
    sn = s / half
    inten = P.smoothstep(1.1, 0.5, np.abs(sn)) * P.smoothstep(0.0, 0.2, t) * P.smoothstep(1.25, 1.0, t)
    inten *= 0.8 + 0.25 * np.sin(sn * 8 + 1.3) * 0.5 + 0.15 * fbm(ah, aw, 120, "4a-beam", octaves=2)
    return (inten[..., None] * MOON * 0.14).astype(np.float32)


# ---------------------------------------------------------------- 4b page

def page_plate():
    cut = "4b"
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    paper = P.paper_for(ah, aw, "4b")
    # warm light from the left (the lantern), the page curving into the gutter at the right
    lightx = P.smoothstep(pw * 1.05, -pw * 0.1, xx)
    gutter = P.smoothstep(pw - 260, pw + 60, xx)
    base = np.array([0.90, 0.84, 0.68], np.float32) * (0.93 + 0.10 * fbm(ah, aw, 260, "4b-v", octaves=3))[..., None]
    tint = np.array([1.0, 0.94, 0.82], np.float32)
    col = base * (0.70 + 0.34 * lightx)[..., None] * tint
    col = col * (1 - 0.45 * gutter)[..., None] + gutter[..., None] * np.array([0.03, 0.03, 0.06], np.float32)
    # foxing: brown age spots in clusters, tide marks, a soft stain
    fox = P.smoothstep(0.80, 0.93, fbm(ah, aw, 26, "4b-fox", octaves=4)) * (0.5 + 0.5 * fbm(ah, aw, 300, "4b-foxc", octaves=2))
    col = P.mix(col, col * np.array([0.84, 0.70, 0.52], np.float32), fox * 0.6)
    tide = P.smoothstep(0.02, 0.0, np.abs(fbm(ah, aw, 340, "4b-tide", octaves=3) - 0.62)) * 0.5
    col = col * (1 - 0.10 * tide)[..., None]
    # fibres: faint long strands in the sheet
    r = rng("4b-fibres")
    st = P.Strokes(max_pts=4)
    for i in range(900):
        x, y = r.random() * aw, r.random() * ah
        a = r.random() * math.pi
        ln = 6 + r.random() * 22
        c = np.clip(col[int(y), int(x)] * (0.9 if r.random() < 0.5 else 1.06), 0, 1)
        st.add([(x, y), (x + math.cos(a) * ln * 0.5 + (r.random() - 0.5) * 2, y + math.sin(a) * ln * 0.5),
                (x + math.cos(a) * ln, y + math.sin(a) * ln)], 0.45 + 0.4 * r.random(), c,
               seed=int(r.integers(1 << 30)), opacity=0.4, dry=0.6, ragged=0.3, bristles=2, pool=0, streak=0,
               paper_k=0.2, pickup=0)
    canvas = P.new_canvas(ah, aw)
    ang = P.orientation(col, 4, 12, base=0.15, base_weight=1.0)
    P.paint(canvas, col.astype(np.float32), ang, [
        dict(radius=30, spacing=0.9, steps=(3, 10), opacity=0.9, dry=0.4, cthresh=0.08, wob=0.2,
             jitter=(0.025, 0.01, 0.03), streak=0.05, pickup=0.35),
        dict(radius=10, spacing=1.0, steps=(2, 8), opacity=0.8, dry=0.5, cthresh=0.06, wob=0.2,
             jitter=(0.025, 0.01, 0.03), streak=0.05, pickup=0.3, use_err=True, err_thresh=0.02),
    ], paper, "4b-page")
    P.draw(canvas, st, paper)
    # a faint ruled margin near the left and a thin top rule
    rule = P.Strokes(max_pts=12)
    xs_ = np.linspace(-M, aw, 14)
    rule.add([(x, 150 + M + 2 * math.sin(x * 0.004)) for x in xs_], 1.0, np.array([0.62, 0.36, 0.30]) * 0.9,
             seed=seed_of("rule-top"), opacity=0.45, dry=0.65, ragged=0.3, bristles=2, pool=0, streak=0, taper0=0,
             taper1=0)
    ys_ = np.linspace(-M, ah, 14)
    rule.add([(230 + M + 1.5 * math.sin(y * 0.005), y) for y in ys_], 1.0, np.array([0.62, 0.36, 0.30]) * 0.9,
             seed=seed_of("rule-left"), opacity=0.4, dry=0.65, ragged=0.3, bristles=2, pool=0, streak=0, taper0=0,
             taper1=0)
    P.draw(canvas, rule, paper)
    P.finish(canvas, paper, grain=0.07, tooth=0.04)
    canvas[..., 3] = 1.0
    return canvas


# ---------------------------------------------------------------- 5a / 5b shelf from below

SH_ROWS = [1080.0, 790.0, 500.0, 210.0]       # board tops (books stand on these)
SH_RH = 262.0
SH_TREMBLE_ROW = 2                            # the row at y 500
SH_SCALE = 3.1


def shelf_low_books():
    cut = "5a"
    pw, ph, aw, ah = dims(cut)
    out = []
    for i, by in enumerate(SH_ROWS):
        for j, (a, b) in enumerate([(-M, 640.0), (680.0, 1440.0), (1480.0, pw + M)]):
            for bk in BK.books_for_segment(a, b, by, SH_RH - 30, ("low", i, j), scale=SH_SCALE, flat_rate=0.04):
                bk["row"] = i
                out.append(bk)
    return out


def shelf_low_light(cut):
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    sc = np.exp(-(((xx - pw * 0.72) / 900.0) ** 2 + ((yy - 200) / 700.0) ** 2))
    climb = P.smoothstep(-200, ph, yy)
    return (AMB[None, None] * (0.70 + 0.3 * climb)[..., None] + MOON * (sc * 0.38)[..., None]).astype(np.float32)


def tremble_index(books):
    """The single book that trembles in 5a: a tall one near the middle of the row."""
    best, bi = 1e9, 0
    for i, b in enumerate(books):
        if b.get("row") == SH_TREMBLE_ROW and b["kind"] == "stand":
            d = abs(b["x"] - 1060)
            if d < best:
                best, bi = d, i
    return bi


def shelf_low_plate(exclude):
    """Paint the low angle shelf plate, leaving out the books in `exclude`
    (indices) so they can be drawn per frame."""
    cut = "5a"
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    Lgt = shelf_low_light(cut)
    ref = np.zeros((ah, aw, 3), np.float32)
    ref[:] = DARK * 1.5 + 0.012
    wood = WOOD * (0.8 + 0.4 * fbm(ah, aw, 30, "low-wood", octaves=3))[..., None]
    for by in SH_ROWS:
        # board front and its underside seen from below
        ref = np.where(((yy >= by) & (yy < by + 26))[..., None], wood * 1.1, ref)
        ref = np.where(((yy >= by + 26) & (yy < by + 52))[..., None], wood * 0.55, ref)
    for ux in [660.0, 1460.0]:
        ref = np.where((np.abs(xx - ux) < 20)[..., None], wood, ref)
    lit = np.clip(ref * Lgt * 2.3, 0, 1)
    paper = P.paper_for(ah, aw, "low")
    canvas = P.new_canvas(ah, aw)
    ang = P.orientation(lit, 2, 6, base=math.pi / 2, base_weight=0.6)
    P.paint(canvas, lit, ang, [
        dict(radius=16, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=6, spacing=1.0, steps=(2, 8), opacity=0.92, dry=0.4, cthresh=0.08, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.02),
    ], paper, "low-struct")
    books = shelf_low_books()
    keep = [b for i, b in enumerate(books) if i not in set(exclude)]

    def lf(x, y):
        return Lgt[int(min(max(y + M, 0), ah - 1)), int(min(max(x + M, 0), aw - 1))] * 2.25
    b1, a1 = BK.book_strokes(keep, lf, offset=(M, M), eye_y=-1e9)
    P.draw(canvas, b1, paper)
    P.draw(canvas, a1, paper)
    st = P.Strokes(max_pts=8)
    for by in SH_ROWS:
        xs_ = np.linspace(-M, pw + M, 8)
        c = np.clip(WOOD * 1.5 * lf(pw * 0.7, by), 0, 1)
        st.add([(x + M, by + 6 + M) for x in xs_], 7.0, c, seed=seed_of("lowb", by), opacity=0.95, dry=0.35,
               ragged=0.1, bristles=4, pool=0.1, streak=0.15, taper0=0, taper1=0, cap=1.0)
        st.add([(x + M, by + 1 + M) for x in xs_], 1.4, np.clip(c * 1.7 + MOON * 0.06, 0, 1), seed=seed_of("lowbh", by),
               opacity=0.6, dry=0.6, ragged=0.2, bristles=2, pool=0, streak=0, cap=1.0)
    P.draw(canvas, st, paper)
    P.finish(canvas, paper, grain=0.05)
    canvas[..., 3] = 1.0
    return canvas


def pose(seed, f, hold=3, amp=1.0):
    """Hard held key pose: a new (dx, dy, angle) every `hold` frames."""
    k = f // hold
    r = np.random.default_rng((seed * 7919 + k * 104729) & 0x7FFFFFFF)
    return ((r.random() - 0.5) * 5 * amp, -abs(r.random() - 0.3) * 4 * amp, (r.random() - 0.5) * 0.05 * amp)


KEY = 0.34


def keystone_xy(cam, xp, yp):
    """Plate to screen for the low angle shots: verticals converge upward."""
    zoom, cx, cy = cam
    sy = T.H / 2 + (yp - cy) * zoom
    k = -KEY * np.clip(sy / T.H, -0.3, 1.3)
    sx = T.W / 2 + (xp - cx) * zoom / (1 + k)
    return sx, sy, zoom / (1 + k)


def keystone_sample(plate, cam):
    from scipy import ndimage
    zoom, cx, cy = cam
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    k = -KEY * ys / T.H
    px = cx + (xs - T.W / 2) / zoom * (1 + k) + M
    py = cy + (ys - T.H / 2) / zoom + M
    out = np.empty((T.H, T.W, 4), np.float32)
    for c in range(4):
        out[..., c] = ndimage.map_coordinates(plate[..., c], [py, px], order=1, mode="nearest")
    return out


def animated_books(cut, f, idx_list, amp):
    books = shelf_low_books()
    Lgt = shelf_low_light("5a")
    pw, ph, aw, ah = dims("5a")

    def lf(x, y):
        return Lgt[int(min(max(y + M, 0), ah - 1)), int(min(max(x + M, 0), aw - 1))] * 2.25
    moved = []
    for i in idx_list:
        b = dict(books[i])
        dx, dy, da = pose(i, f, hold=3 if cut == "5a" else 2, amp=amp)
        b["x"] += dx
        b["y"] += dy
        b["angle"] = b["angle"] + da
        moved.append(b)
    b1, a1 = BK.book_strokes(moved, lf, eye_y=-1e9)
    cam = T.camera(cut, f)
    out = []
    for a in (b1, a1):
        a = dict(a)
        sx, sy, sc = keystone_xy(cam, a["xs"], a["ys"])
        a["xs"], a["ys"] = sx.astype(np.float32), sy.astype(np.float32)
        a["prm"] = a["prm"].copy()
        a["prm"][:, P.P_WIDTH] *= sc[:, 0]
        out.append(a)
    return out


def dust_streams(cut, f, xs_plate, y_top, t0):
    """Thin streams of dust sifting down from a shelf edge, brightest in the
    moonlit air. Returns an additive layer (screen space)."""
    t = (f - (f % 2)) / T.FPS - t0
    out = np.zeros((T.H, T.W, 4), np.float32)
    if t <= 0:
        return out[..., :3]
    cam = T.camera(cut, f)
    st = P.Strokes(max_pts=2)
    for si, sx in enumerate(xs_plate):
        r = np.random.default_rng(si * 31 + 7)
        n = 70
        for k in range(n):
            ph = r.random()
            v = 180 + 120 * r.random()
            y = y_top + ((t * v + ph * 900) % 900)
            if y - y_top > t * v + 5:
                continue
            x = sx + math.sin(y * 0.02 + k) * 3 + (r.random() - 0.5) * 6
            px, py, _sc = keystone_xy(cam, x, y)
            fade = max(0.0, 1.0 - (y - y_top) / 900.0)
            st.add([(px, py), (px + 0.2, py + 2.5)], 0.9 * cam[0], MOON * 0.7 * fade, seed=si * 1000 + k,
                   opacity=0.7 * fade, dry=0, ragged=0.1, bristles=2, pool=0, streak=0, paper_k=0, pickup=0)
    P.draw(out, st, P.paper_for(T.H, T.W, "screen-paper"))
    return out[..., :3]


# ---------------------------------------------------------------- paint

def paint(cuts):
    if "3" in cuts:
        log("cut 3")
        lp = screen_to_plate0("3", *T.ANCHORS["3"]["lantern_glow_from_screen"])
        sharp = close_shelves("3", lp, 0.0, 1010.0, "c3", warm_k=1.6)
        C.save_plate("c3_bg", defocus("3", sharp, lp, "c3", sigma=16.0, dark=0.52, floor_y=1010.0, n_bokeh=22))
        np.save(C.plate_path("c3_glow"), lantern_glow_plate("3", lp, 0.22).astype(np.float16))
    if "5c" in cuts:
        log("cut 5c")
        lp = screen_to_plate0("5c", *T.ANCHORS["5c"]["lantern_screen"])
        lp = (lp[0], lp[1] + 60)
        sharp = close_shelves("5c", lp, 0.0, 900.0, "c5c", rh=170.0, scale=2.1, warm_k=1.3)
        C.save_plate("c5c_bg", defocus("5c", sharp, lp, "c5c", sigma=5.0, dark=0.6, floor_y=900.0, n_bokeh=10))
        np.save(C.plate_path("c5c_glow"), lantern_glow_plate("5c", lp, 0.28).astype(np.float16))
    if "4a" in cuts:
        log("cut 4a")
        C.save_plate("c4a_back", backdrop_from_hall("4a", (560, -40, 2340, 1100), darken=0.85, blur=1.4))
        C.save_plate("c4a_lectern", lectern_layer())
        np.save(C.plate_path("c4a_glow"), beam_4a().astype(np.float16))
    if "4b" in cuts:
        log("cut 4b")
        C.save_plate("c4b_page", page_plate())
    if "5a" in cuts or "5b" in cuts:
        log("cut 5a, 5b")
        books = shelf_low_books()
        ti = tremble_index(books)
        C.save_plate("c5a_bg", shelf_low_plate([ti]))
        row = [i for i, b in enumerate(books) if b.get("row") == SH_TREMBLE_ROW]
        C.save_plate("c5b_bg", shelf_low_plate(row))
    if "6a" in cuts:
        log("cut 6a")
        paper = P.paper_for(H.AH, H.AW, "hall")
        for v in ("half", "open"):
            cv, _ = H.paint_hall(paper, variant=v)
            C.save_plate(f"hall_{v}", cv)


# ---------------------------------------------------------------- render

def render_cut(cut, f):
    cam = T.camera(cut, f)
    if cut == "3":
        bg = C.sample(C.load_plate("c3_bg"), cam)
        return dict(bg=bg, fg=None, glow=C.sample(C.load_plate("c3_glow"), cam))
    if cut == "5c":
        bg = C.sample(C.load_plate("c5c_bg"), cam)
        return dict(bg=bg, fg=None, glow=C.sample(C.load_plate("c5c_glow"), cam))
    if cut == "4a":
        bg = C.sample(C.load_plate("c4a_back"), cam)
        C.over(bg, C.sample(C.load_plate("c4a_lectern"), cam))
        bg[..., 3] = 1.0
        return dict(bg=bg, fg=None, glow=C.sample(C.load_plate("c4a_glow"), cam))
    if cut == "4b":
        bg = C.sample(C.load_plate("c4b_page"), cam)
        return dict(bg=bg, fg=None, glow=None)
    if cut in ("5a", "5b"):
        bg = keystone_sample(C.load_plate("c5a_bg" if cut == "5a" else "c5b_bg"), cam)
        books = shelf_low_books()
        f0 = T.cut_range(cut)[0]
        t = f / T.FPS
        if cut == "5a":
            idx = [tremble_index(books)]
            amp = 1.0 if t >= T.CUES["tremble_one"] else 0.0
        else:
            idx = [i for i, b in enumerate(books) if b.get("row") == SH_TREMBLE_ROW]
            amp = 1.3 if t >= T.CUES["tremble_row"] else 0.0
        paper = P.paper_for(T.H, T.W, "screen-paper")
        for a in animated_books(cut, f if amp else f0, idx, amp):
            P.draw(bg, a, paper)
        glow = None
        if cut == "5b":
            xs = [b["x"] for i, b in enumerate(books) if b.get("row") == SH_TREMBLE_ROW][::6]
            glow = dust_streams(cut, f, xs, SH_ROWS[SH_TREMBLE_ROW - 1] + 40, T.CUES["dust_fall"])
        bg[..., 3] = 1.0
        return dict(bg=bg, fg=None, glow=glow)
    if cut == "6a":
        f0 = T.cut_range("6a")[0]
        k = f - f0
        name = "hall_half" if k < 2 else "hall_open"
        return H.render_cut("6a", f, bg_name=name)
    raise KeyError(cut)


if __name__ == "__main__":
    import time
    t0 = time.time()
    cuts = sys.argv[2:] if len(sys.argv) > 2 else ["3", "5c", "4a", "4b", "5a", "6a"]
    paint(cuts)
    log(f"done {time.time() - t0:.1f}s")
