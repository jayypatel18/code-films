"""The sky family: cuts 7, 8, 9a and 9b, at sunrise and in the morning.

  7   the sun clears the sea of clouds (sun_start to sun_end), gold and rose,
      the stone balcony ledge at lower left with grass whipping in the wind,
      a great ring of wind rolling outward across the cloud tops (wind_ring).
  8   the high golden sky: towering cloud banks, sunbeams, mist, a tiny corner
      of the balcony bottom left.
  9a  the balcony in warm morning light: balustrade, sky and clouds beyond,
      stone floor where the page lands.
  9b  closer and lower: the floor fills the frame around the page quad.

Run `python3 world/sky_family.py paint [cut ...]` through tools/heavy.py.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine import paint_clouds as PC  # noqa: E402
from engine.noise import fbm, value_noise, rng, noise1, seed_of, ridged  # noqa: E402
from engine.paint_shapes import poly_mask, wobble, soft  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402

M = MARGIN

GOLD = np.array([1.00, 0.80, 0.46], np.float32)
ROSE = np.array([0.92, 0.58, 0.58], np.float32)
SKY_TOP = np.array([0.36, 0.42, 0.66], np.float32)
SKY_MID = np.array([0.80, 0.60, 0.62], np.float32)
SKY_LOW = np.array([1.00, 0.80, 0.56], np.float32)
CL_SH = np.array([0.50, 0.36, 0.50], np.float32)
CL_MID = np.array([0.82, 0.60, 0.62], np.float32)
CL_LIT = np.array([1.00, 0.88, 0.66], np.float32)
STONE_LIT = np.array([0.96, 0.78, 0.56], np.float32)
STONE_SH = np.array([0.50, 0.40, 0.50], np.float32)


def dims(cut):
    pw, ph = T.plate_size(cut)
    return pw, ph, pw + 2 * M, ph + 2 * M


def grid(cut):
    pw, ph, aw, ah = dims(cut)
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    return yy - M, xx - M


def lerp_stops(t, stops):
    out = np.zeros(t.shape + (3,), np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        m = (t >= t0) & (t <= t1)
        u = ((t - t0) / (t1 - t0))[m]
        u = u * u * (3 - 2 * u)
        out[m] = np.asarray(c0) * (1 - u[:, None]) + np.asarray(c1) * u[:, None]
    out[t < stops[0][0]] = stops[0][1]
    out[t > stops[-1][0]] = stops[-1][1]
    return out


# ---------------------------------------------------------------- sky

def sky_clouds(ref, cut, specs, sun, key, lit=CL_LIT, mid=CL_MID, sh=CL_SH, haze=None):
    """Soft cumulus banks floating in the sky. specs: (cx, cy, w, h, depth)
    in plate coords; depth 0 near, 1 far (fades into the sky colour). Tops
    and the side toward the sun are lit; bellies sink into rose violet."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    sx, sy = sun
    for i, (cx, cy, w, h, dep) in enumerate(specs):
        x0, x1 = int(max(cx - w * 1.3 + M, 0)), int(min(cx + w * 1.3 + M, aw))
        y0, y1 = int(max(cy - h * 1.6 + M, 0)), int(min(cy + h * 1.2 + M, ah))
        if x1 <= x0 or y1 <= y0:
            continue
        Y, X = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        u = (X - cx) / w
        v = (Y - cy) / h
        n = fbm(y1 - y0, x1 - x0, max(w * 0.18, 6), (key, "cl", i), octaves=4)
        # flat bottom, billowing top
        shape = 1.0 - (u * u + np.where(v < 0, v * v * 0.8, v * v * 3.0))
        dens = shape + (n - 0.5) * 0.9 * np.clip(1 - v, 0, 1.5)
        m = P.smoothstep(0.0, 0.08, dens)
        side = np.clip((sx - X) / max(abs(sx - cx), 1.0) * 0.4, -0.4, 0.4) * np.sign(sx - cx + 1e-3)
        tone = np.clip(0.55 - v * 0.6 + (n - 0.5) * 0.8 + side, 0, 1)
        col = P.mix(sh, mid, P.smoothstep(0.1, 0.5, tone))
        col = P.mix(col, lit, P.smoothstep(0.55, 0.9, tone))
        hz = ref[y0:y1, x0:x1]
        col = P.mix(col, hz, dep * 0.75)
        ref[y0:y1, x0:x1] = P.mix(ref[y0:y1, x0:x1], col, m)
    return ref


def sky_plate(cut, horizon, sun, key, glow_k=1.0, streaks=8, top=SKY_TOP, mid=SKY_MID, low=SKY_LOW, clouds=None):
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    t = np.clip(yy / horizon, -0.2, 1.3)
    ref = lerp_stops(t, [(-0.2, top * 0.9), (0.0, top), (0.55, mid), (0.95, low), (1.3, low)])
    sx, sy = sun
    d = np.sqrt(((xx - sx) / 900.0) ** 2 + ((yy - sy) / 520.0) ** 2)
    ref = ref + (np.exp(-d ** 2 * 1.2) * 0.28 * glow_k)[..., None] * GOLD
    ref = ref * (0.95 + 0.1 * fbm(ah, aw, 400, (key, "var"), octaves=3))[..., None]
    # long streaks of cloud lit from below: gold near the sun, rose higher up
    r = rng(key, "streaks")
    sl = np.zeros((ah, aw), np.float32)
    for i in range(streaks):
        cy = horizon * (0.25 + 0.6 * r.random())
        cx = r.random() * pw
        ln = 500 + r.random() * 1100
        th = 5 + r.random() * 14
        bend = (r.random() - 0.5) * 40
        dd = yy - cy - bend * ((xx - cx) / ln) ** 2
        along = np.clip(1 - np.abs((xx - cx) / ln) ** 2, 0, 1)
        n = value_noise(ah, aw, 70, (key, "st", i)) * 0.8 + 0.2
        sl = np.maximum(sl, np.exp(-(dd / th) ** 2) * along * n)
    near_sun = np.exp(-(((xx - sx) / 800.0) ** 2))
    scol = P.mix(ROSE * 1.05, GOLD * 1.1, near_sun * P.smoothstep(0.2 * horizon, horizon, yy))
    ref = P.mix(ref, scol, sl * 0.6)
    if clouds:
        ref = sky_clouds(ref, cut, clouds, sun, key)
    ref = np.clip(ref, 0, 1).astype(np.float32)
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    ang = P.orientation(ref, 3, 14, base=0.0, base_weight=1.0, edge_gain=0.9)
    # long, wet, overlapping strokes with no pooled edges: no lozenge pattern
    P.paint(canvas, ref, ang, [
        dict(radius=24, spacing=0.75, steps=(6, 22), opacity=0.85, dry=0.3, cthresh=0.10, wob=0.15,
             jitter=(0.012, 0.008, 0.02), streak=0.04, ragged=0.25, pickup=0.5, curve=0.3, pool=0.0),
        dict(radius=10, spacing=0.9, steps=(5, 18), opacity=0.75, dry=0.45, cthresh=0.07, wob=0.15,
             jitter=(0.012, 0.008, 0.02), streak=0.04, pickup=0.45, curve=0.35, pool=0.0, use_err=True,
             err_thresh=0.03),
        dict(radius=4, spacing=1.0, steps=(3, 10), opacity=0.7, dry=0.5, cthresh=0.06, wob=0.2,
             jitter=(0.012, 0.008, 0.02), streak=0.04, pickup=0.35, pool=0.0, use_err=True, err_thresh=0.05),
    ], paper, (key, "sky"))
    P.finish(canvas, paper, grain=0.04)
    canvas[..., 3] = 1.0
    return canvas


# ---------------------------------------------------------------- cloud sea at sunrise

def sunrise_sea(cut, horizon, cam_h, sun, key, base_r=0.10, heap_rate=0.35, heap_height=4.0, z_far=50.0,
                top_y=None, calm=None, warmth=1.0):
    """A cloud sea lit by a low sun ahead: gold rims on every lobe top, rose
    violet bodies, golden haze thickening toward the horizon. Returns a
    painted alpha layer and the buffers."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    H = horizon + M
    zb = PC.band_sea((key, "sea"), ah, aw, 1600.0, H, cam_h, z_near=0.8, z_far=z_far, row_step=0.06,
                     base_r=base_r, heap_rate=heap_rate, heap_height=heap_height, flat=0.72, heap_width=9.0,
                     calm=calm)
    nx, ny, nz = zb["nx"], zb["ny"], zb["nz"]
    sc = np.where(zb["cover"] > 0, zb["scale"][np.maximum(zb["band"], 0)], 2.0).astype(np.float32)
    sm = [P.blur(c, 6.0) for c in (nx, ny, nz)]
    k = 0.45
    nx, ny, nz = nx * (1 - k) + sm[0] * k, ny * (1 - k) + sm[1] * k, nz * (1 - k) + sm[2] * k
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    sx, sy = sun
    # light from the sun ahead and a little to the side: rims, not faces
    lx = np.clip((sx - xx) / 1400.0, -1, 1) * -0.8
    L = np.stack([np.clip(-(lx), -1, 1), np.full_like(lx, -0.55), np.full_like(lx, -0.25)], -1)
    L /= np.linalg.norm(L, axis=-1, keepdims=True)
    ndl = nx * L[..., 0] + ny * L[..., 1] + nz * L[..., 2]
    rim = np.clip(ndl + 0.15, 0, 1) ** 1.5 * (1 - nz) ** 0.6
    top = np.clip(-ny, 0, 1)
    dn = zb["dn"]
    body = P.smoothstep(0.3, 2.4, dn)
    col = P.mix(CL_MID, CL_SH, np.clip(body * 0.9 + (1 - top) * 0.25, 0, 1))
    col = P.mix(col, CL_LIT * warmth + (1 - warmth) * CL_MID, np.clip(rim * 1.4 + top * 0.25, 0, 1))
    cr = PC.creases(zb["order"], zb["band"], zb["scale"], reach=0.45, maxpx=16)
    col = col * (1 - 0.18 * P.blur(cr, 2.0))[..., None]
    depth = zb["depth"]
    fogk = 1.0 - np.exp(-np.maximum(depth - 1.0, 0) / 10.0)
    near_sun = np.exp(-(((xx - sx) / 900.0) ** 2 + ((yy - sy) / 400.0) ** 2))
    hz = P.mix(SKY_LOW * 0.95, GOLD * 1.05, near_sun)
    col = P.mix(col, hz, np.clip(fogk * 0.85, 0, 0.88))
    col = col + (near_sun * 0.12)[..., None] * GOLD
    ref = np.clip(col, 0, 1).astype(np.float32)
    cover = zb["cover"]
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    cm = (P.blur(cover, 2.0) > 0.3).astype(np.float32)
    ang = P.orientation(ref, 2.0, 7.0, base=0.0, base_weight=0.55, edge_gain=1.2)
    P.paint(canvas, ref, ang, [
        dict(radius=20, spacing=0.9, steps=(3, 9), opacity=0.95, dry=0.3, cthresh=0.18, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, ragged=0.22, pickup=0.2, curve=0.5, mask=cm),
        dict(radius=9, spacing=0.95, steps=(2, 9), opacity=0.92, dry=0.4, cthresh=0.13, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.2, mask=cm, use_err=True, err_thresh=0.04),
        dict(radius=4, spacing=1.0, steps=(2, 8), opacity=0.9, dry=0.45, cthresh=0.10, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.15, mask=cm, use_err=True, err_thresh=0.05),
    ], paper, (key, "sea"))
    # bright rims along the lobe tops facing the sun
    st = P.Strokes(max_pts=10)
    Lb = zb["lobes"]
    Zs = zb["Z"]
    r = rng(key, "rims")
    for i in range(len(Lb)):
        kk, cx, cy, rx, ry = Lb[i]
        kk = int(kk)
        if rx < 8 or Zs[kk] > 20 or r.random() < 0.45:
            continue
        side = 1.0 if cx < sx else -1.0
        a0 = math.pi * (1.0 + (0.45 if side > 0 else 0.05) + 0.15 * r.random())
        a1 = a0 + math.pi * (0.25 + 0.3 * r.random())
        pts = []
        for q in range(8):
            a = a0 + (a1 - a0) * q / 7
            px, py = cx + math.cos(a) * rx * 0.97, cy + math.sin(a) * ry * 0.97
            xi, yi = int(px), int(py)
            if not (0 <= xi < aw and 0 <= yi < ah) or zb["band"][yi, xi] != kk or abs(zb["order"][yi, xi] - i) > 0.5:
                if len(pts) >= 3:
                    break
                pts = []
                continue
            pts.append((px, py))
        if len(pts) < 3:
            continue
        w = max(0.9, min(5.0, rx * 0.06))
        st.add(pts, w, np.clip(CL_LIT * 1.05, 0, 1), seed=int(r.integers(1 << 30)), opacity=0.6, dry=0.55,
               ragged=0.3, bristles=3, pool=0, streak=0.05, taper0=0.3, taper1=0.6, pickup=0.2)
    if len(st):
        P.draw(canvas, st.arrays(), paper)
    P.finish(canvas, paper, grain=0.04)
    return canvas, zb


# ---------------------------------------------------------------- stone

def stone_wall(cut, poly, key, light_dir=(0.85, -0.2), course=34.0, lit=STONE_LIT, sh=STONE_SH, face_tone=0.6):
    """A painted stone mass (ledge, parapet, tower wall) as an alpha layer.
    face_tone: how much the flat face receives the key light."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    m = poly_mask(ah, aw, np.asarray(poly) + M)
    c = np.floor(yy / course)
    off = (np.sin(c * 12.9898) * 43758.5453 % 1.0) * 60
    b = np.floor((xx + off) / (course * 1.8))
    sid = (c * 131 + b * 977) % 10007
    rj = (np.sin(sid * 12.9898) * 43758.5453) % 1.0
    tone = face_tone * (0.85 + 0.25 * rj)
    col = P.mix(sh, lit, tone)
    joint = (np.mod(yy, course) < 2.2) | (np.mod(xx + off, course * 1.8) < 2.2)
    col = col * (1 - 0.3 * joint)[..., None]
    col = col * (0.85 + 0.25 * fbm(ah, aw, 18, (key, "st"), octaves=3))[..., None]
    moss = P.smoothstep(0.66, 0.82, fbm(ah, aw, 40, (key, "moss"), octaves=3))
    col = P.mix(col, np.array([0.45, 0.55, 0.30], np.float32), moss * 0.35)
    ref = np.clip(col, 0, 1).astype(np.float32)
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    mm = (m > 0.5).astype(np.float32)
    ang = P.orientation(ref, 1.5, 4, base=0.0, base_weight=0.8)
    P.paint(canvas, ref, ang, [
        dict(radius=10, spacing=0.9, steps=(2, 6), opacity=0.96, dry=0.35, cthresh=0.15, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.1, mask=mm),
        dict(radius=4, spacing=1.0, steps=(2, 5), opacity=0.92, dry=0.45, cthresh=0.1, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.1, mask=mm, use_err=True, err_thresh=0.05),
    ], paper, (key, "stone"), clip=m)
    P.finish(canvas, paper, grain=0.04)
    return canvas


def backlit_stone(cut, poly, key, top_y=None, course=58.0, block=128.0, rim_x=None, cyl=None, curve=0.0,
                  tone=1.0):
    """Stone against a low sun: the faces in deep violet brown shadow, a hot
    rim along the top (and along rim_x for a tower edge), big weathered blocks.
    cyl: (cx, r) shades a round tower."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    m = poly_mask(ah, aw, np.asarray(poly) + M)
    yy_true = yy
    if cyl is not None and curve:
        # courses on a round tower bow as they turn away from us
        u = np.clip((xx - cyl[0]) / cyl[1], -1, 1)
        yy = yy + curve * (1 - np.sqrt(1 - u * u)) * cyl[1]
    # courses of uneven height, blocks of uneven length: no brick grid
    rr = rng(key, "courses")
    edges = [-M - 200.0]
    while edges[-1] < ph + M + 200:
        edges.append(edges[-1] + course * (0.7 + 0.6 * rr.random()))
    edges = np.array(edges)
    c = np.clip(np.searchsorted(edges, yy) - 1, 0, len(edges) - 2).astype(np.float32)
    fy = yy - edges[c.astype(int)]
    chh = (edges[c.astype(int) + 1] - edges[c.astype(int)])
    off = (np.sin(c * 12.9898) * 43758.5453 % 1.0) * block
    blen = block * (0.6 + 0.9 * ((np.sin(c * 7.77) * 9631.7) % 1.0))
    b = np.floor((xx + off) / blen)
    sid = (c * 131 + b * 977) % 10007
    rj = (np.sin(sid * 12.9898) * 43758.5453) % 1.0
    dark = np.array([0.26, 0.18, 0.24], np.float32)
    mid = np.array([0.42, 0.30, 0.34], np.float32)
    col = P.mix(dark, mid, 0.35 + 0.35 * rj)
    if cyl is not None:
        cx, r = cyl
        u = np.clip((xx - cx) / r, -1, 1)
        col = col * (0.75 + 0.35 * np.clip(u, 0, 1) ** 2)[..., None]
    fx = np.mod(xx + off, blen)
    broken = value_noise(ah, aw, 30, (key, "jb")) > 0.35
    joint = ((fy < 2.5) | (fx < 2.5)) & broken
    col = col * (1 - 0.22 * joint)[..., None]
    chip = (fy > chh - 6) & (rj > 0.5)
    col = col * (1 + 0.10 * chip)[..., None]
    col = col * (0.85 + 0.25 * fbm(ah, aw, 14, (key, "st"), octaves=4))[..., None]
    stain = np.clip((np.repeat(fbm(1, aw, 6, (key, "rain"), octaves=2), ah, 0) - 0.55) * 3, 0, 1)
    col = col * (1 - 0.18 * stain)[..., None]
    # warm bounce from the cloud sea below and a little moss in the joints near the top
    col = col + (P.smoothstep(ph * 0.5, ph, yy) * 0.06)[..., None] * np.array([1.0, 0.6, 0.4], np.float32)
    if top_y is not None:
        topband = P.smoothstep(top_y + 26, top_y + 4, yy) * (yy >= top_y)
        col = P.mix(col, np.array([0.98, 0.72, 0.42], np.float32), topband * 0.85)
        moss = joint * P.smoothstep(top_y + 80, top_y + 20, yy) * P.smoothstep(0.55, 0.75, fbm(ah, aw, 20, (key, "m"), octaves=3))
        col = P.mix(col, np.array([0.35, 0.36, 0.18], np.float32), moss * 0.6)
    if rim_x is not None:
        rim = P.smoothstep(rim_x - 16, rim_x - 2, xx) * (xx <= rim_x + 1)
        col = P.mix(col, np.array([1.0, 0.76, 0.46], np.float32), rim * 0.75)
    ref = np.clip(col, 0, 1).astype(np.float32)
    paper = P.paper_for(ah, aw, key)
    canvas = P.new_canvas(ah, aw)
    mm = (m > 0.5).astype(np.float32)
    ang = P.orientation(ref, 1.5, 4, base=0.0, base_weight=0.7)
    P.paint(canvas, ref, ang, [
        dict(radius=12, spacing=0.9, steps=(2, 6), opacity=0.96, dry=0.35, cthresh=0.15, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.1, mask=mm),
        dict(radius=5, spacing=1.0, steps=(2, 5), opacity=0.92, dry=0.45, cthresh=0.1, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.1, mask=mm, use_err=True, err_thresh=0.04),
        dict(radius=2.2, spacing=1.1, steps=(2, 4), opacity=0.9, dry=0.5, cthresh=0.08, wob=0.3,
             jitter=(0.04, 0.03, 0.05), streak=0.08, mask=mm, use_err=True, err_thresh=0.06),
    ], paper, (key, "stone"), clip=m)
    P.finish(canvas, paper, grain=0.04)
    return canvas


def edge_strokes(pts, width, col, key, opacity=0.85):
    st = P.Strokes(max_pts=16)
    st.add([(x + M, y + M) for x, y in pts], width, col, seed=seed_of(key), opacity=opacity, dry=0.5, ragged=0.25,
           bristles=3, pool=0, streak=0.05, taper0=0.05, taper1=0.1)
    return st.arrays()


def flagstones(cut, eye, vp_x, key, scale=1.0, d_near=0.4, d_far=14.0):
    """Worn flagstones of different sizes in perspective, no grid (rows of
    varied depth, stones of varied width, long slabs, broken faded joints).
    Returns (value 0..1 per stone, joint mask, stone id)."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    r = rng("flag", key)
    d = 1600.0 / np.maximum(yy - eye, 1.0)
    X = (xx - vp_x) * d / 1600.0
    bounds = [d_near]
    while bounds[-1] < d_far:
        bounds.append(bounds[-1] + (0.40 + 0.45 * r.random()) * scale)
    bounds = np.array(bounds)
    row = np.clip(np.searchsorted(bounds, d) - 1, 0, len(bounds) - 2)
    joint = np.zeros((ah, aw), np.float32)
    sid = np.zeros((ah, aw), np.int64)
    ybound = eye + 1600.0 / bounds
    for ri in range(len(bounds) - 1):
        m = row == ri
        if not m.any():
            continue
        xs = [-14.0]
        while xs[-1] < 14.0:
            w = (0.45 + 0.75 * r.random()) * scale
            if r.random() < 0.12:
                w *= 1.8
            xs.append(xs[-1] + w)
        xs = np.array(xs) + r.random() * 0.4
        Xm = X[m]
        k = np.clip(np.searchsorted(xs, Xm) - 1, 0, len(xs) - 2)
        dm = d[m]
        lat = np.minimum(Xm - xs[k], xs[k + 1] - Xm) * 1600.0 / dm
        ym = yy[m]
        dep = np.minimum(np.abs(ym - ybound[ri]), np.abs(ym - ybound[ri + 1]))
        sm = ri * 1000 + k
        sid[m] = sm
        wj = 0.5 + 1.1 * np.clip(3.0 / dm, 0.4, 1.6)
        fl = value_noise(1, 4096, 9, (key, "jl", ri))[0][np.clip(((ym - eye) * 2).astype(int), 0, 4095)]
        fd = value_noise(1, 4096, 7, (key, "jd", ri))[0][np.clip(((Xm + 14) * 140).astype(int), 0, 4095)]
        jl = P.smoothstep(wj + 0.9, wj * 0.6, lat) * (0.3 + 0.7 * fl)
        jd = P.smoothstep(wj + 0.7, wj * 0.5, dep) * (0.3 + 0.7 * fd)
        hs = ((sm * 2246822519) % 4294967296) / 4294967296.0
        far = np.abs(ym - ybound[ri + 1]) < np.abs(ym - ybound[ri])
        jd = np.where((hs < 0.28) & far, jd * 0.15, jd)
        joint[m] = np.maximum(jl, jd)
    val = ((sid * 2654435761) % 4294967296 / 4294967296.0).astype(np.float32)
    return val, joint, sid


# ---------------------------------------------------------------- grass (animated)

def grass_tufts(key, base_pts, height, n_per, lean0=0.25):
    r = rng("grass", key)
    out = []
    for (bx, by) in base_pts:
        for k in range(n_per):
            out.append(dict(x=bx + (r.random() - 0.5) * 14, y=by + r.random() * 3, h=height * (0.5 + 0.7 * r.random()),
                            lean=lean0 + (r.random() - 0.5) * 0.5, ph=r.random() * 6.28, seed=int(r.integers(1 << 30)),
                            w=1.2 + 1.4 * r.random(), lit=r.random() ** 3))
    return out


def grass_strokes(tufts, t, cam, wind, lit_col, dark_col):
    """Blades bending in the wind (on twos), tip lagging the base."""
    sl = P.Strokes(max_pts=8)
    z = cam[0]
    for g in tufts:
        bx, by = C.to_screen(cam, g["x"], g["y"])
        h = g["h"] * z
        flutter = 0.12 * math.sin(t * 9.0 + g["ph"]) + 0.06 * math.sin(t * 17.0 + g["ph"] * 2)
        bend = g["lean"] + wind * (0.6 + flutter * 2.0) + flutter * 0.5
        pts = []
        for q in range(7):
            u = q / 6
            a = bend * u ** 1.6
            pts.append((bx + h * math.sin(a) * u, by - h * math.cos(a) * u))
        c = P.mix(dark_col, lit_col, g["lit"])
        sl.add(pts, g["w"] * max(z, 0.5), c, seed=g["seed"], opacity=0.95, dry=0.25, ragged=0.12, bristles=2,
               pool=0.0, streak=0.05, taper0=0.05, taper1=0.85, pickup=0.0)
    return sl.arrays()


# ---------------------------------------------------------------- cut 7

A7 = T.ANCHORS["7"]
H7 = float(A7["cloud_sea_top_y"]) - 20


def sun_pos(f):
    t = f / T.FPS
    t0, t1 = T.cut_range("7")
    u = (f - t0) / max(t1 - 1 - t0, 1)
    u = min(max(u, 0.0), 1.0)
    u = u * u * (3 - 2 * u)
    a, b = A7["sun_start"], A7["sun_end"]
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def ledge_poly_7():
    x0, x1 = A7["ledge_x"]
    ty = float(A7["ledge_top_y"])
    return [(x0 - 20, ty), (x1, ty + 4), (x1 + 6, ty + 30), (x1, PH7 + M + 10), (-M - 10, PH7 + M + 10),
            (-M - 10, ty + 40), (x0 - 40, ty + 26)]


PW7, PH7 = T.plate_size("7")


def paint_7(sea_only=False):
    sun0 = A7["sun_start"]
    sky = sky_plate("7", H7, (sun0[0], sun0[1] - 40), "c7", glow_k=1.2,
                    clouds=[(700, 420, 380, 70, 0.45), (1250, 300, 300, 50, 0.55), (2050, 380, 340, 60, 0.5),
                            (420, 640, 260, 60, 0.7)]) if not sea_only else None
    if sky is not None:
        C.save_plate("c7_sky", sky)
    import clouds_sunrise as CS
    sun_l = (A7["sun_end"][0], 790.0)
    sky_at = CS.sky_fn(800 + M, (sun_l[0] + M, sun_l[1] + M), SKY_LOW, GOLD, SKY_MID)
    sea = CS.paint_banks("7", CS.banks_7(), sun_l, CS.PAL_SUN, sky_at, "c7-cum", wisps=None,
                         floor=CS.floor_7())
    C.save_plate("c7_sea", sea)
    if not sea_only:
        C.save_plate("c7_ledge", parapet_7())


def ivy_strokes(st, r, x, y, length, scale=1.0, dark=(0.10, 0.14, 0.10), lit=(0.62, 0.66, 0.30), droop=True):
    """A trailing ivy strand from (x, y): a thin stem and leaf pairs, lit leaves
    catching the warm light."""
    n = 7
    pts = [(x + 8 * math.sin(q * 0.9 + r.random()) * scale, y + q * length / (n - 1) * (1 if droop else -1)) for q in range(n)]
    st.add(pts, 0.9 * scale, np.array(dark), seed=int(r.integers(1 << 30)), opacity=0.9, dry=0.3, ragged=0.2,
           bristles=2, pool=0, streak=0, taper0=0.05, taper1=0.6)
    for (px, py) in pts[1:]:
        for q in range(2):
            a = r.random() * math.pi
            ox = (r.random() - 0.5) * 8 * scale
            c = P.mix(np.array(dark), np.array(lit), r.random() ** 2)
            st.add([(px + ox - math.cos(a) * 3.2 * scale, py - math.sin(a) * 3.2 * scale),
                    (px + ox + math.cos(a) * 3.2 * scale, py + math.sin(a) * 3.2 * scale)], 2.6 * scale, c,
                   seed=int(r.integers(1 << 30)), opacity=0.93, dry=0.2, ragged=0.35, bristles=2, pool=0.12,
                   streak=0.05, pickup=0)


def parapet_7():
    """The stone balcony against the rising sun, turned a little so its top
    and thickness show: a lit coping slab raked by gold light, its shadowed
    front with an overhang, a moulding, a weathered ashlar face with cracks
    and moss in the joints, the sunlit end, and the curved tower wall behind
    with its lancet window and ivy."""
    cut = "7"
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    ty = float(A7["ledge_top_y"])
    x0, x1 = A7["ledge_x"]
    paper = P.paper_for(ah, aw, "c7-par")
    # --- tower behind: round, courses bowing, a gold rim down its sun side
    tower_poly = [(-M - 10, -M - 10), (198, -M - 10), (206, 300), (214, 700), (224, ty + 80), (228, ph + M + 10),
                  (-M - 10, ph + M + 10)]
    canvas = backlit_stone(cut, tower_poly, "c7-tower", course=46.0, block=92.0, cyl=(30.0, 190.0), curve=0.22)
    # --- geometry of the parapet (plate coords)
    xa, xb = x0 - 40, x1 + 14            # left end hidden behind the tower
    back_l, back_r = ty - 54, ty - 44    # back edge of the top surface
    front_l, front_r = ty + 12, ty + 16  # front edge (the fox stands between)
    face_t = 38.0                        # coping thickness
    endw = 34.0                          # the end face (facing the sun) width at the front
    top_poly = [(xa, back_l), (xb - 10, back_r), (xb + endw * 0.55, back_r + 6), (xb + endw, front_r), (xb, front_r),
                (xa, front_l)]
    cope_poly = [(xa, front_l), (xb, front_r), (xb, front_r + face_t), (xa, front_l + face_t)]
    end_cope = [(xb, front_r), (xb + endw, front_r), (xb + endw - 2, front_r + face_t + 2), (xb, front_r + face_t)]
    wall_top = front_l + face_t + 8
    wall_poly = [(xa + 14, wall_top), (xb - 10, wall_top + 3), (xb - 10, ph + M + 10), (xa + 14, ph + M + 10)]
    end_wall = [(xb - 10, wall_top + 3), (xb + endw - 14, wall_top + 5), (xb + endw - 16, ph + M + 10), (xb - 10, ph + M + 10)]
    # --- the wall face: weathered ashlar in shadow, warm bounce from below
    wall = backlit_stone(cut, wall_poly, "c7-wall", course=58.0, block=136.0)
    C.over(canvas, wall)
    # --- reference for the coping and the end faces, then painted
    ref = np.zeros((ah, aw, 3), np.float32)
    alpha = np.zeros((ah, aw), np.float32)

    def put(poly, col):
        nonlocal ref, alpha
        mm = poly_mask(ah, aw, np.array(poly) + M)
        ref = ref * (1 - mm[..., None]) + col * mm[..., None]
        alpha = np.maximum(alpha, mm)
        return mm
    u = np.clip((xx - xa) / (xb - xa), 0, 1)
    bumps = fbm(ah, aw, 5, "c7-top-b", octaves=3)
    mott = fbm(ah, aw, 30, "c7-top-m", octaves=3)
    # top: raked by gold light from the right, pits and cracks catching it
    topc = np.array([0.98, 0.74, 0.46], np.float32) * (0.62 + 0.42 * u ** 0.7)[..., None]
    topc = topc * (0.80 + 0.30 * mott)[..., None] * (0.86 + 0.24 * bumps)[..., None]
    put(top_poly, topc)
    copec = np.array([0.40, 0.27, 0.30], np.float32) * (0.85 + 0.25 * mott)[..., None]
    copec = copec + (u * 0.08)[..., None] * np.array([1.0, 0.6, 0.4], np.float32)
    put(cope_poly, copec)
    endc = np.array([1.0, 0.70, 0.44], np.float32) * (0.85 + 0.2 * mott)[..., None]
    put(end_cope, endc)
    put(end_wall, np.array([0.92, 0.60, 0.40], np.float32) * (0.8 + 0.25 * mott)[..., None])
    mm_ = (alpha > 0.5).astype(np.float32)
    lay = P.new_canvas(ah, aw)
    ang = P.orientation(ref, 1.5, 4, base=0.03, base_weight=1.0)
    P.paint(lay, np.clip(ref, 0, 1), ang, [
        dict(radius=8, spacing=0.9, steps=(3, 9), opacity=0.96, dry=0.4, cthresh=0.15, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.12, mask=mm_),
        dict(radius=3.5, spacing=1.0, steps=(2, 7), opacity=0.92, dry=0.5, cthresh=0.1, wob=0.3,
             jitter=(0.05, 0.03, 0.06), streak=0.12, mask=mm_, use_err=True, err_thresh=0.04),
        dict(radius=1.6, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.55, cthresh=0.08, wob=0.3,
             jitter=(0.04, 0.03, 0.05), streak=0.1, mask=mm_, use_err=True, err_thresh=0.06),
    ], paper, "c7-cope", clip=alpha)
    C.over(canvas, lay)
    canvas[..., 3] = np.maximum(canvas[..., 3], alpha)
    st = P.Strokes(max_pts=12)
    r = rng("c7-parapet2")

    def line(pts, w, col, op=0.85, **kw):
        prm = dict(dry=0.45, ragged=0.25, bristles=3, pool=0, streak=0.05, taper0=0.05, taper1=0.1)
        prm.update(kw)
        st.add([(x + M, y + M) for x, y in pts], w, np.asarray(col, np.float32), seed=int(r.integers(1 << 30)),
               opacity=op, **prm)
    # crisp edges: the lit front lip of the top, the back edge, the end arris
    line([(xa, front_l - 1), ((xa + xb) / 2, (front_l + front_r) / 2 - 1), (xb, front_r - 1)], 2.2, (1.0, 0.86, 0.60), 0.9)
    line([(xa, back_l + 1), (xb - 10, back_r + 1)], 1.4, (0.70, 0.46, 0.38), 0.7)
    line([(xb, front_r), (xb, front_r + face_t)], 1.6, (1.0, 0.82, 0.56), 0.85, cap=1.0)
    line([(xb + endw, front_r), (xb + endw * 0.55, back_r + 6), (xb - 10, back_r)], 1.4, (1.0, 0.90, 0.66), 0.8)
    # the overhang's shadow on the moulding and the moulding itself
    line([(xa + 14, front_l + face_t + 3), (xb - 10, front_r + face_t + 3)], 5.0, (0.12, 0.08, 0.12), 0.85, cap=1.0)
    line([(xa + 14, wall_top + 2), (xb - 10, wall_top + 5)], 3.0, (0.55, 0.36, 0.34), 0.85, cap=1.0)
    line([(xa + 14, wall_top + 7), (xb - 10, wall_top + 10)], 2.0, (0.20, 0.13, 0.17), 0.7, cap=1.0)
    # rain stains running down the wall face, and long cracks through the courses
    for i in range(16):
        sx_ = xa + 30 + (xb - xa - 60) * r.random()
        ln = 60 + 220 * r.random()
        line([(sx_, wall_top + 12), (sx_ + (r.random() - 0.5) * 4, wall_top + 12 + ln)], 4 + 6 * r.random(),
             (0.20, 0.13, 0.16), 0.25 + 0.2 * r.random(), dry=0.7, ragged=0.4, taper0=0.1, taper1=0.8)
    for i in range(4):
        cx_ = xa + 80 + (xb - xa - 160) * r.random()
        pts = [(cx_, wall_top + 14)]
        for q in range(6):
            pts.append((pts[-1][0] + (r.random() - 0.5) * 22, pts[-1][1] + 18 + 16 * r.random()))
        line(pts, 1.4, (0.12, 0.07, 0.10), 0.85, ragged=0.5)
        line([(x - 1.5, y) for x, y in pts], 0.8, (0.62, 0.42, 0.38), 0.5, ragged=0.5)
    # cracks across the top and down the coping face, each with a lit lip
    for i in range(7):
        cx_ = xa + 60 + (xb - xa - 120) * r.random()
        pts = []
        yv = back_l + 6 + r.random() * 10
        xv = cx_
        for q in range(6):
            pts.append((xv, yv))
            xv += (r.random() - 0.4) * 14
            yv += (front_l - back_l) / 5.5
        line(pts, 1.0, (0.30, 0.18, 0.20), 0.85, ragged=0.5)
        line([(x - 1.2, y - 0.8) for x, y in pts], 0.7, (1.0, 0.88, 0.62), 0.6, ragged=0.5)
        if r.random() < 0.6:
            yv2 = front_l + 2
            line([(xv, yv2), (xv + (r.random() - 0.5) * 8, yv2 + face_t * 0.5), (xv + (r.random() - 0.5) * 10, yv2 + face_t)],
                 0.9, (0.16, 0.10, 0.12), 0.8, ragged=0.5)
    # chips knocked out of the front lip
    for i in range(10):
        cxp = xa + 30 + (xb - xa - 60) * r.random()
        fy = front_l + (front_r - front_l) * (cxp - xa) / (xb - xa)
        line([(cxp, fy + 2), (cxp + 5 + 9 * r.random(), fy + 3)], 3.0, (0.30, 0.20, 0.22), 0.9, ragged=0.5)
    # moss in the joints of the top and along the back edge, catching the light
    for i in range(320):
        mx = xa + (xb - xa) * r.random()
        t = r.random()
        my = back_l + (front_l - back_l) * (t ** 3 if r.random() < 0.6 else 1 - t ** 4 * 0.1)
        c = P.mix(np.array([0.24, 0.27, 0.12]), np.array([0.92, 0.84, 0.40]), r.random() ** 2 * (0.4 + 0.6 * (mx - xa) / (xb - xa)))
        line([(mx, my), (mx + 2 + 4 * r.random(), my - r.random() * 1.5)], 1.6 + 1.8 * r.random(), c, 0.9, ragged=0.45,
             pool=0.1)
    P.draw(canvas, st.arrays(), paper)
    # ivy: hanging over the face, climbing the tower, leaves catching the gold light
    mv = P.Strokes(max_pts=8)
    for (ix, ln) in [(300, 190), (330, 110), (365, 240), (610, 140), (640, 80), (900, 200), (935, 120)]:
        ivy_strokes(mv, r, ix + M, front_l + face_t + 4 + M, ln, scale=1.3)
    for (ix, iy, ln) in [(150, 980, 420), (175, 900, 300), (120, 1040, 520), (60, 760, 260), (185, 1000, 360),
                         (95, 1010, 470), (200, 1060, 380)]:
        ivy_strokes(mv, r, ix + M, iy + M, ln, droop=False, scale=1.8)
    wx, wtop, wbot = 92.0, 330.0, 560.0
    mv.add([(wx + M, wbot + M), (wx + M, wtop + 26 + M)], 22.0, np.array([0.10, 0.07, 0.11]), seed=seed_of("lancet"),
           opacity=0.96, dry=0.15, ragged=0.08, bristles=3, pool=0.1, streak=0, taper0=0, taper1=0, cap=1.0)
    mv.add([(wx + M, wtop + 30 + M), (wx + M, wtop + M)], 13.0, np.array([0.10, 0.07, 0.11]), seed=seed_of("lancet2"),
           opacity=0.96, dry=0.15, ragged=0.08, bristles=3, pool=0.1, streak=0, taper0=0, taper1=0.9)
    mv.add([(wx - 30 + M, wbot + 6 + M), (wx + 30 + M, wbot + 6 + M)], 5.0, np.array([0.80, 0.56, 0.42]),
           seed=seed_of("lsill"), opacity=0.9, dry=0.4, ragged=0.2, bristles=3, pool=0, streak=0.1, cap=1.0)
    for side in (-1, 1):
        mv.add([(wx + side * 30 + M, wbot + M), (wx + side * 30 + M, wtop + 24 + M), (wx + side * 12 + M, wtop - 14 + M)],
               3.0, np.array([0.46, 0.32, 0.34]), seed=seed_of("lsur", side), opacity=0.85, dry=0.4, ragged=0.2,
               bristles=2, pool=0, streak=0.1)
    mv.add([(205 + M, -60 + M), (209 + M, 300 + M), (217 + M, 700 + M), (226 + M, ty + 30 + M)], 2.4,
           np.array([1.0, 0.80, 0.52]), seed=seed_of("tower-rim"), opacity=0.75, dry=0.6, ragged=0.3, bristles=2,
           pool=0, streak=0, taper0=0.05, taper1=0.05)
    P.draw(canvas, mv, paper)
    P.finish(canvas, paper, grain=0.04)
    return canvas


def wind_ring_layer(f, cam):
    t = f / T.FPS
    t0 = T.CUES["wind_ring"]
    out = np.zeros((T.H, T.W, 3), np.float32)
    if t < t0:
        return out
    u = (t - t0) / 2.5
    if u > 1.2:
        return out
    cx, cy = A7["sun_end"][0], H7 + 40
    R = 120 + 2100 * u ** 0.8
    ry = R * 0.10
    sx, sy = C.to_screen(cam, cx, cy)
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    d = np.sqrt(((xs - sx) / (R * cam[0])) ** 2 + ((ys - sy) / (ry * cam[0])) ** 2)
    w = 0.10 + 0.10 * u
    band = np.exp(-((d - 1.0) / w) ** 2) + 0.5 * np.exp(-((d - 0.88) / (w * 1.6)) ** 2)
    ang = np.arctan2((ys - sy) / max(ry, 1), (xs - sx) / max(R, 1))
    k = ((ang + math.pi) / (2 * math.pi) * 512).astype(int) % 512
    nz = value_noise(1, 512, 6, "ring-n")[0][k]
    streak = 0.45 + 0.75 * nz
    a = band * streak * (1 - min(u, 1.0)) ** 1.3 * 0.42
    out += a[..., None] * np.array([1.0, 0.92, 0.80], np.float32)
    return out


def render_7(f):
    cam = T.camera("7", f)
    bg = C.sample(C.load_plate("c7_sky"), C.parallax_cam("7", f, 0.6))
    # the sun disc sits behind the cloud tops and climbs clear of them
    sx, sy = sun_pos(f)
    ssx, ssy = C.to_screen(C.parallax_cam("7", f, 0.85), sx, sy)
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    dd = np.sqrt((xs - ssx) ** 2 + (ys - ssy) ** 2)
    rad = 46 * cam[0]
    disc = np.clip(rad - dd + 0.5, 0, 1)
    bg[..., :3] = bg[..., :3] * (1 - disc[..., None]) + disc[..., None] * np.array([1.0, 0.97, 0.86], np.float32)
    C.over(bg, C.sample(C.load_plate("c7_sea"), C.parallax_cam("7", f, 0.85)))
    C.over(bg, C.sample(C.load_plate("c7_ledge"), cam))
    bg[..., 3] = 1.0
    glow = (np.exp(-(dd / (90 * cam[0])) ** 2) * 0.55 + np.exp(-(dd / (380 * cam[0])) ** 2) * 0.22)[..., None] * GOLD
    # sunbeams fanning up from behind the cloud tops
    ang = np.arctan2(ys - ssy, xs - ssx)
    rays = (0.5 + 0.5 * np.sin(ang * 23 + 1.3)) * (0.5 + 0.5 * np.sin(ang * 9 + 0.4))
    rays = rays * P.smoothstep(80, 360, dd) * P.smoothstep(ssy + 60, ssy - 120, ys)
    glow += (rays * np.exp(-dd / (900 * cam[0])) * 0.09)[..., None] * GOLD
    glow += wind_ring_layer(f, C.parallax_cam("7", f, 0.85))
    # grass on the ledge whips harder once the wind ring passes
    t = (f - (f % 2)) / T.FPS
    tw = T.CUES["wind_ring"]
    wind = 0.25 + 0.9 * P.smoothstep(tw - 0.2, tw + 0.6, t) * (1 - 0.3 * P.smoothstep(tw + 1.5, tw + 2.5, t))
    x0, x1 = A7["ledge_x"]
    ty = float(A7["ledge_top_y"])
    rr = rng("c7-clumps")
    tufts = []
    for ci in range(11):
        cxg = x0 + 20 + (x1 - x0 - 60) * (ci + rr.random() * 0.8) / 11
        n = int(6 + rr.integers(0, 12))
        hgt = 50 + 110 * rr.random()
        tufts += grass_tufts(("c7", ci), [(cxg, ty + 3)], hgt, n, lean0=-0.1 + 0.3 * rr.random())
    tufts += grass_tufts("c7-short", [(x, ty + 3) for x in np.linspace(x0 + 10, x1 - 20, 30)], 28, 2)
    fg = np.zeros((T.H, T.W, 4), np.float32)
    P.draw(fg, grass_strokes(tufts, t, cam, float(wind), np.array([1.0, 0.80, 0.42]), np.array([0.20, 0.15, 0.17])),
           P.paper_for(T.H, T.W, "screen-paper"))
    return dict(bg=bg, fg=fg, glow=glow.astype(np.float32))


# ---------------------------------------------------------------- cut 8

A8 = T.ANCHORS["8"]
H8 = 1080.0


def paint_8():
    sky = sky_plate("8", H8, (2300, 900), "c8", glow_k=1.4, streaks=10,
                    clouds=[(250, 860, 420, 190, 0.45), (820, 980, 320, 90, 0.7), (1760, 990, 380, 100, 0.68),
                            (2450, 760, 300, 150, 0.35), (560, 280, 560, 46, 0.55), (2050, 190, 460, 40, 0.6),
                            (1500, 820, 220, 60, 0.8)])
    C.save_plate("c8_sky", sky)
    import clouds_sunrise as CS
    sky_at = CS.sky_fn(H8 + M, (2300 + M, 900 + M), SKY_LOW, GOLD, SKY_MID)
    sea = CS.paint_banks("8", CS.banks_8(), (2300.0, 900.0), CS.PAL_SUN, sky_at, "c8-cum", wisps=None,
                         floor=CS.floor_8(), stroke_scale=1.15)
    C.save_plate("c8_sea", sea)
    pw, ph, aw, ah = dims("8")
    cx, cy = A8["balcony_corner"]
    fy = float(A8["fox_paws"][1]) + 28           # the floor's near edge, in front of the fox
    # the balcony floor seen a little from above, its stone face below, a short
    # balustrade along the far edge for scale
    face = [(-M - 10, fy), (560, fy - 6), (566, fy + 30), (560, ph + M + 10), (-M - 10, ph + M + 10)]
    para = backlit_stone("8", face, "c8-par", course=22.0, block=48.0)
    floor_poly = [(-M - 10, cy + 40), (540, cy + 52), (560, fy - 6), (-M - 10, fy)]
    yy, xx = grid("8")
    fm = poly_mask(ah, aw, np.array(floor_poly) + M)
    fl = np.array([0.92, 0.70, 0.52], np.float32) * (0.85 + 0.25 * fbm(ah, aw, 14, "c8-fl", octaves=3))[..., None]
    joints = (np.abs(np.mod(xx + (yy - cy) * 0.6, 70.0)) < 1.6) | (np.abs(np.mod(yy - cy, 26.0)) < 1.4)
    fl = fl * (1 - 0.25 * joints)[..., None]
    flc = P.new_canvas(ah, aw)
    flc[..., :3] = fl * fm[..., None]
    flc[..., 3] = fm
    C.over(para, flc)
    st = P.Strokes(max_pts=8)
    st.add([(-M + M, fy + M), (300 + M, fy - 3 + M), (560 + M, fy - 6 + M)], 2.0, np.array([1.0, 0.84, 0.56]),
           seed=seed_of("c8-edge"), opacity=0.85, dry=0.4, ragged=0.2, bristles=3, pool=0, streak=0, cap=1.0)
    st.add([(560 + M, fy - 6 + M), (566 + M, fy + 28 + M), (562 + M, ph + M)], 4.0, np.array([0.95, 0.66, 0.44]),
           seed=seed_of("c8-end"), opacity=0.9, dry=0.35, ragged=0.15, bristles=3, pool=0.1, streak=0.1, cap=1.0)
    rail_y = cy - 22
    st.add([(-M + M, rail_y + M), (540 + M, rail_y + 12 + M)], 5.0, np.array([0.50, 0.36, 0.38]),
           seed=seed_of("c8-rail"), opacity=0.96, dry=0.3, ragged=0.1, bristles=3, pool=0.1, streak=0.1, cap=1.0)
    st.add([(-M + M, rail_y - 3 + M), (540 + M, rail_y + 9 + M)], 1.4, np.array([1.0, 0.82, 0.54]),
           seed=seed_of("c8-railh"), opacity=0.8, dry=0.5, ragged=0.2, bristles=2, pool=0, streak=0, cap=1.0)
    for i, bxp in enumerate(np.arange(-40, 530, 21.0)):
        yb = rail_y + 12 * (bxp + M) / (540 + M)
        st.add([(bxp + M, yb + 4 + M), (bxp + 0.5 + M, yb + 24 + M), (bxp + M, yb + 58 + M)], 3.2,
               np.array([0.46, 0.33, 0.36]), seed=seed_of("c8-bal", i), opacity=0.96, dry=0.25, ragged=0.1,
               bristles=2, pool=0.12, streak=0.1, cap=1.0)
        st.add([(bxp + 2 + M, yb + 6 + M), (bxp + 2 + M, yb + 56 + M)], 0.7, np.array([1.0, 0.80, 0.52]),
               seed=seed_of("c8-balh", i), opacity=0.6, dry=0.6, ragged=0.3, bristles=2, pool=0, streak=0)
    P.draw(para, st, P.paper_for(ah, aw, "c8-par"))
    C.save_plate("c8_par", para)


def render_8(f):
    cam = T.camera("8", f)
    bg = C.sample(C.load_plate("c8_sky"), C.parallax_cam("8", f, 0.5))
    C.over(bg, C.sample(C.load_plate("c8_sea"), C.parallax_cam("8", f, 0.8)))
    C.over(bg, C.sample(C.load_plate("c8_par"), cam))
    bg[..., 3] = 1.0
    sx, sy = C.to_screen(C.parallax_cam("8", f, 0.5), 2300, 900)
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    dd = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
    ang = np.arctan2(ys - sy, xs - sx)
    rays = (0.5 + 0.5 * np.sin(ang * 17 + 0.7)) * (0.5 + 0.5 * np.sin(ang * 7 + 2.0))
    rays = rays * P.smoothstep(120, 520, dd)
    glow = (rays * np.exp(-dd / 1400.0) * 0.10 + np.exp(-(dd / 500.0) ** 2) * 0.22)[..., None] * GOLD
    return dict(bg=bg, fg=None, glow=glow.astype(np.float32))


# ---------------------------------------------------------------- 9a / 9b balcony

def balcony_plate(cut, rail_top_s, floor_top_s, key, scale=1.0):
    """The balcony in warm morning light. rail_top_s, floor_top_s: screen y of
    the balustrade rail and of the floor's far edge at the cut's first frame."""
    pw, ph, aw, ah = dims(cut)
    yy, xx = grid(cut)
    f0 = T.cut_range(cut)[0]
    _, rail_y = T.screen_to_plate(cut, f0, 0, rail_top_s)
    _, floor_y = T.screen_to_plate(cut, f0, 0, floor_top_s)
    paper = P.paper_for(ah, aw, key)
    sky = sky_plate(cut, rail_y + 120, (pw * 0.85, rail_y - 200), key + "-sky", glow_k=0.8, streaks=5,
                    top=np.array([0.46, 0.58, 0.84]), mid=np.array([0.86, 0.76, 0.74]),
                    low=np.array([1.0, 0.88, 0.70]),
                    clouds=[(pw * 0.22, rail_y - 300 * scale, 420, 120, 0.3), (pw * 0.62, rail_y - 420 * scale, 360, 70, 0.4),
                            (pw * 0.95, rail_y - 180 * scale, 300, 90, 0.35), (pw * 0.45, rail_y - 110, 260, 50, 0.6)])
    import clouds_sunrise as CS
    hz_y = rail_y - 40
    sun9 = (pw * 0.85, hz_y - 200)
    sky_at = CS.sky_fn(hz_y + M, (sun9[0] + M, sun9[1] + M), np.array([1.0, 0.90, 0.74]), np.array([1.0, 0.92, 0.72]),
                       np.array([0.86, 0.76, 0.74]))
    sea = CS.paint_banks(cut, CS.banks_far(hz_y, key, scale=scale, width=pw), sun9, CS.PAL_MORNING, sky_at,
                         key + "-cum", floor=(hz_y + M, np.array([0.78, 0.70, 0.76], np.float32),
                                              np.array([1.0, 0.92, 0.78], np.float32), 6.0))
    C.over(sky, sea)
    # floor: worn flagstones in perspective, warm, with the balusters' shadows
    eye = floor_y - 260 * scale
    val, joint, sid = flagstones(cut, eye, pw * 0.55, key, scale=scale)
    h2 = (sid * 40503) % 65536 / 65536.0
    fcol = P.mix(STONE_SH, STONE_LIT, 0.50 + 0.28 * val)
    fcol = fcol + ((h2 > 0.84)[..., None] * np.array([0.03, 0.0, -0.03], np.float32))
    fcol = fcol * (0.92 + 0.12 * fbm(ah, aw, 30, (key, "fm"), octaves=3))[..., None] * (1 - 0.40 * joint)[..., None]
    wear = fbm(ah, aw, 90, (key, "wear"), octaves=2)
    fcol = fcol * (0.94 + 0.10 * wear)[..., None]
    moss = joint * P.smoothstep(0.6, 0.8, fbm(ah, aw, 40, (key, "moss"), octaves=3))
    fcol = P.mix(fcol, np.array([0.45, 0.55, 0.28], np.float32), moss * 0.5)
    # shadows cast by the actual balustrade (plinth, vase balusters, piers,
    # rail), thrown toward us and to the left, broadening and softening with
    # distance, dappled where ivy breaks the light; warm light between
    dyf = np.maximum(yy - floor_y, 0)
    reach = 300.0 * scale ** 0.5
    hf = dyf / reach                                   # height on the balustrade that casts here
    wob = (value_noise(ah, aw, 40, (key, "shw")) - 0.5) * 14 * (0.4 + hf)
    x0s = xx + dyf * 0.62 + 10 + wob
    bxs = np.mod(x0s, 58.0) - 29
    bidx_s = np.floor(x0s / 58.0)
    PL, BT = 0.14, 0.84                                # plinth top, rail bottom (fractions of height)
    tb = np.clip(1 - (hf - PL) / (BT - PL), 0, 1)
    profs = np.interp(tb, [0.0, 0.06, 0.07, 0.14, 0.16, 0.30, 0.48, 0.70, 0.84, 0.86, 0.92, 0.93, 1.0],
                      [14, 14, 10, 10, 7, 6, 8, 14, 11, 9, 9, 15, 15]) * scale ** 0.3 * (1 + 0.5 * hf)
    sh_bal = (hf >= PL) & (hf < BT) & (np.abs(bxs) < profs) & ~((bidx_s == 21) & (tb > 0.35))
    sh_pier = (np.mod(bidx_s, 9) == 4) & (np.abs(bxs) < 27 * (1 + 0.5 * hf)) & (hf < 1.0)
    sh_pl = hf < PL
    sh_rail = (hf >= BT) & (hf < 1.0)
    shadow = ((sh_bal | sh_pier | sh_pl | sh_rail) & (yy >= floor_y)).astype(np.float32)
    near_s = P.blur(shadow, 2.0)
    far_s = P.blur(shadow, 9.0)
    kk = np.clip(hf, 0, 1)
    shadow = near_s * (1 - kk) + far_s * kk
    dapple = P.smoothstep(0.35, 0.65, fbm(ah, aw, 26, (key, "dapple"), octaves=3))
    shadow = np.clip(shadow + dapple * 0.35 * (hf < 1.1) * (yy >= floor_y) * (xx < pw * 0.35), 0, 1)
    # the light itself is uneven (cloud breaks, the ivy, the rail's own dents)
    patch = 0.55 + 0.45 * P.smoothstep(0.3, 0.7, fbm(ah, aw, 180, (key, "lightpatch"), octaves=2))
    shadow = shadow * (0.92 - 0.30 * kk) * patch
    warm = np.array([1.06, 0.98, 0.86], np.float32)
    cool = np.array([0.66, 0.62, 0.78], np.float32)
    fcol = fcol * (warm + (cool - warm) * shadow[..., None])
    floor = (yy >= floor_y).astype(np.float32)
    # balustrade: rail, balusters, base
    rail = ((yy >= rail_y) & (yy < rail_y + 26)).astype(np.float32)
    base_y = floor_y - 6
    plinth = ((yy >= base_y - 24) & (yy < floor_y)).astype(np.float32)
    bx = np.mod(xx + 10, 58.0) - 29
    t = np.clip((yy - rail_y - 26) / max(base_y - 24 - rail_y - 26, 1), 0, 1)
    # a classic vase baluster: top block, cap, neck, swelling vase, base block
    prof = np.interp(t, [0.0, 0.06, 0.07, 0.14, 0.16, 0.30, 0.48, 0.70, 0.84, 0.86, 0.92, 0.93, 1.0],
                     [14, 14, 10, 10, 7, 6, 8, 14, 11, 9, 9, 15, 15]) * scale ** 0.3
    bal = ((np.abs(bx) < prof) & (yy >= rail_y + 26) & (yy < base_y - 24)).astype(np.float32)
    # a solid pier every nine balusters, and one baluster broken away
    bidx = np.floor((xx + 10) / 58.0)
    pier = (np.mod(bidx, 9) == 4) & (np.abs(bx) < 27) & (yy >= rail_y) & (yy < floor_y)
    broken = (bidx == 21) & (yy > rail_y + 26 + (base_y - rail_y) * 0.35)
    bal = np.where(broken, 0.0, bal)
    bal = np.maximum(bal, pier.astype(np.float32))
    bl = np.clip(0.45 + bx / prof * 0.5, 0.12, 1.0)
    bl = np.where(pier, np.clip(0.5 + bx / 27 * 0.35, 0.2, 1.0), bl)
    bcol = P.mix(STONE_SH, STONE_LIT, bl)
    bcol = bcol * (0.9 + 0.15 * fbm(ah, aw, 10, (key, "balv"), octaves=3))[..., None]
    lich = P.smoothstep(0.7, 0.85, fbm(ah, aw, 14, (key, "lich"), octaves=3))
    bcol = P.mix(bcol, np.array([0.62, 0.66, 0.40], np.float32), lich * 0.4)
    ref = sky[..., :3].copy()
    ref = P.mix(ref, bcol, bal)
    ref = P.mix(ref, P.mix(STONE_SH, STONE_LIT, 0.75 + 0.0 * xx), rail)
    ref = P.mix(ref, P.mix(STONE_SH, STONE_LIT, 0.45 + 0.0 * xx), plinth)
    ref = P.mix(ref, fcol, floor)
    ref = np.clip(ref, 0, 1).astype(np.float32)
    canvas = sky.copy()
    struct = np.clip(bal + rail + plinth + floor, 0, 1)
    m = (P.blur(struct, 1.0) > 0.3).astype(np.float32)
    ang = P.orientation(ref, 1.5, 4, base=np.where(bal > 0.5, math.pi / 2, 0.0), base_weight=0.8)
    P.paint(canvas, ref, ang, [
        dict(radius=10, spacing=0.9, steps=(2, 7), opacity=0.96, dry=0.35, cthresh=0.12, wob=0.25,
             jitter=(0.04, 0.02, 0.05), streak=0.08, mask=m),
        dict(radius=4, spacing=1.0, steps=(2, 6), opacity=0.92, dry=0.45, cthresh=0.09, wob=0.25,
             jitter=(0.04, 0.02, 0.05), streak=0.08, mask=m, use_err=True, err_thresh=0.04),
        dict(radius=1.8, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.5, cthresh=0.07, wob=0.25,
             jitter=(0.03, 0.02, 0.04), streak=0.06, mask=m, use_err=True, err_thresh=0.06),
    ], paper, (key, "bal"), clip=struct)
    st = P.Strokes(max_pts=12)
    xs_ = np.linspace(-M, pw + M, 12)
    st.add([(x + M, rail_y + 1 + M) for x in xs_], 2.0, np.clip(STONE_LIT * 1.12, 0, 1), seed=seed_of(key, "railh"),
           opacity=0.8, dry=0.5, ragged=0.2, bristles=3, pool=0, streak=0.05, taper0=0, taper1=0, cap=1.0)
    st.add([(x + M, floor_y + 1 + M) for x in xs_], 2.0, np.clip(STONE_SH * 0.8, 0, 1), seed=seed_of(key, "flr"),
           opacity=0.7, dry=0.4, ragged=0.2, bristles=3, pool=0, streak=0.05, taper0=0, taper1=0, cap=1.0)
    P.draw(canvas, st, paper)
    # small weeds in the joints near the balustrade, ivy trailing over the rail at the left end
    wd = P.Strokes(max_pts=6)
    rw = rng(key, "weeds")
    jy, jx = np.nonzero((joint > 0.5) & (yy > floor_y + 4) & (yy < floor_y + 260 * scale) & (rw.random(joint.shape) < 0.004))
    for (y0_, x0_) in zip(jy[:400], jx[:400]):
        for q in range(int(2 + rw.integers(0, 4))):
            hgt = (5 + 12 * rw.random()) * scale
            a_ = -math.pi / 2 + (rw.random() - 0.5) * 1.4
            c = P.mix(np.array([0.30, 0.38, 0.16]), np.array([0.75, 0.80, 0.40]), rw.random() * 0.7)
            wd.add([(x0_, y0_), (x0_ + math.cos(a_) * hgt * 0.5, y0_ + math.sin(a_) * hgt * 0.6),
                    (x0_ + math.cos(a_) * hgt, y0_ + math.sin(a_) * hgt)], 0.8 * scale, c,
                   seed=int(rw.integers(1 << 30)), opacity=0.9, dry=0.3, ragged=0.15, bristles=2, pool=0, streak=0,
                   taper0=0.05, taper1=0.8, pickup=0)
    # ivy: a dense clump over the rail at the left end, strands trailing down
    for i in range(160):
        px_ = M + 20 + rw.random() * 230 * (1 - rw.random() * 0.4)
        py_ = rail_y + M - 6 + rw.random() * 30 * rw.random()
        a_ = rw.random() * math.pi
        c = P.mix(np.array([0.20, 0.26, 0.14]), np.array([0.78, 0.80, 0.38]), rw.random() ** 1.6)
        wd.add([(px_ - math.cos(a_) * 6 * scale, py_ - math.sin(a_) * 6 * scale),
                (px_ + math.cos(a_) * 6 * scale, py_ + math.sin(a_) * 6 * scale)], 4.5 * scale, c,
               seed=int(rw.integers(1 << 30)), opacity=0.94, dry=0.2, ragged=0.35, bristles=3, pool=0.12,
               streak=0.05, pickup=0)
    for v_ in range(9):
        x0_ = 30 + v_ * 24 + rw.random() * 16 + M
        ln = min((60 + 150 * rw.random()) * scale, (floor_y - rail_y) - 14) * (0.25 + 0.75 * rw.random() ** 1.5)
        pts = [(x0_ + 6 * math.sin(q * 0.8 + v_), rail_y + M + 10 + q * ln / 6) for q in range(7)]
        wd.add(pts, 1.2 * scale, np.array([0.18, 0.22, 0.12]), seed=int(rw.integers(1 << 30)), opacity=0.9, dry=0.3,
               ragged=0.2, bristles=2, pool=0, streak=0, taper0=0.05, taper1=0.6)
        for (px_, py_) in pts[1:]:
            for q in range(2):
                a_ = rw.random() * math.pi
                ox = (rw.random() - 0.5) * 10 * scale
                c = P.mix(np.array([0.18, 0.24, 0.12]), np.array([0.70, 0.74, 0.34]), rw.random() ** 1.5)
                wd.add([(px_ + ox - math.cos(a_) * 5 * scale, py_ - math.sin(a_) * 5 * scale),
                        (px_ + ox + math.cos(a_) * 5 * scale, py_ + math.sin(a_) * 5 * scale)], 4.0 * scale, c,
                       seed=int(rw.integers(1 << 30)), opacity=0.93, dry=0.2, ragged=0.35, bristles=2, pool=0.1,
                       streak=0.05, pickup=0)
    P.draw(canvas, wd, paper)
    P.finish(canvas, paper, grain=0.04)
    canvas[..., 3] = 1.0
    return canvas


def beams_glow(cut, f):
    cam = T.camera(cut, f)
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    q = (xs * 0.62 + ys * 0.78)
    b = (0.5 + 0.5 * np.sin(q * 0.006 + 0.5)) ** 6 * 0.10 + (0.5 + 0.5 * np.sin(q * 0.013 + 2.0)) ** 8 * 0.06
    b *= P.smoothstep(-200, T.H, ys) * 0.7 + 0.3
    return (b[..., None] * np.array([1.0, 0.88, 0.62], np.float32)).astype(np.float32)


def paint_9():
    C.save_plate("c9a_bg", balcony_plate("9a", 470, 660, "c9a", scale=1.0))
    C.save_plate("c9b_bg", balcony_plate("9b", 250, 400, "c9b", scale=1.5))


def render_9(cut, f):
    cam = T.camera(cut, f)
    bg = C.sample(C.load_plate(f"c{cut}_bg"), cam)
    bg[..., 3] = 1.0
    return dict(bg=bg, fg=None, glow=beams_glow(cut, f))


# ---------------------------------------------------------------- dispatch

def render_cut(cut, f):
    if cut == "7":
        return render_7(f)
    if cut == "8":
        return render_8(f)
    if cut in ("9a", "9b"):
        return render_9(cut, f)
    raise KeyError(cut)


if __name__ == "__main__":
    import time
    t0 = time.time()
    cuts = sys.argv[2:] if len(sys.argv) > 2 else ["7", "8", "9"]
    if "7" in cuts:
        log("cut 7")
        paint_7()
    if "7sea" in cuts:
        log("cut 7 sea only")
        paint_7(sea_only=True)
    if "8sea" in cuts:
        log("cut 8 sea only")
        import clouds_sunrise as CS
        sky_at = CS.sky_fn(H8 + M, (2300 + M, 900 + M), SKY_LOW, GOLD, SKY_MID)
        C.save_plate("c8_sea", CS.paint_banks("8", CS.banks_8(), (2300.0, 900.0), CS.PAL_SUN, sky_at, "c8-cum",
                                              wisps=None, floor=CS.floor_8(), stroke_scale=1.15))
    if "7ledge" in cuts:
        log("cut 7 ledge only")
        C.save_plate("c7_ledge", parapet_7())
    if "8" in cuts:
        log("cut 8")
        paint_8()
    if "9" in cuts or "9a" in cuts or "9b" in cuts:
        log("cut 9")
        paint_9()
    log(f"done {time.time() - t0:.1f}s")
