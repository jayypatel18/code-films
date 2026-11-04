"""Cuts 6b and 6c: the stairwell and the dome, seen straight up.

  6b  a dramatic low angle up the circular stairwell: rings of shelves
      wrapping the walls storey above storey, a spiral stair winding up,
      galleries with railings, the dome far above. The plate turns slowly.
  6c  the dome from below: the old copper shutters break open on hard key
      poses with falling debris, and the first gold light crashes down.

Run `python3 world/dome.py paint` through tools/heavy.py.
"""

import math
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise, rng, seed_of, ridged  # noqa: E402

from engine.paint_shapes import poly_mask  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402
import books as BK  # noqa: E402

M = MARGIN
MOON = np.array(T.LIGHTS["hall"]["key"], np.float32)
AMB = np.array(T.LIGHTS["hall"]["ambient"], np.float32)
GOLD = np.array(T.LIGHTS["gold_burst"]["key"], np.float32)
WOOD = np.array([0.300, 0.200, 0.150], np.float32)
STONE = np.array([0.420, 0.405, 0.470], np.float32)
COPPER = np.array([0.22, 0.40, 0.40], np.float32)
DARK = np.array([0.030, 0.034, 0.075], np.float32)


def dims(cut):
    pw, ph = T.plate_size(cut)
    return pw, ph, pw + 2 * M, ph + 2 * M


# ---------------------------------------------------------------- 6b stairwell

# Storey rings seen from below: nested circles that are NOT concentric (the
# camera is off the axis, so each higher storey's ring drifts toward the upper
# right) and NOT evenly spaced (storeys of different heights). Plate coords.
RINGS_R = [1700.0, 1060.0, 760.0, 540.0, 395.0, 285.0, 205.0, 150.0]
C_NEAR = (860.0, 900.0)        # centre of the nearest ring
C_TOP = (1150.0, 520.0)        # centre of the dome (off axis, upper right)
R_DOME = 112.0
WIN_6B = [(1, -2.30), (2, 0.55), (3, -1.10), (4, 2.30), (5, -0.30)]   # (storey, angle) of tall windows


def ring_centres():
    out = []
    for R in RINGS_R:
        t = (math.log(RINGS_R[0] / R) / math.log(RINGS_R[0] / RINGS_R[-1])) ** 0.85
        out.append((C_NEAR[0] + (C_TOP[0] - C_NEAR[0]) * t, C_NEAR[1] + (C_TOP[1] - C_NEAR[1]) * t))
    return out


def stair_coords(aw, ah):
    """Per pixel: storey k, position v across it (0 at its outer, lower edge,
    1 at the inner, higher edge), angle th about the local axis, s = k + v."""
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    xx -= M
    yy -= M
    cs = ring_centres()
    n = len(RINGS_R)
    dn = [np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / R for c, R in zip(cs, RINGS_R)]
    count = np.zeros((ah, aw), np.int32)
    for d in dn:
        count += (d < 1.0)
    k = np.clip(count - 1, -1, n - 1)
    v = np.zeros((ah, aw), np.float32)
    th = np.zeros((ah, aw), np.float32)
    for i in range(n - 1):
        m = k == i
        if not m.any():
            continue
        a_ = 1 - dn[i][m]
        b_ = dn[i + 1][m] - 1
        vv = a_ / np.maximum(a_ + b_, 1e-6)
        v[m] = vv
        cxm = cs[i][0] * (1 - vv) + cs[i + 1][0] * vv
        cym = cs[i][1] * (1 - vv) + cs[i + 1][1] * vv
        th[m] = np.arctan2(yy[m] - cym, xx[m] - cxm)
    outside = k < 0
    th[outside] = np.arctan2(yy[outside] - cs[0][1], xx[outside] - cs[0][0])
    v[outside] = 0.0
    top = k == n - 1
    th[top] = np.arctan2(yy[top] - cs[-1][1], xx[top] - cs[-1][0])
    v[top] = np.clip(1 - dn[-1][top], 0, 1)
    s_ = np.where(outside, -0.5, k + v).astype(np.float32)
    return xx, yy, k, v, th, s_, cs, dn


def stair_angle(s_):
    """The helix: the angle of the spiral stair at storey coordinate s."""
    return 0.4 + 2 * math.pi * 0.42 * s_


def stairwell():
    cut = "6b"
    pw, ph, aw, ah = dims(cut)
    xx, yy, k, v, th, s_, cs, dn = stair_coords(aw, ah)
    n = len(RINGS_R)
    up = np.clip(s_ / (n - 1), 0, 1)
    # light: the tall windows throw cold moonlight across their storey and the next
    win = np.zeros((ah, aw), np.float32)
    for (ks, ta) in WIN_6B:
        dth = np.angle(np.exp(1j * (th - ta)))
        win = np.maximum(win, np.exp(-((s_ - ks - 0.55) / 0.55) ** 2) * np.exp(-(dth / 0.6) ** 2))
    L = AMB[None, None] * (0.85 + 0.5 * up)[..., None] + MOON * (win * 0.55 + 0.08)[..., None]
    ref = np.zeros((ah, aw, 3), np.float32)
    ref[:] = DARK * 1.6 + 0.012
    wood = WOOD * (0.8 + 0.4 * fbm(ah, aw, 24, "6b-wood", octaves=3))[..., None]
    # bands across each storey: gallery slab underside with joists, railing, shelves, cornice
    # galleries only on some storeys, of different depths: the second one is the
    # deepest and reads closest to camera; storeys without one show bare wall
    SW = np.array([0.24, 0.0, 0.17, 0.0, 0.13, 0.0, 0.10, 0.0], np.float32)
    sw = SW[np.clip(k, 0, len(SW) - 1)]
    slab = (v < sw) & (k >= 0)
    joist = slab & (np.abs(np.mod(th / (2 * math.pi) * 64, 1.0) - 0.5) < 0.08)
    ref = np.where(slab[..., None], wood * 0.55, ref)
    ref = np.where(joist[..., None], wood * 0.35, ref)
    edge = (v >= sw) & (v < sw + 0.035) & (sw > 0)
    ref = np.where(edge[..., None], wood * 1.15, ref)
    railb = (v >= sw + 0.035) & (v < sw + 0.11) & (sw > 0) & (np.abs(np.mod(th / (2 * math.pi) * 180 + 0.5 * k, 1.0) - 0.5) < 0.15)
    # bare stone wall where there is no gallery, courses following the curve
    bare = (sw == 0) & (k >= 0) & (v < 0.30)
    stonec = STONE * 0.7 * (0.85 + 0.25 * fbm(ah, aw, 14, "6b-wall", octaves=3))[..., None]
    stonec = np.where((np.abs(np.mod(v * 30, 1.0) - 0.5) > 0.44)[..., None], stonec * 0.6, stonec)
    ref = np.where(bare[..., None], stonec, ref)
    ref = np.where(railb[..., None], wood * 0.9, ref)
    # shelf boards (two or three rows per storey, uneven)
    for vb in (0.42, 0.62, 0.82):
        b = (np.abs(v - vb - 0.02 * np.sin(k * 3.1)) < 0.012) & (k >= 0) & (k < n - 1)
        ref = np.where(b[..., None], wood, ref)
    # the tall windows: arched openings full of night sky
    for (ks, ta) in WIN_6B:
        dth = np.angle(np.exp(1j * (th - ta)))
        wv = (k == ks) & (v > 0.34) & (v < 0.94) & (np.abs(dth) < 0.13 * (1 - 0.5 * np.clip((v - 0.8) / 0.14, 0, 1)))
        frame = (k == ks) & (v > 0.31) & (v < 0.96) & (np.abs(dth) < 0.17) & ~wv
        sky = np.array([0.36, 0.44, 0.70], np.float32) * (0.9 + 0.2 * v)[..., None]
        lead = np.abs(np.mod(v * 30, 1.0) - 0.5) > 0.44
        ref = np.where(wv[..., None], np.where(lead[..., None], np.array([0.05, 0.05, 0.09], np.float32), sky / 2.0), ref)
        ref = np.where(frame[..., None], STONE * 0.7, ref)
    # dome at the top: ribbed copper with the shutters' ring and thin gold seams
    top = k == n - 1
    rib = np.abs(np.mod(th / (2 * math.pi) * 18, 1.0) - 0.5) < 0.05
    dcol = COPPER * (0.6 + 0.5 * v)[..., None]
    dcol = np.where(rib[..., None], dcol * 0.55, dcol)
    ref = np.where(top[..., None], dcol, ref)
    lit = np.clip(ref * L * 2.3, 0, 1)
    dr = np.sqrt((xx - cs[-1][0]) ** 2 + (yy - cs[-1][1]) ** 2)
    shut = (dr < R_DOME) & top
    lit = np.where(shut[..., None], np.array([0.10, 0.11, 0.12], np.float32), lit)
    seam = (np.abs(np.mod((th + math.pi) / (2 * math.pi) * 10, 1.0) - 0.5) > 0.47) & shut
    lit = np.where(seam[..., None], GOLD * 0.8, lit)
    lit = lit + (np.exp(-((dr - R_DOME) / 3.0) ** 2) * top * 0.4)[..., None] * GOLD
    # windows themselves stay luminous
    lit = np.clip(lit, 0, 1).astype(np.float32)
    paper = P.paper_for(ah, aw, "6b")
    canvas = P.new_canvas(ah, aw)
    ang = (th + math.pi / 2).astype(np.float32)
    P.paint(canvas, lit, ang, [
        dict(radius=12, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=5, spacing=1.0, steps=(2, 7), opacity=0.92, dry=0.4, cthresh=0.08, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.025),
        dict(radius=2.2, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.45, cthresh=0.06, wob=0.2,
             jitter=(0.04, 0.02, 0.04), streak=0.06, use_err=True, err_thresh=0.04),
    ], paper, "6b-struct")

    def at(si, ang_, vv):
        """Plate point at storey coordinate (si + vv) and angle ang_."""
        i = int(min(max(si, 0), n - 2))
        R0, R1 = RINGS_R[i], RINGS_R[i + 1]
        c0, c1 = cs[i], cs[i + 1]
        R = R0 * (R1 / R0) ** vv
        cx_ = c0[0] + (c1[0] - c0[0]) * vv
        cy_ = c0[1] + (c1[1] - c0[1]) * vv
        return (cx_ + math.cos(ang_) * R + M, cy_ + math.sin(ang_) * R + M)

    def inward(si, ang_, vv, frac):
        i = int(min(max(si, 0), n - 2))
        R0, R1 = RINGS_R[i], RINGS_R[i + 1]
        c0, c1 = cs[i], cs[i + 1]
        R = R0 * (R1 / R0) ** vv * frac
        cx_ = c0[0] + (c1[0] - c0[0]) * vv
        cy_ = c0[1] + (c1[1] - c0[1]) * vv
        return (cx_ + math.cos(ang_) * R + M, cy_ + math.sin(ang_) * R + M)

    def Lat(pt):
        xi, yi = int(min(max(pt[0], 0), aw - 1)), int(min(max(pt[1], 0), ah - 1))
        return L[yi, xi] * 2.25
    # books standing on the shelves, bending with the curved walls, varied heights
    st = P.Strokes(max_pts=3)
    acc = P.Strokes(max_pts=3)
    rb = rng("6b-books")
    for ks in range(n - 1):
        for (v0, v1) in ((0.31, 0.415), (0.435, 0.615), (0.635, 0.815)):
            Rm = RINGS_R[ks] * (RINGS_R[ks + 1] / RINGS_R[ks]) ** v0
            spine = 11.0 * Rm / 800.0
            circ = 2 * math.pi * Rm
            nb = int(circ / max(spine * 1.1, 1.2))
            a_ = rb.random() * 2 * math.pi
            row_h = 0.78 + 0.14 * rb.random()
            for i in range(nb):
                wb = spine * (0.6 + 0.9 * rb.random())
                a_ += wb / Rm * 1.1
                # skip where a window or the stair is
                if any(ks == w[0] and abs(math.remainder(a_ - w[1], 2 * math.pi)) < 0.2 for w in WIN_6B):
                    continue
                if abs(math.remainder(a_ - stair_angle(ks + (v0 + v1) / 2), 2 * math.pi)) < 0.32:
                    continue
                if rb.random() < 0.02:
                    a_ += wb / Rm * 2
                    continue
                hgt = row_h + 0.07 * (rb.random() - 0.5)
                p0 = at(ks, a_, v0)
                p1 = at(ks, a_ + (rb.random() - 0.5) * 0.004, v0 + (v1 - v0) * hgt)
                bc = BK.BOOK_COLS[int(rb.integers(0, len(BK.BOOK_COLS)))] * (0.85 + 0.3 * rb.random())
                bc = bc * 0.75 + bc.mean() * 0.25
                col = np.clip(bc * Lat(p0) * 0.85, 0, 1)
                st.add([p0, p1], max(0.6, wb / 2), col, seed=int(rb.integers(1 << 30)), opacity=0.96, dry=0.18,
                       ragged=0.05, bristles=2, pool=0.12, streak=0.1, taper0=0, taper1=0, cap=1.0)
                if rb.random() < 0.45:
                    acc.add([at(ks, a_ + wb / Rm * 0.3, v0), at(ks, a_ + wb / Rm * 0.3, v0 + (v1 - v0) * hgt)],
                            max(0.4, wb * 0.12), np.clip(col * 1.5 + MOON * 0.03, 0, 1), seed=int(rb.integers(1 << 30)),
                            opacity=0.5, dry=0.6, ragged=0.2, bristles=2, pool=0, streak=0, cap=1.0)
    P.draw(canvas, P.shuffle(st.arrays(), "6b-b"), paper)
    P.draw(canvas, acc.arrays(), paper)
    # the spiral stair seen from beneath: a continuous stone soffit hugging the
    # wall as a helix, its inner edge stepped (one notch per tread), and a
    # railing with balusters along that edge
    s_vals = np.arange(-0.25, n - 1.5, 0.008)
    chunk = 0.25
    soffit = np.zeros((ah, aw), np.float32)
    for c0 in np.arange(s_vals[0], s_vals[-1], chunk):
        seg = s_vals[(s_vals >= c0) & (s_vals <= c0 + chunk + 0.01)]
        if len(seg) < 2:
            continue
        outer, inner = [], []
        for sv in seg:
            ks = int(math.floor(max(sv, 0)))
            vv = max(sv, 0) - ks
            a_ = stair_angle(sv)
            step = (sv / 0.055) % 1.0
            fin = 0.885 + (0.015 if step < 0.5 else 0.0)
            outer.append(inward(ks, a_, vv, 1.02))
            inner.append(inward(ks, a_, vv, fin))
        poly = np.array(outer + inner[::-1]) - M
        soffit = np.maximum(soffit, poly_mask(ah, aw, poly + M))
    sof_ref = np.clip(STONE * 0.62 * L * 2.3, 0, 1)
    sof_ref = sof_ref * (0.85 + 0.25 * fbm(ah, aw, 18, "6b-sof", octaves=3))[..., None]
    sof_canvas = P.new_canvas(ah, aw)
    P.paint(sof_canvas, sof_ref.astype(np.float32), (th + math.pi / 2).astype(np.float32), [
        dict(radius=6, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.12, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, mask=(soffit > 0.5).astype(np.float32)),
        dict(radius=2.5, spacing=1.0, steps=(2, 6), opacity=0.92, dry=0.4, cthresh=0.1, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, mask=(soffit > 0.5).astype(np.float32), use_err=True,
             err_thresh=0.04),
    ], paper, "6b-soffit", clip=soffit)
    C.over(canvas, sof_canvas)
    canvas[..., 3] = 1.0
    sst = P.Strokes(max_pts=8)
    rail = []
    edge = []
    for sv in s_vals[::2]:
        ks = int(math.floor(max(sv, 0)))
        vv = max(sv, 0) - ks
        rail.append(inward(ks, stair_angle(sv), vv, 0.86))
        edge.append(inward(ks, stair_angle(sv), vv, 0.885))
    for jj in range(0, len(edge) - 6, 5):
        seg = edge[jj:jj + 7]
        Ls = Lat(seg[0])
        sst.add(seg, 1.4, np.clip(STONE * 1.3 * Ls + MOON * 0.06, 0, 1), seed=8000 + jj, opacity=0.85, dry=0.4,
                ragged=0.15, bristles=2, pool=0, streak=0, taper0=0, taper1=0)
        rs = rail[jj:jj + 7]
        sst.add(rs, 2.0, np.clip(WOOD * 1.1 * Ls + MOON * 0.04, 0, 1), seed=9000 + jj, opacity=0.95, dry=0.3,
                ragged=0.08, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0)
    for jj in range(0, len(edge), 3):
        Ls = Lat(edge[jj])
        sst.add([edge[jj], rail[jj]], 1.0, np.clip(WOOD * 0.9 * Ls, 0, 1), seed=9500 + jj, opacity=0.9, dry=0.3,
                ragged=0.08, bristles=2, pool=0.1, streak=0, taper0=0, taper1=0)
    P.draw(canvas, sst.arrays(), paper)
    # shelf boards restated as arcs along each storey
    sb = P.Strokes(max_pts=40)
    for ks in range(n - 1):
        for vb in (0.31, 0.43, 0.63, 0.83):
            pts = [at(ks, 2 * math.pi * q / 39, vb) for q in range(40)]
            sb.add(pts, max(1.0, RINGS_R[ks] * 0.003), np.clip(WOOD * 0.8 * Lat(pts[0]), 0, 1), seed=ks * 10 + int(vb * 100),
                   opacity=0.55, dry=0.3, ragged=0.1, bristles=2, pool=0.05, streak=0.1, taper0=0, taper1=0)
    P.draw(canvas, sb.arrays(), paper)
    # moonlit rims along the gallery edges facing the windows
    rim = P.Strokes(max_pts=24)
    for ks in range(n - 1):
        if SW[ks] == 0:
            continue
        for (_, ta) in WIN_6B:
            pts = [at(ks, ta + 2.2 + 0.08 * q, float(SW[ks]) + 0.02) for q in range(20)]
            rim.add(pts, max(0.7, RINGS_R[ks] * 0.0018), np.clip(MOON * 0.55, 0, 1), seed=ks * 100 + int(ta * 10),
                    opacity=0.45, dry=0.6, ragged=0.3, bristles=2, pool=0, streak=0, taper0=0.3, taper1=0.3)
    P.draw(canvas, rim.arrays(), paper)
    # blue air thickens toward the top
    air = np.clip(s_ / (n - 1), 0, 1) ** 1.6 * (k < n - 1)
    canvas[..., :3] = canvas[..., :3] * (1 - air[..., None] * 0.45) + np.array([0.08, 0.10, 0.20], np.float32) * air[..., None] * 0.45
    P.finish(canvas, paper, grain=0.05)
    canvas[..., 3] = 1.0
    # glow: shafts of moonlight slanting in from the windows across the well
    glow = np.zeros((ah, aw, 3), np.float32)
    for (ks, ta) in WIN_6B:
        wp = at(ks, ta, 0.62)
        tgt = at(ks, ta + math.pi * 0.8, 0.5)
        dx, dy = tgt[0] - wp[0], tgt[1] - wp[1]
        ln = math.hypot(dx, dy) + 1e-6
        ux, uy = dx / ln, dy / ln
        X, Y = xx + M - wp[0], yy + M - wp[1]
        t = (X * ux + Y * uy) / ln
        sd = np.abs(X * -uy + Y * ux)
        wdt = RINGS_R[ks] * 0.10 * (1 + 0.6 * np.clip(t, 0, 1))
        sh = P.smoothstep(1.0, 0.4, sd / wdt) * P.smoothstep(0.0, 0.08, t) * P.smoothstep(1.0, 0.5, t)
        glow += (sh * 0.10)[..., None] * MOON
    glow += (np.exp(-((dr - R_DOME) / 10.0) ** 2) * 0.25 * top)[..., None] * GOLD
    return canvas, glow.astype(np.float32)


def render_6b(f):
    cut = "6b"
    pl = C.load_plate("c6b_bg")
    gl = C.load_plate("c6b_glow")
    zoom, cx, cy = T.camera(cut, f)
    f0, f1 = T.cut_range(cut)
    u = (f - f0) / max(f1 - 1 - f0, 1)
    rot = math.radians(-14.0 * u)
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    px0 = (xs - T.W / 2) / zoom
    py0 = (ys - T.H / 2) / zoom
    cr, sr = math.cos(rot), math.sin(rot)
    px = cr * px0 - sr * py0 + cx + M
    py = sr * px0 + cr * py0 + cy + M
    bg = np.empty((T.H, T.W, 4), np.float32)
    for c in range(4):
        bg[..., c] = ndimage.map_coordinates(pl[..., c], [py, px], order=1, mode="nearest")
    bg[..., 3] = 1.0
    glow = np.empty((T.H, T.W, 3), np.float32)
    for c in range(3):
        glow[..., c] = ndimage.map_coordinates(gl[..., c], [py, px], order=1, mode="nearest")
    return dict(bg=bg, fg=None, glow=glow)


# ---------------------------------------------------------------- 6c dome

N_SHUT = 10
R_LANT = 170.0      # lantern ring radius (the shutters)
R_SHUT_IN = 70.0


DOME_C = (1150.0, 520.0)      # plate position of the lantern opening (off axis, as in 6b)


def dome_plate(state):
    """state: 'closed', 'half' (shutters mid swing) or 'open'."""
    cut = "6c"
    pw, ph, aw, ah = dims(cut)
    cx, cy = DOME_C[0] + M, DOME_C[1] + M
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    th = np.arctan2(dy, dx)
    # dome interior: ribbed copper and stone, darker at the rim, a moonlit side
    t = np.clip(r / 1300.0, 0, 1.4)
    rib = np.abs(np.mod(th / (2 * math.pi) * 20 + 0.5, 1.0) - 0.5)
    ribm = rib < 0.035 + 0.02 * t
    panel = COPPER * (0.55 + 0.35 * fbm(ah, aw, 30, "6c-pat", octaves=3))[..., None]
    side = 0.5 + 0.5 * np.cos(th - (-0.6))
    L = AMB[None, None] * (1.0 + 0.4 * side)[..., None] + MOON * (0.15 * side)[..., None]
    ref = panel * (1.2 - 0.5 * t)[..., None]
    ref = np.where(ribm[..., None], ref * 0.55 + STONE * 0.15, ref)
    # coffered rings, softened and broken so they read as painted mouldings
    rv = np.abs(np.mod(np.log(r / 60.0) * 4.2 + 0.15 * np.sin(th * 3), 1.0) - 0.5)
    ring = P.smoothstep(0.06, 0.02, rv) * (r > R_LANT * 1.2) * (0.6 + 0.4 * value_noise(ah, aw, 40, "6c-rb"))
    ref = ref * (1 - 0.45 * ring)[..., None]
    lit = np.clip(ref * L * 2.4, 0, 1)
    gold_sky = np.array([1.0, 0.86, 0.55], np.float32)
    # the lantern ring of shutters
    sector = np.floor((th + math.pi) / (2 * math.pi) * N_SHUT)
    sfrac = (th + math.pi) / (2 * math.pi) * N_SHUT - sector
    inring = (r > R_SHUT_IN) & (r < R_LANT)
    rr = rng("6c-panels")
    tilt = rr.random(N_SHUT)
    if state == "closed":
        shut = inring
        # old copper panels gone green and brown, riveted bands, plank seams
        scol = np.array([0.20, 0.30, 0.28], np.float32) * (0.7 + 0.4 * fbm(ah, aw, 10, "6c-sh", octaves=3))[..., None]
        brown = P.smoothstep(0.6, 0.8, fbm(ah, aw, 16, "6c-br", octaves=3))
        scol = P.mix(scol, np.array([0.30, 0.20, 0.14], np.float32), brown * 0.7)
        seam = (np.abs(sfrac - 0.5) > 0.46) | (np.abs(np.mod(sfrac * 3, 1.0) - 0.5) > 0.47)
        scol = np.where(seam[..., None], scol * 0.45, scol)
        band = (np.abs(r - R_SHUT_IN * 1.5) < 3) | (np.abs(r - R_LANT * 0.85) < 3)
        scol = np.where(band[..., None], scol * 0.6, scol)
        rivet = band & (np.abs(np.mod(sfrac * 6, 1.0) - 0.5) < 0.08)
        scol = np.where(rivet[..., None], scol * 2.0, scol)
        lit = np.where(shut[..., None], np.clip(scol * L * 2.2, 0, 1), lit)
        crack = inring & (np.abs(sfrac - 0.5) > 0.488)
        lit = np.where(crack[..., None], gold_sky * 0.85, lit)
        lit = np.where((r <= R_SHUT_IN)[..., None], np.array([0.10, 0.10, 0.11], np.float32), lit)
    else:
        # shutters flung outward and up: each panel seen nearly edge on, hinged at
        # the rim, each at its own angle; two have torn away completely
        gone = np.isin(sector, [2, 7]) if state == "open" else np.zeros_like(sector, bool)
        wdt = (0.20 + 0.22 * tilt)[np.clip(sector.astype(int), 0, N_SHUT - 1)]
        if state == "half":
            wdt = 0.45 + 0.35 * tilt[np.clip(sector.astype(int), 0, N_SHUT - 1)]
        sl = inring & (sfrac < wdt) & (r > R_LANT * (0.35 + 0.3 * tilt[np.clip(sector.astype(int), 0, N_SHUT - 1)])) & ~gone
        open_ = (r < R_LANT) & ~sl
        sky = gold_sky * (1.0 + 0.12 * (1 - r / R_LANT))[..., None]
        lit = np.where(open_[..., None], np.clip(sky, 0, 1), lit)
        pc = np.array([0.24, 0.20, 0.16], np.float32) * (0.8 + 0.3 * fbm(ah, aw, 8, "6c-op", octaves=2))[..., None]
        lit = np.where(sl[..., None], pc, lit)
        edge = sl & (sfrac > wdt - 0.02)
        lit = np.where(edge[..., None], np.array([1.0, 0.82, 0.50], np.float32), lit)
        # the panels' planks and the lantern frame: radial posts between sectors
        plank = sl & (np.abs(np.mod(sfrac / np.maximum(wdt, 0.05) * 3, 1.0) - 0.5) > 0.45)
        lit = np.where(plank[..., None], lit * 0.6, lit)
        post = (r < R_LANT * 1.02) & (np.abs(sfrac - 0.0) < 0.02 * (R_LANT / np.maximum(r, 20)) ** 0.5)
        lit = np.where(post[..., None], np.array([0.16, 0.12, 0.10], np.float32), lit)
        ringf = np.abs(r - R_LANT) < 6
        lit = np.where(ringf[..., None], np.array([0.30, 0.24, 0.18], np.float32), lit)
        # the interior of the dome now catches gold near the opening
        warm = np.exp(-((r - R_LANT) / 420.0) ** 2) * (r >= R_LANT)
        lit = np.clip(lit + warm[..., None] * GOLD * 0.35, 0, 1)
    paper = P.paper_for(ah, aw, "6c")
    canvas = P.new_canvas(ah, aw)
    ang = np.where(r < R_LANT * 1.1, th, th + math.pi / 2).astype(np.float32)
    P.paint(canvas, lit.astype(np.float32), ang, [
        dict(radius=12, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=5, spacing=1.0, steps=(2, 7), opacity=0.92, dry=0.4, cthresh=0.08, wob=0.2,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.025),
        dict(radius=2.2, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.45, cthresh=0.06, wob=0.2,
             jitter=(0.04, 0.02, 0.04), streak=0.06, use_err=True, err_thresh=0.04),
    ], paper, ("6c", state))
    P.finish(canvas, paper, grain=0.05)
    canvas[..., 3] = 1.0
    return canvas


def debris(f, f_break, cam):
    """Fragments of copper and stone tumbling down toward us, drawn per frame."""
    out = np.zeros((T.H, T.W, 4), np.float32)
    t = (f - f_break) / T.FPS
    if t < 0 or t > 0.9:
        return out
    st = P.Strokes(max_pts=3)
    r = rng("6c-debris")
    for i in range(40):
        a = r.random() * 2 * math.pi
        sp = 300 + 900 * r.random()
        rad = R_LANT * 0.9 + sp * t
        grow = 1 + 3.5 * t * (0.5 + r.random())
        dcx, dcy = C.to_screen(cam, DOME_C[0], DOME_C[1])
        x = dcx + math.cos(a) * rad * cam[0]
        y = dcy + math.sin(a) * rad * cam[0] + 200 * t * t
        sz = (2.5 + 4 * r.random()) * (1 + 1.6 * t)
        rot = r.random() * 3 + t * 8 * (r.random() - 0.5)
        c = np.array([0.18, 0.14, 0.12]) if r.random() < 0.6 else np.array([0.20, 0.32, 0.30])
        st.add([(x - math.cos(rot) * sz, y - math.sin(rot) * sz), (x + math.cos(rot) * sz, y + math.sin(rot) * sz)],
               sz * 0.45, c, seed=i, opacity=0.95, dry=0.2, ragged=0.4, bristles=2, pool=0.1, streak=0, pickup=0)
    P.draw(out, st, P.paper_for(T.H, T.W, "screen-paper"))
    return out


def shaft_gaps(state):
    """(centre angle, half width) of every gap the light pours through."""
    rr = rng("6c-panels")
    tilt = rr.random(N_SHUT)
    gaps = []
    for i in range(N_SHUT):
        a0 = -math.pi + (i + 0.0) / N_SHUT * 2 * math.pi
        span = 2 * math.pi / N_SHUT
        if state == "open":
            if i in (2, 7):
                gaps.append((a0 + span * 0.5, span * 0.5, 1.0))
            else:
                w = 0.05 + 0.12 * tilt[i]
                gaps.append((a0 + span * (w + (1 - w) / 2), span * (1 - w) / 2 * 0.8, 0.2 + 0.5 * tilt[i] ** 2))
        elif state == "half":
            w = 0.45 + 0.35 * tilt[i]
            gaps.append((a0 + span * (w + (1 - w) / 2), span * (1 - w) / 2 * 0.7, 0.4))
    return gaps


def render_6c(f):
    cut = "6c"
    cam = T.camera(cut, f)
    fb = int(round(T.CUES["shutters_break"] * T.FPS))
    k = f - fb
    # held key poses: closed, a two frame snap mid swing, then flung open
    state = "closed" if k < 0 else ("half" if k < 2 else "open")
    bg = C.sample(C.load_plate("c6c_" + state), cam)
    bg[..., 3] = 1.0
    C.over(bg, debris(f, fb, cam))
    bg[..., 3] = 1.0
    ys, xs = np.mgrid[0:T.H, 0:T.W].astype(np.float32)
    cx, cy = C.to_screen(cam, DOME_C[0], DOME_C[1])
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    ang = np.arctan2(ys - cy, xs - cx)
    rl = R_LANT * cam[0]
    glow = np.zeros((T.H, T.W, 3), np.float32)
    if state == "closed":
        # only thin gold lines leak round the shutters before they break
        glow += (np.exp(-(d / (rl * 1.2)) ** 2) * 0.12)[..., None] * GOLD
        return dict(bg=bg, fg=None, glow=glow)
    t_after = max(0.0, (f - fb) / T.FPS)
    reach = rl + 2400 * min(1.0, t_after / 0.5) ** 0.7
    # broad shafts through the gaps: soft edged wedges, streaked along their length,
    # dimming as they fall toward us, with dust caught in them
    ki = ((ang + math.pi) / (2 * math.pi) * 2048).astype(int) % 2048
    streak = value_noise(1, 2048, 5, "6c-streak")[0][ki]
    shafts = np.zeros((T.H, T.W), np.float32)
    for (ca, hw, strength) in shaft_gaps(state):
        da = np.abs(np.angle(np.exp(1j * (ang - ca))))
        wedge = P.smoothstep(hw * 1.15, hw * 0.55, da)
        shafts = np.maximum(shafts, wedge * strength)
    vol = (0.65 + 0.35 * streak) * (0.8 + 0.2 * value_noise(T.H, T.W, 120, "6c-vol"))
    along = P.smoothstep(rl * 0.6, rl * 1.4, d) * np.exp(-np.maximum(d - rl, 0) / (reach * 0.55)) * (d < reach)
    glow += (shafts * vol * along * 0.55)[..., None] * GOLD
    glow += (np.exp(-(d / (rl * 1.4)) ** 2) * 0.45 + np.exp(-(d / (rl * 3.5)) ** 2) * 0.16)[..., None] * GOLD
    # dust motes sparkling inside the shafts
    r = rng("6c-dust")
    n = 260
    md = np.zeros((T.H, T.W, 4), np.float32)
    stp = P.Strokes(max_pts=2)
    for i in range(n):
        a = r.random() * 2 * math.pi
        rad = rl * (1.1 + 6 * r.random() ** 1.5) + 40 * t_after * (1 + r.random())
        x, y = cx + math.cos(a) * rad, cy + math.sin(a) * rad
        if not (0 <= x < T.W and 0 <= y < T.H):
            continue
        b = float(shafts[int(y), int(x)])
        if b < 0.2:
            continue
        stp.add([(x, y), (x + 0.4, y + 0.3)], 0.8 + 1.2 * r.random(), np.array([1.0, 0.92, 0.70]) * b, seed=i,
                opacity=min(1.0, b), dry=0, ragged=0.05, bristles=2, pool=0, streak=0, paper_k=0, pickup=0)
    P.draw(md, stp, P.paper_for(T.H, T.W, "screen-paper"))
    glow += md[..., :3]
    return dict(bg=bg, fg=None, glow=glow.astype(np.float32))


def render_cut(cut, f):
    return render_6b(f) if cut == "6b" else render_6c(f)


if __name__ == "__main__":
    import time
    t0 = time.time()
    log("6b")
    c6b, g6b = stairwell()
    C.save_plate("c6b_bg", c6b)
    np.save(C.plate_path("c6b_glow"), g6b.astype(np.float16))
    log("6c")
    C.save_plate("c6c_closed", dome_plate("closed"))
    C.save_plate("c6c_half", dome_plate("half"))
    C.save_plate("c6c_open", dome_plate("open"))
    log(f"done {time.time() - t0:.1f}s")
