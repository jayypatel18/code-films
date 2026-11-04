"""Cut 1, Before Dawn: the library tower on its cliff above a sea of clouds.

Layers (all plates carry common.MARGIN on every side):
  cut1_sky    opaque. Night sky, the faint dawn band, stars. Parallax 0.55.
  cut1_stage  alpha.  Cloud sea, cliff, tower, pines, ivy. Follows the camera.
  cut1_bank   alpha.  The dark grassy bank lower left. Parallax 1.35.
Animated in render(): the lantern light crossing the window row, the pampas
plumes on twos (drawn per drawing with the stroke kernel), a slow cloud drift.

Run `python3 world/cut1_dawn.py paint` (through tools/heavy.py) to paint plates.
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
from engine.paint_shapes import poly_mask, wobble, ellipse_mask, soft, rough_edge  # noqa: E402

import common as C  # noqa: E402
import cut1_tower as TW  # noqa: E402
from engine import paint_flora as F  # noqa: E402
from common import T, MARGIN, log  # noqa: E402

CUT = "1"
PW, PH = T.plate_size(CUT)
AW, AH = PW + 2 * MARGIN, PH + 2 * MARGIN     # array size
A = T.ANCHORS[CUT]
M = MARGIN

# camera of the cloud world (array coordinates)
HORIZON = 688 + M
CXS = PW / 2 + M
FOC = 1600.0
CAMH = 0.50
TOWER_Z = 6.2

SKY_PARALLAX = 0.55
BANK_PARALLAX = 1.35

# palette (display space)
ZENITH = np.array([0.045, 0.050, 0.140], np.float32)
SKY_MID = np.array([0.110, 0.110, 0.290], np.float32)
SKY_LOW = np.array([0.270, 0.215, 0.420], np.float32)
DAWN = np.array([0.560, 0.420, 0.520], np.float32)
DAWN_HOT = np.array([0.700, 0.520, 0.540], np.float32)

CL_SHADOW = np.array([0.072, 0.058, 0.185], np.float32)
CL_MID = np.array([0.195, 0.168, 0.370], np.float32)
CL_LIT = np.array([0.560, 0.570, 0.820], np.float32)
CL_ROSE = np.array([0.700, 0.560, 0.680], np.float32)
HAZE = np.array([0.470, 0.385, 0.565], np.float32)

LIGHT = np.array([-0.86, -0.42, 0.30], np.float32)
MOON_POS = (400.0, 235.0)     # plate coords: a low full moon opposite the dawn
MOON_R = 40.0
CL_RIM = np.array([0.80, 0.82, 1.00], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def _grid():
    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float32)
    return yy, xx


# ---------------------------------------------------------------- sky

def sky_reference():
    yy, xx = _grid()
    y = yy - M
    x = xx - M
    t = np.clip(y / (HORIZON - M), 0, 1.2)
    ref = np.empty((AH, AW, 3), np.float32)
    stops = [(0.0, ZENITH), (0.45, SKY_MID), (0.82, SKY_LOW), (1.0, DAWN), (1.2, DAWN)]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        m = (t >= t0) & (t <= t1)
        u = ((t - t0) / (t1 - t0))[m]
        u = u * u * (3 - 2 * u)
        ref[m] = c0 * (1 - u[:, None]) + c1 * u[:, None]
    # the dawn band is warmer and brighter to the right, behind the tower
    hot = np.exp(-((x - 1850) / 950.0) ** 2) * np.exp(-((y - (HORIZON - M)) / 150.0) ** 2)
    ref = P.mix(ref, DAWN_HOT, hot * 0.85)
    glow = np.exp(-((x - 1700) / 420.0) ** 2) * np.exp(-((y - (HORIZON - M) + 20) / 110.0) ** 2)
    ref = ref + glow[..., None] * np.array([0.10, 0.06, 0.04], np.float32)
    # moonlight from off frame upper left lifts that corner a little
    mx, my = MOON_POS
    dm = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
    ref = ref + (np.exp(-(dm / 520.0) ** 2) * 0.10 + np.exp(-(dm / 160.0) ** 2) * 0.10)[..., None] * np.array([0.55, 0.62, 0.95], np.float32)
    # broad soft variation so the gradient is never a computer gradient
    v = fbm(AH, AW, 420, "sky-var", octaves=3)
    ref = ref * (0.93 + 0.14 * v[..., None])
    # long low cloud streaks near the horizon: dark violet, rose underlit
    streak = np.zeros((AH, AW), np.float32)
    lit = np.zeros((AH, AW), np.float32)
    r = rng("sky-streaks")
    for i in range(9):
        cy = HORIZON - M - 30 - r.random() * 230
        cx = r.random() * PW
        ln = 500 + r.random() * 900
        th = 4 + r.random() * 9
        bend = (r.random() - 0.5) * 30
        d = (y - cy - bend * ((x - cx) / ln) ** 2)
        along = np.clip(1 - np.abs((x - cx) / ln) ** 2, 0, 1)
        n = value_noise(AH, AW, 60, ("streak", i)) * 0.8 + 0.2
        band = np.exp(-(d / th) ** 2) * along * n
        streak = np.maximum(streak, band)
        lit = np.maximum(lit, np.exp(-((d - th * 0.9) / (th * 0.45)) ** 2) * along * n * np.clip(1 - (HORIZON - M - cy) / 260, 0, 1))
    ref = P.mix(ref, ref * np.array([0.62, 0.58, 0.78], np.float32), streak * 0.85)
    ref = P.mix(ref, DAWN_HOT * 1.05, lit * 0.55)
    return np.clip(ref, 0, 1)


def paint_sky(paper):
    log("sky: reference")
    ref = sky_reference()
    ang = P.orientation(ref, sigma=3, rho=14, base=0.0, base_weight=1.2, edge_gain=0.5)
    canvas = P.new_canvas(AH, AW)
    passes = [
        dict(radius=24, spacing=0.75, steps=(6, 22), opacity=0.85, dry=0.3, cthresh=0.10, wob=0.15,
             jitter=(0.012, 0.008, 0.02), streak=0.04, ragged=0.25, pickup=0.5, curve=0.3, pool=0.0),
        dict(radius=10, spacing=0.9, steps=(5, 18), opacity=0.75, dry=0.45, cthresh=0.07, wob=0.15,
             jitter=(0.012, 0.008, 0.02), streak=0.04, pickup=0.45, curve=0.35, pool=0.0, use_err=True,
             err_thresh=0.03),
        dict(radius=4, spacing=1.0, steps=(3, 10), opacity=0.7, dry=0.5, cthresh=0.06, wob=0.2,
             jitter=(0.012, 0.008, 0.02), streak=0.04, pickup=0.35, pool=0.0, use_err=True, err_thresh=0.05),
    ]
    P.paint(canvas, ref, ang, passes, paper, "cut1-sky")
    # stars: hand placed dabs, thinning toward the horizon and the dawn
    st = P.Strokes(max_pts=2)
    r = rng("stars")
    n = 2600
    xs = r.random(n) * AW
    ys = r.random(n) * (HORIZON - 40)
    yrel = (ys - M) / (HORIZON - M)
    dawn = np.exp(-((xs - M - 1950) / 1000.0) ** 2)
    keep = r.random(n) < np.clip(1.05 - yrel * 1.25 - dawn * yrel * 0.6, 0, 1) ** 1.6
    for x, y in zip(xs[keep], ys[keep]):
        b = r.random() ** 3
        w = 0.55 + 1.1 * b
        c = np.array([0.80, 0.82, 1.0]) * (0.45 + 0.6 * b)
        if r.random() < 0.15:
            c = c * np.array([1.05, 0.95, 0.85])
        st.add([(x, y), (x + r.random() - 0.5, y + r.random() - 0.5)], w, c, seed=int(r.integers(1 << 30)),
               opacity=0.55 + 0.4 * b, dry=0.0, ragged=0.1, bristles=2, pool=0, streak=0, paper_k=0.1, pickup=0.0)
    P.draw(canvas, st, paper)
    # the moon: a painted disc with soft maria, low on the left
    mx, my = MOON_POS[0] + M, MOON_POS[1] + M
    yy0, xx0 = _grid()
    dm = np.sqrt((xx0 - mx) ** 2 + (yy0 - my) ** 2)
    disc = np.clip(MOON_R - dm + 0.7, 0, 1)
    maria = fbm(AH, AW, 9, "maria", octaves=3)
    mcol = np.array([0.93, 0.93, 0.98], np.float32) * (0.86 + 0.14 * P.smoothstep(0.35, 0.65, maria))[..., None]
    mcol = mcol * (1 - 0.10 * np.clip((xx0 - mx) / MOON_R, 0, 1))[..., None]
    canvas[..., :3] = canvas[..., :3] * (1 - disc[..., None]) + mcol * disc[..., None]
    # a few bright stars get a soft halo
    halo = np.zeros((AH, AW), np.float32)
    bright = r.random(n) < 0.012
    for x, y in zip(xs[bright & keep], ys[bright & keep]):
        xi, yi = int(x), int(y)
        if 0 <= xi < AW and 0 <= yi < AH:
            halo[yi, xi] = 1.0
    halo = soft(halo, 2.0) * 25
    canvas[..., :3] += np.clip(halo, 0, 0.12)[..., None] * np.array([0.7, 0.75, 1.0], np.float32)
    P.finish(canvas, paper, grain=0.05, tooth=0.02)
    return canvas


# ---------------------------------------------------------------- clouds

def _calm(xs, Z):
    # rows passing in front of the cliff stay low so the rock and pines show
    if Z > 5.0:
        return np.ones_like(xs)
    near = np.exp(-(((xs - M - 1560) / 520.0) ** 2))
    return 1.0 - 0.85 * near


def cloud_buffers():
    return PC.band_sea("cut1-sea", AH, AW, FOC, HORIZON, CAMH, z_near=0.85, z_far=48.0,
                       row_step=0.06, base_r=0.100, heap_rate=0.30, heap_height=5.2, flat=0.70, near_step=1.7,
                       near_size=0.8, near_z=2.6,
                       heap_width=9.0, calm=_calm)


def cloud_reference(zb):
    yy, xx = _grid()
    cover = zb["cover"]
    depth = zb["depth"]
    nx, ny, nz = zb["nx"], zb["ny"], zb["nz"]
    dn = zb["dn"]
    # a painter shades the mass first and the lobes second: blend each lobe's
    # normal with a normal blurred at the local lobe scale
    sc = np.where(cover > 0, zb["scale"][np.maximum(zb["band"], 0)], 2.0).astype(np.float32)
    sig = [4.0, 12.0, 32.0]
    sm = [[P.blur(c, g) for g in sig] for c in (nx, ny, nz)]
    t = np.clip(np.log2(np.maximum(sc * 0.7, 2.0) / 2.0) / np.log2(3.0), 0, 2)
    def pick(lv):
        a = np.where(t < 1, lv[0] * (1 - t) + lv[1] * t, lv[1] * (2 - t) + lv[2] * (t - 1))
        return a
    k = 0.78
    nx = nx * (1 - k) + pick(sm[0]) * k
    ny = ny * (1 - k) + pick(sm[1]) * k
    nz = nz * (1 - k) + pick(sm[2]) * k
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    u = nx
    ndl = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    wrap = np.clip((ndl + 0.25) / 1.25, 0, 1)
    tone = P.smoothstep(0.40, 0.70, wrap)
    # the body below each scalloped top sinks into shadow
    body = P.smoothstep(0.35, 2.6, dn)
    # broad light: moonlight breaks through in patches, brightest in the
    # middle distance behind the tower, the left side quieter
    lmap = fbm(AH, AW, 520, "cut1-lightmap", octaves=3)
    focus = np.exp(-(((xx - M - 1500) / 1100.0) ** 2 + ((yy - HORIZON - 120) / 420.0) ** 2))
    lk = np.clip(0.30 + 0.9 * lmap + 0.45 * focus, 0.2, 1.3)
    tone = tone * np.clip(lk, 0, 1)
    # the near sea is mostly in shadow, only its rims catch the moon: it frames the lit middle
    fgd = P.smoothstep(HORIZON + 200, AH - 120, yy)
    tone = tone * (1.0 - 0.85 * fgd)
    col = P.mix(CL_MID, CL_LIT * np.clip(lk, 0.6, 1.1)[..., None], tone)
    col = P.mix(col, CL_SHADOW, np.clip(body * 0.95 + (1 - wrap) * 0.35, 0, 1))
    # moon rims: lobe edges facing the low moon catch a bright cool line
    facing = np.clip(-(nx * 0.88 + ny * 0.47), 0, 1)
    rimm = (1 - nz) ** 1.4 * facing * (dn < 0.05) * (0.55 + 0.45 * P.smoothstep(HORIZON + 700, HORIZON, yy))
    col = P.mix(col, CL_RIM, np.clip(rimm * 0.85, 0, 0.75))
    valley = P.smoothstep(0.6, 2.6, dn)
    col = col * (1 - 0.30 * valley)[..., None]
    # masses that heap up catch the moon; the low lobes between them sink into violet shadow
    lift = P.blur(zb["lift"], 3.0)
    col = col * (0.62 + 0.45 * np.clip(lift, 0, 1))[..., None] + ((1 - np.clip(lift, 0, 1)) * 0.03)[..., None] * np.array([0.3, 0.2, 0.6], np.float32)
    # folds where one lobe tucks behind the next
    cr = PC.creases(zb["order"], zb["band"], zb["scale"], reach=0.45, maxpx=18)
    cr = P.blur(cr, 2.0) * (0.5 + 0.5 * tone)
    nearz = P.smoothstep(7.0, 2.0, depth)
    col = col * (1 - (0.10 * (1 - tone) + 0.27 * nearz) * cr)[..., None]
    # near lobes keep a moonlit upper rim so each billow reads against the one behind
    nrim = P.smoothstep(0.55, 0.92, 1 - nz) * np.clip(-ny, 0, 1) * (dn < 0.05) * nearz
    col = col + (nrim * 0.10)[..., None] * np.array([0.55, 0.58, 0.95], np.float32)
    # a thin lit rim just under the top contour, strongest on light facing slopes
    rim = (1 - nz) ** 3 * np.clip(-ny, 0, 1) * np.clip(0.4 + 0.9 * (-u), 0, 1) * (dn < 0.05)
    col = col + (rim * 0.10)[..., None] * np.array([0.6, 0.65, 1.0], np.float32)
    # rose light from the dawn band on the far tops
    far = P.smoothstep(4.0, 24.0, depth)
    dawnx = np.exp(-((xx - M - 1950) / 1100.0) ** 2)
    rose = np.clip(-ny, 0, 1) ** 2 * (dn < 0.05) * (0.25 + 0.75 * far) * (0.35 + 0.65 * dawnx)
    col = P.mix(col, CL_ROSE, np.clip(rose * 0.55, 0, 0.6))
    # atmosphere: far rows melt into a rose violet haze
    fogk = 1.0 - np.exp(-np.maximum(depth - 1.2, 0) / 9.0)
    hz = P.mix(HAZE, DAWN * 0.95, dawnx * P.smoothstep(HORIZON - 160, HORIZON + 10, yy) * 0.6)
    col = P.mix(col, hz, np.clip(fogk * 0.9, 0, 0.9))
    # value design: the foreground sea sinks into shadow so the lit middle
    # distance around the tower carries the light
    fgdark = P.smoothstep(HORIZON + 150, AH - 60, yy)
    col = col * (1.0 - 0.66 * fgdark)[..., None]
    # low mist lying in the valleys, slightly glowing
    mist = fbm(AH, AW, 260, "cut1-mist", octaves=4)
    mband = P.smoothstep(HORIZON - 10, HORIZON + 120, yy) * P.smoothstep(HORIZON + 520, HORIZON + 200, yy)
    mk = np.clip((mist - 0.45) * 1.6, 0, 1) * mband * body
    col = P.mix(col, HAZE * 1.1, mk * 0.5)
    col = np.where(cover[..., None] > 0, col, HAZE)
    return np.clip(col, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- stage

Z_FRONT = 2.40   # cloud rows nearer than this pass in front of the cliff


def stage_parts():
    zb = cloud_buffers()
    cl = cloud_reference(zb)
    front = ((zb["depth"] < Z_FRONT) & (zb["cover"] > 0)).astype(np.float32)
    front = soft(front, 1.2)
    cliff = TW.cliff_reference()
    tower = TW.tower_reference()
    iv = TW.ivy(tower)
    pines = TW.pines_reference()
    return dict(zb=zb, clouds=cl, front=front, cliff=cliff, tower=tower, ivy=iv, pines=pines)


def compose_stage(parts):
    """The stage reference (what the painter samples), straight colour + alpha."""
    zb = parts["zb"]
    ref = parts["clouds"].copy()
    alpha = np.clip(zb["cover"], 0, 1).copy()
    for key in ("cliff", "tower", "ivy", "pines"):
        pr = parts[key]
        m = pr["alpha"] * (1 - parts["front"])
        ref = ref * (1 - m[..., None]) + pr["rgb"] * m[..., None]
        alpha = np.maximum(alpha, m)
    return ref, alpha


def paint_stage(paper, parts):
    zb = parts["zb"]
    cl = parts["clouds"]
    cover = zb["cover"]
    canvas = P.new_canvas(AH, AW)
    # clouds: strokes follow the billow contours, flattening toward the horizon
    log("stage: clouds")
    ang_c = P.orientation(cl, sigma=2.0, rho=7.0, base=0.0, base_weight=0.55, edge_gain=1.2)
    cmask = (P.blur(cover, 2.0) > 0.3).astype(np.float32)
    passes = [
        dict(radius=22, spacing=0.9, steps=(3, 9), opacity=0.95, dry=0.3, cthresh=0.18, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, ragged=0.22, pickup=0.2, curve=0.5, mask=cmask),
        dict(radius=10, spacing=0.95, steps=(2, 9), opacity=0.92, dry=0.4, cthresh=0.13, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.2, mask=cmask, use_err=True, err_thresh=0.04),
        dict(radius=5, spacing=1.0, steps=(2, 8), opacity=0.9, dry=0.45, cthresh=0.10, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.15, mask=cmask, use_err=True, err_thresh=0.05),
        dict(radius=2.5, spacing=1.1, steps=(2, 7), opacity=0.88, dry=0.5, cthresh=0.08, wob=0.3,
             jitter=(0.03, 0.02, 0.05), streak=0.06, pickup=0.1, mask=cmask, use_err=True, err_thresh=0.07),
    ]
    P.paint(canvas, cl, ang_c, passes, paper, "cut1-clouds")
    log("stage: cloud edges")
    cloud_edges(canvas, paper, parts)
    # a few long wisps of stratus lying over the sea, lit by the moon
    ws = P.Strokes(max_pts=14)
    rw = rng("wisps")
    for i in range(9):
        y0 = HORIZON + 30 + rw.random() * 260
        x0 = rw.random() * AW - 300
        ln = 500 + rw.random() * 900
        pts = [(x0 + ln * q / 12, y0 + 6 * math.sin(q * 0.7 + i) + (q / 12) * (rw.random() - 0.5) * 20) for q in range(13)]
        c = np.clip(CL_LIT * (0.75 + 0.2 * rw.random()), 0, 1)
        ws.add(pts, 2.5 + 5 * rw.random(), c, seed=int(rw.integers(1 << 30)), opacity=0.35 + 0.2 * rw.random(),
               dry=0.75, ragged=0.35, bristles=8, pool=0, streak=0.1, taper0=0.3, taper1=0.5, pickup=0.3)
    P.draw(canvas, ws, paper)
    ref, _ = compose_stage(parts)
    # the cliff
    log("stage: cliff")
    cf = parts["cliff"]
    clip = cf["alpha"] * (1 - parts["front"])
    m = (clip > 0.5).astype(np.float32)
    P.paint(canvas, ref, cf["ang"], [
        dict(radius=11, spacing=0.9, steps=(2, 6), opacity=0.95, dry=0.35, cthresh=0.2, wob=0.5,
             jitter=(0.05, 0.03, 0.06), streak=0.1, ragged=0.35, pickup=0.15, mask=m),
        dict(radius=5, spacing=1.0, steps=(2, 5), opacity=0.92, dry=0.45, cthresh=0.15, wob=0.6,
             jitter=(0.05, 0.03, 0.06), streak=0.1, ragged=0.35, mask=m, use_err=True, err_thresh=0.06),
        dict(radius=2.4, spacing=1.1, steps=(2, 4), opacity=0.9, dry=0.5, cthresh=0.12, wob=0.6,
             jitter=(0.05, 0.03, 0.06), streak=0.1, ragged=0.35, mask=m, use_err=True, err_thresh=0.09),
    ], paper, "cut1-cliff", clip=clip)
    # the tower
    log("stage: tower")
    tw = parts["tower"]
    clip = tw["alpha"] * (1 - parts["front"])
    m = (clip > 0.5).astype(np.float32)
    ang_t = P.blend_angles(P.orientation(ref, sigma=1.0, rho=3.0, base=tw["ang"], base_weight=1.5), tw["ang"], 0.5)
    P.paint(canvas, ref, ang_t, [
        dict(radius=6, spacing=0.9, steps=(2, 6), opacity=0.95, dry=0.3, cthresh=0.12, wob=0.12,
             jitter=(0.04, 0.02, 0.05), streak=0.08, ragged=0.2, pickup=0.15, mask=m, curve=0.7),
        dict(radius=3.0, spacing=1.0, steps=(2, 6), opacity=0.92, dry=0.4, cthresh=0.10, wob=0.12,
             jitter=(0.04, 0.02, 0.05), streak=0.08, ragged=0.2, pickup=0.1, mask=m, use_err=True,
             err_thresh=0.04, curve=0.7),
        dict(radius=1.5, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.45, cthresh=0.08, wob=0.12,
             jitter=(0.03, 0.02, 0.04), streak=0.05, ragged=0.2, pickup=0.05, mask=m, use_err=True,
             err_thresh=0.06, curve=0.7),
    ], paper, "cut1-tower", clip=clip)
    P.draw(canvas, tw["details"], paper)
    # ivy: the clump body, then leaves
    log("stage: ivy")
    iv = parts["ivy"]
    clip = iv["alpha"] * (1 - parts["front"])
    m = (clip > 0.5).astype(np.float32)
    P.paint(canvas, ref, P.orientation(ref, 1.0, 3.0, base=1.4, base_weight=0.8), [
        dict(radius=2.6, spacing=1.0, steps=(1, 3), opacity=0.9, dry=0.35, cthresh=0.2, wob=1.2,
             jitter=(0.08, 0.04, 0.08), streak=0.05, ragged=0.45, mask=m),
    ], paper, "cut1-ivy", clip=clip)
    P.draw(canvas, P.shuffle(iv["leaves"], "ivyleaves"), paper)
    P.draw(canvas, iv["vines"], paper)
    # pines
    log("stage: pines")
    pn = parts["pines"]
    clip = pn["alpha"] * (1 - parts["front"])
    m = (clip > 0.5).astype(np.float32)
    P.paint(canvas, ref, P.orientation(ref, 1.0, 3.0, base=0.0, base_weight=0.8), [
        dict(radius=3.0, spacing=1.0, steps=(2, 4), opacity=0.92, dry=0.4, cthresh=0.15, wob=0.8,
             jitter=(0.05, 0.03, 0.06), streak=0.08, ragged=0.45, mask=m),
    ], paper, "cut1-pines", clip=clip)
    P.draw(canvas, pn["trunk"], paper)
    P.draw(canvas, P.shuffle(pn["pads"], "pads"), paper)
    P.draw(canvas, P.shuffle(pn["needles"], "needles"), paper)
    # front clouds wrap the cliff base again
    log("stage: front clouds")
    fr = parts["front"]
    fm = (fr > 0.5).astype(np.float32)
    P.paint(canvas, cl, ang_c, [
        dict(radius=20, spacing=0.9, steps=(3, 9), opacity=0.95, dry=0.3, cthresh=0.2, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.2, mask=fm),
        dict(radius=8, spacing=1.0, steps=(2, 8), opacity=0.9, dry=0.45, cthresh=0.12, wob=0.3,
             jitter=(0.035, 0.02, 0.05), streak=0.07, pickup=0.15, mask=fm, use_err=True, err_thresh=0.05),
        dict(radius=3.5, spacing=1.1, steps=(2, 7), opacity=0.88, dry=0.5, cthresh=0.10, wob=0.3,
             jitter=(0.03, 0.02, 0.05), streak=0.06, pickup=0.1, mask=fm, use_err=True, err_thresh=0.08),
    ], paper, "cut1-front", clip=fr)
    P.finish(canvas, paper, grain=0.05, tooth=0.025)
    return canvas


def cloud_edges(canvas, paper, parts):
    """Crisp lit rims along the upper left outlines of the lobes, the way a
    background painter sharpens cumulus: a thin bright edge and a softer lit
    band just inside it, only where the outline is really visible."""
    zb = parts["zb"]
    L = zb["lobes"]
    band = zb["band"]
    order = zb["order"]
    Zs = zb["Z"]
    cl = parts["clouds"]
    st = P.Strokes(max_pts=10)
    r = rng("cloud-edges")
    for i in range(len(L)):
        k, cx, cy, rx, ry = L[i]
        k = int(k)
        if rx < 9 or Zs[k] > 22 or r.random() < 0.5:
            continue
        a0 = math.pi * (0.88 + 0.16 * r.random())
        a1 = a0 + math.pi * (0.25 + 0.3 * r.random())
        pts = []
        for q in range(9):
            a = a0 + (a1 - a0) * q / 8
            px = cx + math.cos(a) * rx * 0.97
            py = cy + math.sin(a) * ry * 0.97
            xi, yi = int(px), int(py)
            if not (0 <= xi < AW and 0 <= yi < AH) or band[yi, xi] != k or abs(order[yi, xi] - i) > 0.5:
                if len(pts) >= 3:
                    break
                pts = []
                continue
            pts.append((px, py))
        if len(pts) < 3:
            continue
        mx, my = pts[len(pts) // 2]
        base = cl[min(int(my) + 2, AH - 1), int(mx)]
        lum = float(base.mean())
        if Zs[k] < 4.5:
            # near lobes sit in shadow: give them a cool moonlit rim instead
            nearw = float(np.clip((4.5 - Zs[k]) / 2.5, 0, 1))
            base = base * (1 - nearw) + CL_LIT * 0.62 * nearw
            lum = float(base.mean())
        if lum < 0.16:
            continue
        w = max(0.9, min(6.0, rx * 0.07))
        inner = [(x + (cx - x) * 0.06, y + (cy - y) * 0.10) for (x, y) in pts]
        st.add(inner, w * 1.6, np.clip(base * 1.06, 0, 1), seed=int(r.integers(1 << 30)), opacity=0.45, dry=0.6,
               ragged=0.3, bristles=4, pool=0.0, streak=0.06, taper0=0.3, taper1=0.6, pickup=0.35)
        st.add(pts, w * 0.5, np.clip(base * 0.6 + CL_RIM * 0.5, 0, 1), seed=int(r.integers(1 << 30)),
               opacity=0.65, dry=0.55, ragged=0.35, bristles=3, pool=0.0, streak=0.05, taper0=0.25, taper1=0.6,
               pickup=0.1)
    if len(st):
        P.draw(canvas, st.arrays(), paper)
    return canvas


# ---------------------------------------------------------------- bank (fg)

BANK_DARK = np.array([0.035, 0.040, 0.085], np.float32)
BANK_MID = np.array([0.075, 0.085, 0.150], np.float32)
BLADE_LIT = np.array([0.300, 0.320, 0.480], np.float32)


def bank_outline():
    pts = [(-M - 10, 860), (60, 880), (180, 915), (300, 965), (420, 1030), (520, 1100), (600, 1170),
           (660, 1240), (700, 1300), (720, PH + M + 10), (-M - 10, PH + M + 10)]
    pts = wobble(np.array(pts, np.float64), 6.0, 50.0, "bank", closed=True)
    return pts + M


def paint_bank(paper):
    log("bank")
    outline = bank_outline()
    mask = poly_mask(AH, AW, outline)
    yy, xx = _grid()
    ref = P.mix(BANK_MID, BANK_DARK, P.smoothstep(900 + M, 1250 + M, yy))
    ref = ref * (0.85 + 0.3 * fbm(AH, AW, 40, "bank-v", octaves=3))[..., None]
    canvas = P.new_canvas(AH, AW)
    ang = P.orientation(ref, 2, 6, base=-1.2, base_weight=1.0)
    m = (mask > 0.5).astype(np.float32)
    P.paint(canvas, ref, ang, [
        dict(radius=14, spacing=0.9, steps=(2, 6), opacity=0.96, dry=0.3, cthresh=0.2, wob=0.6,
             jitter=(0.06, 0.03, 0.06), streak=0.08, mask=m),
        dict(radius=5, spacing=1.0, steps=(2, 6), opacity=0.92, dry=0.5, cthresh=0.15, wob=0.6,
             jitter=(0.07, 0.03, 0.06), streak=0.08, mask=m),
    ], paper, "cut1-bank", clip=mask)
    # grass blades along the crest, arching downwind, a few catching the moon
    st = P.Strokes(max_pts=10)
    r = rng("bank-blades")
    crest = outline[(outline[:, 1] < PH + M) & (outline[:, 0] > 0) & (outline[:, 0] < 760 + M)]
    for i in range(1400):
        j = int(r.integers(0, len(crest) - 1))
        bx = crest[j, 0] + (r.random() - 0.5) * 8
        by = crest[j, 1] + r.random() * 60 * r.random()
        ln = 30 + r.random() * 120 * (1 - r.random() * 0.5)
        lean = 0.25 + 0.5 * r.random()
        pts = []
        for q in range(8):
            t = q / 7
            pts.append((bx + ln * lean * t ** 1.6 + (r.random() - 0.5) * 0.6, by - ln * t * (1 - 0.35 * t)))
        lit = r.random() ** 2.5
        c = P.mix(BANK_DARK * 1.3, BLADE_LIT, lit * 0.8)
        st.add(pts, 0.8 + 1.4 * r.random(), c, seed=int(r.integers(1 << 30)), opacity=0.92, dry=0.3,
               ragged=0.15, bristles=2, pool=0.0, streak=0.05, taper0=0.05, taper1=0.85, pickup=0.0)
    P.draw(canvas, P.shuffle(st.arrays(), "blades"), paper)
    P.finish(canvas, paper, grain=0.04)
    return canvas


# ---------------------------------------------------------------- pampas (animated)

_PAMPAS = []


def pampas_set():
    if _PAMPAS:
        return _PAMPAS
    r = rng("pampas-set")
    outline = bank_outline() - M
    crest = outline[(outline[:, 0] > -40) & (outline[:, 0] < 640) & (outline[:, 1] < PH)]
    out = []
    for i in range(24):
        j = int(r.integers(0, len(crest) - 1))
        bx = crest[j, 0] + (r.random() - 0.5) * 30
        by = crest[j, 1] + 20 + r.random() * 90
        hgt = 260 + r.random() * 220
        out.append(F.pampas_stalk((bx, by), hgt, 0.18 + 0.2 * r.random(), ("pampas", i)))
    out.sort(key=lambda s: s["base"][1])
    _PAMPAS.extend(out)
    return _PAMPAS


def _stalk_pose(st, t):
    """Bend of a stalk at time t (s): slow sway plus a gust that travels left
    to right, the plume lagging behind the stem (overlap)."""
    ph = st["phase"]
    fr = st["freq"]
    gust_t = t - 0.0016 * st["base"][0]

    def gust(tt):
        return 0.6 * math.exp(-(((tt % 3.4) - 1.2) ** 2) / 0.35)
    stem = 0.05 * math.sin(2 * math.pi * fr * 0.35 * t + ph) + 0.035 * math.sin(2 * math.pi * fr * 0.9 * t + 1.7 * ph)
    stem += 0.06 * gust(gust_t)
    lag = 0.25
    tl = t - lag
    plume = 0.08 * math.sin(2 * math.pi * fr * 0.35 * tl + ph) + 0.05 * math.sin(2 * math.pi * fr * 0.9 * tl + 1.7 * ph)
    plume += 0.12 * gust(gust_t - lag)
    return stem, plume


PLUME_LIT = np.array([0.860, 0.850, 0.960], np.float32)
PLUME_MID = np.array([0.520, 0.500, 0.680], np.float32)
PLUME_SH = np.array([0.200, 0.180, 0.320], np.float32)
STEM = np.array([0.120, 0.120, 0.200], np.float32)


def pampas_strokes(t, cam):
    """Strokes for every stalk at time t, in screen coordinates."""
    zoom = cam[0]
    sl = P.Strokes(max_pts=12)
    for st in pampas_set():
        stem_b, plume_b = _stalk_pose(st, t)
        bx, by = C.to_screen(cam, st["base"][0], st["base"][1])
        hgt = st["height"] * zoom
        lean = st["lean"] + stem_b
        pts = []
        n = 10
        for q in range(n):
            u = q / (n - 1)
            a = lean * u ** 1.4 + (plume_b * (u - 0.6) * 2.5 if u > 0.6 else 0)
            pts.append((bx + hgt * math.sin(a) * u, by - hgt * math.cos(a) * u))
        sl.add(pts, 1.1 * zoom + 0.3, STEM, seed=st["seed"], opacity=0.95, dry=0.2, ragged=0.1, bristles=2,
               pool=0.0, streak=0.0, taper0=0.05, taper1=0.4, pickup=0.0)
        # plume: a spine from the stem top, drooping downwind
        tx, ty = pts[-1]
        ang = lean + plume_b * 1.6
        L = st["plume_len"] * zoom
        spine = [(tx, ty)]
        for q in range(1, 9):
            u = q / 8
            a = ang + st["droop"] * u ** 1.5 + plume_b * u * 1.2
            px, py = spine[-1]
            spine.append((px + math.sin(a) * L / 8, py - math.cos(a) * L / 8))
        rr = np.random.default_rng(st["seed"])
        Wd = st["plume_w"] * zoom
        for k in range(st["hairs"]):
            u = rr.random() ** 0.8
            i = min(int(u * 8), 7)
            fu = u * 8 - i
            sx = spine[i][0] * (1 - fu) + spine[i + 1][0] * fu
            sy = spine[i][1] * (1 - fu) + spine[i + 1][1] * fu
            prof = math.sin(math.pi * min(u * 1.1, 1.0)) ** 0.7
            side = 1 if rr.random() < 0.5 else -1
            a = ang + st["droop"] * u ** 1.5 + plume_b * (1.0 + u) + side * (0.5 + 0.5 * rr.random()) - 0.35
            hl = Wd * (0.6 + 0.8 * rr.random()) * prof + 2.0
            mx = sx + math.sin(a) * hl * 0.5 + math.sin(ang + 1.2) * hl * 0.25
            my = sy - math.cos(a) * hl * 0.5 + 1.0
            ex = sx + math.sin(a + 0.4 * plume_b) * hl + math.sin(ang + 1.4) * hl * 0.4
            ey = sy - math.cos(a + 0.4 * plume_b) * hl + hl * 0.25
            lit = (0.5 + 0.5 * (-side)) * (0.4 + 0.6 * rr.random()) * (1.0 - 0.3 * u)
            c = P.mix(PLUME_SH, PLUME_MID, min(1.0, 0.4 + lit))
            c = P.mix(c, PLUME_LIT, max(0.0, lit - 0.35) * 1.4)
            sl.add([(sx, sy), (mx, my), (ex, ey)], (0.55 + 0.6 * rr.random()) * max(zoom, 0.6), c,
                   seed=st["seed"] + k, opacity=0.6 + 0.3 * rr.random(), dry=0.5, ragged=0.2, bristles=2,
                   pool=0.0, streak=0.1, taper0=0.1, taper1=0.8, pickup=0.0)
    return sl.arrays()


# ---------------------------------------------------------------- window light (animated)

_WIN = {}


def _windows():
    if not _WIN:
        d = np.load(C.plate_path("cut1_windows", "npz"))
        for k in ("mask", "ds", "h", "angles"):
            _WIN[k] = d[k]
        _WIN["origin"] = tuple(int(v) for v in d["origin"])
    return _WIN


def lantern_x(f):
    """Plate x of the lantern behind the window row at global frame f (on twos)
    and its intensity."""
    f = f - (f % 2)
    t = f / T.FPS
    t0, t1 = T.CUES["window_light_start"], T.CUES["window_light_end"]
    p0, p1 = A["lit_window_path"]
    if t < t0:
        return None, 0.0
    u = min((t - t0) / (t1 - t0), 1.0)
    # a small creature walking: it lingers a moment at the windows
    w = u + 0.035 * math.sin(u * math.pi * 6.0)
    x = (p0[0] - 30) + ((p1[0] + 25) - (p0[0] - 30)) * w
    inten = min(1.0, (t - t0) / 0.5)
    if t > t1:
        inten = max(0.35, 1.0 - (t - t1) * 0.8)
    return x, inten


def window_light(f):
    """Plate space crop of warm light inside the openings."""
    wd = _windows()
    x, inten = lantern_x(f)
    nw, hgt, wid = wd["mask"].shape
    out = np.zeros((hgt, wid, 3), np.float32)
    if x is None or inten <= 0:
        return out
    warm = np.array(T.LIGHTS["night_out"]["warm"], np.float32)
    flick = 0.94 + 0.06 * noise1((f // 2) * 0.9, 77)
    hh = wd["h"]
    g = np.clip((hh - TW.WIN_SPRING + 10) / 48.0, 0.3, 1.0)
    for i, a in enumerate(wd["angles"]):
        R = TW.body_radius(520)
        wx = TW.CX + R * math.sin(math.radians(a))
        d = (wx - x) / 22.0
        k = math.exp(-d * d) * inten * flick * math.cos(math.radians(a)) ** 0.5
        if k < 0.01:
            continue
        side = np.clip(0.65 - wd["ds"][i] * np.sign(x - wx + 1e-3) * 0.05, 0.3, 1.0)
        # the lantern's hot core sits low in the opening, the arch stays deeper orange
        core = np.exp(-((hh - (TW.WIN_SILL - 10)) / 16.0) ** 2) * np.exp(-((wd["ds"][i] + (x - wx) * 0.15) / 5.0) ** 2)
        m = wd["mask"][i] * g * side * k
        out += m[..., None] * (warm * np.array([0.9, 0.62, 0.42], np.float32))
        out += (m * core)[..., None] * np.array([0.35, 0.30, 0.18], np.float32)
    return out


# ---------------------------------------------------------------- render

def render(f):
    sky = C.load_plate("cut1_sky")
    stage = C.load_plate("cut1_stage")
    bank = C.load_plate("cut1_bank")
    cam = T.camera(CUT, f)
    cam_sky = C.parallax_cam(CUT, f, SKY_PARALLAX)
    cam_bank = C.parallax_cam(CUT, f, BANK_PARALLAX)
    bg = C.sample(sky, cam_sky)
    wl = window_light(f)
    glow = None
    if wl.max() > 0:
        # the lantern light is put into the stage plate before sampling so it
        # rides the tower exactly, then the plate is restored
        wd = _windows()
        ox, oy = wd["origin"]
        hgt, wid = wl.shape[:2]
        region = stage[oy:oy + hgt, ox:ox + wid].copy()
        lit = wl.max(-1, keepdims=True)
        stage[oy:oy + hgt, ox:ox + wid, :3] = region[..., :3] * (1 - np.clip(lit, 0, 1)) + wl * region[..., 3:4]
        st_s = C.sample(stage, cam)
        stage[oy:oy + hgt, ox:ox + wid] = region
        # glow: a small warm halo, placed through the same camera
        halo = np.zeros((stage.shape[0], stage.shape[1], 3), np.float32)
        hl = np.stack([P.blur(wl[..., c], 6.0) * 1.2 + P.blur(wl[..., c], 20.0) * 0.9 for c in range(3)], -1)
        halo[oy:oy + hgt, ox:ox + wid] = hl
        glow = C.sample(halo, cam) * 0.5
    else:
        st_s = C.sample(stage, cam)
    C.over(bg, st_s)
    bg[..., 3] = 1.0
    # foreground: the bank and the pampas, drawn on twos
    fg = C.sample(bank, cam_bank)
    tdraw = (f - (f % 2)) / T.FPS
    paper = P.paper_for(T.H, T.W, "screen-paper")
    P.draw(fg, pampas_strokes(tdraw, cam_bank), paper)
    return dict(bg=bg, fg=fg, glow=glow)


# ---------------------------------------------------------------- main

def paint_all(only=None):
    paper = P.paper_for(AH, AW, "cut1")
    if only in (None, "sky"):
        C.save_plate("cut1_sky", paint_sky(paper))
    if only in (None, "stage"):
        parts = stage_parts()
        C.save_plate("cut1_stage", paint_stage(paper, parts))
        tw = parts["tower"]
        np.savez_compressed(C.plate_path("cut1_windows", "npz"),
                            mask=np.stack([w["mask"] for w in tw["windows"]]),
                            ds=np.stack([w["ds"] for w in tw["windows"]]),
                            h=tw["windows"][0]["h"], origin=np.array(tw["windows"][0]["origin"]),
                            angles=np.array([w["angle"] for w in tw["windows"]]))
    if only in (None, "bank"):
        C.save_plate("cut1_bank", paint_bank(paper))


if __name__ == "__main__":
    import time
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ref"
    out = os.path.join(C.REVIEW, "a2-dev")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if cmd == "ref":
        sky = sky_reference()
        parts = stage_parts()
        ref, alpha = compose_stage(parts)
        comp = sky * (1 - alpha[..., None]) + ref * alpha[..., None]
        C.save_review(os.path.join(out, "cut1_ref.png"), comp)
        C.save_review(os.path.join(out, "cut1_ref_tower.png"), comp[150 + M:850 + M, 1250 + M:1950 + M])
    elif cmd == "paint":
        paint_all(sys.argv[2] if len(sys.argv) > 2 else None)
    log(f"done {time.time() - t0:.1f}s")


