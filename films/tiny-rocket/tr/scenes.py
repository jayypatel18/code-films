"""Scene builders: painted backgrounds and cel layers for each shot (working scale 2x)."""

import numpy as np
from scipy.ndimage import gaussian_filter
from . import raster as R
from . import bg as B
from .cel import Cel
from .geom import catmull, wobble, ellipse, stroke_outline, resample
from .panda import Pose, Transform, draw_head, pose_stand, mult_shade, _ink_w, DARK, DARK_D, SHADOW_TINT, CREAM, RUSSET
RUSSET_D_LINE = (0.55, 0.25, 0.12)
from .rocket import draw_rocket, BAMBOO, BAMBOO_D, PAPER, PAPER_D, BRASS

W2, H2 = 3840, 2160


# ----------------------------------------------------------------------------
# shot 2: the hill at dusk, wide
# ----------------------------------------------------------------------------

def hill_crest(x):
    """Hill silhouette height (canvas y) as a function of x, 2x canvas."""
    t = x / W2
    y = 1590 - 230 * np.exp(-((t - 0.40) / 0.30) ** 2) + 170 * (t - 0.5) + 40 * np.sin(t * 9.0) * np.exp(-((t - 0.7) / 0.4) ** 2)
    return y


PALETTES = {
    "dusk": dict(roof_mul=1.0, sky=[(0.0, (0.10, 0.12, 0.34)), (0.25, (0.24, 0.24, 0.48)), (0.45, (0.62, 0.42, 0.52)), (0.58, (0.96, 0.66, 0.42)), (0.66, (0.99, 0.80, 0.52)), (1.0, (0.98, 0.84, 0.60))],
                glow=[((0.35, 0.18, 0.02), 1250, 1320, 1500, 0.9, 1.8), ((0.30, 0.10, 0.0), 1250, 1360, 700, 0.5, 1.4)],
                ridge1=(0.62, 0.46, 0.60), ridge2=(0.48, 0.36, 0.52), valley=(0.40, 0.34, 0.46), river=(0.99, 0.80, 0.58),
                hill=[(0.0, (0.30, 0.36, 0.20)), (0.68, (0.34, 0.40, 0.20)), (0.85, (0.22, 0.26, 0.16)), (1.0, (0.14, 0.16, 0.12))],
                crest=(0.80, 0.58, 0.28), cloud_top=((0.66, 0.46, 0.62), (0.86, 0.60, 0.58)), cloud_lit=((1.0, 0.84, 0.62), (1.0, 0.90, 0.70)),
                foliage=((0.10, 0.16, 0.18), (0.42, 0.40, 0.22), (0.62, 0.52, 0.26), (0.80, 0.62, 0.30)), trunk=((0.22, 0.16, 0.20), (0.52, 0.36, 0.30)),
                wall=((0.62, 0.50, 0.50), (0.36, 0.28, 0.40)), grass_tip=(0.18, 0.10, -0.05)),
    "night": dict(roof_mul=0.2, sky=[(0.0, (0.05, 0.07, 0.22)), (0.3, (0.08, 0.11, 0.30)), (0.55, (0.12, 0.15, 0.36)), (0.66, (0.18, 0.18, 0.38)), (1.0, (0.14, 0.14, 0.28))],
                 glow=[((0.04, 0.05, 0.10), 1250, 1320, 1500, 0.6, 1.8)],
                 ridge1=(0.14, 0.14, 0.28), ridge2=(0.10, 0.10, 0.22), valley=(0.08, 0.09, 0.18), river=(0.22, 0.26, 0.40),
                 hill=[(0.0, (0.08, 0.12, 0.14)), (0.68, (0.10, 0.14, 0.14)), (0.85, (0.06, 0.09, 0.10)), (1.0, (0.04, 0.05, 0.07))],
                 crest=(0.16, 0.20, 0.26), cloud_top=((0.10, 0.10, 0.22), (0.14, 0.13, 0.24)), cloud_lit=((0.22, 0.22, 0.34), (0.26, 0.24, 0.36)),
                 foliage=((0.03, 0.05, 0.08), (0.08, 0.12, 0.14), (0.12, 0.16, 0.18), (0.16, 0.20, 0.24)), trunk=((0.05, 0.05, 0.08), (0.14, 0.12, 0.16)),
                 wall=((0.16, 0.14, 0.22), (0.08, 0.08, 0.14)), grass_tip=(0.02, 0.04, 0.04)),
    "predawn": dict(roof_mul=0.4, sky=[(0.0, (0.06, 0.08, 0.22)), (0.3, (0.12, 0.16, 0.34)), (0.5, (0.26, 0.32, 0.50)), (0.62, (0.52, 0.52, 0.62)), (0.7, (0.70, 0.62, 0.62)), (1.0, (0.60, 0.56, 0.60))],
                   glow=[((0.20, 0.14, 0.10), 2900, 1340, 1300, 0.7, 1.8)],
                   ridge1=(0.30, 0.32, 0.46), ridge2=(0.22, 0.24, 0.38), valley=(0.20, 0.22, 0.32), river=(0.70, 0.66, 0.66),
                   hill=[(0.0, (0.14, 0.20, 0.22)), (0.68, (0.16, 0.22, 0.22)), (0.85, (0.10, 0.14, 0.14)), (1.0, (0.06, 0.08, 0.09))],
                   crest=(0.34, 0.40, 0.44), cloud_top=((0.26, 0.28, 0.42), (0.40, 0.36, 0.46)), cloud_lit=((0.62, 0.52, 0.56), (0.80, 0.64, 0.60)),
                   foliage=((0.05, 0.08, 0.11), (0.12, 0.18, 0.20), (0.18, 0.24, 0.26), (0.26, 0.30, 0.32)), trunk=((0.08, 0.08, 0.12), (0.20, 0.18, 0.22)),
                   wall=((0.30, 0.28, 0.36), (0.14, 0.14, 0.22)), grass_tip=(0.06, 0.08, 0.08)),
    "morning": dict(roof_mul=1.15, sky=[(0.0, (0.36, 0.58, 0.86)), (0.3, (0.56, 0.74, 0.92)), (0.5, (0.78, 0.86, 0.94)), (0.62, (0.94, 0.90, 0.84)), (1.0, (0.96, 0.92, 0.80))],
                   glow=[((0.30, 0.22, 0.08), 2900, 1250, 1500, 0.7, 1.8)],
                   ridge1=(0.60, 0.66, 0.78), ridge2=(0.48, 0.56, 0.68), valley=(0.52, 0.60, 0.50), river=(0.92, 0.94, 0.96),
                   hill=[(0.0, (0.42, 0.58, 0.26)), (0.68, (0.46, 0.62, 0.26)), (0.85, (0.30, 0.44, 0.20)), (1.0, (0.20, 0.30, 0.14))],
                   crest=(0.90, 0.86, 0.50), cloud_top=((0.80, 0.84, 0.92), (0.88, 0.88, 0.92)), cloud_lit=((1.0, 0.98, 0.92), (1.0, 0.98, 0.94)),
                   foliage=((0.14, 0.26, 0.18), (0.46, 0.62, 0.26), (0.66, 0.78, 0.32), (0.86, 0.90, 0.46)), trunk=((0.28, 0.22, 0.20), (0.62, 0.52, 0.40)),
                   wall=((0.88, 0.82, 0.74), (0.52, 0.48, 0.56)), grass_tip=(0.22, 0.20, 0.0)),
}


def compose_hill_ref(seed=2, pal=None):
    """Reference image: layout, values, light. Painted over afterwards."""
    pal = pal or PALETTES["dusk"]
    H, W = H2, W2
    img = B.vgrad(H, W, pal["sky"])
    for (col, gx, gy, rad, st, fo) in pal["glow"]:
        B.soft_light(img, col, gx, gy, rad, st, H, W, fo)
    masks = {}
    # far mountains (two ridges), atmospheric
    g = B.rng(seed)
    ridge1 = [(0, 1250)] + [(x, 1250 - 140 * max(0, np.sin(x / 520.0 + 1.0)) - 60 * max(0, np.sin(x / 190.0)) - 30 * g.uniform(0, 1)) for x in range(0, W + 200, 100)] + [(W, 1250), (W, 1500), (0, 1500)]
    m1 = B.mask_of([ridge1], W, H)
    B.blend(img, pal["ridge1"], m1)
    ridge2 = [(0, 1340)] + [(x, 1340 - 100 * max(0, np.sin(x / 400.0 + 2.5)) - 50 * max(0, np.sin(x / 150.0 + 1)) - 25 * g.uniform(0, 1)) for x in range(0, W + 200, 100)] + [(W, 1340), (W, 1500), (0, 1500)]
    m2 = B.mask_of([ridge2], W, H)
    B.blend(img, pal["ridge2"], m2)
    masks["ridge"] = np.maximum(m1, m2)
    # valley floor (fields and the river)
    valley = [(0, 1380), (W, 1360), (W, 1700), (0, 1700)]
    mv = B.mask_of([valley], W, H)
    B.blend(img, pal["valley"], mv)
    masks["valley"] = mv
    # river winding from far right to lower left, catching the sky
    river = []
    for i, x in enumerate(np.linspace(W + 100, 300, 16)):
        t = i / 15
        yc = 1370 + 300 * t ** 1.4
        wdt = 8 + 90 * t ** 1.3
        river.append((x + 60 * np.sin(t * 5), yc - wdt * 0.5 + 25 * np.sin(t * 7)))
    for i, x in enumerate(np.linspace(300, W + 100, 16)):
        t = 1 - i / 15
        yc = 1370 + 300 * t ** 1.4
        wdt = 8 + 90 * t ** 1.3
        river.append((x + 60 * np.sin(t * 5), yc + wdt * 0.5 + 25 * np.sin(t * 7)))
    mr = B.mask_of([catmull(river, closed=True, n=4)], W, H)
    B.blend(img, pal["river"], mr)
    masks["river"] = mr
    # the hill
    xs = np.arange(-50, W + 100, 40)
    hill = [(x, hill_crest(x)) for x in xs] + [(W + 50, H + 50), (-50, H + 50)]
    mh = B.mask_of([hill], W, H)
    hill_col = B.vgrad(H, W, pal["hill"])
    img[:] = img * (1 - mh[..., None]) + hill_col * mh[..., None]
    masks["hill"] = mh
    # warm rim along the crest where the dusk light grazes the grass
    crest_band = np.clip((mh - np.roll(mh, 70, axis=0)), 0, 1)
    B.blend(img, pal["crest"], gaussian_filter(crest_band, 18) * 0.9)
    return img, masks


def paint_hill(seed=2, stars=True, windows=True, palette="dusk"):
    H, W = H2, W2
    pal = PALETTES[palette]
    ref, masks = compose_hill_ref(seed, pal)
    img = ref.copy()
    sky = 1 - np.maximum.reduce([masks["ridge"], masks["valley"], masks["hill"]])
    # sky: long horizontal strokes, big to small
    B.paint_region(img, ref, sky, 60, 260, 70, 0.75, seed + 1, angle=0.0, angle_jit=0.06, val=0.03, hue=0.02, hard=0.3, region=(0, 0, W, 1560))
    B.paint_region(img, ref, sky, 26, 110, 30, 0.6, seed + 2, angle=0.0, angle_jit=0.08, val=0.025, hue=0.02, hard=0.4, region=(0, 0, W, 1560))
    # dusk clouds: real masses of overlapping puffs, lit from below, soft edges
    cloud_masses(img, seed + 3, pal)
    # ridges: diagonal dabs following the slopes
    B.paint_region(img, ref, masks["ridge"], 30, 130, 38, 0.7, seed + 4, angle=0.0, angle_jit=0.12, val=0.035, hue=0.03, hard=0.4)
    # valley: soft horizontal dabs, hazy
    B.paint_region(img, ref, masks["valley"], 18, 60, 18, 0.65, seed + 5, angle=0.0, angle_jit=0.1, val=0.05, hue=0.04, hard=0.4)
    B.paint_region(img, ref, masks["river"], 10, 34, 9, 0.8, seed + 6, angle=0.1, angle_jit=0.12, val=0.05, hue=0.03, hard=0.5, min_cov=0.3)
    # village: roofs along the river, small warm windows
    village(img, seed + 7, windows=windows, pal=pal)
    # hill: dabs oriented along the slope, big then small, then grass blades
    B.paint_region(img, ref, masks["hill"], 34, 90, 30, 0.65, seed + 8, angle=0.15, angle_jit=0.2, val=0.04, hue=0.03, hard=0.4)
    B.paint_region(img, ref, masks["hill"], 16, 48, 16, 0.7, seed + 9, angle=0.2, angle_jit=0.35, val=0.06, hue=0.04, hard=0.6)
    camphor(img, seed + 11, pal)
    grass_on_hill(img, ref, masks["hill"], seed + 10, pal)
    # first stars in the dark part of the sky, fading toward the glow
    if stars:
        B.stars(img, 260, seed + 12, (0, 0, W, 900), rmin=1.2, rmax=3.0, strength=0.8,
                fade=lambda xs, ys: np.clip(1 - ys / 900, 0, 1) ** 1.5)
    if palette == "morning":
        # morning mist lying in the valley, thinning upward
        mist = np.clip(1 - np.abs(np.arange(H) - 1560) / 220.0, 0, 1)[:, None] ** 1.5 * 0.55
        mist = mist * (1 - masks["hill"])
        mist = gaussian_filter(mist, 25) * (1 + 0.3 * (R.noise_field(H, W, 160.0, seed + 44, 3) - 0.5))
        img[:] = img * (1 - mist[..., None]) + np.array([0.96, 0.94, 0.92], np.float32) * mist[..., None]
    # paper grain
    img *= B.paper(H, W, seed + 13, 0.06)[..., None]
    return np.clip(img, 0, 1).astype(np.float32), masks


def cloud_masses(img, seed, pal):
    g = B.rng(seed)
    specs = []
    for k in range(7):
        cx = g.uniform(100, W2 - 100)
        cy = g.uniform(560, 1250)
        far = (cy - 560) / 700  # lower clouds are nearer the horizon: flatter, warmer
        ln = g.uniform(500, 1300) * (0.7 + 0.6 * far)
        hgt = g.uniform(70, 160) * (1.1 - 0.6 * far)
        specs.append((cx, cy, ln, hgt, far))
    specs.sort(key=lambda c: c[1])
    for k, (cx, cy, ln, hgt, far) in enumerate(specs):
        npuff = int(ln / (hgt * 0.9)) + 2
        xs, ys, rs = [], [], []
        for i in range(npuff):
            t = (i + 0.5) / npuff
            env = np.sin(np.pi * t) ** 0.5
            xs.append(cx - ln / 2 + t * ln + g.normal(0, hgt * 0.2))
            ys.append(cy - hgt * 0.35 * env * g.uniform(0.3, 1.2))
            rs.append(hgt * env * g.uniform(0.6, 1.0))
        top_col = np.array(pal["cloud_top"][0]) * (1 - far) + np.array(pal["cloud_top"][1]) * far
        lit_col = np.array(pal["cloud_lit"][0]) * (1 - far) + np.array(pal["cloud_lit"][1]) * far
        allx, ally, cols, L, Wd, al = [], [], [], [], [], []
        for (px, py, r) in zip(xs, ys, rs):
            n = int(r * r / 120) + 10
            a = g.uniform(0, 2 * np.pi, n)
            rr = r * np.sqrt(g.uniform(0, 1, n))
            dx = rr * np.cos(a)
            dy = rr * np.sin(a) * 0.65
            lit = np.clip(dy / (r + 1e-6) * 0.5 + 0.55, 0, 1) ** 1.5
            c = top_col[None] * (1 - lit[:, None]) + lit_col[None] * lit[:, None]
            allx.append(px + dx)
            ally.append(py + dy)
            cols.append(B.jitter_colors(c, g, 0.03, 0.02))
            L.append(g.uniform(0.5, 0.9, n) * r * 0.6)
            Wd.append(g.uniform(0.3, 0.6, n) * r * 0.6)
            al.append(g.uniform(0.35, 0.7, n))
        n = sum(len(x) for x in allx)
        d = B.make_dabs(np.concatenate(allx), np.concatenate(ally), g.normal(0, 0.12, n), np.concatenate(L), np.concatenate(Wd),
                        np.concatenate(cols), np.concatenate(al), 0.15)
        R.paint_dabs(img, d, seed + 10 + k)
        # a few crisp lit dabs along the underside
        m = npuff * 6
        ux = cx - ln / 2 + g.uniform(0.05, 0.95, m) * ln
        uy = cy + hgt * 0.18 + g.normal(0, hgt * 0.06, m)
        d2 = B.make_dabs(ux, uy, g.normal(0, 0.08, m), g.uniform(30, 70, m), g.uniform(10, 22, m),
                         B.jitter_colors(np.tile(lit_col, (m, 1)), g, 0.03, 0.02), g.uniform(0.5, 0.8, m), 0.5)
        R.paint_dabs(img, d2, seed + 30 + k)


def village(img, seed, windows=True, pal=None):
    """Houses clustered along the river: tile and thatch roofs, lit and shaded walls, warm windows."""
    pal = pal or PALETTES["dusk"]
    g = B.rng(seed)
    pts = []
    # hamlets: a few cluster centres on the river banks, houses spread round each
    centres = []
    for i in range(7):
        t = g.uniform(0.12, 0.98)
        cx = 300 + (W2 - 300) * t
        cy = 1370 + 300 * (1 - t) ** 1.4 + g.uniform(-40, 60) * (0.4 + 0.6 * (1 - t))
        centres.append((cx, cy, t, g.integers(5, 14)))
    for (cx, cy, t, cnt) in centres:
        for j in range(cnt):
            x = cx + g.normal(0, 90 + 140 * (1 - t))
            y = cy + g.normal(0, 30 + 60 * (1 - t))
            s = 14 + 60 * (1 - t) ** 1.3 * g.uniform(0.8, 1.25)
            pts.append((x, y, s))
    pts.sort(key=lambda p: p[1])
    glows = []
    for (x, y, s) in pts:
        if y > hill_crest(x) - 10:
            continue
        thatch = g.uniform() < 0.3
        hw = s * g.uniform(0.85, 1.25)
        wh = s * g.uniform(0.5, 0.8)
        rm = pal["roof_mul"]
        wall_l = np.array([(x - hw, y), (x, y + wh * 0.15), (x, y + wh), (x - hw, y + wh * 0.9)])
        wall_r = np.array([(x, y + wh * 0.15), (x + hw, y), (x + hw, y + wh * 0.9), (x, y + wh)])
        R.fill_poly(img, wobble(wall_l, s * 0.03, seed + int(x), closed=True), pal["wall"][0], 0.95)
        R.fill_poly(img, wobble(wall_r, s * 0.03, seed + int(y), closed=True), pal["wall"][1], 0.95)
        if thatch:
            rh = s * 0.8
            roof = catmull([(x - hw * 1.2, y + wh * 0.05), (x - hw * 0.85, y - rh * 0.8), (x, y - rh), (x + hw * 0.85, y - rh * 0.8), (x + hw * 1.2, y + wh * 0.05)], closed=True, n=5)
            R.fill_poly(img, wobble(roof, s * 0.04, seed + int(x * 3), closed=True), (0.50 * rm, 0.36 * rm, 0.24 * rm), 1.0)
            R.fill_poly(img, wobble(catmull([(x - hw * 1.15, y), (x - hw * 0.7, y - rh * 0.62), (x, y - rh * 0.85), (x - hw * 0.05, y)], closed=True, n=4), s * 0.04, seed + 9, closed=True), (0.66 * rm, 0.50 * rm, 0.30 * rm), 0.9)
            if s > 26:
                for k in range(4):
                    xa = x - hw * 0.9 + k * hw * 0.5
                    R.fill_poly(img, stroke_outline(np.array([(xa, y - rh * 0.1 - k * 4), (xa + hw * 0.3, y - rh * 0.55)]), np.array([s * 0.035, s * 0.03])), (0.40 * rm, 0.28 * rm, 0.18 * rm), 0.6)
        else:
            rh = s * 0.75
            ridge = (x, y - rh)
            roof_l = np.array([(x - hw * 1.2, y + wh * 0.02), ridge, (x + hw * 0.12, y - rh * 0.95), (x, y + wh * 0.1)])
            roof_r = np.array([(x, y + wh * 0.1), (x + hw * 0.12, y - rh * 0.95), (x + hw * 1.2, y + wh * 0.02)])
            R.fill_poly(img, wobble(roof_l, s * 0.03, seed + int(x * 7), closed=True), (0.44 * rm, 0.34 * rm, 0.46 * rm), 1.0)
            R.fill_poly(img, wobble(roof_r, s * 0.03, seed + int(y * 7), closed=True), (0.24 * rm, 0.18 * rm, 0.28 * rm), 1.0)
            if s > 24:
                for k in range(1, 5):
                    tt = k / 5
                    a = np.array([x - hw * 1.2 + tt * hw * 1.2, y + wh * 0.02 - tt * (rh + wh * 0.02)])
                    b = np.array([x + tt * hw * 0.12, y + wh * 0.1 - tt * (rh * 0.95 + wh * 0.1)])
                    R.fill_poly(img, stroke_outline(np.array([a, b]), np.array([s * 0.035, s * 0.035])), (0.32 * rm, 0.24 * rm, 0.34 * rm), 0.7)
            # ridge line
            R.fill_poly(img, stroke_outline(np.array([(x - hw * 1.2, y + wh * 0.02), ridge]), np.array([s * 0.05, s * 0.05])), (0.52 * rm, 0.42 * rm, 0.52 * rm), 0.8)
        R.fill_poly(img, np.array([(x - hw, y), (x + hw, y), (x + hw, y + wh * 0.18), (x - hw, y + wh * 0.18)]), (0.25 * rm, 0.18 * rm, 0.28 * rm), 0.5)
        if g.uniform() < 0.8:
            wx = x + g.uniform(-0.55, 0.55) * hw
            wy = y + wh * 0.55
            ws = s * 0.14
            glows.append((wx, wy, s * 1.1, 1.0, 0.72, 0.30, 0.9 * g.uniform(0.5, 1.0), ws, g.uniform(0, 1)))
    if windows and glows:
        village_lights(img, glows, 1.0)
    return glows


def village_lights(img, glows, t_on):
    """Draw the windows whose turn-on time (last column, 0..1) is below t_on."""
    on = [gl for gl in glows if gl[8] <= t_on]
    for (wx, wy, rad, r, g_, b, st, ws, _) in on:
        R.fill_poly(img, [(wx - ws, wy - ws), (wx + ws, wy - ws), (wx + ws, wy + ws), (wx - ws, wy + ws)], (1.0, 0.88, 0.55), 1.0)
    if on:
        R.add_glows(img, np.array([gl[:7] for gl in on], np.float64))


def village_glow_list(seed):
    """The same window list the painter made, without painting (deterministic)."""
    scratch = np.zeros((4, 4, 3), np.float32)
    return village(scratch, seed, windows=False)


def grass_on_hill(img, ref, mh, seed, pal=None):
    pal = pal or PALETTES["dusk"]
    g = B.rng(seed)
    H, W = img.shape[:2]
    # density and size grow toward the camera
    for band, (y0, y1, n, hmin, hmax, wd) in enumerate([(1200, 1600, 9000, 10, 26, 3), (1550, 1900, 7000, 24, 60, 5), (1850, H, 4500, 60, 150, 8)]):
        xs = g.uniform(-20, W + 20, n)
        ys = g.uniform(y0, y1, n)
        keep = B.sample(mh, xs, ys) > 0.5
        xs, ys = xs[keep], ys[keep]
        hs = g.uniform(hmin, hmax, xs.size)
        base = B.sample(ref, xs, ys)
        # blades: darker than the ground with warm tips near the crest
        crest = np.clip(1 - (ys - hill_crest(xs)) / 260, 0, 1)
        cols = base * g.uniform(0.75, 1.05, (xs.size, 1)) + np.array(pal["grass_tip"]) * crest[:, None] * g.uniform(0, 1, (xs.size, 1))
        cols = np.clip(cols + g.normal(0, 0.03, (xs.size, 3)), 0, 1)
        B.grass_blades(img, xs, ys, hs, cols, seed + band, lean=0.18, width=wd, alpha=0.85, curve=0.4)
    # a few wildflowers (tiny warm dots) near the crest
    n = 180
    xs = g.uniform(200, W - 200, n)
    ys = hill_crest(xs) + g.uniform(20, 240, n)
    cols = np.array([(0.98, 0.86, 0.55), (0.95, 0.70, 0.75), (0.98, 0.95, 0.90)])[g.integers(0, 3, n)]
    d = B.make_dabs(xs, ys, g.uniform(0, 3.14, n), g.uniform(7, 12, n), g.uniform(6, 10, n), cols, np.full(n, 0.9), 1.0)
    R.paint_dabs(img, d, seed + 99)


def camphor(img, seed, pal=None):
    pal = pal or PALETTES["dusk"]
    """Old camphor tree leaning in from the left over the hilltop."""
    g = B.rng(seed)
    base = (560, hill_crest(560) + 110)
    trunk = [base, (520, 1250), (560, 1050), (700, 880), (900, 760)]
    tp = catmull(trunk, n=10)
    widths = np.linspace(130, 46, len(tp))
    poly = stroke_outline(tp, widths)
    R.fill_poly(img, wobble(poly, 4, seed, closed=True), tuple(np.array(pal["trunk"][0]) * 0.5 + np.array(pal["trunk"][1]) * 0.5), 1.0)
    # bark strokes along the trunk
    n = 600
    t = g.uniform(0, 1, n)
    idx = (t * (len(tp) - 1)).astype(int)
    d = np.gradient(tp, axis=0)
    ang = np.arctan2(d[idx, 1], d[idx, 0])
    off = g.uniform(-1, 1, n)
    nx, ny = -np.sin(ang), np.cos(ang)
    xs = tp[idx, 0] + nx * off * widths[idx] * 0.48
    ys = tp[idx, 1] + ny * off * widths[idx] * 0.48
    lit = np.clip(-off * 0.5 + 0.5, 0, 1)  # left side catches the dusk glow
    cols = np.array(pal["trunk"][0])[None] * (1 - lit[:, None]) + np.array(pal["trunk"][1])[None] * lit[:, None]
    cols = B.jitter_colors(cols, g, 0.05, 0.03)
    dd = B.make_dabs(xs, ys, ang + g.normal(0, 0.1, n), g.uniform(20, 60, n), g.uniform(5, 12, n), cols, g.uniform(0.5, 0.9, n), 0.7)
    R.paint_dabs(img, dd, seed + 1)
    # branches
    branches = [[(720, 860), (640, 700), (520, 600)], [(800, 800), (820, 640), (760, 520)], [(880, 770), (1050, 700), (1180, 720)],
                [(900, 760), (980, 620), (1060, 560)], [(600, 1000), (420, 940), (300, 900)]]
    for k, br in enumerate(branches):
        bp = catmull(br, n=8)
        w = np.linspace(34, 12, len(bp))
        R.fill_poly(img, wobble(stroke_outline(bp, w), 2, seed + k, closed=True), tuple(np.array(pal["trunk"][0]) * 0.6 + np.array(pal["trunk"][1]) * 0.4), 1.0)
    # foliage clumps: dark teal masses, warm lit tops
    centers = [(480, 560), (700, 470), (930, 520), (1130, 640), (1000, 700), (640, 700), (330, 820), (840, 380), (560, 420), (1240, 760)]
    radii = [150, 170, 160, 130, 120, 110, 110, 120, 110, 90]
    fo = pal["foliage"]
    B.foliage_clumps(img, None, centers, radii, seed + 2, fo[0], fo[1], light_dir=(-0.4, -0.9), leaf=26, density=1.3)
    B.foliage_clumps(img, None, centers, [r * 0.8 for r in radii], seed + 3, tuple((np.array(fo[0]) + np.array(fo[1])) * 0.5), fo[2], light_dir=(-0.4, -0.9), leaf=16, density=1.0, alpha=0.8)
    B.foliage_clumps(img, None, [(c[0] - 20, c[1] - r * 0.5) for c, r in zip(centers, radii)], [r * 0.5 for r in radii], seed + 4, tuple((np.array(fo[0]) + np.array(fo[2])) * 0.5), fo[3], light_dir=(-0.4, -0.9), leaf=11, density=0.8, alpha=0.9)


def shot2_props(light=(-0.7, -0.6), seed=2):
    """Rocket and star map stick as one static cel."""
    cel = Cel(W2, H2)
    T = Transform(0, 0, 1.0)
    rx = 2330
    draw_rocket(cel, T, rx, hill_crest(rx) + 20, 640.0, light, seed=seed)
    sx = 1560
    sy = hill_crest(sx) + 40
    stick = stroke_outline(np.array([(sx, sy), (sx + 6, sy - 150)]), np.array([8.0, 6.0]))
    cel.fill(stick, BAMBOO_D)
    cel.ink_closed(stick, 2.0, light=light, seed=seed + 1)
    card = np.array([(sx - 60, sy - 150), (sx + 62, sy - 158), (sx + 66, sy - 70), (sx - 56, sy - 64)])
    cm = cel.fill(wobble(card, 2, seed + 2, closed=True), PAPER)
    mult_shade(cel, cm, SHADOW_TINT, -light[0] * 14, -light[1] * 14, 0.7)
    cel.ink_closed(card, 2.2, light=light, seed=seed + 3)
    g = B.rng(seed + 4)
    for k in range(9):
        px, py = g.uniform(sx - 45, sx + 48), g.uniform(sy - 140, sy - 80)
        cel.fill(ellipse(px, py, 2.5, 2.5, n=8), (0.25, 0.30, 0.50))
    for k in range(4):
        cel.ink([(g.uniform(sx - 45, sx + 48), g.uniform(sy - 140, sy - 80)), (g.uniform(sx - 45, sx + 48), g.uniform(sy - 140, sy - 80))], 1.4, color=(0.35, 0.40, 0.60), seed=seed + 10 + k, a=0.7)
    return cel


def shot2_panda(light=(-0.7, -0.6), seed=2, px=1840, yaw=0.65, pitch=-0.45, stride=0.0, bob=0.0, flip=False,
                look=(0.3, -0.3), mouth="smile", ear_l=0.0, ear_r=0.0, eye_open=1.0, head_up=0.3, tail_swish=0.0):
    cel = Cel(W2, H2)
    Tp = Transform(px, hill_crest(px) - 100 - bob, 62.0, flip=flip)
    pose = Pose(yaw=yaw, pitch=pitch, mouth=mouth, seed=seed + 20, look=look, ear_l=ear_l, ear_r=ear_r, eye_open=eye_open)
    pose_stand(cel, Tp, pose, light, stride=stride, head_up=head_up)
    return cel


def shot2_cels(light=(-0.7, -0.6), seed=2):
    cel = shot2_props(light, seed)
    p = shot2_panda(light, seed)
    p.over(cel.img, 0, 0)
    cel.alpha = np.maximum(cel.alpha, p.alpha)
    return cel


# ----------------------------------------------------------------------------
# shot 1: the last knot, close
# ----------------------------------------------------------------------------

def paint_closeup_bg(seed=1):
    """Out of focus dusk grass and sky behind the panda."""
    H, W = H2, W2
    ref = B.vgrad(H, W, [(0.0, (0.95, 0.70, 0.45)), (0.22, (0.98, 0.78, 0.50)), (0.36, (0.70, 0.62, 0.34)),
                         (0.6, (0.36, 0.38, 0.18)), (1.0, (0.18, 0.20, 0.12))])
    B.soft_light(ref, (0.30, 0.16, 0.02), 3100, 500, 1500, 0.9, H, W, 1.6)
    img = ref.copy()
    B.paint_region(img, ref, None, 70, 300, 90, 0.8, seed + 1, angle=0.0, angle_jit=0.1, val=0.04, hue=0.03, hard=0.3)
    B.paint_region(img, ref, None, 30, 120, 40, 0.6, seed + 2, angle=0.1, angle_jit=0.3, val=0.05, hue=0.03, hard=0.4)
    g = B.rng(seed + 3)
    # grass blades across the lower two thirds, lit from behind
    n = 2600
    xs = g.uniform(-50, W + 50, n)
    ys = g.uniform(700, H + 100, n)
    hs = g.uniform(120, 520, n) * (0.5 + ys / H)
    t = (ys - 700) / (H - 700)
    cols = np.array([0.52, 0.52, 0.22])[None] * (1 - t[:, None]) + np.array([0.14, 0.18, 0.10])[None] * t[:, None]
    cols = np.clip(cols + g.normal(0, 0.04, (n, 3)), 0, 1)
    B.grass_blades(img, xs, ys, hs, cols, seed + 4, lean=0.25, width=14, alpha=0.8, curve=0.5)
    # depth of field: soft everything, then bokeh discs of sun through the grass
    img = np.stack([gaussian_filter(img[..., c], 22) for c in range(3)], axis=2)
    # nearer blades, less out of focus, dark against the glow
    n2 = 500
    xs2 = g.uniform(-50, W + 50, n2)
    ys2 = g.uniform(1500, H + 200, n2)
    hs2 = g.uniform(350, 900, n2)
    cols2 = np.clip(np.array([0.16, 0.19, 0.11])[None] + g.normal(0, 0.03, (n2, 3)), 0, 1)
    layer = img.copy()
    B.grass_blades(layer, xs2, ys2, hs2, cols2, seed + 6, lean=0.3, width=22, alpha=0.9, curve=0.6)
    layer = np.stack([gaussian_filter(layer[..., c], 9) for c in range(3)], axis=2)
    img = layer
    nb = 40
    bx = g.uniform(1800, W, nb)
    by = g.uniform(200, 1100, nb)
    gl = np.empty((nb, 7))
    gl[:, 0] = bx
    gl[:, 1] = by
    gl[:, 2] = g.uniform(40, 110, nb)
    gl[:, 3:6] = np.array([1.0, 0.78, 0.45])
    gl[:, 6] = g.uniform(0.15, 0.4, nb)
    R.add_glows(img, gl)
    img *= B.paper(H, W, seed + 5, 0.04)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def shot1_cels(light=(0.75, -0.55), seed=1, pull=1.0, blink=1.0, ear_flick=0.0, tail_t=0.0, head_dy=0.0,
               pitch=0.45, yaw=-0.45, look=(-0.35, 0.45), mouth="teeth", ear_l=None, ear_r=None, smear=0.0, string_in_mouth=True, tail_swish=0.0, brow=0.4):
    """Panda close: head large at right, paws and the knot large at the left, string end in its teeth."""
    from .panda import part, body_shape, draw_leg, ruff, cast, head_poly_u
    from .cel import fluffy
    cel = Cel(W2, H2)
    Tp = Transform(2420, 1000 + head_dy, 640.0)
    pcx, pcy = 0.0, 0.0
    knot = np.array([1280.0, 1560.0])
    # bamboo leg of the rocket, leaning, the knot tied round it
    leg = np.array([(1060, 2300), (1440, 1000)])
    d = (leg[1] - leg[0]) / np.hypot(*(leg[1] - leg[0]))
    nrm = np.array([-d[1], d[0]])
    lp = stroke_outline(leg, np.array([170.0, 140.0]))
    lm = cel.fill(wobble(lp, 3, seed, closed=True), BAMBOO)
    mult_shade(cel, lm, SHADOW_TINT, -light[0] * 44, -light[1] * 44, 0.8)
    cel.ink_closed(lp, 5.0, light=light, seed=seed + 1)
    for t in (0.2, 0.78):
        nx, ny = leg[0] + (leg[1] - leg[0]) * t
        cel.ink([(nx - nrm[0] * 80, ny - nrm[1] * 80), (nx + nrm[0] * 80, ny + nrm[1] * 80)], 4.5, seed=seed + 2)
        cel.ink([(nx - nrm[0] * 74 + d[0] * 16, ny - nrm[1] * 74 + d[1] * 16), (nx + nrm[0] * 74 + d[0] * 16, ny + nrm[1] * 74 + d[1] * 16)], 2.2, color=BAMBOO_D, seed=seed + 3)
    # body: shoulders and chest, fluffy, with the belly dark lower down
    chest = [(pcx - 1.35, pcy + 0.5), (pcx + 1.2, pcy + 0.6), (pcx + 1.7, pcy + 1.2), (pcx + 1.85, pcy + 2.2), (pcx - 1.9, pcy + 2.2), (pcx - 1.8, pcy + 1.1)]
    cm = part(cel, Tp, body_shape(chest, seed + 7, amp=0.1), RUSSET, SHADOW_TINT, light, 0.034, seed + 7)
    # dark chest fur with a tufted upper edge
    cel.paint_in(cm, Tp(fluffy([(pcx - 0.75, pcy + 1.15), (pcx - 0.3, pcy + 1.05), (pcx + 0.3, pcy + 1.1), (pcx + 0.75, pcy + 1.25), (pcx + 1.0, pcy + 1.7), (pcx + 0.95, pcy + 2.2), (pcx - 1.0, pcy + 2.2), (pcx - 1.05, pcy + 1.6)], seed + 8, amp=0.09, step=0.05, n=6, spikes=1.4)), DARK)
    # shoulder fur direction strokes
    for k in range(5):
        ax = pcx + 1.0 + 0.12 * k
        cel.ink(Tp(catmull([(ax, pcy + 0.75 + 0.08 * k), (ax + 0.1, pcy + 0.95 + 0.08 * k), (ax + 0.14, pcy + 1.15 + 0.08 * k)], n=6)), 4.0, color=RUSSET_D_LINE, seed=seed + 60 + k, a=0.5, taper_in=0.1, taper_out=0.5)
    cast(cel, cm, Tp(head_poly_u(pcx, pcy, Pose(yaw=-0.45), seed + 20)), -Tp.px(0.1), Tp.px(0.3), 0.8)
    # far forearm: reaches round the bamboo and holds the string's standing part
    far = draw_leg(cel, Tp, [(pcx - 0.9, pcy + 1.15), (pcx - 1.55, pcy + 1.05), ((knot[0] - 150 - 2420) / 640.0, (knot[1] + 60 - 1000 - head_dy) / 640.0)], 0.40, light, seed + 8, color=DARK_D)
    # the knot: four fat loops round the bamboo and an overhand twist in front
    STR = (0.93, 0.86, 0.68)
    STR_D = (0.55, 0.45, 0.30)
    for k in range(3):
        c = knot + d * (k * 46 - 50)
        loop = ellipse(c[0], c[1], 96, 40, n=26, rot=np.arctan2(nrm[1], nrm[0]) + 0.1)
        loop = wobble(loop, 2, seed + k, closed=True)
        cel.ink(np.concatenate([loop, loop[:1]]), 24.0, color=STR, seed=seed + 12 + k, var=0.1, taper_in=0.02, taper_out=0.02)
        cel.ink(np.concatenate([loop, loop[:1]]) + np.array([0, 8]), 6.0, color=STR_D, seed=seed + 32 + k, var=0.3, taper_in=0.02, taper_out=0.02, a=0.5)
    # the overhand knot: two bights crossing in front of the loops
    kc = knot + np.array([80, 30])
    b1 = catmull([kc + (-70, -10), kc + (-30, -50), kc + (30, -40), kc + (60, 10), kc + (20, 50), kc + (-40, 40)], closed=True, n=6)
    b2 = catmull([kc + (-20, -70), kc + (40, -60), kc + (70, -10), kc + (40, 30), kc + (-10, 10), kc + (-40, -30)], closed=True, n=6)
    for bight in (b1,):
        cel.ink(np.concatenate([bight, bight[:1]]), 24.0, color=STR, seed=seed + 17, var=0.1, taper_in=0.02, taper_out=0.02)
        cel.ink(np.concatenate([bight, bight[:1]]) + np.array([0, 8]), 6.0, color=STR_D, seed=seed + 18, var=0.3, taper_in=0.02, taper_out=0.02, a=0.5)
    # loose end hanging from the knot
    tail = catmull([kc + (10, 45), kc + (-30, 170), kc + (-120, 280)], n=8)
    cel.ink(tail, 22.0, color=STR, seed=seed + 16, taper_in=0.02, taper_out=0.3)
    cel.ink(tail + np.array([0, 6]), 4.5, color=STR_D, seed=seed + 19, taper_in=0.02, taper_out=0.3, a=0.5)
    # near forearm: grips the working end between the knot and the mouth, pulling
    grip = kc + np.array([250, -60])
    gx, gy = (grip[0] - 2420) / 640.0, (grip[1] - 1000 - head_dy) / 640.0
    near = draw_leg(cel, Tp, [(pcx - 0.35, pcy + 1.35), (pcx - 0.9, pcy + 1.5), (gx + 0.12, gy + 0.1)], 0.40, light, seed + 9)
    # head and ruff
    pose = Pose(yaw=yaw, pitch=pitch, mouth=mouth, seed=seed + 20, look=look, eye_open=blink,
                ear_l=(-0.2 * ear_flick if ear_l is None else ear_l), ear_r=(0.25 * ear_flick if ear_r is None else ear_r), brow=brow)
    # tail tip swishing into frame at the right
    from .panda import draw_tail
    draw_tail(cel, Tp, [(pcx + 1.5, pcy + 2.1), (pcx + 2.1, pcy + 1.9 - 0.25 * tail_swish), (pcx + 2.5, pcy + 1.4 - 0.5 * tail_swish), (pcx + 2.6, pcy + 0.8 - 0.6 * tail_swish)],
              0.55, light, seed + 200, rings=5, taper=0.4)
    ruff(cel, Tp, pcx, pcy, pose.yaw, light, seed + 20, rx=1.25, ry=0.5, dy=0.7)
    draw_head(cel, Tp, pcx, pcy, pose, light)
    # the string: taut from the knot through the grip to the teeth, and a short end past the mouth
    mouth = Tp([(pcx - 0.46, pcy + 0.43)])[0]
    if string_in_mouth:
        sag = 90 * (1 - pull)
        sp = catmull([kc + (30, -10), grip + (-40, 20), (grip + mouth) / 2 + np.array([0, sag]), mouth], n=14)
        if smear > 0:
            for k in range(3):
                cel.ink(sp + np.array([0, (k + 1) * 14 * smear]), 10.0, color=STR, seed=seed + 40 + k, var=0.1, taper_in=0.02, taper_out=0.02, a=0.35)
        cel.ink(sp, 22.0, color=STR, seed=seed + 10, var=0.1, taper_in=0.02, taper_out=0.02)
        cel.ink(sp + np.array([0, 8]), 6.0, color=STR_D, seed=seed + 11, var=0.3, taper_in=0.02, taper_out=0.02, a=0.55)
        end = catmull([mouth, mouth + (70, 30), mouth + (120, 90)], n=8)
        cel.ink(end, 20.0, color=STR, seed=seed + 13, taper_in=0.02, taper_out=0.4)
    else:
        # released: the end hangs from the grip
        sp = catmull([kc + (30, -10), grip + (-40, 20), grip + (60, 120), grip + (40, 260)], n=14)
        cel.ink(sp, 16.0, color=STR, seed=seed + 10, var=0.12, taper_in=0.02, taper_out=0.3)
    return cel


RUSSET_D_LINE = (0.55, 0.25, 0.12)


def rim_light(frame, cel, color, dx, dy, strength=0.8, width=14, ink_w=7, seed=3):
    """Rim light inside the ink line: thin, only on the side facing (dx, dy), broken in places."""
    from scipy.ndimage import minimum_filter
    a = cel.alpha
    core = minimum_filter(a, size=int(ink_w * 2 + 1))  # step inside past the outline ink
    inner = R.shift_full(core, -dx * width, -dy * width)
    rim = np.clip(core - inner, 0, 1)
    # break the rim with noise along the edge
    H, W = a.shape
    nz = R.noise_field(H // 8, W // 8, 6.0, seed, 2)
    nz = np.kron(nz, np.ones((8, 8), np.float32))[:H, :W]
    rim = rim * np.clip((nz - 0.42) * 3.0, 0, 1)
    rim = gaussian_filter(rim, 1.2)
    frame[:] = frame + np.asarray(color, np.float32) * (rim * strength)[..., None]


# ----------------------------------------------------------------------------
# shot 3: night
# ----------------------------------------------------------------------------

def star_field(seed=3, n=1800, band=2600):
    """Fixed star catalogue in polar coords round the pole, plus a milky way band."""
    g = B.rng(seed)
    r = np.sqrt(g.uniform(0, 1, n)) * 4200
    a = g.uniform(0, 2 * np.pi, n)
    size = g.uniform(1.0, 3.2, n) ** 1.4
    bright = g.uniform(0.25, 1.0, n) ** 1.5
    # milky way: stars concentrated along a diagonal band in sky coords
    u = g.uniform(-2600, 2600, band)
    v = g.normal(0, 260, band)
    ang = -0.55
    bx = 2000 + u * np.cos(ang) - v * np.sin(ang)
    by = 700 + u * np.sin(ang) + v * np.cos(ang)
    br = np.hypot(bx - 650, by - 250)
    ba = np.arctan2(by - 250, bx - 650)
    r = np.concatenate([r, br])
    a = np.concatenate([a, ba])
    size = np.concatenate([size, g.uniform(0.8, 1.8, band)])
    bright = np.concatenate([bright, g.uniform(0.1, 0.5, band)])
    return r, a, size, bright


def draw_stars(img, cat, t, rot_rate=0.022, pole=(650, 250), twinkle_seed=0, fade_y=1500, strength=1.0, color=(0.92, 0.94, 1.0)):
    r, a, size, bright = cat
    ang = a + t * rot_rate
    xs = pole[0] + r * np.cos(ang)
    ys = pole[1] + r * np.sin(ang)
    keep = (xs > -10) & (xs < W2 + 10) & (ys > -10) & (ys < fade_y)
    xs, ys, sz, br = xs[keep], ys[keep], size[keep], bright[keep]
    n = xs.shape[0]
    tw = 0.75 + 0.25 * np.sin(t * 3.0 + np.arange(n) * 1.7 + twinkle_seed)
    fade = np.clip((fade_y - ys) / 500.0, 0, 1)
    gl = np.empty((n, 7))
    gl[:, 0] = xs
    gl[:, 1] = ys
    gl[:, 2] = sz * 3.0
    gl[:, 3:6] = np.asarray(color)
    gl[:, 6] = br * tw * fade * strength * 2.6
    R.add_glows(img, gl)


def shooting_star(img, t, t0, dur=0.6, p0=(2500, 260), p1=(3500, 760)):
    u = (t - t0) / dur
    if u < 0 or u > 1.3:
        return
    p0 = np.array(p0, np.float64)
    p1 = np.array(p1, np.float64)
    head = p0 + (p1 - p0) * min(u, 1.0)
    n = 40
    gl = np.empty((n, 7))
    for i in range(n):
        k = i / (n - 1)
        pos = head - (p1 - p0) * 0.22 * k
        gl[i, 0] = pos[0]
        gl[i, 1] = pos[1]
        gl[i, 2] = 10 * (1 - k) + 3
        gl[i, 3:6] = (1.0, 0.95, 0.85)
        gl[i, 6] = (1 - k) ** 2 * 1.2 * (1 if u <= 1 else max(0, 1 - (u - 1) / 0.3))
    R.add_glows(img, gl)


def fireflies(img, t, seed=31, n=30):
    g = B.rng(seed)
    x0 = g.uniform(200, W2 - 200, n)
    y0 = g.uniform(1480, 2000, n)
    ph = g.uniform(0, 6.28, n)
    sp = g.uniform(0.3, 0.8, n)
    xs = x0 + 60 * np.sin(t * sp + ph) + 30 * np.sin(t * 1.7 * sp + ph * 2)
    ys = y0 + 25 * np.sin(t * sp * 1.3 + ph * 1.5) - t * 4
    blink = np.clip(np.sin(t * 2.2 * sp + ph * 3), 0, 1) ** 3
    gl = np.empty((n, 7))
    gl[:, 0] = xs
    gl[:, 1] = ys
    gl[:, 2] = 26
    gl[:, 3:6] = (0.85, 1.0, 0.45)
    gl[:, 6] = blink * 1.2
    R.add_glows(img, gl)
    gl2 = gl.copy()
    gl2[:, 2] = 4
    gl2[:, 6] = blink * 3
    R.add_glows(img, gl2)


def shot3_panda(light=(-0.3, -0.9), seed=3, ear_r=0.0, ear_l=0.0, px=2110):
    cel = Cel(W2, H2)
    Tp = Transform(px, hill_crest(px) - 40, 58.0)
    pose = Pose(yaw=0.5, pitch=0.3, eye_open=0.0, mouth="line", ear_r=ear_r, ear_l=ear_l, seed=seed + 20)
    from .panda import pose_sleep
    pose_sleep(cel, Tp, pose, light)
    return cel


# ----------------------------------------------------------------------------
# shot 4: first light
# ----------------------------------------------------------------------------

def shot4_panda(phase, u, light=(0.5, -0.8), seed=4, px=2110, ear_r=0.0, blink=1.0):
    """phase: sleep, sit, stand, walk, crouch, leap, inside. u: 0..1 progress inside the phase."""
    from .panda import pose_sleep, pose_sit_front, pose_stand
    cel = Cel(W2, H2)
    rx = 2330
    base_y = hill_crest(px) - 100
    if phase == "sleep":
        Tp = Transform(px, hill_crest(px) - 40, 58.0)
        pose_sleep(cel, Tp, Pose(yaw=0.5, pitch=0.3, eye_open=0.0, mouth="line", ear_r=ear_r, seed=seed + 20), light)
    elif phase == "sit":
        Tp = Transform(px + 40, hill_crest(px) - 130 + 20 * (1 - u), 58.0)
        pose_sit_front(cel, Tp, Pose(yaw=0.3, pitch=-0.1, eye_open=min(blink, 0.35 + 0.65 * u), mouth="o", mouth_open=1.0 - u, ear_r=ear_r, seed=seed + 21), light)
    elif phase in ("stand", "walk", "crouch"):
        stride = float(np.sin(2 * np.pi * u * 2.2)) if phase == "walk" else 0.0
        bob = abs(np.sin(2 * np.pi * u * 2.2)) * 5 if phase == "walk" else 0.0
        crouch = u if phase == "crouch" else 0.0
        Tp = Transform(px, base_y - bob + 18 * crouch, 58.0)
        pose_stand(cel, Tp, Pose(yaw=0.65, pitch=-0.5 if phase != "crouch" else -0.8, eye_open=blink, mouth="smile", seed=seed + 22), light, stride=stride, head_up=0.3, crouch=crouch)
    elif phase == "leap":
        # arc from the ground to the window, body stretched and nose up
        wx, wy = rx - 10, hill_crest(rx) + 20 - 0.84 * 640 + 0.3 * 640
        x0, y0 = px, base_y
        x = x0 + (wx - 60 - x0) * u
        y = y0 + (wy + 40 - y0) * u - 260 * np.sin(np.pi * u)
        if u < 0.3:
            rot, stride, pitch_, hu = -1.0, 1.0, -0.7, 0.5    # launch: stretched, nose up
        elif u < 0.7:
            rot, stride, pitch_, hu = -0.45, -0.6, -0.3, 0.2  # tuck at the top
        else:
            rot, stride, pitch_, hu = 0.05, 0.4, 0.1, 0.1     # reach for the rim
        Tp = Transform(x, y, 58.0, rot=rot)
        pose_stand(cel, Tp, Pose(yaw=0.7, pitch=pitch_, eye_open=1.0, mouth="o", mouth_open=0.4, seed=seed + 23), light, stride=stride, head_up=hu)
    elif phase == "inside":
        wx, wy = rx - 10, hill_crest(rx) + 20 - 0.84 * 640 + 0.3 * 640
        Tp = Transform(wx - 5, wy + 8 + 30 * (1 - u), 48.0)
        draw_head(cel, Tp, 0, 0, Pose(yaw=0.0, pitch=-0.1, eye_open=blink, mouth="smile", goggles=True, goggles_up=True, seed=seed + 24), light)
    return cel


def shot4b_frame(goggles_up=True, pitch=0.0, brow=-0.5, mouth="line", look=(0.0, -0.1), head_dy=0.0, blink=1.0, swallow=0.0, seed=4, light=(0.6, -0.7), sky=None, **pose_kw):
    """Close-up through the porthole: the paper body fills the frame, the panda's face in the glass."""
    from .cel import Cel as _Cel
    H, W = H2, W2
    frame = B.vgrad(H, W, [(0.0, (0.80, 0.72, 0.58)), (1.0, (0.62, 0.52, 0.42))])
    frame *= B.paper(H, W, seed + 9, 0.08)[..., None]
    cel = _Cel(W, H)
    cx, cy, rr = 1920, 1080, 820
    # seams and tape on the paper
    for k, yy in enumerate([260, 1900]):
        cel.ink(catmull([(0, yy + 10), (1900, yy - 10), (3840, yy + 14)], n=8), 6.0, color=PAPER_D, seed=seed + 30 + k, a=0.8)
    # dark cockpit behind the glass
    inner = ellipse(cx, cy, rr - 50, rr - 50, n=40)
    cel.fill(inner, (0.22, 0.14, 0.10))
    # the face, clipped to the glass
    face = _Cel(W, H)
    Tp = Transform(cx + 30, cy + 110 + head_dy, 520.0)
    pose = Pose(yaw=0.1, pitch=pitch, brow=brow, mouth=mouth, look=look, eye_open=blink, goggles=True, goggles_up=goggles_up, seed=seed + 20, **pose_kw)
    from .panda import ruff
    ruff(face, Tp, 0, 0, 0.1, light, seed + 20, rx=1.3, ry=0.5, dy=0.7)
    draw_head(face, Tp, 0, 0, pose, light)
    if swallow > 0:
        # throat bob: a short curved line under the chin
        face.ink(Tp(catmull([(-0.15, 1.1 + 0.05 * swallow), (0.0, 1.16 + 0.08 * swallow), (0.15, 1.1 + 0.05 * swallow)], n=6)), 5.0, seed=seed + 40, a=0.6 * swallow)
    gm = R.poly_mask(ellipse(cx, cy, rr - 50, rr - 50, n=40), W, H)
    full = R.full_mask_from(gm, W, H)
    face.alpha *= full
    face.over(cel.img, 0, 0)
    cel.alpha = np.maximum(cel.alpha, face.alpha)
    # glass tint and two reflections of the dawn sky
    glass = ellipse(cx, cy, rr - 50, rr - 50, n=40)
    cel.fill(glass, (0.55, 0.70, 0.80), 0.18)
    cel.fill(ellipse(cx - 300, cy - 380, 260, 70, n=16, rot=-0.6), (0.95, 0.97, 1.0), 0.35)
    cel.fill(ellipse(cx + 380, cy + 420, 120, 36, n=12, rot=-0.6), (0.95, 0.97, 1.0), 0.25)
    # brass ring with rivets
    ring_o = ellipse(cx, cy, rr + 40, rr + 40, n=48)
    ring_i = ellipse(cx, cy, rr - 50, rr - 50, n=48)
    ring = np.concatenate([ring_o, ring_i[::-1]])
    rm = cel.fill(wobble(ring, 3, seed + 5, closed=True), BRASS)
    mult_shade(cel, rm, SHADOW_TINT, -light[0] * 30, -light[1] * 30, 0.8)
    cel.ink_closed(ring_o, 7.0, light=light, seed=seed + 6)
    cel.ink_closed(ring_i, 5.0, light=light, seed=seed + 7, gap=False)
    for k in range(8):
        a = k * np.pi / 4 + 0.3
        cel.fill(ellipse(cx + (rr - 5) * np.cos(a), cy + (rr - 5) * np.sin(a), 14, 14, n=12), (0.45, 0.34, 0.18))
    cel.over(frame, 0, 0)
    return frame, cel


def glass_mask():
    """Full mask of the rocket window glass in shot 2 layout."""
    rx = 2330
    h = 640.0
    w = 0.155 * h
    top = hill_crest(rx) + 20 - 0.84 * h
    wx, wy = rx - 0.1 * w, top + 0.3 * h
    m = R.poly_mask(ellipse(wx, wy, 0.32 * w, 0.32 * w, n=22), W2, H2)
    return R.full_mask_from(m, W2, H2)


# ----------------------------------------------------------------------------
# shot 5: the spark
# ----------------------------------------------------------------------------

def paper_wall_bg(seed, with_band=True):
    H, W = H2, W2
    frame = B.vgrad(H, W, [(0.0, (0.74, 0.66, 0.52)), (1.0, (0.56, 0.47, 0.38))])
    frame *= B.paper(H, W, seed + 9, 0.08)[..., None]
    cel = Cel(W, H)
    for k, yy in enumerate([300, 760, 1700]):
        cel.ink(catmull([(0, yy + 10), (1900, yy - 12), (3840, yy + 16)], n=8), 6.0, color=PAPER_D, seed=seed + 30 + k, a=0.8)
    if with_band:
        band = np.array([(0, 1000), (3840, 980), (3840, 1240), (0, 1260)])
        from .rocket import TIN, TIN_D
        bm = cel.fill(wobble(band, 3, seed + 1, closed=True), TIN)
        mult_shade(cel, bm, SHADOW_TINT, 0, -60, 0.7)
        cel.ink_closed(band, 6.0, light=(0.3, -0.9), seed=seed + 2, gap=False)
        for k in range(9):
            cel.fill(ellipse(240 + k * 420, 1120, 22, 22, n=14), TIN_D)
            cel.ink_closed(ellipse(240 + k * 420, 1120, 22, 22, n=14), 3.0, light=(0.3, -0.9), seed=seed + 10 + k, gap=False)
        # scratch marks on the band from earlier strikes
        for k in range(5):
            cel.ink([(1500 + k * 90, 1150 + k * 6), (1700 + k * 90, 1120 + k * 6)], 2.5, color=(0.35, 0.36, 0.40), seed=seed + 40 + k, a=0.6)
    cel.over(frame)
    return frame


def shot5a_cel(u, lit, flare, seed=5, light=(0.3, -0.9)):
    """Paw holding a match, striking along the tin band. u: paw x progress 0..1 (right to left)."""
    from .panda import draw_leg
    cel = Cel(W2, H2)
    T = Transform(0, 0, 1.0)
    px = 2600 - 1300 * u
    py = 1130
    # forearm from the lower right, paw at (px, py)
    Tp = Transform(px, py, 420.0)
    draw_leg(cel, Tp, [(1.6, 1.9), (1.0, 1.0), (0.1, 0.1)], 0.42, light, seed + 1)
    # match: stick from the paw toward the upper left, head on the band
    mx0, my0 = px - 30, py - 20
    ang = -0.35 - 0.25 * u
    mx1, my1 = mx0 + 300 * np.cos(ang), my0 + 300 * np.sin(ang)
    stick = stroke_outline(np.array([(mx0, my0), (mx1, my1)]), np.array([22.0, 18.0]))
    sm = cel.fill(stick, (0.88, 0.80, 0.62))
    cel.ink_closed(stick, 3.5, light=light, seed=seed + 3, gap=True)
    head = ellipse(mx1, my1, 20, 26, n=14, rot=ang)
    cel.fill(head, (0.60, 0.16, 0.12) if not lit else (0.25, 0.14, 0.10))
    cel.ink_closed(head, 3.0, light=light, seed=seed + 4, gap=False)
    if lit:
        # flame: teardrop with a yellow core, flickers with the seed
        g = B.rng(seed * 7)
        fl = 1.0 + 0.2 * g.uniform(-1, 1)
        tip = (mx1 - 10 * g.uniform(-1, 1), my1 - 95 * fl)
        fpoly = catmull([(mx1 - 30, my1 - 10), (mx1 - 22, my1 - 50 * fl), tip, (mx1 + 24, my1 - 50 * fl), (mx1 + 30, my1 - 10)], closed=True, n=6)
        cel.fill(fpoly, (1.0, 0.55, 0.15), 0.95)
        core = catmull([(mx1 - 14, my1 - 8), (mx1 - 10, my1 - 35 * fl), (mx1, my1 - 55 * fl), (mx1 + 10, my1 - 35 * fl), (mx1 + 14, my1 - 8)], closed=True, n=6)
        cel.fill(core, (1.0, 0.92, 0.55), 0.95)
    return cel, (mx1, my1)


def fuse_bg(seed=5):
    """Grass close to the ground at predawn with the rocket legs at the right."""
    H, W = H2, W2
    pal = PALETTES["predawn"]
    img = B.vgrad(H, W, [(0.0, (0.30, 0.36, 0.50)), (0.25, (0.22, 0.28, 0.32)), (0.6, (0.12, 0.18, 0.18)), (1.0, (0.06, 0.09, 0.10))])
    B.soft_light(img, (0.30, 0.22, 0.14), 3300, 300, 1600, 0.8, H, W, 1.8)
    ref = img.copy()
    B.paint_region(img, ref, None, 50, 200, 60, 0.8, seed + 1, angle=0.05, angle_jit=0.15, val=0.04, hue=0.03, hard=0.3)
    g = B.rng(seed + 2)
    for band, (y0, y1, n, hmin, hmax, wd) in enumerate([(500, 1100, 5000, 40, 120, 5), (1000, 1700, 4000, 120, 320, 9), (1600, H + 50, 2500, 300, 700, 16)]):
        xs = g.uniform(-50, W + 50, n)
        ys = g.uniform(y0, y1, n)
        hs = g.uniform(hmin, hmax, n)
        t = (ys - 500) / (H - 500)
        cols = np.array([0.22, 0.30, 0.26])[None] * (1 - t[:, None]) + np.array([0.05, 0.08, 0.08])[None] * t[:, None]
        cols = np.clip(cols + g.normal(0, 0.03, (n, 3)), 0, 1)
        B.grass_blades(img, xs, ys, hs, cols, seed + 10 + band, lean=0.2, width=wd, alpha=0.85, curve=0.5)
    img *= B.paper(H, W, seed + 13, 0.05)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def fuse_path():
    return catmull([(-100, 1500), (500, 1560), (1100, 1480), (1700, 1600), (2300, 1520), (2750, 1620), (3000, 1500)], n=16)


def shot5b_cels(spark_t, seed=5, light=(0.3, -0.9)):
    """Fuse cord across the grass, burnt behind the spark; rocket legs at the right."""
    cel = Cel(W2, H2)
    path = fuse_path()
    n = len(path)
    k = int(np.clip(spark_t, 0, 1) * (n - 1))
    # unburnt cord (dark) ahead, burnt (grey ash) behind
    if k < n - 1:
        cel.ink(path[k:], 14.0, color=(0.18, 0.12, 0.10), seed=seed, taper_in=0.0, taper_out=0.0, var=0.2)
    if k > 1:
        cel.ink(path[:k + 1], 10.0, color=(0.36, 0.34, 0.34), seed=seed + 1, taper_in=0.0, taper_out=0.0, var=0.3, a=0.8)
    # rocket legs and exhaust can at the right edge (big, near)
    T = Transform(0, 0, 1.0)
    draw_rocket(cel, T, 3350, 1560, 2600.0, light, seed=seed + 5)
    return cel, path[k]


def sparks(img, pos, seed, n=26, size=1.0, color=(1.0, 0.85, 0.45)):
    g = B.rng(seed)
    a = g.uniform(0, 2 * np.pi, n)
    r = g.uniform(10, 120, n) ** 1.1 * size
    gl = np.empty((n + 1, 7))
    gl[:n, 0] = pos[0] + r * np.cos(a)
    gl[:n, 1] = pos[1] + r * np.sin(a) * 0.7 - 20
    gl[:n, 2] = g.uniform(4, 11, n) * size
    gl[:n, 3:6] = np.asarray(color)
    gl[:n, 6] = g.uniform(0.8, 2.0, n)
    gl[n] = (pos[0], pos[1], 90 * size, 1.0, 0.75, 0.35, 2.0)
    R.add_glows(img, gl)


# ----------------------------------------------------------------------------
# shot 6: lift
# ----------------------------------------------------------------------------

def smoke(img, puffs, seed, base=(0.78, 0.76, 0.78), lit=(1.0, 0.80, 0.55), light_dir=(0.0, 1.0)):
    """Painted smoke: each puff (x, y, r, alpha) is a cluster of soft dabs, lit on one side."""
    g = B.rng(seed)
    allx, ally, cols, L, Wd, al = [], [], [], [], [], []
    for (x, y, r, a) in puffs:
        if a <= 0.02 or r < 4:
            continue
        n = int(r * r / 260) + 8
        ang = g.uniform(0, 2 * np.pi, n)
        rr = r * np.sqrt(g.uniform(0, 1, n))
        dx = rr * np.cos(ang)
        dy = rr * np.sin(ang) * 0.85
        litf = np.clip((dx * light_dir[0] + dy * light_dir[1]) / (r + 1e-6) * 0.5 + 0.5, 0, 1) ** 1.4
        c = np.asarray(base)[None] * (1 - litf[:, None]) + np.asarray(lit)[None] * litf[:, None]
        allx.append(x + dx)
        ally.append(y + dy)
        cols.append(B.jitter_colors(c, g, 0.04, 0.02))
        L.append(g.uniform(0.5, 0.9, n) * r * 0.7)
        Wd.append(g.uniform(0.3, 0.6, n) * r * 0.7)
        al.append(g.uniform(0.5, 0.9, n) * a)
    if not allx:
        return
    n = sum(len(x) for x in allx)
    d = B.make_dabs(np.concatenate(allx), np.concatenate(ally), g.normal(0, 0.3, n), np.concatenate(L), np.concatenate(Wd),
                    np.concatenate(cols), np.concatenate(al), 0.15)
    R.paint_dabs(img, d, seed)


def ignition_puffs(t, seed=6, base=(2330, 0)):
    """Ground smoke burst: puffs born in the first 0.8 s, spreading outward and up."""
    g = B.rng(seed)
    bx = base[0]
    by = hill_crest(bx) + 10
    puffs = []
    for i in range(70):
        birth = g.uniform(0.0, 0.8)
        age = t - birth
        if age < 0:
            continue
        ang = g.uniform(np.pi * 0.05, np.pi * 0.95)  # upward half
        side = g.uniform(-1, 1)
        speed = g.uniform(300, 900)
        x = bx + side * speed * age ** 0.6 * 1.4
        y = by - abs(np.sin(ang)) * speed * 0.35 * age ** 0.7 + 40 * age
        r = 40 + 220 * age ** 0.6
        a = max(0.0, 1.0 - age / 2.6) * 0.9
        puffs.append((x, y, r, a))
    return puffs


def grass_ring(img, t, seed=6, base=(2330, 0)):
    """Grass flattened outward in a ring: pale streaks radiating, expanding."""
    if t <= 0:
        return
    g = B.rng(seed)
    bx = base[0]
    by = hill_crest(bx) + 10
    rad = 120 + 900 * (1 - np.exp(-t * 2.5))
    n = 260
    ang = g.uniform(0, 2 * np.pi, n)
    rr = rad * g.uniform(0.8, 1.1, n)
    xs = bx + rr * np.cos(ang)
    ys = by + rr * np.sin(ang) * 0.32 + 30
    keep = ys > hill_crest(xs) + 5
    xs, ys, ang = xs[keep], ys[keep], ang[keep]
    n = xs.shape[0]
    cols = B.jitter_colors(np.tile(np.array([0.44, 0.50, 0.30]), (n, 1)), g, 0.05, 0.03)
    d = B.make_dabs(xs, ys, np.arctan2(np.sin(ang) * 0.32, np.cos(ang)), g.uniform(30, 70, n), g.uniform(5, 10, n), cols, np.full(n, 0.7) * max(0.0, 1 - t / 2.5), 0.6)
    R.paint_dabs(img, d, seed + 1)


def rocket_cel(dy=0.0, shudder=0.0, light=(0.5, -0.8), seed=2, fin_flap=0.0, ghost=0.0):
    cel = Cel(W2, H2)
    T = Transform(0, dy, 1.0)
    rx = 2330
    if ghost > 0:
        for k in (1, 2):
            Tg = Transform(0, dy + k * ghost, 1.0)
            draw_rocket(cel, Tg, rx, hill_crest(rx) + 20, 640.0, light, seed=seed + k, shudder=shudder)
        cel.img *= 1.0
        cel.alpha *= 0.35
    draw_rocket(cel, T, rx, hill_crest(rx) + 20, 640.0, light, seed=seed, shudder=shudder, fin_flap=fin_flap)
    return cel


def low_angle_bg(seed=6):
    """Looking up past the camphor canopy into the dawn sky."""
    H, W = H2, W2
    img = B.vgrad(H, W, [(0.0, (0.10, 0.14, 0.38)), (0.35, (0.26, 0.30, 0.56)), (0.65, (0.66, 0.52, 0.60)), (0.85, (0.98, 0.74, 0.52)), (1.0, (1.0, 0.86, 0.60))])
    B.soft_light(img, (0.35, 0.22, 0.08), 2600, 2100, 1800, 0.8, H, W, 1.8)
    ref = img.copy()
    B.paint_region(img, ref, None, 60, 260, 70, 0.75, seed + 1, angle=0.0, angle_jit=0.06, val=0.03, hue=0.02, hard=0.3)
    B.paint_region(img, ref, None, 26, 110, 30, 0.6, seed + 2, angle=0.0, angle_jit=0.08, val=0.025, hue=0.02, hard=0.4)
    cloud_masses(img, seed + 3, PALETTES["predawn"])
    # the trunk sweeping in from the lower left, branches fanning over the top
    g = B.rng(seed + 4)
    trunk = [(200, 2300), (500, 1500), (900, 900), (1500, 500)]
    tp = catmull(trunk, n=10)
    widths = np.linspace(260, 90, len(tp))
    poly = stroke_outline(tp, widths)
    R.fill_poly(img, wobble(poly, 5, seed, closed=True), (0.16, 0.12, 0.16), 1.0)
    n = 900
    t = g.uniform(0, 1, n)
    idx = (t * (len(tp) - 1)).astype(int)
    d = np.gradient(tp, axis=0)
    ang = np.arctan2(d[idx, 1], d[idx, 0])
    off = g.uniform(-1, 1, n)
    nx, ny = -np.sin(ang), np.cos(ang)
    xs = tp[idx, 0] + nx * off * widths[idx] * 0.48
    ys = tp[idx, 1] + ny * off * widths[idx] * 0.48
    lit = np.clip(off * 0.5 + 0.5, 0, 1)
    cols = np.array([0.10, 0.08, 0.12])[None] * (1 - lit[:, None]) + np.array([0.42, 0.30, 0.28])[None] * lit[:, None]
    dd = B.make_dabs(xs, ys, ang + g.normal(0, 0.1, n), g.uniform(30, 90, n), g.uniform(8, 18, n), B.jitter_colors(cols, g, 0.05, 0.03), g.uniform(0.5, 0.9, n), 0.7)
    R.paint_dabs(img, dd, seed + 5)
    branches = [[(1400, 560), (2000, 300), (2600, 150)], [(1300, 620), (1700, 200), (1900, -50)], [(900, 900), (700, 500), (650, 200)],
                [(1500, 500), (2200, 600), (2900, 450)], [(600, 1350), (300, 1100), (100, 1000)]]
    for k, br in enumerate(branches):
        bp = catmull(br, n=8)
        w = np.linspace(60, 18, len(bp))
        R.fill_poly(img, wobble(stroke_outline(bp, w), 3, seed + k, closed=True), (0.16, 0.12, 0.16), 1.0)
    centers = [(2200, 250), (1700, 120), (2700, 150), (700, 300), (300, 900), (1200, 350), (2900, 480), (3300, 200), (100, 1300), (1000, 150)]
    radii = [320, 300, 280, 300, 260, 260, 220, 240, 260, 220]
    fo = PALETTES["predawn"]["foliage"]
    B.foliage_clumps(img, None, centers, radii, seed + 6, fo[0], fo[1], light_dir=(0.3, 0.9), leaf=40, density=1.2)
    B.foliage_clumps(img, None, centers, [r * 0.75 for r in radii], seed + 7, fo[1], fo[3], light_dir=(0.3, 0.9), leaf=24, density=0.9, alpha=0.8)
    img *= B.paper(H, W, seed + 13, 0.05)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def birds_cel(t, t0=0.6, seed=6, n=16):
    """Small birds bursting out of the canopy, flapping on threes."""
    cel = Cel(W2, H2)
    if t < t0:
        return cel
    g = B.rng(seed)
    for i in range(n):
        delay = g.uniform(0, 0.5)
        u = t - t0 - delay
        if u < 0:
            continue
        ox, oy = g.uniform(600, 2900), g.uniform(250, 750)
        side = -1 if ox < 1800 else 1
        vx, vy = side * g.uniform(350, 700), g.uniform(-120, 220)
        x = ox + vx * u
        y = oy + vy * u + 60 * np.sin(u * 7 + i)
        size = g.uniform(26, 44) * (1 + 0.5 * u)
        flap = np.sin((u * 14 + i) // 1 * 0.8)  # held poses
        wing = 0.9 * flap
        for sgn in (-1, 1):
            tip = (x + sgn * size * np.cos(wing * 0.6), y - size * np.sin(wing * 0.6 + 0.2) * sgn * sgn)
            cel.ink(catmull([(x, y), (x + sgn * size * 0.5, y - size * 0.25 * (0.4 + wing)), tip], n=6), 5.0, color=(0.18, 0.12, 0.14), seed=seed + i, taper_in=0.1, taper_out=0.3)
        cel.fill(ellipse(x, y, size * 0.16, size * 0.1, n=8), (0.18, 0.12, 0.14))
    return cel


def trail_puffs(path_fn, t, t0, seed, life=1.6, every=0.06, r0=30, grow=120, drift=0.0):
    """Smoke puffs left along a moving object's path."""
    g = B.rng(seed)
    puffs = []
    k = 0
    tb = t0
    while tb < t:
        age = t - tb
        if age < life:
            x, y = path_fn(tb)
            jx, jy = g.normal(0, 18), g.normal(0, 12)
            sw = np.sin(tb * 9.0 + k) * drift
            puffs.append((x + jx + 30 * age + sw * age, y + jy + 60 * age, r0 + grow * age ** 0.7, max(0.0, 1 - age / life) * 0.85))
        else:
            g.normal(0, 18)
            g.normal(0, 12)
        tb += every
        k += 1
    return puffs


def flame_cel(x, y, size, seed, angle=np.pi / 2):
    """Exhaust flame pointing along angle (down by default)."""
    cel = Cel(W2, H2)
    g = B.rng(seed)
    fl = 1 + 0.25 * g.uniform(-1, 1)
    ca, sa = np.cos(angle), np.sin(angle)
    def P(u, v):
        return (x + u * size * (-sa) + v * size * ca, y + u * size * ca + v * size * sa)
    outer = catmull([P(-0.32, 0), P(-0.25, 0.6 * fl), P(0, 1.3 * fl), P(0.25, 0.6 * fl), P(0.32, 0)], closed=True, n=6)
    cel.fill(outer, (1.0, 0.55, 0.15), 0.95)
    core = catmull([P(-0.16, 0), P(-0.12, 0.45 * fl), P(0, 0.8 * fl), P(0.12, 0.45 * fl), P(0.16, 0)], closed=True, n=6)
    cel.fill(core, (1.0, 0.92, 0.6), 0.95)
    return cel


def small_rocket_cel(x, y, h, light, seed, rot=0.0, fin_flap=0.0):
    cel = Cel(W2, H2)
    T = Transform(x, y, 1.0, rot=rot)
    draw_rocket(cel, T, 0, 0, h, light, seed=seed, fin_flap=fin_flap)
    return cel


# ----------------------------------------------------------------------------
# shot 7: through the clouds
# ----------------------------------------------------------------------------

def sky_climb_bg(seed=7):
    H, W = H2, W2
    img = B.vgrad(H, W, [(0.0, (0.16, 0.26, 0.56)), (0.4, (0.40, 0.56, 0.80)), (0.7, (0.86, 0.74, 0.64)), (0.9, (1.0, 0.84, 0.52)), (1.0, (1.0, 0.92, 0.66))])
    B.soft_light(img, (0.4, 0.26, 0.08), 1900, 2300, 2200, 0.9, H, W, 1.8)
    ref = img.copy()
    B.paint_region(img, ref, None, 60, 260, 70, 0.75, seed + 1, angle=0.0, angle_jit=0.06, val=0.03, hue=0.02, hard=0.3)
    B.paint_region(img, ref, None, 26, 110, 30, 0.6, seed + 2, angle=0.0, angle_jit=0.08, val=0.025, hue=0.02, hard=0.4)
    img *= B.paper(H, W, seed + 13, 0.05)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def cloud_layer(seed, y_center, thickness, n_masses, top_col, lit_col, alpha=0.9):
    """An RGBA cloud band (img, alpha) painted with soft puffs, lit from below."""
    H, W = H2, W2
    img = np.zeros((H, W, 3), np.float32)
    al = np.zeros((H, W), np.float32)
    g = B.rng(seed)
    allx, ally, cols, L, Wd, aa = [], [], [], [], [], []
    for k in range(n_masses):
        cx = g.uniform(-200, W + 200)
        cy = y_center + g.normal(0, thickness * 0.25)
        ln = g.uniform(500, 1400)
        hgt = thickness * g.uniform(0.6, 1.1)
        npuff = int(ln / (hgt * 0.8)) + 2
        for i in range(npuff):
            t = (i + 0.5) / npuff
            env = np.sin(np.pi * t) ** 0.5
            px = cx - ln / 2 + t * ln + g.normal(0, hgt * 0.2)
            py = cy - hgt * 0.3 * env * g.uniform(0.3, 1.2)
            r = hgt * env * g.uniform(0.6, 1.0)
            n = int(r * r / 140) + 10
            a = g.uniform(0, 2 * np.pi, n)
            rr = r * np.sqrt(g.uniform(0, 1, n))
            dx = rr * np.cos(a)
            dy = rr * np.sin(a) * 0.7
            lit = np.clip(dy / (r + 1e-6) * 0.5 + 0.55, 0, 1) ** 1.4
            c = np.asarray(top_col)[None] * (1 - lit[:, None]) + np.asarray(lit_col)[None] * lit[:, None]
            allx.append(px + dx)
            ally.append(py + dy)
            cols.append(B.jitter_colors(c, g, 0.03, 0.02))
            L.append(g.uniform(0.5, 0.9, n) * r * 0.6)
            Wd.append(g.uniform(0.3, 0.6, n) * r * 0.6)
            aa.append(g.uniform(0.5, 0.9, n) * alpha)
    n = sum(len(x) for x in allx)
    d = B.make_dabs(np.concatenate(allx), np.concatenate(ally), g.normal(0, 0.12, n), np.concatenate(L), np.concatenate(Wd), np.concatenate(cols), np.concatenate(aa), 0.15)
    R.paint_dabs_alpha(img, al, d, seed)
    return img, al


def over_layer(frame, layer, dy, a=1.0):
    """Composite an RGBA layer shifted vertically by dy (wraps vertically)."""
    img, al = layer
    H = frame.shape[0]
    dy = int(round(dy)) % H
    img2 = np.roll(img, dy, axis=0)
    al2 = np.roll(al, dy, axis=0)[..., None] * a
    frame[:] = frame * (1 - al2) + img2 * al2


# ----------------------------------------------------------------------------
# shot 8: the shooting star (the widest image)
# ----------------------------------------------------------------------------

def world_bg(seed=8):
    """Very high above the valley at dawn: the curve of the world, a sea of clouds, peaks, rivers, the rising sun."""
    H, W = H2, W2
    img = B.vgrad(H, W, [(0.0, (0.03, 0.05, 0.18)), (0.25, (0.08, 0.12, 0.34)), (0.45, (0.26, 0.30, 0.58)), (0.56, (0.74, 0.50, 0.56)), (0.62, (1.0, 0.72, 0.42)), (1.0, (1.0, 0.86, 0.60))])
    cx_sun, cy_sun = 2500, 1330
    B.soft_light(img, (0.45, 0.28, 0.08), cx_sun, cy_sun, 2000, 1.0, H, W, 2.0)
    masks = {}
    # the curve of the world: everything below this arc is the ground
    rc = 9000.0
    cyc = 1300 + rc
    arc = [(x, cyc - np.sqrt(max(rc * rc - (x - 1920) ** 2, 0))) for x in range(-100, W + 200, 60)]
    ground = arc + [(W + 100, H + 100), (-100, H + 100)]
    mg = B.mask_of([ground], W, H)
    ground_col = B.vgrad(H, W, [(0.0, (0.80, 0.60, 0.62)), (0.6, (0.60, 0.50, 0.62)), (0.75, (0.42, 0.40, 0.56)), (1.0, (0.26, 0.26, 0.40))])
    img[:] = img * (1 - mg[..., None]) + ground_col * mg[..., None]
    masks["ground"] = mg
    ref = img.copy()
    B.paint_region(img, ref, 1 - mg, 60, 260, 70, 0.75, seed + 1, angle=0.0, angle_jit=0.06, val=0.03, hue=0.02, hard=0.3)
    B.paint_region(img, ref, 1 - mg, 26, 110, 30, 0.6, seed + 2, angle=0.0, angle_jit=0.08, val=0.025, hue=0.02, hard=0.4)
    B.paint_region(img, ref, mg, 30, 120, 36, 0.7, seed + 3, angle=0.0, angle_jit=0.1, val=0.04, hue=0.03, hard=0.4)
    g = B.rng(seed + 4)
    # rivers: bright winding threads catching the light, fanning down from the horizon
    for k in range(5):
        x0 = g.uniform(300, 3500)
        pts = [(x0, 1340 + g.uniform(0, 60))]
        x, y = pts[0]
        while y < H + 100:
            x += g.normal(0, 90) + (1920 - x0) * 0.02
            y += g.uniform(90, 170)
            pts.append((x, y))
        path = catmull(pts, n=6)
        wdt = np.linspace(3, 26, len(path))
        R.fill_poly(img, stroke_outline(path, wdt), (1.0, 0.82, 0.58), 0.9)
        R.fill_poly(img, stroke_outline(path, wdt * 0.4), (1.0, 0.94, 0.80), 0.9)
    # mountain peaks poking through the cloud sea
    for k in range(26):
        px = g.uniform(-100, W + 100)
        py = 1400 + 700 * g.uniform(0, 1) ** 1.3
        ssz = 60 + (py - 1400) * 0.5 * g.uniform(0.6, 1.4)
        peak = [(px - ssz, py + ssz * 0.6), (px - ssz * 0.3, py - ssz * 0.1), (px, py - ssz * 0.75), (px + ssz * 0.35, py - ssz * 0.2), (px + ssz, py + ssz * 0.6)]
        R.fill_poly(img, wobble(np.array(peak), ssz * 0.04, seed + k, closed=True), (0.30, 0.26, 0.44), 1.0)
        lit = [(px, py - ssz * 0.75), (px + ssz * 0.35, py - ssz * 0.2), (px + ssz * 0.6, py + ssz * 0.1), (px + ssz * 0.15, py - ssz * 0.1)]
        R.fill_poly(img, wobble(np.array(lit), ssz * 0.04, seed + 40 + k, closed=True), (0.96, 0.72, 0.56), 0.9)
    # sea of clouds: rows of masses, larger and lower toward the camera
    for row in range(5):
        yc = 1420 + row * 170 + 40 * row * row
        thick = 90 + row * 70
        n_m = 9 - row
        for k in range(n_m):
            cx = g.uniform(-200, W + 200)
            ln = g.uniform(600, 1500) * (1 + row * 0.3)
            far = 1 - row / 4
            top_col = np.array([0.56, 0.46, 0.62]) * far + np.array([0.70, 0.60, 0.70]) * (1 - far)
            lit_col = np.array([1.0, 0.82, 0.60]) * far + np.array([1.0, 0.92, 0.76]) * (1 - far)
            npuff = int(ln / (thick * 0.8)) + 2
            allx, ally, cols, L, Wd, al = [], [], [], [], [], []
            for i in range(npuff):
                t = (i + 0.5) / npuff
                env = np.sin(np.pi * t) ** 0.5
                x = cx - ln / 2 + t * ln + g.normal(0, thick * 0.2)
                y = yc - thick * 0.3 * env * g.uniform(0.3, 1.2)
                r = thick * env * g.uniform(0.6, 1.0)
                n = int(r * r / 160) + 10
                a = g.uniform(0, 2 * np.pi, n)
                rr = r * np.sqrt(g.uniform(0, 1, n))
                dx = rr * np.cos(a)
                dy = rr * np.sin(a) * 0.6
                litf = np.clip(-dy / (r + 1e-6) * 0.5 + 0.5, 0, 1) ** 1.3  # lit on top by the low sun
                c = top_col[None] * (1 - litf[:, None]) + lit_col[None] * litf[:, None]
                allx.append(x + dx)
                ally.append(y + dy)
                cols.append(B.jitter_colors(c, g, 0.03, 0.02))
                L.append(g.uniform(0.5, 0.9, n) * r * 0.6)
                Wd.append(g.uniform(0.3, 0.6, n) * r * 0.6)
                al.append(g.uniform(0.4, 0.8, n))
            n = sum(len(x) for x in allx)
            d = B.make_dabs(np.concatenate(allx), np.concatenate(ally), g.normal(0, 0.1, n), np.concatenate(L), np.concatenate(Wd), np.concatenate(cols), np.concatenate(al), 0.2)
            R.paint_dabs(img, d, seed + 100 + row * 10 + k)
    # haze toward the horizon and the sun disc
    haze = np.clip(1 - np.abs(np.arange(H) - 1340) / 260.0, 0, 1)[:, None] ** 2 * 0.45
    img[:] = img * (1 - haze[..., None]) + np.array([1.0, 0.80, 0.56], np.float32) * haze[..., None]
    R.add_glows(img, np.array([[cx_sun, cy_sun + 30, 520, 1.0, 0.80, 0.5, 0.45], [cx_sun, cy_sun + 40, 110, 1.0, 0.95, 0.8, 1.6]]))
    B.stars(img, 320, seed + 12, (0, 0, W, 800), rmin=1.0, rmax=2.6, strength=0.9, fade=lambda xs, ys: np.clip(1 - ys / 800, 0, 1) ** 1.5)
    img *= B.paper(H, W, seed + 13, 0.05)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def star_arc(u):
    """The shooting star path across the sky: a tall arc from the lower left to the upper right."""
    x = 400 + 3200 * u
    y = 1500 - 1250 * np.sin(np.pi * min(max(u, 0), 1)) ** 0.8 + 150 * u
    return x, y


def shooting_trail(img, u, seed, head_size=1.0):
    """Bright head with a long glowing tail and sparkle along the arc."""
    if u <= 0:
        return
    n = 90
    gl = []
    g = B.rng(seed)
    for i in range(n):
        k = i / (n - 1)
        uu = u - k * 0.45
        if uu < 0:
            continue
        x, y = star_arc(uu)
        s_ = (1 - k) ** 1.6
        gl.append((x, y, 110 * s_ * head_size + 14, 1.0, 0.92, 0.70, 1.4 * s_))
        if i % 3 == 0:
            gl.append((x + g.normal(0, 40), y + g.normal(0, 40), 8 + 10 * s_, 1.0, 0.85, 0.5, g.uniform(0.3, 1.2) * s_))
    hx, hy = star_arc(u)
    gl.append((hx, hy, 170 * head_size, 1.0, 0.88, 0.62, 2.6))
    gl.append((hx, hy, 40 * head_size, 1.0, 0.98, 0.9, 5.0))
    R.add_glows(img, np.array(gl, np.float64))


# ----------------------------------------------------------------------------
# shot 9: home
# ----------------------------------------------------------------------------

def parachute_cel(x, y, open_u, swing, light, seed, rocket_dy):
    """Umbrella parachute over the rocket: a dome of alternating panels, ribs and lines."""
    cel = Cel(W2, H2)
    T = Transform(0, 0, 1.0)
    rw = 0.155 * 640
    top = y - 0.84 * 640
    handle_y = top - 0.31 * 640
    cx = x + 400 * np.sin(swing) * 0.4
    cy = handle_y - 200 * open_u
    rad = 400 * open_u + 40
    rh = rad * 0.55
    dome = catmull([(cx - rad, cy), (cx - rad * 0.7, cy - rh * 0.75), (cx, cy - rh), (cx + rad * 0.7, cy - rh * 0.75), (cx + rad, cy)], closed=False, n=8)
    # scalloped lower edge
    edge = []
    for k in range(8):
        a = cx - rad + (2 * rad) * (k + 0.5) / 8
        edge.append((a, cy + rh * 0.12))
        edge.append((cx - rad + (2 * rad) * (k + 1) / 8, cy))
    poly = np.concatenate([dome, np.array(edge[::-1])])
    m = cel.fill(wobble(poly, 3, seed, closed=True), (0.92, 0.86, 0.70))
    for k in range(0, 8, 2):
        xa = cx - rad + (2 * rad) * k / 8
        xb = cx - rad + (2 * rad) * (k + 1) / 8
        panel = [(xa, cy), (xb, cy), (cx + (xb - cx) * 0.2, cy - rh * 0.9), (cx + (xa - cx) * 0.2, cy - rh * 0.9)]
        cel.paint_in(m, wobble(np.array(panel), 2, seed + k, closed=True), (0.76, 0.26, 0.22))
    for k in range(9):
        xa = cx - rad + (2 * rad) * k / 8
        cel.ink(catmull([(xa, cy), (cx + (xa - cx) * 0.5, cy - rh * 0.7), (cx, cy - rh)], n=6), 3.0, color=(0.45, 0.35, 0.3), seed=seed + 20 + k, a=0.7)
    mult_shade(cel, m, SHADOW_TINT, -light[0] * 40, -light[1] * 40, 0.7)
    cel.ink_closed(poly, 4.5, light=light, seed=seed + 1)
    for k in range(0, 9, 2):
        xa = cx - rad + (2 * rad) * k / 8
        cel.ink([(xa, cy), (x, handle_y + 20)], 3.0, color=(0.55, 0.45, 0.30), seed=seed + 30 + k, taper_in=0.02, taper_out=0.02)
    return cel


def soot_and_goggles(cel, Tp, pose, light, seed):
    """Sooty smudges on the white face and crooked goggles, drawn over the head."""
    g = B.rng(seed)
    for k in range(6):
        px = g.uniform(-0.6, 0.6)
        py = g.uniform(-0.1, 0.55)
        if abs(px) < 0.25 and py < 0.15:
            continue
        sm = ellipse(px, py, g.uniform(0.12, 0.24), g.uniform(0.07, 0.13), n=10, rot=g.uniform(0, 3))
        cel.fill(Tp(wobble(sm, 0.02, seed + k, closed=True)), (0.18, 0.14, 0.14), 0.85)


# ----------------------------------------------------------------------------
# brush lettering (no fonts: every letter is a few strokes)
# ----------------------------------------------------------------------------

GLYPHS = {
    "T": [[(0.08, 0.14), (0.5, 0.1), (0.92, 0.12)], [(0.5, 0.1), (0.47, 0.95)]],
    "i": [[(0.24, 0.42), (0.24, 0.95)], [(0.21, 0.19), (0.27, 0.23)]],
    "n": [[(0.2, 0.42), (0.2, 0.95)], [(0.22, 0.6), (0.36, 0.42), (0.56, 0.42), (0.7, 0.58), (0.7, 0.95)]],
    "y": [[(0.15, 0.42), (0.47, 0.95)], [(0.85, 0.42), (0.5, 1.05), (0.28, 1.3)]],
    "R": [[(0.2, 0.1), (0.2, 0.95)], [(0.2, 0.1), (0.62, 0.1), (0.78, 0.28), (0.6, 0.5), (0.2, 0.5)], [(0.5, 0.5), (0.82, 0.95)]],
    "o": [[(0.7, 0.5), (0.5, 0.42), (0.25, 0.6), (0.3, 0.88), (0.55, 0.96), (0.72, 0.78), (0.68, 0.55)]],
    "c": [[(0.74, 0.5), (0.5, 0.42), (0.25, 0.6), (0.3, 0.88), (0.55, 0.96), (0.76, 0.84)]],
    "k": [[(0.2, 0.1), (0.2, 0.95)], [(0.66, 0.42), (0.24, 0.72)], [(0.38, 0.64), (0.72, 0.95)]],
    "e": [[(0.22, 0.7), (0.7, 0.68), (0.68, 0.5), (0.45, 0.42), (0.22, 0.58), (0.24, 0.84), (0.5, 0.96), (0.74, 0.86)]],
    "t": [[(0.45, 0.16), (0.44, 0.84), (0.58, 0.96), (0.76, 0.86)], [(0.24, 0.42), (0.72, 0.4)]],
    "F": [[(0.2, 0.1), (0.2, 0.95)], [(0.2, 0.1), (0.82, 0.1)], [(0.2, 0.5), (0.66, 0.5)]],
    "a": [[(0.66, 0.48), (0.42, 0.42), (0.22, 0.62), (0.26, 0.88), (0.5, 0.96), (0.68, 0.8)], [(0.68, 0.42), (0.7, 0.95)]],
    "b": [[(0.2, 0.08), (0.2, 0.95)], [(0.22, 0.58), (0.46, 0.42), (0.72, 0.56), (0.72, 0.82), (0.48, 0.96), (0.22, 0.86)]],
    "l": [[(0.26, 0.08), (0.24, 0.95)]],
    "5": [[(0.76, 0.1), (0.3, 0.1), (0.25, 0.5), (0.56, 0.46), (0.78, 0.66), (0.62, 0.95), (0.24, 0.9)]],
    ".": [[(0.18, 0.88), (0.26, 0.95)]],
    "1": [[(0.28, 0.28), (0.5, 0.1), (0.5, 0.95)]],
    " ": [],
}
GLYPH_W = {"i": 0.42, "l": 0.42, "t": 0.62, ".": 0.3, " ": 0.5, "T": 0.95, "R": 0.95, "F": 0.85, "1": 0.58, "y": 0.78, "c": 0.74, "e": 0.78, "a": 0.78, "n": 0.82, "o": 0.78, "k": 0.78, "b": 0.8, "5": 0.76}


def brush_stroke(cel, pts, w, color, seed, frac=1.0, a=1.0):
    """One brush stroke: loaded start, pressure swelling through the body, dry split ends at the lift."""
    pts = np.asarray(pts, np.float64)
    pts = resample(pts, max(w * 0.25, 1.5))
    n = len(pts)
    if n < 3:
        return
    k = max(2, int(n * frac))
    pts = pts[:k]
    n = k
    t = np.linspace(0, 1, n)
    g = B.rng(seed)
    prof = 1.0 + 0.3 * np.exp(-t / 0.08) - 0.25 * t  # loaded start, thinning as the ink runs out
    prof *= 1 + 0.12 * np.sin(t * np.pi * 1.0 + g.uniform(0, 1))  # a swell through the body
    prof *= 1 + 0.08 * np.array([R.vnoise(i * 0.15, seed * 0.1, seed) - 0.5 for i in range(n)]) * 2
    if frac >= 1.0:
        prof *= np.clip((1 - t) / 0.18, 0, 1) ** 0.5 * 0.85 + 0.15
    body = stroke_outline(pts, w * prof)
    R.fill_poly(cel.img, body, color, a, cel.alpha)
    # bristle tails: thin parallel sub strokes that lift at different lengths, the dry brush
    if frac >= 1.0 and n > 6:
        d = np.gradient(pts, axis=0)
        L = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        nx, ny = -d[:, 1] / L, d[:, 0] / L
        for b in range(5):
            off = g.uniform(-0.45, 0.45) * w
            start = int(n * g.uniform(0.55, 0.75))
            end = int(n * g.uniform(0.85, 1.0))
            if end - start < 3:
                continue
            sub = pts[start:end] + np.stack([nx[start:end] * off, ny[start:end] * off], axis=1)
            ww = w * g.uniform(0.12, 0.22) * np.linspace(1.0, 0.3, end - start)
            poly = stroke_outline(sub, ww)
            if poly is not None:
                R.fill_poly(cel.img, poly, color, a * g.uniform(0.6, 1.0), cel.alpha)
    # a darker pool where the brush first touched
    R.fill_poly(cel.img, ellipse(pts[0, 0], pts[0, 1], w * 0.55, w * 0.45, n=10), tuple(np.asarray(color) * 0.8), a * 0.5, cel.alpha)


def brush_text(cel, text, x, y, size, reveal=1.0, color=(0.12, 0.08, 0.08), seed=9, weight=0.11):
    """Brush lettering on a steady baseline, strokes revealed in writing order by reveal (0..1)."""
    strokes = []
    cx = x
    for ch in text:
        w = GLYPH_W.get(ch, 0.8) * size
        for st in GLYPHS.get(ch, []):
            strokes.append([(cx + px * size, y + py * size) for px, py in st])
        cx += w
    total = len(strokes)
    for i, st in enumerate(strokes):
        frac = np.clip(reveal * total - i, 0, 1)
        if frac <= 0:
            break
        pts = catmull(st, n=10) if len(st) > 2 else resample(np.array(st, np.float64), size * 0.05)
        pts = wobble(pts, size * 0.008, seed + i)
        brush_stroke(cel, pts, size * weight, color, seed + i, frac=frac)


def hanko(cel, x, y, size, press=1.0, seed=9):
    """Red seal reading 星, pressed like a stamp: stamp texture breaks the red."""
    g = B.rng(seed)
    sq = catmull([(x - size / 2, y - size / 2), (x + size / 2, y - size / 2), (x + size / 2, y + size / 2), (x - size / 2, y + size / 2)], closed=True, n=8)
    sq = wobble(sq, size * 0.02, seed, closed=True)
    m = cel.fill(sq, (0.78, 0.14, 0.12), 0.92 * press)
    if m is None:
        return
    # stamp breakup: paper grain shows through
    sub, x0, y0 = m
    H, W = sub.shape
    nz = R.noise_field(H, W, 3.5, seed + 3, 2)
    hole = np.clip((nz - 0.52) * 6, 0, 1)
    cel.alpha[y0:y0 + H, x0:x0 + W] *= (1 - hole * sub * 0.8)
    s_ = size
    cream = (0.96, 0.92, 0.82)
    w = s_ * 0.075
    def L(pts):
        brush_stroke(cel, [(x + px * s_, y + py * s_) for px, py in pts], w, cream, seed + int(abs(pts[0][0] * 100 + pts[0][1] * 37)), a=press)
    # 日 on top
    L([(-0.22, -0.4), (0.22, -0.4)])
    L([(-0.22, -0.4), (-0.22, -0.08)])
    L([(0.22, -0.4), (0.22, -0.08)])
    L([(-0.22, -0.24), (0.22, -0.24)])
    L([(-0.22, -0.08), (0.22, -0.08)])
    # 生 below
    L([(-0.18, 0.02), (-0.08, 0.1)])
    L([(-0.3, 0.1), (0.3, 0.1)])
    L([(-0.22, 0.26), (0.22, 0.26)])
    L([(0.0, 0.1), (0.0, 0.42)])
    L([(-0.34, 0.42), (0.34, 0.42)])
