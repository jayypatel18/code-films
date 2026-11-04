"""Shot 8: the world from very high at dawn. Layered ranges, cloud seas, rivers, the sun, the star."""

import numpy as np
from scipy.ndimage import gaussian_filter
from . import raster as R
from . import bg as B
from .geom import catmull, wobble, ellipse, stroke_outline

W2, H2 = 3840, 2160
SUN = (2560, 1290)


def arc_y(x, rc=11000.0, top=1180):
    """The curve of the world: horizon height at x."""
    return top + rc - np.sqrt(np.maximum(rc * rc - (x - 1920.0) ** 2, 0))


def ridge_line(seed, base, amp, freq, n=400, sharp=1.6):
    """A mountain ridge: y(x) across the width, peaks sharpened, following the world's curve."""
    xs = np.linspace(-200, W2 + 200, n)
    ys = np.empty(n)
    for i, x in enumerate(xs):
        v = R.fbm(x / freq, seed * 0.37, seed, 5)
        v2 = R.fbm(x / (freq * 0.23), seed * 0.11 + 3, seed + 9, 3)
        h = (v - 0.45) * 2.0
        h = np.sign(h) * abs(h) ** (1 / sharp) if h > 0 else h * 0.6
        ys[i] = base + arc_y(x) - 1180 - amp * h - amp * 0.25 * (v2 - 0.5)
    return xs, ys


def paint_range(img, xs, ys, col_dark, col_lit, col_haze, haze, seed, dab=28, snow=None, bottom=H2 + 100):
    """Fill a range below its ridge and paint it like the hill: a reference of lit faces, depth and haze,
    then oriented dabs sampled from it, then crisp crest highlights."""
    H, W = img.shape[:2]
    poly = np.concatenate([np.stack([xs, ys], axis=1), [[xs[-1], bottom], [xs[0], bottom]]], axis=0)
    mask = B.mask_of([poly], W, H)
    base = np.asarray(col_dark, np.float32) * (1 - haze) + np.asarray(col_haze, np.float32) * haze
    lit = np.asarray(col_lit, np.float32) * (1 - haze) + np.asarray(col_haze, np.float32) * haze
    # per column: ridge height and facing (descending to the right faces the low sun on the right)
    cols_x = np.arange(W)
    ridge = np.interp(cols_x, xs, ys)
    slope = np.interp(cols_x, xs, gaussian_filter(np.gradient(ys, xs), 3.0))
    facing = np.clip(slope * 1.4 + 0.15, -1, 1) * 0.5 + 0.5
    side = 1 - np.minimum(1.0, np.abs(cols_x - SUN[0]) / 2800.0)
    f_col = (facing ** 1.2 * 0.8 + 0.2 * side).astype(np.float32)
    yy = np.arange(H)[:, None].astype(np.float32)
    depth = np.clip((yy - ridge[None, :]) / 700.0, 0, 1)
    f = f_col[None, :] * (1 - depth) ** 0.7
    ref = base[None, None, :] * (1 - f[..., None]) + lit[None, None, :] * f[..., None]
    ref = ref * (1 - 0.35 * depth[..., None]) * (1 - 0.15 * haze)
    # low valleys pick up the warm haze
    ref = ref * (1 - depth[..., None] * haze * 0.5) + np.asarray(col_haze, np.float32)[None, None, :] * depth[..., None] * haze * 0.5
    ref = np.clip(ref, 0, 1).astype(np.float32)
    img[:] = img * (1 - mask[..., None]) + ref * mask[..., None]
    ang_field = np.tile(np.arctan2(slope, 1.0).astype(np.float32)[None, :], (H, 1))
    B.paint_region(img, ref, mask, dab * 1.4, dab * 3.2, dab * 1.1, 0.75, seed + 1, angle_field=ang_field, angle_jit=0.2, val=0.035, hue=0.02, hard=0.45)
    B.paint_region(img, ref, mask, dab * 0.6, dab * 1.4, dab * 0.5, 0.6, seed + 2, angle_field=ang_field, angle_jit=0.3, val=0.04, hue=0.025, hard=0.55)
    g = B.rng(seed)
    dy = gaussian_filter(np.gradient(ys, xs), 2.0)
    # crisp lit ridge line: short bright strokes on sun-facing crests
    allx, ally, ang, L, Wd, cols, al = [], [], [], [], [], [], []
    for i in range(2, len(xs) - 2, 2):
        sl = dy[i]
        if sl > 0.12 and g.uniform() < 0.8:
            fl = min(1.0, sl * 1.5)
            allx.append(xs[i])
            ally.append(ys[i] + 4)
            ang.append(np.arctan2(sl, 1.0))
            L.append(dab * g.uniform(1.0, 2.0))
            Wd.append(dab * g.uniform(0.18, 0.3))
            cols.append(np.clip(lit * (0.9 + 0.3 * fl) + np.array([0.08, 0.03, 0.0]) * fl, 0, 1))
            al.append(0.65 * (1 - haze * 0.7))
    if allx:
        d = B.make_dabs(np.array(allx), np.array(ally), np.array(ang), np.array(L), np.array(Wd), np.array(cols), np.array(al), 0.8)
        R.paint_dabs(img, d, seed + 3)
    if snow is not None:
        allx, ally, ang, L, Wd, cols, al = [], [], [], [], [], [], []
        for i in range(1, len(xs) - 1):
            if ys[i] < snow and g.uniform() < 0.9:
                for k in range(3):
                    allx.append(xs[i] + g.normal(0, 6))
                    ally.append(ys[i] + g.uniform(0, 40) * (snow - ys[i]) / 120)
                    ang.append(np.arctan2(dy[i], 1.0))
                    L.append(dab * 0.9)
                    Wd.append(dab * 0.3)
                    cols.append(np.array([0.98, 0.92, 0.88]) * (1 - haze * 0.5) + np.asarray(col_haze) * haze * 0.5)
                    al.append(0.75)
        if allx:
            d = B.make_dabs(np.array(allx), np.array(ally), np.array(ang), np.array(L), np.array(Wd), np.array(cols), np.array(al), 0.7)
            R.paint_dabs(img, d, seed + 4)


def cloud_sea(img, seed, y0, y1, n_masses, top_col, lit_col, shade_col, thick, alpha=0.9, sun_x=SUN[0]):
    """A band of cloud masses between y0 and y1: puffs lit on their sun side and top, shaded undersides."""
    g = B.rng(seed)
    allx, ally, ang, L, Wd, cols, al = [], [], [], [], [], [], []
    for k in range(n_masses):
        cx = g.uniform(-300, W2 + 300)
        cy = g.uniform(y0, y1)
        ln = g.uniform(500, 1600) * (0.5 + (cy - y0) / max(1, y1 - y0))
        hgt = thick * g.uniform(0.8, 1.4)
        npuff = int(ln / (hgt * 1.1)) + 2
        for i in range(npuff):
            t = (i + 0.5) / npuff
            env = np.sin(np.pi * t) ** 0.6
            px = cx - ln / 2 + t * ln + g.normal(0, hgt * 0.25)
            py = cy - hgt * 0.45 * env * g.uniform(0.3, 1.3)
            r = hgt * env * g.uniform(0.7, 1.15)
            n = int(r * r / 230) + 10
            a = g.uniform(0, 2 * np.pi, n)
            rr = r * np.sqrt(g.uniform(0, 1, n))
            dx = rr * np.cos(a)
            dyy = rr * np.sin(a) * 0.62
            toward_sun = np.clip((dx * np.sign(sun_x - px)) / (r + 1e-6) * 0.5 + 0.5, 0, 1)
            up = np.clip(-dyy / (r + 1e-6) * 0.5 + 0.5, 0, 1)
            lit = (0.55 * up + 0.45 * toward_sun) ** 1.5
            shade = np.clip(dyy / (r + 1e-6), 0, 1) ** 1.2
            c = np.asarray(top_col)[None] * (1 - lit[:, None]) + np.asarray(lit_col)[None] * lit[:, None]
            c = c * (1 - shade[:, None] * 0.35) + np.asarray(shade_col)[None] * shade[:, None] * 0.35
            allx.append(px + dx)
            ally.append(py + dyy)
            ang.append(g.normal(0, 0.15, n))
            L.append(g.uniform(0.6, 1.0, n) * r * 0.7)
            Wd.append(g.uniform(0.35, 0.7, n) * r * 0.7)
            cols.append(B.jitter_colors(c, g, 0.03, 0.02))
            al.append(g.uniform(0.45, 0.85, n) * alpha)
    if not allx:
        return
    n = sum(len(x) for x in allx)
    d = B.make_dabs(np.concatenate(allx), np.concatenate(ally), np.concatenate(ang), np.concatenate(L), np.concatenate(Wd), np.concatenate(cols), np.concatenate(al), 0.1)
    R.paint_dabs(img, d, seed + 5)


def rivers(img, seed, n=4, y_start=1420):
    """Glinting ribbons winding down the lowest valley: dark banks, bright water, soft glow."""
    g = B.rng(seed)
    for k in range(n):
        x0 = g.uniform(300, W2 - 300)
        pts = [(x0, y_start + g.uniform(0, 80))]
        x, y = pts[0]
        while y < H2 + 100:
            x += g.normal(0, 110) + (1920 - x0) * 0.015 + 70 * np.sin(y / 190.0 + k)
            y += g.uniform(80, 140)
            pts.append((x, y))
        path = catmull(pts, n=8)
        t = np.linspace(0, 1, len(path))
        wdt = 3 + 16 * t ** 1.3
        bank = stroke_outline(path, wdt * 2.2)
        R.fill_poly(img, bank, (0.34, 0.28, 0.42), 0.4)
        water = stroke_outline(path, wdt)
        R.fill_poly(img, water, (0.98, 0.84, 0.62), 0.8)
        # glints: short bright dabs along the water
        m = len(path) // 2
        idx = g.integers(0, len(path), m)
        d = np.gradient(path, axis=0)
        ang = np.arctan2(d[idx, 1], d[idx, 0])
        cols = B.jitter_colors(np.tile(np.array([1.0, 0.95, 0.80]), (m, 1)), g, 0.03, 0.02)
        dd = B.make_dabs(path[idx, 0] + g.normal(0, 3, m), path[idx, 1], ang, wdt[idx] * 1.6, wdt[idx] * 0.35, cols, g.uniform(0.5, 0.95, m), 0.8)
        R.paint_dabs(img, dd, seed + 10 + k)
        # glow of the water
        sel = path[::4]
        gl = np.empty((len(sel), 7))
        gl[:, 0] = sel[:, 0]
        gl[:, 1] = sel[:, 1]
        gl[:, 2] = wdt[::4] * 2.5 + 10
        gl[:, 3:6] = (1.0, 0.8, 0.5)
        gl[:, 6] = 0.18
        R.add_glows(img, gl)


def world_bg(seed=8):
    H, W = H2, W2
    img = B.vgrad(H, W, [(0.0, (0.03, 0.04, 0.16)), (0.22, (0.07, 0.10, 0.32)), (0.42, (0.24, 0.26, 0.56)), (0.52, (0.62, 0.42, 0.56)),
                         (0.57, (0.98, 0.66, 0.46)), (0.62, (1.0, 0.80, 0.52)), (1.0, (0.90, 0.78, 0.62))])
    B.soft_light(img, (0.40, 0.24, 0.06), SUN[0], SUN[1], 1900, 0.9, H, W, 2.2)
    ref = img.copy()
    B.paint_region(img, ref, None, 60, 280, 70, 0.7, seed + 1, angle=0.0, angle_jit=0.05, val=0.03, hue=0.02, hard=0.3, region=(0, 0, W, 1400))
    B.paint_region(img, ref, None, 26, 110, 30, 0.55, seed + 2, angle=0.0, angle_jit=0.07, val=0.025, hue=0.02, hard=0.4, region=(0, 0, W, 1400))
    B.stars(img, 420, seed + 12, (0, 0, W, 900), rmin=1.0, rmax=2.8, strength=1.0, fade=lambda xs, ys: np.clip(1 - ys / 850, 0, 1) ** 1.4)
    # high thin cloud streaks in the dawn band
    g = B.rng(seed + 3)
    for k in range(10):
        cy = g.uniform(1000, 1230)
        cx = g.uniform(0, W)
        ln = g.uniform(400, 1400)
        n = int(ln / 16)
        xs = cx - ln / 2 + np.linspace(0, ln, n)
        ys = cy + g.normal(0, 5, n)
        cols = B.jitter_colors(np.tile(np.array([0.92, 0.62, 0.62]), (n, 1)), g, 0.04, 0.02)
        R.paint_dabs(img, B.make_dabs(xs, ys, g.normal(0, 0.03, n), g.uniform(30, 60, n), g.uniform(8, 18, n), cols, g.uniform(0.3, 0.6, n), 0.3), seed + 20 + k)
    # the world: ranges from far to near, cloud seas between them
    haze_col = (0.96, 0.72, 0.56)
    # far: at the horizon, almost lost in light
    xs, ys = ridge_line(seed + 31, 1290, 90, 520)
    paint_range(img, xs, ys, (0.52, 0.40, 0.56), (0.98, 0.78, 0.62), haze_col, 0.65, seed + 32, dab=16, snow=1225)
    cloud_sea(img, seed + 40, 1330, 1380, 12, (0.80, 0.62, 0.66), (1.0, 0.86, 0.68), (0.60, 0.48, 0.62), 70, 0.85)
    xs, ys = ridge_line(seed + 33, 1420, 170, 430, sharp=1.9)
    paint_range(img, xs, ys, (0.34, 0.28, 0.48), (0.96, 0.70, 0.52), haze_col, 0.42, seed + 34, dab=22, snow=1290)
    rivers(img, seed + 50, n=5, y_start=1470)
    cloud_sea(img, seed + 41, 1480, 1560, 11, (0.76, 0.60, 0.68), (1.0, 0.86, 0.66), (0.52, 0.42, 0.60), 110, 0.9)
    xs, ys = ridge_line(seed + 35, 1620, 230, 360, sharp=2.0)
    paint_range(img, xs, ys, (0.24, 0.20, 0.38), (0.82, 0.58, 0.50), haze_col, 0.22, seed + 36, dab=28, snow=1420)
    cloud_sea(img, seed + 42, 1740, 1860, 8, (0.72, 0.58, 0.68), (1.0, 0.88, 0.70), (0.46, 0.38, 0.56), 150, 0.9)
    xs, ys = ridge_line(seed + 37, 1980, 300, 300, sharp=2.2)
    paint_range(img, xs, ys, (0.16, 0.13, 0.28), (0.72, 0.50, 0.46), haze_col, 0.08, seed + 38, dab=36)
    cloud_sea(img, seed + 43, 2080, 2220, 6, (0.66, 0.54, 0.66), (1.0, 0.90, 0.74), (0.40, 0.34, 0.52), 200, 0.9)
    # haze band at the horizon and the sun
    haze = np.clip(1 - np.abs(np.arange(H) - 1300) / 240.0, 0, 1)[:, None] ** 2 * 0.5
    img[:] = img * (1 - haze[..., None]) + np.array([1.0, 0.82, 0.58], np.float32) * haze[..., None]
    R.add_glows(img, np.array([[SUN[0], SUN[1] + 20, 620, 1.0, 0.78, 0.46, 0.28], [SUN[0], SUN[1] + 30, 130, 1.0, 0.95, 0.80, 0.9]]))
    # faint rays
    yy, xx = np.mgrid[0:H, 0:W]
    ang = np.arctan2(yy - SUN[1], xx - SUN[0])
    rays = (np.sin(ang * 14 + 1.0) * 0.5 + 0.5) ** 6 * np.clip(1 - np.hypot(xx - SUN[0], yy - SUN[1]) / 1500.0, 0, 1) ** 1.5 * (yy < 1320)
    img += np.array([1.0, 0.8, 0.5], np.float32) * (rays * 0.07)[..., None]
    img *= B.paper(H, W, seed + 13, 0.045)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# the star streak
# ----------------------------------------------------------------------------

def star_arc(u):
    x = 350 + 3300 * u
    y = 1500 - 1250 * np.sin(np.pi * min(max(u, 0), 1)) ** 0.8 + 150 * u
    return x, y


def star_streak(img, u, seed, head=1.0):
    """Sparkling head, coloured fringe, long tapering trail with sparks, light on the clouds below."""
    if u <= 0:
        return
    g = B.rng(seed)
    hx, hy = star_arc(u)
    # trail: dense glows along the arc behind the head, tapering, gold to rose to violet
    n = 140
    gl = []
    for i in range(n):
        k = i / (n - 1)
        uu = u - k * 0.5
        if uu < 0:
            continue
        x, y = star_arc(uu)
        s_ = (1 - k) ** 1.5
        col = np.array([1.0, 0.95, 0.80]) * (1 - k) + np.array([0.9, 0.45, 0.65]) * k
        gl.append((x + g.normal(0, 3), y + g.normal(0, 3), 18 + 70 * s_ * head, col[0], col[1], col[2], 1.2 * s_ + 0.05))
        if i % 2 == 0:
            # sparks drifting off the trail
            sx = x + g.normal(0, 60) * (0.3 + k)
            sy = y + g.normal(0, 60) * (0.3 + k) + 30 * k
            gl.append((sx, sy, 5 + 9 * g.uniform(), 1.0, 0.85, 0.5, g.uniform(0.3, 1.5) * s_))
    # fringe: warm orange halo, then rose outer
    gl.append((hx, hy, 300 * head, 1.0, 0.55, 0.30, 0.55))
    gl.append((hx, hy, 480 * head, 0.9, 0.40, 0.55, 0.22))
    # head: white gold core
    gl.append((hx, hy, 120 * head, 1.0, 0.92, 0.70, 2.2))
    gl.append((hx, hy, 46 * head, 1.0, 1.0, 0.95, 5.0))
    # light on the clouds beneath the head
    gl.append((hx, min(hy + 500, 1900), 700, 1.0, 0.7, 0.4, 0.22))
    R.add_glows(img, np.array(gl, np.float64))
    # sparkle: four thin flares through the head
    for k, a in enumerate([0.0, np.pi / 2, np.pi / 4, -np.pi / 4]):
        ln = (190 if k < 2 else 100) * head * (0.8 + 0.4 * g.uniform())
        pts = np.array([(hx - ln * np.cos(a), hy - ln * np.sin(a)), (hx, hy), (hx + ln * np.cos(a), hy + ln * np.sin(a))])
        wd = np.array([1.0, 12.0 * head, 1.0])
        R.fill_poly(img, stroke_outline(pts, wd), (1.0, 0.97, 0.9), 0.7)
