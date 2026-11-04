"""Background painter: compose a reference, then paint it in gouache dabs."""

import numpy as np
from scipy.ndimage import gaussian_filter
from . import raster as R
from .geom import catmull, wobble, ellipse


def rng(seed):
    return np.random.default_rng(seed)


def jitter_colors(cols, g, val=0.06, hue=0.04):
    """Per dab colour variation: value shift plus independent channel wobble (hue drift)."""
    n = cols.shape[0]
    v = g.normal(0, val, (n, 1))
    h = g.normal(0, hue, (n, 3))
    return np.clip(cols + v + h, 0, 1)


def sample(ref, xs, ys):
    H, W = ref.shape[:2]
    xi = np.clip(xs.astype(np.int64), 0, W - 1)
    yi = np.clip(ys.astype(np.int64), 0, H - 1)
    return ref[yi, xi]


def grid_points(x0, y0, x1, y1, spacing, g, jit=0.6):
    xs = np.arange(x0, x1, spacing, dtype=np.float64)
    ys = np.arange(y0, y1, spacing, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    X = X.ravel() + g.uniform(-jit, jit, X.size) * spacing
    Y = Y.ravel() + g.uniform(-jit, jit, Y.size) * spacing
    return X, Y


def make_dabs(xs, ys, angles, lengths, widths, cols, alpha, hard=0.6):
    n = xs.shape[0]
    d = np.empty((n, 10), np.float64)
    d[:, 0] = xs
    d[:, 1] = ys
    d[:, 2] = angles
    d[:, 3] = lengths
    d[:, 4] = widths
    d[:, 5:8] = cols
    d[:, 8] = alpha
    d[:, 9] = hard
    return d


def paint_region(img, ref, mask, spacing, length, width, alpha, seed, angle=0.0, angle_jit=0.15,
                 val=0.06, hue=0.04, hard=0.6, len_jit=0.35, angle_field=None, darken_mask=None,
                 min_cov=0.5, region=None):
    """Lay oriented dabs on every grid point where mask > min_cov, colours sampled from ref."""
    g = rng(seed)
    H, W = img.shape[:2]
    if region is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = region
    X, Y = grid_points(x0, y0, x1, y1, spacing, g)
    if mask is not None:
        keep = sample(mask, X, Y) > min_cov
        X, Y = X[keep], Y[keep]
    n = X.shape[0]
    if n == 0:
        return
    cols = sample(ref, X, Y)
    cols = jitter_colors(cols, g, val, hue)
    if angle_field is not None:
        ang = sample(angle_field, X, Y) + g.normal(0, angle_jit, n)
    else:
        ang = angle + g.normal(0, angle_jit, n)
    L = length * (1 + g.uniform(-len_jit, len_jit, n))
    Wd = width * (1 + g.uniform(-0.3, 0.3, n))
    a = alpha * (1 + g.uniform(-0.25, 0.1, n))
    order = g.permutation(n)
    d = make_dabs(X[order], Y[order], ang[order], L[order], Wd[order], cols[order], a[order], hard)
    R.paint_dabs(img, d, seed)


def paper(H, W, seed, amount=0.05):
    """Paper grain to modulate paint, centred on 1."""
    n = R.noise_field(H, W, 3.0, seed, 3)
    n2 = R.noise_field(H, W, 60.0, seed + 5, 2)
    return 1.0 + (n - 0.5) * amount + (n2 - 0.5) * amount * 0.8


def vgrad(H, W, stops):
    """Vertical gradient from (t, rgb) stops, t in 0..1."""
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], np.float32)
    y = np.linspace(0, 1, H)
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(y, ts, cs[:, c])[:, None]
    return out


def mask_of(polys, W, H, blur=0.0):
    m = R.polys_mask(polys, W, H)
    if m is None:
        m = np.zeros((H, W), np.float32)
    if blur > 0:
        m = gaussian_filter(m, blur)
    return m


def blend(img, color, mask, a=1.0):
    m = (mask * a)[..., None]
    img[:] = img * (1 - m) + np.asarray(color, np.float32) * m


def soft_light(img, color, cx, cy, radius, strength, H, W, falloff=2.0):
    """Additive glow (sun below the horizon, lantern)."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    f = np.clip(1 - d, 0, 1) ** falloff * strength
    img[:] = img + np.asarray(color, np.float32) * f[..., None]


def foliage_clumps(img, ref, centers, radii, seed, base_col, light_col, light_dir=(-0.6, -0.8),
                   leaf=18, density=1.0, alpha=0.9):
    """Paint leaf clusters: each clump gets overlapping small dabs, lit side lighter."""
    g = rng(seed)
    allx, ally, ang, L, Wd, cols, al = [], [], [], [], [], [], []
    lx, ly = light_dir
    ln = np.hypot(lx, ly)
    lx, ly = lx / ln, ly / ln
    for (cx, cy), r in zip(centers, radii):
        n = int(density * r * r / (leaf * leaf) * 6) + 6
        a = g.uniform(0, 2 * np.pi, n)
        rr = r * np.sqrt(g.uniform(0, 1, n)) * (0.85 + 0.3 * g.uniform(0, 1, n))
        xs = cx + rr * np.cos(a)
        ys = cy + rr * np.sin(a) * 0.8
        # lit fraction: dot of offset with light direction
        lit = np.clip(((xs - cx) * lx + (ys - cy) * ly) / (r + 1e-6) * 0.5 + 0.5, 0, 1)
        lit = lit ** 1.6 * g.uniform(0.6, 1.2, n)
        c = np.asarray(base_col)[None] * (1 - lit[:, None]) + np.asarray(light_col)[None] * lit[:, None]
        c = jitter_colors(c, g, 0.05, 0.03)
        allx.append(xs)
        ally.append(ys)
        ang.append(g.normal(-0.4, 0.5, n))
        L.append(leaf * g.uniform(0.8, 1.5, n))
        Wd.append(leaf * 0.55 * g.uniform(0.7, 1.3, n))
        cols.append(c)
        al.append(alpha * g.uniform(0.7, 1.0, n))
    if not allx:
        return
    d = make_dabs(np.concatenate(allx), np.concatenate(ally), np.concatenate(ang), np.concatenate(L),
                  np.concatenate(Wd), np.concatenate(cols), np.concatenate(al), 0.9)
    R.paint_dabs(img, d, seed)


def grass_blades(img, xs, ys, heights, colors, seed, lean=0.0, width=4.0, alpha=0.9, curve=0.3):
    """Individual blades as short tapered dab chains."""
    g = rng(seed)
    allx, ally, ang, L, Wd, cols, al = [], [], [], [], [], [], []
    for i in range(xs.shape[0]):
        h = heights[i]
        segs = max(2, int(h / 8))
        ln = lean + g.normal(0, 0.25)
        bend = curve * g.normal(0, 1)
        for k in range(segs):
            t = (k + 0.5) / segs
            x = xs[i] + ln * t * h + bend * t * t * h * 0.5
            y = ys[i] - t * h
            allx.append(x)
            ally.append(y)
            ang.append(-np.pi / 2 + ln + bend * t)
            L.append(h / segs * 1.4)
            Wd.append(width * (1.1 - t))
            cols.append(colors[i])
            al.append(alpha)
    if not allx:
        return
    cols = np.array(cols)
    d = make_dabs(np.array(allx), np.array(ally), np.array(ang), np.array(L), np.array(Wd), cols,
                  np.array(al), 0.8)
    R.paint_dabs(img, d, seed)


def stars(img, n, seed, region, color=(1.0, 0.97, 0.9), rmin=1.5, rmax=4.0, strength=0.9, fade=None):
    g = rng(seed)
    x0, y0, x1, y1 = region
    xs = g.uniform(x0, x1, n)
    ys = g.uniform(y0, y1, n)
    r = g.uniform(rmin, rmax, n) ** 1.3
    s = strength * g.uniform(0.3, 1.0, n)
    if fade is not None:
        s = s * fade(xs, ys)
    gl = np.empty((n, 7), np.float64)
    gl[:, 0] = xs
    gl[:, 1] = ys
    gl[:, 2] = r * 2.2
    gl[:, 3:6] = np.asarray(color)[None] * (1 + g.normal(0, 0.04, (n, 3)))
    gl[:, 6] = s
    R.add_glows(img, gl)
