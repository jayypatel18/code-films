"""Deterministic noise for painting and animation.

Everything here is seeded from string or tuple keys, so a plate painted today
paints the same tomorrow and any frame renders the same on its own.

Two families:
- numpy field generators (value noise, fbm, ridged, warped) for whole images.
- numba scalar helpers (hash, 1D and 2D value noise) for use inside stroke
  kernels, where per pixel calls must be cheap.
"""

import zlib

import numpy as np
from numba import njit
from scipy import ndimage


def seed_of(*keys):
    """Stable 32 bit seed from any printable keys, e.g. seed_of('cut1', 'pine', 3)."""
    return zlib.crc32(repr(keys).encode("utf-8")) & 0x7FFFFFFF


def rng(*keys):
    return np.random.default_rng(seed_of(*keys))


# ---------------------------------------------------------------- numpy fields

def value_noise(h, w, cell, key, order=3):
    """Smooth value noise in roughly [0, 1] with features about `cell` px wide."""
    cell = max(float(cell), 1.0)
    gh = int(np.ceil(h / cell)) + 4
    gw = int(np.ceil(w / cell)) + 4
    g = rng("vn", key, gh, gw).random((gh, gw)).astype(np.float32)
    ys = (np.arange(h, dtype=np.float32) / cell + 1.5)
    xs = (np.arange(w, dtype=np.float32) / cell + 1.5)
    # separable cubic interpolation: interpolate rows then columns
    if order == 1:
        out = ndimage.map_coordinates(g, np.meshgrid(ys, xs, indexing="ij"), order=1)
    else:
        coef = ndimage.spline_filter(g, order=3)
        out = ndimage.map_coordinates(coef, np.meshgrid(ys, xs, indexing="ij"),
                                      order=3, prefilter=False)
    return out.astype(np.float32)


def fbm(h, w, cell, key, octaves=5, persistence=0.5, lacunarity=2.0):
    """Fractal sum of value noise, normalised to [0, 1]."""
    total = np.zeros((h, w), np.float32)
    amp = 1.0
    norm = 0.0
    c = float(cell)
    for o in range(octaves):
        if c < 1.0:
            break
        total += amp * value_noise(h, w, c, (key, o))
        norm += amp
        amp *= persistence
        c /= lacunarity
    total /= max(norm, 1e-6)
    lo, hi = np.percentile(total, 0.5), np.percentile(total, 99.5)
    return np.clip((total - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def ridged(h, w, cell, key, octaves=4):
    """Ridged noise: sharp creases, good for rock cracks and cloud seams."""
    n = fbm(h, w, cell, ("ridge", key), octaves=octaves)
    r = 1.0 - np.abs(n * 2.0 - 1.0)
    return (r * r).astype(np.float32)


def warp(field, dx, dy, order=1):
    """Sample field at (y + dy, x + dx)."""
    h, w = field.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = [yy + dy, xx + dx]
    if field.ndim == 2:
        return ndimage.map_coordinates(field, coords, order=order, mode="nearest").astype(np.float32)
    out = np.empty_like(field)
    for c in range(field.shape[2]):
        out[..., c] = ndimage.map_coordinates(field[..., c], coords, order=order, mode="nearest")
    return out


def paper_grain(h, w, key, fibre=True):
    """Fine paper or board texture in [0, 1], mean about 0.5.

    A mix of very fine tooth, a medium mottle, and faint fibres running in
    random directions. Used both to modulate the finished paint and inside the
    stroke kernel so dry brush catches only on the paper's high points.
    """
    tooth = value_noise(h, w, 1.3, ("tooth", key), order=1)
    tooth2 = value_noise(h, w, 2.7, ("tooth2", key))
    mottle = fbm(h, w, 24, ("mottle", key), octaves=3)
    g = 0.45 * tooth + 0.35 * tooth2 + 0.20 * mottle
    if fibre:
        r = rng("fibre", key, h, w)
        fib = np.zeros((h, w), np.float32)
        n = int(h * w / 900)
        ys = r.integers(0, h, n)
        xs = r.integers(0, w, n)
        fib[ys, xs] = r.random(n).astype(np.float32)
        ang = r.random() * np.pi
        k1 = ndimage.gaussian_filter(fib, (0.6, 5.0))
        k2 = ndimage.gaussian_filter(fib, (5.0, 0.6))
        fibres = k1 * abs(np.cos(ang)) + k2 * abs(np.sin(ang))
        fibres /= max(float(fibres.max()), 1e-6)
        g = g + 0.25 * fibres
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- numba scalars

@njit(cache=True, inline="always")
def hash_u32(x):
    x = (x ^ 61) ^ (x >> 16)
    x = (x + (x << 3)) & 0xFFFFFFFF
    x = x ^ (x >> 4)
    x = (x * 0x27D4EB2D) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    return x


@njit(cache=True, inline="always")
def hash1(i, seed):
    """Integer lattice to float in [0, 1)."""
    h = hash_u32((i * 374761393 + seed * 668265263) & 0xFFFFFFFF)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, inline="always")
def hash2(i, j, seed):
    h = hash_u32((i * 374761393 + j * 1274126177 + seed * 668265263) & 0xFFFFFFFF)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, inline="always")
def noise1(x, seed):
    """Smooth 1D value noise in [0, 1]."""
    fi = np.floor(x)
    i = np.int64(fi)
    f = x - fi
    u = f * f * (3.0 - 2.0 * f)
    a = hash1(i, seed)
    b = hash1(i + 1, seed)
    return a + (b - a) * u


@njit(cache=True, inline="always")
def noise2(x, y, seed):
    """Smooth 2D value noise in [0, 1]."""
    fx = np.floor(x)
    fy = np.floor(y)
    i = np.int64(fx)
    j = np.int64(fy)
    u = x - fx
    v = y - fy
    u = u * u * (3.0 - 2.0 * u)
    v = v * v * (3.0 - 2.0 * v)
    a = hash2(i, j, seed)
    b = hash2(i + 1, j, seed)
    c = hash2(i, j + 1, seed)
    d = hash2(i + 1, j + 1, seed)
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v
