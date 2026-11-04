"""Low level rasterizer for TINY ROCKET. Written from scratch, numba hot loops.

Images are float32 arrays (H, W, 3) in linear-ish 0..1 display space.
Masks are float32 (h, w) coverage arrays with an offset (x0, y0).
"""

import numpy as np
from numba import njit

SUB = 5  # sub scanlines per pixel row for polygon anti-aliasing


# ----------------------------------------------------------------------------
# hashing and noise
# ----------------------------------------------------------------------------

@njit(cache=True, inline="always")
def hash2(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777215.0


@njit(cache=True, inline="always")
def smooth(t):
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def vnoise(x, y, seed):
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    fx = x - ix
    fy = y - iy
    a = hash2(ix, iy, seed)
    b = hash2(ix + 1, iy, seed)
    c = hash2(ix, iy + 1, seed)
    d = hash2(ix + 1, iy + 1, seed)
    sx = smooth(fx)
    sy = smooth(fy)
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy


@njit(cache=True)
def fbm(x, y, seed, octaves):
    v = 0.0
    amp = 0.5
    f = 1.0
    for o in range(octaves):
        v += amp * vnoise(x * f, y * f, seed + o * 17)
        amp *= 0.5
        f *= 2.03
    return v


@njit(cache=True)
def noise_field(h, w, scale, seed, octaves):
    out = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            out[y, x] = fbm(x / scale, y / scale, seed, octaves)
    return out


# ----------------------------------------------------------------------------
# polygon coverage (nonzero winding, anti aliased by sub scanlines)
# ----------------------------------------------------------------------------

@njit(cache=True)
def _poly_cov(px, py, x0, y0, w, h, out):
    n = px.shape[0]
    xs = np.empty(n, np.float64)
    ds = np.empty(n, np.int32)
    inv = 1.0 / SUB
    for row in range(h):
        for s in range(SUB):
            sy = y0 + row + (s + 0.5) * inv
            m = 0
            for i in range(n):
                j = i + 1
                if j == n:
                    j = 0
                ya = py[i]
                yb = py[j]
                if ya == yb:
                    continue
                if (sy >= ya and sy < yb) or (sy >= yb and sy < ya):
                    t = (sy - ya) / (yb - ya)
                    xs[m] = px[i] + t * (px[j] - px[i])
                    ds[m] = 1 if yb > ya else -1
                    m += 1
            if m < 2:
                continue
            # insertion sort crossings
            for i in range(1, m):
                kx = xs[i]
                kd = ds[i]
                j = i - 1
                while j >= 0 and xs[j] > kx:
                    xs[j + 1] = xs[j]
                    ds[j + 1] = ds[j]
                    j -= 1
                xs[j + 1] = kx
                ds[j + 1] = kd
            wind = 0
            for i in range(m - 1):
                wind += ds[i]
                if wind != 0:
                    xa = xs[i] - x0
                    xb = xs[i + 1] - x0
                    if xb <= 0 or xa >= w:
                        continue
                    if xa < 0:
                        xa = 0.0
                    if xb > w:
                        xb = float(w)
                    ia = int(xa)
                    ib = int(xb)
                    if ib >= w:
                        ib = w - 1
                    if ia == ib:
                        out[row, ia] += (xb - xa) * inv
                    else:
                        out[row, ia] += (ia + 1 - xa) * inv
                        for k in range(ia + 1, ib):
                            out[row, k] += inv
                        out[row, ib] += (xb - ib) * inv
    for row in range(h):
        for col in range(w):
            if out[row, col] > 1.0:
                out[row, col] = 1.0


def poly_mask(pts, W, H):
    """Coverage mask for a closed polygon. Returns (mask, x0, y0) or None."""
    pts = np.asarray(pts, np.float64)
    if pts.shape[0] < 3:
        return None
    x0 = int(np.floor(pts[:, 0].min())) - 1
    y0 = int(np.floor(pts[:, 1].min())) - 1
    x1 = int(np.ceil(pts[:, 0].max())) + 2
    y1 = int(np.ceil(pts[:, 1].max())) + 2
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, W)
    y1 = min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    _poly_cov(np.ascontiguousarray(pts[:, 0]), np.ascontiguousarray(pts[:, 1]),
              float(x0), float(y0), x1 - x0, y1 - y0, out)
    return out, x0, y0


def polys_mask(polys, W, H):
    """Union coverage of several polygons (max blend)."""
    res = None
    for p in polys:
        m = poly_mask(p, W, H)
        if m is None:
            continue
        if res is None:
            res = full_mask_from(m, W, H)
        else:
            sub, x0, y0 = m
            h, w = sub.shape
            np.maximum(res[y0:y0 + h, x0:x0 + w], sub, out=res[y0:y0 + h, x0:x0 + w])
    return res


def full_mask_from(m, W, H):
    full = np.zeros((H, W), np.float32)
    sub, x0, y0 = m
    h, w = sub.shape
    full[y0:y0 + h, x0:x0 + w] = sub
    return full


# ----------------------------------------------------------------------------
# compositing
# ----------------------------------------------------------------------------

@njit(cache=True)
def _blend_sub(img, alpha, sub, x0, y0, r, g, b, a, write_alpha):
    h, w = sub.shape
    for yy in range(h):
        for xx in range(w):
            c = sub[yy, xx] * a
            if c <= 0.0:
                continue
            y = y0 + yy
            x = x0 + xx
            img[y, x, 0] = img[y, x, 0] * (1 - c) + r * c
            img[y, x, 1] = img[y, x, 1] * (1 - c) + g * c
            img[y, x, 2] = img[y, x, 2] * (1 - c) + b * c
            if write_alpha:
                alpha[y, x] = alpha[y, x] + c - alpha[y, x] * c


def fill_mask(img, m, color, a=1.0, alpha=None):
    if m is None:
        return
    sub, x0, y0 = m
    wa = alpha is not None
    if not wa:
        alpha = np.zeros((1, 1), np.float32)
    _blend_sub(img, alpha, sub, x0, y0, float(color[0]), float(color[1]), float(color[2]), float(a), wa)


def fill_poly(img, pts, color, a=1.0, alpha=None):
    m = poly_mask(pts, img.shape[1], img.shape[0])
    fill_mask(img, m, color, a, alpha)
    return m


def fill_full(img, full, color, a=1.0, alpha=None):
    """Blend using a full size mask."""
    ys, xs = np.nonzero(full > 0.001)
    if ys.size == 0:
        return
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    fill_mask(img, (np.ascontiguousarray(full[y0:y1, x0:x1]), x0, y0), color, a, alpha)


def clip_mask(m, full):
    """Multiply a sub mask by a full size mask (intersection)."""
    if m is None:
        return None
    sub, x0, y0 = m
    h, w = sub.shape
    return sub * full[y0:y0 + h, x0:x0 + w], x0, y0


def shift_full(full, dx, dy):
    """Translate a full mask by integer pixels, zero fill."""
    out = np.zeros_like(full)
    H, W = full.shape
    dx = int(round(dx))
    dy = int(round(dy))
    sx0 = max(0, -dx)
    sx1 = min(W, W - dx)
    sy0 = max(0, -dy)
    sy1 = min(H, H - dy)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = full[sy0:sy1, sx0:sx1]
    return out


# ----------------------------------------------------------------------------
# gouache dabs
# ----------------------------------------------------------------------------

@njit(cache=True)
def paint_dabs(img, dabs, seed):
    """dabs: (N, 10) rows of x, y, angle, length, width, r, g, b, alpha, hardness.
    A dab is an oriented elliptical stamp with bristle streaks and a broken tip."""
    H = img.shape[0]
    W = img.shape[1]
    n = dabs.shape[0]
    for i in range(n):
        x = dabs[i, 0]
        y = dabs[i, 1]
        ang = dabs[i, 2]
        L = dabs[i, 3] * 0.5
        Wd = dabs[i, 4] * 0.5
        r = dabs[i, 5]
        g = dabs[i, 6]
        b = dabs[i, 7]
        a = dabs[i, 8]
        hard = dabs[i, 9]
        if L < 0.5:
            L = 0.5
        if Wd < 0.5:
            Wd = 0.5
        ca = np.cos(ang)
        sa = np.sin(ang)
        ext = np.sqrt(L * L + Wd * Wd) + 1.0
        x0 = int(x - ext)
        x1 = int(x + ext) + 1
        y0 = int(y - ext)
        y1 = int(y + ext) + 1
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > W:
            x1 = W
        if y1 > H:
            y1 = H
        ds = seed + i * 7919
        # per dab colour variation along the bristles
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                dx = xx + 0.5 - x
                dy = yy + 0.5 - y
                u = (dx * ca + dy * sa) / L   # along stroke
                v = (-dx * sa + dy * ca) / Wd  # across stroke
                d = u * u + v * v
                if d > 1.0:
                    continue
                # bristle streaks across the width, breakup along the length
                br = vnoise(v * Wd * 0.45 + 3.1, u * L * 0.08, ds)
                tip = vnoise(u * L * 0.25, v * Wd * 0.25, ds + 1)
                edge = 1.0 - d
                e = edge * (hard + 2.5) if edge * (hard + 2.5) < 1.0 else 1.0
                cov = e * (0.55 + 0.45 * br) * (0.7 + 0.5 * tip)
                # break the ends of the stroke
                if abs(u) > 0.6:
                    cov *= 1.0 - (abs(u) - 0.6) * 1.6 * (0.3 + tip)
                if cov <= 0.0:
                    continue
                cov *= a
                if cov > 1.0:
                    cov = 1.0
                # paint pools a little darker where bristles are sparse
                k = 0.93 + 0.14 * br
                img[yy, xx, 0] = img[yy, xx, 0] * (1 - cov) + r * k * cov
                img[yy, xx, 1] = img[yy, xx, 1] * (1 - cov) + g * k * cov
                img[yy, xx, 2] = img[yy, xx, 2] * (1 - cov) + b * k * cov


@njit(cache=True)
def paint_dabs_alpha(img, alpha, dabs, seed):
    """Same as paint_dabs but also accumulates alpha (for painted elements on a layer)."""
    H = img.shape[0]
    W = img.shape[1]
    n = dabs.shape[0]
    for i in range(n):
        x = dabs[i, 0]
        y = dabs[i, 1]
        ang = dabs[i, 2]
        L = dabs[i, 3] * 0.5
        Wd = dabs[i, 4] * 0.5
        r = dabs[i, 5]
        g = dabs[i, 6]
        b = dabs[i, 7]
        a = dabs[i, 8]
        hard = dabs[i, 9]
        if L < 0.5:
            L = 0.5
        if Wd < 0.5:
            Wd = 0.5
        ca = np.cos(ang)
        sa = np.sin(ang)
        ext = np.sqrt(L * L + Wd * Wd) + 1.0
        x0 = int(x - ext)
        x1 = int(x + ext) + 1
        y0 = int(y - ext)
        y1 = int(y + ext) + 1
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > W:
            x1 = W
        if y1 > H:
            y1 = H
        ds = seed + i * 7919
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                dx = xx + 0.5 - x
                dy = yy + 0.5 - y
                u = (dx * ca + dy * sa) / L
                v = (-dx * sa + dy * ca) / Wd
                d = u * u + v * v
                if d > 1.0:
                    continue
                br = vnoise(v * Wd * 0.45 + 3.1, u * L * 0.08, ds)
                tip = vnoise(u * L * 0.25, v * Wd * 0.25, ds + 1)
                edge = 1.0 - d
                e = edge * (hard + 2.5) if edge * (hard + 2.5) < 1.0 else 1.0
                cov = e * (0.55 + 0.45 * br) * (0.7 + 0.5 * tip)
                if abs(u) > 0.6:
                    cov *= 1.0 - (abs(u) - 0.6) * 1.6 * (0.3 + tip)
                if cov <= 0.0:
                    continue
                cov *= a
                if cov > 1.0:
                    cov = 1.0
                k = 0.93 + 0.14 * br
                img[yy, xx, 0] = img[yy, xx, 0] * (1 - cov) + r * k * cov
                img[yy, xx, 1] = img[yy, xx, 1] * (1 - cov) + g * k * cov
                img[yy, xx, 2] = img[yy, xx, 2] * (1 - cov) + b * k * cov
                alpha[yy, xx] = alpha[yy, xx] + cov - alpha[yy, xx] * cov


# ----------------------------------------------------------------------------
# soft round glows (stars, lights, fireflies, bloom sources)
# ----------------------------------------------------------------------------

@njit(cache=True)
def add_glows(img, glows):
    """glows: (N, 7) rows x, y, radius, r, g, b, strength. Additive soft discs."""
    H = img.shape[0]
    W = img.shape[1]
    for i in range(glows.shape[0]):
        x = glows[i, 0]
        y = glows[i, 1]
        rad = glows[i, 2]
        r = glows[i, 3]
        g = glows[i, 4]
        b = glows[i, 5]
        s = glows[i, 6]
        x0 = int(x - rad) - 1
        x1 = int(x + rad) + 2
        y0 = int(y - rad) - 1
        y1 = int(y + rad) + 2
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > W:
            x1 = W
        if y1 > H:
            y1 = H
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                dx = xx + 0.5 - x
                dy = yy + 0.5 - y
                d = np.sqrt(dx * dx + dy * dy) / rad
                if d >= 1.0:
                    continue
                f = (1.0 - d)
                f = f * f * s
                img[yy, xx, 0] += r * f
                img[yy, xx, 1] += g * f
                img[yy, xx, 2] += b * f


# ----------------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------------

def downsample(img, k=2):
    H, W = img.shape[0] // k * k, img.shape[1] // k * k
    a = img[:H, :W]
    if a.ndim == 3:
        return a.reshape(H // k, k, W // k, k, a.shape[2]).mean(axis=(1, 3))
    return a.reshape(H // k, k, W // k, k).mean(axis=(1, 3))


@njit(cache=True)
def grain_dither(img, seed, amount):
    """Fine film grain plus dither, returns uint8."""
    H = img.shape[0]
    W = img.shape[1]
    out = np.empty((H, W, 3), np.uint8)
    for y in range(H):
        for x in range(W):
            g = (hash2(x, y, seed) - 0.5) * amount
            for c in range(3):
                v = img[y, x, c] + g + (hash2(x, y, seed + 11 + c) - 0.5) / 255.0
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                out[y, x, c] = int(v * 255.0 + 0.5)
    return out
