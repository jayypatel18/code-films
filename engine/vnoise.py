"""Small deterministic value noise for the drawing side (A2 owns engine/noise.py).

Used to wobble ink lines, break up shadow terminators and boil drawings. Noise
is a pure function of coordinates and an integer seed, so any frame can be
rendered alone and comes out the same.
"""

import numpy as np


def _hash(ix, iy, iz, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + iz.astype(np.int64) * 2147483647 + np.int64(seed) * 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def _fade(t):
    return t * t * (3.0 - 2.0 * t)


def value3(x, y, z, seed=0):
    """Value noise in [-1, 1] at float arrays x, y, z."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64) + np.zeros_like(x)
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = _fade(x - xi), _fade(y - yi), _fade(z - zi)
    xi, yi, zi = xi.astype(np.int64), yi.astype(np.int64), zi.astype(np.int64)
    acc = 0.0
    for dz in (0, 1):
        wz = fz if dz else 1 - fz
        for dy in (0, 1):
            wy = fy if dy else 1 - fy
            for dx in (0, 1):
                wx = fx if dx else 1 - fx
                acc = acc + wx * wy * wz * _hash(xi + dx, yi + dy, zi + dz, seed)
    return acc * 2.0 - 1.0


def value1(t, seed=0):
    t = np.asarray(t, dtype=np.float64)
    return value3(t, np.zeros_like(t), np.zeros_like(t), seed)


def fbm3(x, y, z, seed=0, octaves=3, lacunarity=2.0, gain=0.5):
    amp, freq, tot, norm = 1.0, 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * value3(x * freq, y * freq, z * freq, seed + 101 * o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return tot / norm


def rng(*keys):
    """numpy Generator seeded from a tuple of ints or strings."""
    acc = 1469598103934665603
    for k in keys:
        for ch in str(k).encode():
            acc = ((acc ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        acc = ((acc ^ 0x9E) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(acc)
