"""Tubes: swept spheres along a 3D centre line (legs, tails, necks).

The silhouette is the 2D envelope of the projected circles; per pixel we find
the nearest centre-line point to get depth, normal and the parameter along the
tube (used for socks, tail tips and fur patterns).
"""

import math

import numpy as np
from numba import njit

from . import geom


@njit(cache=True)
def _nearest(xs, ys, C, R, S):
    n = xs.shape[0]
    m = C.shape[0]
    z = np.empty(n)
    nx = np.empty(n)
    ny = np.empty(n)
    nz = np.empty(n)
    sp = np.empty(n)
    side = np.empty(n)
    for i in range(n):
        px = xs[i]
        py = ys[i]
        best = 1e30
        bz = 0.0
        bdx = 0.0
        bdy = 0.0
        br = 1.0
        bs = 0.0
        for k in range(m - 1):
            ax = C[k, 0]
            ay = C[k, 1]
            vx = C[k + 1, 0] - ax
            vy = C[k + 1, 1] - ay
            vv = vx * vx + vy * vy
            t = 0.0
            if vv > 1e-12:
                t = ((px - ax) * vx + (py - ay) * vy) / vv
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            qx = ax + vx * t
            qy = ay + vy * t
            r = R[k] + (R[k + 1] - R[k]) * t
            dx = px - qx
            dy = py - qy
            d = math.sqrt(dx * dx + dy * dy)
            sd = d - r
            if sd < best:
                best = sd
                bz = C[k, 2] + (C[k + 1, 2] - C[k, 2]) * t
                bdx = dx
                bdy = dy
                br = r
                bs = S[k] + (S[k + 1] - S[k]) * t
                # which side of the centre line (for top/bottom patterns)
                side[i] = (vx * dy - vy * dx) / (math.sqrt(vv) + 1e-12) / (r + 1e-12)
        h2 = br * br - bdx * bdx - bdy * bdy
        h = math.sqrt(h2) if h2 > 0 else 0.0
        z[i] = bz + h
        inv = 1.0 / (br + 1e-12)
        nx[i] = bdx * inv
        ny[i] = bdy * inv
        nz[i] = h * inv
        sp[i] = bs
    return z, nx, ny, nz, sp, side


class Tube:
    """Centre line P (n, 3) in a local space with radii r (n,)."""

    def __init__(self, P, r, samples=8):
        P = np.asarray(P, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        if len(P) >= 3 and samples > 1:
            Pd = geom.catmull_rom(np.hstack([P, r[:, None]]), samples)
            self.P, self.r = Pd[:, :3], np.maximum(Pd[:, 3], 1e-4)
        else:
            self.P, self.r = P, r

    def project(self, X):
        return ProjTube(self, X)


class ProjTube:
    def __init__(self, tube, X):
        self.src = tube
        C = X.apply(tube.P)
        scale = float(np.cbrt(abs(np.linalg.det(X.A))))
        self.C = C
        self.R = tube.r * scale
        s = geom.arclen(C[:, :2])
        s3 = geom.arclen(tube.P)
        self.S = s3 / max(s3[-1], 1e-9)   # parameter 0..1 along the tube (3D arc length)

    def outline(self, cap_samples=10):
        return geom.tube_outline(self.C[:, :2], self.R, cap_samples)

    def surface(self, xs, ys):
        xs = np.ascontiguousarray(xs, dtype=np.float64)
        ys = np.ascontiguousarray(ys, dtype=np.float64)
        z, nx, ny, nz, sp, side = _nearest(xs, ys, np.ascontiguousarray(self.C), np.ascontiguousarray(self.R),
                                           np.ascontiguousarray(self.S))
        return z, [nx, ny, nz], sp, side

    def depth(self):
        return float(np.mean(self.C[:, 2]))
