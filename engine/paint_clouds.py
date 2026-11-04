"""Cloud seas for painted references.

A sea of cumulus is built the way a layout artist would block it in 3D and
then throw the 3D away: heaps of spheres on a plane, seen through a pinhole
camera, z buffered, and shaded with a soft terminator, crevice darkening,
height falloff and atmospheric haze. The result is only a reference: the
stroke painter repaints it so no sphere survives as a sphere.
"""

import math

import numpy as np
from numba import njit
from scipy import ndimage

from engine.noise import noise2, rng


def heap_field(key, f, horizon_y, cx_screen, cam_h, z_near, z_far, half_w_px,
               row_step=0.11, spacing=0.34, base_r=0.11, tall=1.0, height_cell=1.6):
    """Spheres (X, Y, Z, R, heap id) for a cumulus sea seen from a camera at
    height cam_h above the mean cloud top plane, looking at the horizon.

    f: focal length in px. half_w_px: half the plate width plus margins.
    tall scales how high heaps tower. height_cell: world size of heap height
    variation (bigger means broader swells)."""
    r = rng("heaps", key)
    out = []
    Z = z_near
    heap = 0
    hseed = int(r.integers(1, 1 << 20))
    while Z < z_far:
        xmax = (half_w_px + 260) * Z / f
        sp = spacing * (1.0 + 0.012 * Z)
        X = -xmax + r.random() * sp
        while X < xmax:
            Zc = Z * (1.0 + (r.random() - 0.5) * row_step * 1.1)
            h = noise2(X / height_cell, Zc / height_cell, hseed)
            h2 = noise2(X / (height_cell * 0.37), Zc / (height_cell * 0.37), hseed + 7)
            hh = max(0.0, min(1.0, 0.75 * h + 0.45 * h2 - 0.15))
            R0 = base_r * (0.75 + 0.6 * r.random()) * (0.8 + 0.6 * hh)
            detail = 1.0 if Z < 14 else (0.5 if Z < 28 else 0.25)
            # base lobes
            nb = 2 + int(r.integers(0, 3))
            for i in range(nb):
                out.append((X + (r.random() - 0.5) * R0 * 3.0, -R0 * 0.25 + (r.random() - 0.5) * R0 * 0.3,
                            Zc + (r.random() - 0.5) * R0 * 1.6, R0 * (0.7 + 0.35 * r.random()), heap))
            # the heap rising
            top = R0 * (0.2 + 2.6 * hh * tall)
            ns = int(1 + 5 * hh * detail + r.integers(0, 2))
            for i in range(ns):
                t = (i + 1) / (ns + 1)
                rr = R0 * (0.95 - 0.45 * t) * (0.8 + 0.4 * r.random())
                out.append((X + (r.random() - 0.5) * R0 * (1.6 - t), top * t, Zc + (r.random() - 0.5) * R0 * 0.8,
                            rr, heap))
            # cauliflower bumps on the upper surface
            nbumps = int((4 + 9 * hh) * detail)
            for i in range(nbumps):
                a = math.pi * (0.08 + 0.84 * r.random())  # angle from +x across the top
                pr = R0 * (0.75 + 0.3 * r.random())
                yb = top * (0.25 + 0.75 * r.random())
                rb = R0 * (0.22 + 0.28 * r.random())
                out.append((X + math.cos(a) * pr, yb + math.sin(a) * pr * 0.75,
                            Zc - (0.2 + 0.6 * r.random()) * pr * 0.6, rb, heap))
            heap += 1
            X += sp * (0.65 + 0.7 * r.random())
        Z *= 1.0 + row_step
    return np.array(out, np.float64)


@njit(cache=True, fastmath=True)
def _zbuffer(Hh, Ww, sx, sy, sr, sq, sz, sR, sY, occ, zbuf, nxb, nyb, nzb, ybuf, idb):
    n = sx.shape[0]
    for i in range(n):
        r = sr[i]
        q = sq[i]
        if r < 0.4:
            continue
        ry = r * q
        x0 = int(max(0.0, math.floor(sx[i] - r)))
        x1 = int(min(Ww - 1.0, math.ceil(sx[i] + r)))
        y0 = int(max(0.0, math.floor(sy[i] - ry)))
        y1 = int(min(Hh - 1.0, math.ceil(sy[i] + ry)))
        if x0 > x1 or y0 > y1:
            continue
        inv = 1.0 / r
        invy = 1.0 / max(ry, 0.3)
        for y in range(y0, y1 + 1):
            dy = (y - sy[i]) * invy
            for x in range(x0, x1 + 1):
                dx = (x - sx[i]) * inv
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                nz = math.sqrt(1.0 - d2)
                depth = sz[i] - nz * sR[i]
                if depth < zbuf[y, x] and depth < occ[y, x]:
                    zbuf[y, x] = depth
                    gx = dx
                    gy = dy / q
                    ln = math.sqrt(gx * gx + gy * gy + nz * nz)
                    nxb[y, x] = gx / ln
                    nyb[y, x] = gy / ln
                    nzb[y, x] = nz / ln
                    ybuf[y, x] = sY[i] - dy * sR[i] * q
                    idb[y, x] = i


def render_heaps(spheres, h, w, f, horizon_y, cx_screen, cam_h, occ=None, squash=0.62, key="sq"):
    """Z buffer the spheres onto an (h, w) image. Coordinates are array
    coordinates (margins included). Returns a dict of per pixel buffers."""
    X, Y, Z, R = spheres[:, 0], spheres[:, 1], spheres[:, 2], spheres[:, 3]
    ok = Z > 0.05
    X, Y, Z, R = X[ok], Y[ok], Z[ok], R[ok]
    sq = (squash * (0.8 + 0.4 * rng("squash", key).random(len(X)))).astype(np.float64)
    sx = cx_screen + f * X / Z
    sy = horizon_y + f * (cam_h - Y) / Z
    sr = f * R / Z
    zbuf = np.full((h, w), 1e9, np.float64)
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)
    nz = np.zeros((h, w), np.float32)
    yb = np.zeros((h, w), np.float32)
    idb = np.full((h, w), -1, np.int64)
    if occ is None:
        occ = np.full((h, w), 1e10, np.float64)
    _zbuffer(h, w, sx, sy, sr, sq, Z, R, Y, occ.astype(np.float64), zbuf, nx, ny, nz, yb, idb)
    cover = (idb >= 0).astype(np.float32)
    return dict(depth=zbuf, nx=nx, ny=ny, nz=nz, y=yb, id=idb, cover=cover,
                sx=sx, sy=sy, sr=sr, sq=sq, sz=Z, sR=R, sY=Y)


def crevices(depth, cover, sigma):
    """Positive where a pixel sits deeper than its neighbourhood (in inverse
    depth), i.e. in the folds between billows."""
    inv = np.where(cover > 0, 1.0 / np.maximum(depth, 1e-3), 0.0)
    wsum = ndimage.gaussian_filter(cover, sigma) + 1e-6
    avg = ndimage.gaussian_filter(inv * cover, sigma) / wsum
    return np.clip((avg - inv) / np.maximum(inv, 1e-6), 0, 4).astype(np.float32)


# ---------------------------------------------------------------- banded seas

def band_sea(key, h, w, f, horizon_y, cam_h, z_near, z_far, row_step=0.06, base_r=0.10,
             heap_rate=0.35, heap_height=1.6, flat=0.55, heap_width=10.0, calm=None, near_step=1.0,
             near_size=0.0, near_z=3.0):
    """A cloud sea as overlapping rows, painted far to near the way a
    background painter lays in a sea of clouds. Each row is a continuous body
    with lobes (full ellipses) along its top; lobes riding higher are drawn
    first so lower, nearer lobes overlap them with curved seams.

    Returns per pixel buffers: band index, normal (nx, ny, nz), depth into the
    body in lobe units (dn, 0 on lobes), row depth Z and coverage."""
    r = rng("bands", key)
    Zs = []
    Z = z_far
    while Z > z_near:
        Zs.append(Z)
        # near rows are fewer and bigger: large billowing masses in the foreground
        boost = 1.0 + (near_step - 1.0) * math.exp(-Z / near_z)
        Z /= (1.0 + row_step * boost * (0.7 + 0.6 * r.random()))
    nb = len(Zs)
    xs = np.arange(w, dtype=np.float64)
    lob = []          # (band, x, y, rx, ry)
    body = np.full((nb, w), 1e9)
    S = np.zeros(nb)
    lifts = []
    for k, Z in enumerate(Zs):
        s = f * base_r / Z
        nb_ = 1.0 + near_size * math.exp(-Z / near_z)
        S[k] = s
        yk = horizon_y + f * cam_h / Z
        cell = s * heap_width * (0.7 + 0.6 * r.random()) * nb_
        env = np.zeros(w)
        x = -r.random() * cell
        while x < w + cell:
            if r.random() < heap_rate:
                hw = cell * (0.45 + 0.6 * r.random())
                hh = s * heap_height * (0.25 + r.random() ** 1.6) * nb_
                u = (xs - x) / hw
                env = np.maximum(env, hh * np.clip(1 - u * u, 0, 1) ** 0.8)
            x += cell * (0.55 + 0.8 * r.random())
        if calm is not None:
            env = env * calm(xs, Z)
        row = []
        x = -s * 2 * r.random()
        while x < w + s * 2:
            xi = int(min(max(x, 0), w - 1))
            e = env[xi]
            # lobes get smaller as they climb a heap
            bw = s * (0.65 + 0.8 * r.random()) * (1.0 - 0.35 * min(e / (s * heap_height + 1e-6), 1.0))
            bh = bw * flat * (0.75 + 0.5 * r.random())
            cy = yk - e * (0.85 + 0.25 * r.random()) + (r.random() - 0.3) * bh * 0.5
            row.append((k, x, cy, bw, bh, min(e / (s * heap_height * 0.6 + 1e-6), 1.0)))
            # extra lobes to round the shoulders of tall heaps
            if e > s * 0.9:
                for j in range(int(r.integers(1, 3))):
                    bw2 = bw * (0.45 + 0.35 * r.random())
                    row.append((k, x + (r.random() - 0.5) * bw * 1.2,
                                cy - bh * (0.4 + 0.5 * r.random()), bw2, bw2 * flat * (0.8 + 0.4 * r.random()),
                                min(e / (s * heap_height * 0.6 + 1e-6), 1.0) + 0.15))
            # cauliflower: small lobes budding along the upper outline
            if bw > 3.0:
                for j in range(int(r.integers(1, 5))):
                    a = math.pi * (0.12 + 0.76 * r.random())
                    sb = bw * (0.18 + 0.3 * r.random() ** 1.5)
                    row.append((k, x - math.cos(a) * bw * 0.82, cy - math.sin(a) * bh * 0.82,
                                sb, sb * flat * (0.9 + 0.4 * r.random()), min(e / (s * heap_height * 0.6 + 1e-6), 1.0) + 0.1))
            x += bw * (0.6 + 0.55 * r.random())
        # body top: a smoothed envelope through the lobe centres
        bt = np.full(w, yk + s * 0.2)
        for (_, lx, ly, rx, ry, _lf) in row:
            lo = max(int(lx - rx * 0.8), 0)
            hi = min(int(lx + rx * 0.8) + 1, w)
            if hi > lo:
                bt[lo:hi] = np.minimum(bt[lo:hi], ly + ry * 0.25)
        bt = ndimage.gaussian_filter1d(bt, max(s * 0.6, 1.0), mode="nearest")
        body[k] = bt
        row.sort(key=lambda t: t[2])   # higher lobes first, lower ones overlap them
        lob.extend(row)
    LL = np.array(lob, np.float64)
    L = np.ascontiguousarray(LL[:, :5])
    band = np.full((h, w), -1, np.int32)
    NX = np.zeros((h, w), np.float32)
    NY = np.zeros((h, w), np.float32)
    NZ = np.ones((h, w), np.float32)
    DN = np.zeros((h, w), np.float32)
    ORD = np.full((h, w), -1.0, np.float32)
    _lobes(L, body, S, band, NX, NY, NZ, DN, ORD)
    Zarr = np.array(Zs)
    is_lobe = (ORD >= 0) & (np.abs(ORD - np.round(ORD)) < 0.25)
    lift = np.where(is_lobe, LL[np.clip(np.round(ORD).astype(np.int64), 0, len(LL) - 1), 5], 0.0).astype(np.float32)
    depth = np.where(band >= 0, Zarr[np.maximum(band, 0)], 1e3).astype(np.float32)
    return dict(band=band, nx=NX, ny=NY, nz=NZ, dn=DN, depth=depth, order=ORD, lift=lift,
                cover=(band >= 0).astype(np.float32), body=body, scale=S, Z=Zarr, lobes=L)


@njit(cache=True, fastmath=True)
def _lobes(L, body, S, band, NX, NY, NZ, DN, ORD):
    nb = body.shape[0]
    h = band.shape[0]
    w = band.shape[1]
    nl = L.shape[0]
    j = 0
    for k in range(nb):
        s = S[k]
        # the continuous body of the row
        for x in range(w):
            y0 = int(math.ceil(body[k, x]))
            if y0 < 0:
                y0 = 0
            for y in range(y0, h):
                band[y, x] = k
                d = (y - body[k, x]) / s
                NX[y, x] = 0.0
                NY[y, x] = 0.35 + 0.4 * min(d / 3.0, 1.0)
                NZ[y, x] = 0.85
                DN[y, x] = d + 0.25
                ORD[y, x] = j - 0.5
        # the lobes along its top
        while j < nl and int(L[j, 0]) == k:
            cx = L[j, 1]
            cy = L[j, 2]
            rx = L[j, 3]
            ry = L[j, 4]
            x0 = int(max(0.0, math.floor(cx - rx)))
            x1 = int(min(w - 1.0, math.ceil(cx + rx)))
            y0 = int(max(0.0, math.floor(cy - ry)))
            y1 = int(min(h - 1.0, math.ceil(cy + ry)))
            for y in range(y0, y1 + 1):
                v = (y - cy) / ry
                for x in range(x0, x1 + 1):
                    u = (x - cx) / rx
                    d2 = u * u + v * v
                    if d2 >= 1.0:
                        continue
                    nz = math.sqrt(1.0 - d2)
                    gy = v / 0.8
                    ln = math.sqrt(u * u + gy * gy + nz * nz)
                    # the lower half of a lobe melts into the body
                    t = (v - 0.05) / 0.8
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    t = t * t * (3.0 - 2.0 * t)
                    band[y, x] = k
                    NX[y, x] = (u / ln) * (1.0 - t)
                    NY[y, x] = (gy / ln) * (1.0 - t) + 0.4 * t
                    NZ[y, x] = (nz / ln) * (1.0 - t) + 0.85 * t
                    DN[y, x] = max(v, 0.0) * 0.6 + t * 0.3
                    ORD[y, x] = j
            j += 1


def creases(order, band, scale, reach=0.18, maxpx=7):
    """Darkening just above the upper outline of every lobe that overlaps
    another: the fold where one billow tucks behind the next."""
    h, w = order.shape
    out = np.zeros((h, w), np.float32)
    for dy in range(1, maxpx + 1):
        below = np.empty_like(order)
        below[:-dy] = order[dy:]
        below[-dy:] = order[-dy:]
        bb = np.empty_like(band)
        bb[:-dy] = band[dy:]
        bb[-dy:] = band[-dy:]
        sc = scale[np.maximum(band, 0)]
        lim = np.maximum(sc * reach, 1.5)
        hit = (below > order + 0.5) & (dy <= lim)
        out = np.maximum(out, hit * (1.0 - dy / (lim + 1.0)))
    return out
