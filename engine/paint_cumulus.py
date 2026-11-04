"""Cumulus banks for painted skies.

A background painter does not paint a sea of clouds as a field of equal
scallops. They design a few masses: towering banks close by, lower heaps in
the middle distance, flat hazy layers at the horizon, each with a flat base and
a cauliflower top built from puffs of very different sizes. This module builds
such banks as soft density fields, lights them from a low sun (gold flanks,
cool tops, violet shadows thrown away from the sun, bright linings where the
light spills over an edge) and composites them far to near into a reference
the stroke painter repaints.

Banks are dicts:
    cx, base   centre x and base line y (plate coordinates, margins included)
    width      base width
    height     tallest point above the base
    rmax, rmin largest and smallest puff radius
    flat       vertical squash of the puffs (1 round, 0.3 a flat stratus)
    depth      0 near (full contrast) .. 1 far (melted into the haze)
    towers     optional list of (x offset fraction, height fraction, width fraction)
    back_reach how far (px) from the sun a bank counts as backlit (dark body, bright outline)

Usage:

    from engine import paint_cumulus as PCU
    rgb, alpha, rim, idx = PCU.compose(banks, (h, w), sun_xy, palette, sky_at, key, floor=None)

palette: dict lit, high, mid, shadow, deep, top, rim (RGB arrays).
sky_at(X, Y) -> haze colour arrays, used to melt far banks (depth near 1).
floor: optional (y_top, near_rgb, far_rgb, wave_px) for a continuous cloud floor
under the banks. Coordinates are array coordinates (plate margins included).
Paint the result with films/the-last-page/world/clouds_sunrise.paint_banks, which
runs the stroke painter over it and adds painted linings. Keep rmin at 8 to 16
px for near banks, or the puff texture reads as scales; keep depth 0.5 to 0.85
for the horizon rows so they flatten into haze. Design a few big banks of very
different sizes rather than many equal ones.
"""

import math

import numpy as np
from numba import njit
from scipy import ndimage

from engine.noise import fbm, rng, value_noise


@njit(cache=True, fastmath=True)
def _splat(D, DM, xs, ys, rs, fl, x0, y0):
    h, w = D.shape
    for i in range(xs.shape[0]):
        r = rs[i]
        ry = r * fl[i]
        cx = xs[i] - x0
        cy = ys[i] - y0
        xa = int(max(0.0, math.floor(cx - r * 1.2)))
        xb = int(min(w - 1.0, math.ceil(cx + r * 1.2)))
        ya = int(max(0.0, math.floor(cy - ry * 1.2)))
        yb = int(min(h - 1.0, math.ceil(cy + ry * 1.2)))
        ir2 = 1.0 / (r * r)
        iry2 = 1.0 / (ry * ry)
        for y in range(ya, yb + 1):
            dy2 = (y - cy) * (y - cy) * iry2
            for x in range(xa, xb + 1):
                q = (x - cx) * (x - cx) * ir2 + dy2
                if q < 1.44:
                    t = 1.0 - q / 1.44
                    D[y, x] += t * t * t
                    m = 1.0 - q
                    if m > DM[y, x]:
                        DM[y, x] = m


@njit(cache=True, fastmath=True)
def _lobes(NX, NY, NZ, ORD, xs, ys, rs, fl, x0, y0):
    """Paint puffs as overlapping spheres in the given order (later in front)."""
    h, w = NX.shape
    for i in range(xs.shape[0]):
        r = rs[i]
        ry = r * fl
        cx = xs[i] - x0
        cy = ys[i] - y0
        xa = int(max(0.0, math.floor(cx - r)))
        xb = int(min(w - 1.0, math.ceil(cx + r)))
        ya = int(max(0.0, math.floor(cy - ry)))
        yb = int(min(h - 1.0, math.ceil(cy + ry)))
        for y in range(ya, yb + 1):
            v = (y - cy) / ry
            for x in range(xa, xb + 1):
                u = (x - cx) / r
                q = u * u + v * v
                if q >= 1.0:
                    continue
                nz = math.sqrt(1.0 - q)
                gy = v / fl
                ln = math.sqrt(u * u + gy * gy + nz * nz)
                NX[y, x] = u / ln
                NY[y, x] = gy / ln
                NZ[y, x] = nz / ln
                ORD[y, x] = i


def puffs(b, key):
    """Puff list (x, y, r) of one bank: big core puffs along the base, then
    generations of smaller puffs budding on the upper surface, each kept
    inside the bank's height envelope (with optional towers)."""
    r = rng("puffs", key)
    cx, base, width, height = b["cx"], b["base"], b["width"], b["height"]
    rmax, rmin = b["rmax"], b["rmin"]
    fl = b.get("flat", 1.0)
    towers = b.get("towers") or [(0.0, 1.0, 0.6)]
    xs = np.linspace(cx - width / 2, cx + width / 2, 400)

    def env(x):
        e = 0.0
        for (ox, hf, wf) in towers:
            u = (x - (cx + ox * width / 2)) / max(wf * width / 2, 1.0)
            e = max(e, hf * height * max(0.0, 1.0 - u * u) ** 0.65)
        # a low skirt so the bank does not end in a point
        u = (x - cx) / (width / 2)
        e = max(e, 0.22 * height * max(0.0, 1.0 - u * u) ** 0.5)
        return e
    out = []
    # core along the base
    x = cx - width / 2 + r.random() * rmax * 0.5
    while x < cx + width / 2:
        e = env(x)
        rr = min(rmax, max(rmin * 2, e * 0.55)) * (0.7 + 0.5 * r.random())
        out.append((x, base - rr * fl * 0.55, rr))
        x += rr * (0.7 + 0.6 * r.random())
    gen = list(out)
    for g in range(14):
        nxt = []
        for (px, py, pr) in gen:
            head = env(px) - (base - (py - pr * fl))
            k = int(r.integers(2, 5))
            for j in range(k):
                if head > pr * 2.5:
                    # plenty of room: grow upward at nearly the same size (a tower)
                    a = -math.pi / 2 + (r.random() - 0.5) * 1.3
                    cr = pr * (0.72 + 0.24 * r.random())
                else:
                    a = -math.pi * (0.08 + 0.84 * r.random())
                    cr = pr * (0.45 + 0.35 * r.random())
                if cr < rmin:
                    continue
                nx = px + math.cos(a) * pr * 0.78
                ny = py + math.sin(a) * pr * 0.78 * fl
                if base - (ny - cr * fl) > env(nx) * (0.9 + 0.2 * r.random()):
                    continue
                nxt.append((nx, ny, cr))
        if len(nxt) > 2500:
            nxt = [nxt[i] for i in r.choice(len(nxt), 2500, replace=False)]
        if not nxt:
            break
        out.extend(nxt)
        gen = nxt
    # a scatter of small puffs over the whole surface breaks any remaining curve
    n0 = len(out)
    for i in range(n0 // 2):
        px, py, pr = out[int(r.integers(0, n0))]
        a = -math.pi * r.random()
        cr = max(rmin, pr * (0.2 + 0.25 * r.random()))
        nx, ny = px + math.cos(a) * pr * 0.9, py + math.sin(a) * pr * 0.9 * fl
        if base - (ny - cr * fl) <= env(nx) * 1.05:
            out.append((nx, ny, cr))
    return np.array(out, np.float64)


def bank_density(b, key, shape):
    """Density and coverage of a bank inside its bounding box.
    Returns (x0, y0, D, cover)."""
    P_ = puffs(b, key)
    fl = b.get("flat", 1.0)
    pad = b["rmax"] * 1.4
    x0 = int(max(0, b["cx"] - b["width"] / 2 - pad))
    x1 = int(min(shape[1], b["cx"] + b["width"] / 2 + pad))
    y0 = int(max(0, b["base"] - b["height"] - pad * fl - 20))
    y1 = int(min(shape[0], b["base"] + pad * fl * 0.6 + 10))
    if x1 <= x0 or y1 <= y0:
        return x0, y0, None, None, None
    D = np.zeros((y1 - y0, x1 - x0), np.float64)
    DM = np.full((y1 - y0, x1 - x0), -1.0, np.float64)
    _splat(D, DM, P_[:, 0], P_[:, 1], P_[:, 2], np.full(len(P_), fl), float(x0), float(y0))
    D = D.astype(np.float32)
    DM = DM.astype(np.float32)
    hh, ww = D.shape
    # cauliflower edge: noise eats the outline at two scales, more on top than at the base
    yy = np.arange(hh, dtype=np.float32)[:, None] + y0
    topness = np.clip((b["base"] - yy) / max(b["height"], 1.0), 0, 1)
    n1 = fbm(hh, ww, max(b["rmin"] * 3.0, 4.0), (key, "e1"), octaves=3)
    n2 = fbm(hh, ww, max(b["rmax"] * 0.6, 6.0), (key, "e2"), octaves=2)
    Dn = D + ((n1 - 0.5) * 0.28 + (n2 - 0.5) * 0.22) * (0.5 + 0.5 * topness)
    # flat base: density fades out just below the base line
    Dn = Dn * np.clip((b["base"] + b["rmax"] * fl * 0.25 - yy) / (b["rmax"] * fl * 0.35 + 1.0), 0, 1)
    # the silhouette follows the puffs themselves (a bumpy cauliflower outline),
    # nibbled a little by noise
    DMn = DM + ((n1 - 0.5) * 0.16) * (0.5 + 0.5 * topness)
    DMn = DMn * np.clip((b["base"] + b["rmax"] * fl * 0.25 - yy) / (b["rmax"] * fl * 0.35 + 1.0), 0, 1) + \
        (np.clip((b["base"] + b["rmax"] * fl * 0.25 - yy) / (b["rmax"] * fl * 0.35 + 1.0), 0, 1) - 1) * 0.5
    cover = np.clip((DMn - 0.02) / 0.10, 0, 1).astype(np.float32)
    # lobes: lower puffs overlap the ones above them (the near face of the dome)
    order = np.argsort(-(P_[:, 1] + P_[:, 2] * fl), kind="stable")[::-1]
    Q = P_[order]
    NX = np.zeros_like(D)
    NY = np.zeros_like(D)
    NZ = np.ones_like(D)
    ORD = np.full(D.shape, -1, np.int64)
    _lobes(NX, NY, NZ, ORD, Q[:, 0], Q[:, 1], Q[:, 2], float(fl), float(x0), float(y0))
    RAD = np.where(ORD >= 0, Q[np.maximum(ORD, 0), 2], 0.0).astype(np.float32)
    lobes = dict(nx=NX, ny=NY, nz=NZ, order=ORD, r=Q[:, 2], rad=RAD)
    return x0, y0, Dn.astype(np.float32), cover, lobes


def shade_bank(b, key, x0, y0, Dn, cover, sun, pal, sky_at, lobes=None):
    """Colour a bank. sun: (x, y) plate position of the sun (light source).
    pal: dict of colours lit, high, mid, shadow, deep, top, rim.
    sky_at(x, y) -> haze colour arrays for the depth melt."""
    hh, ww = Dn.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    X = xx + x0
    Y = yy + y0
    # a soft height field from the density: the bank's form, then the puffs' form
    H1 = ndimage.gaussian_filter(np.clip(Dn, 0, 2), max(b["rmax"] * 0.35, 2.0))
    H2 = ndimage.gaussian_filter(np.clip(Dn, 0, 2), max(b["rmin"] * 0.9, 1.2))
    gy1, gx1 = np.gradient(H1)
    gy2, gx2 = np.gradient(H2)
    s1 = b["rmax"] * 0.9
    s2 = b["rmin"] * 2.2
    nx = -(gx1 * s1 * 0.55 + gx2 * s2 * 0.45)
    ny = -(gy1 * s1 * 0.55 + gy2 * s2 * 0.45)
    ln = np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx / ln, ny / ln, 1.0 / ln
    lob_edge = np.zeros_like(Dn)
    crease = np.zeros_like(Dn)
    edt = ndimage.distance_transform_edt(cover > 0.5).astype(np.float32)
    edt = ndimage.gaussian_filter(edt, 0.8)
    # cauliflower detail lives near the outline; the interior reads as one broad mass
    near_edge = np.exp(-edt / max(b["rmax"] * 0.9, 4.0))
    if lobes is not None:
        # each lobe's own roundness, blended with the mass
        inl = lobes["order"] >= 0
        # big lobes melt into the mass, small ones keep their roundness
        k = np.clip(0.75 - lobes["rad"] / max(b["rmax"], 1.0) * 0.6, 0.15, 0.55)
        yyb = yy + y0
        k = k * np.clip((b["base"] - yyb) / (b["rmax"] * b.get("flat", 1.0) * 1.2 + 1.0), 0.15, 1.0)
        k = k * (0.25 + 0.75 * near_edge)
        nx = np.where(inl, nx * (1 - k) + lobes["nx"] * k, nx)
        ny = np.where(inl, ny * (1 - k) + lobes["ny"] * k, ny)
        nz = np.where(inl, nz * (1 - k) + lobes["nz"] * k, nz)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        lob_edge = np.where(inl, (1 - lobes["nz"]) ** 3, 0.0) * k / 0.7
        # folds: just above where a lower lobe overlaps one behind it
        o = lobes["order"]
        for d in (2, 4, 7):
            below = np.empty_like(o)
            below[:-d] = o[d:]
            below[-d:] = o[-d:]
            crease = np.maximum(crease, ((below > o) & (o >= 0)).astype(np.float32) * (1.0 - d / 9.0))
        crease = ndimage.gaussian_filter(crease, 1.2) * (0.3 + 0.7 * near_edge)
    # direction toward the sun in the picture
    sx, sy = sun
    dx, dy = sx - X, sy - Y
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl
    # how much the sun is behind this bank (contre-jour) versus beside it
    back = np.exp(-(dl / (b.get("back_reach", 450.0))) ** 2)
    side = np.clip(nx * ux + ny * uy, -1, 1) / 0.6
    top = np.clip(-ny, 0, 1)
    lit = np.clip(0.46 + 0.85 * side + 0.22 * top, 0, 1)
    # short self shadow toward the sun: crevices and the shadow side of each tower
    sh = np.zeros_like(Dn)
    step = max(b["rmax"] * 0.18, 3.0)
    for k in range(1, 6):
        sh += ndimage.map_coordinates(np.clip(Dn - 0.5, 0, 1), [yy + uy * step * k, xx + ux * step * k],
                                      order=1, mode="constant", cval=0.0)
    shadow = np.clip(sh / 2.5, 0, 1)
    light = np.clip(lit * (1 - 0.55 * shadow) * (1 - 0.30 * crease * np.clip(lit * 1.5, 0, 1)), 0, 1)
    # a sun right behind the bank lights only its outline: the body stays in shadow
    light = light * (1 - 0.8 * back * (1 - near_edge))
    col = pal["deep"] * (1 - light[..., None]) + pal["shadow"] * light[..., None]
    t = np.clip((light - 0.25) / 0.45, 0, 1)
    col = col * (1 - t[..., None]) + pal["mid"] * t[..., None]
    t2 = np.clip((light - 0.55) / 0.35, 0, 1) ** 1.2
    col = col * (1 - t2[..., None]) + pal["lit"] * t2[..., None]
    t3 = np.clip((light - 0.85) / 0.15, 0, 1)
    col = col * (1 - t3[..., None]) + pal["high"] * t3[..., None]
    # cool sky light on the tops that face up, away from the sun
    col = col + (top * (1 - t2) * 0.10)[..., None] * pal["top"]
    # contre-jour: banks between us and the sun darken, their edges blaze
    col = col * (1 - 0.35 * back)[..., None] + (back * 0.06)[..., None] * pal["shadow"]
    facing = np.clip(side, 0, 1)
    rimw = max(b["rmin"] * 0.6, 1.5)
    sil = np.exp(-edt / rimw) * (cover > 0.05)
    # linings only along the outline: an inner lobe edge facing a sun right behind
    # the bank would otherwise light up as a stray spot in the middle of the mass
    rim = (sil * (0.2 + 0.8 * facing) * (0.4 + 0.8 * back) + lob_edge * facing * 0.8 * near_edge * (1 - 0.8 * back)) * \
        (0.5 + 0.5 * t2 + 0.5 * back)
    rim = np.clip(rim, 0, 1)
    col = col * (1 - np.clip(rim, 0, 1)[..., None] * 0.85) + pal["rim"] * np.clip(rim, 0, 1)[..., None] * 0.85
    # atmospheric depth: the far banks melt into the haze, keeping a little shape
    dep = b.get("depth", 0.0)
    hz = sky_at(X, Y)
    col = col * (1 - dep * 0.82) + hz * dep * 0.82
    return np.clip(col, 0, 1).astype(np.float32), rim, edt


def compose(banks, shape, sun, pal, sky_at, key, floor=None):
    """Composite banks far to near (higher depth first). Returns rgb, alpha,
    a rim map and a per pixel bank index."""
    h, w = shape
    rgb = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    rimmap = np.zeros((h, w), np.float32)
    order = sorted(range(len(banks)), key=lambda i: -banks[i].get("depth", 0.0))
    idx = np.full((h, w), -1, np.int32)
    if floor is not None:
        # a continuous cloud floor below a gently uneven horizon so no sky shows through
        ytop, near_col, far_col, wav = floor
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        line = ytop + wav * (value_noise(1, w, 260, (key, "floorline"))[0][None, :] - 0.5)
        t = np.clip((yy - line) / max(h - ytop, 1.0), 0, 1) ** 0.7
        hz = sky_at(xx, np.full_like(yy, ytop))
        fc = hz * (1 - t[..., None]) * 0.9 + np.asarray(far_col) * (1 - t[..., None]) * 0.1 + np.asarray(near_col) * t[..., None]
        fc = fc * (0.94 + 0.12 * fbm(h, w, 120, (key, "floortex"), octaves=3))[..., None]
        a = np.clip((yy - line) / 6.0, 0, 1)
        rgb = fc * a[..., None]
        rgb = rgb.astype(np.float32)
        alpha = a.astype(np.float32)
    for i in order:
        b = banks[i]
        x0, y0, Dn, cover, lobes = bank_density(b, (key, i), shape)
        if Dn is None:
            continue
        col, rim, edt = shade_bank(b, (key, i), x0, y0, Dn, cover, sun, pal, sky_at, lobes)
        hh, ww = cover.shape
        sl = (slice(y0, y0 + hh), slice(x0, x0 + ww))
        a = cover
        # nearer banks throw a soft violet shadow on what lies behind them, away from the sun
        rgb[sl] = rgb[sl] * (1 - a[..., None]) + col * a[..., None]
        alpha[sl] = np.maximum(alpha[sl], a)
        rimmap[sl] = rimmap[sl] * (1 - a) + rim * a
        idx[sl] = np.where(a > 0.5, i, idx[sl])
    return rgb, alpha, rimmap, idx
