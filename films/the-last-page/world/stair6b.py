"""Cut 6b: looking straight up the circular stairwell.

A real cylinder of storeys ray cast through a pinhole camera standing off
the axis: shelves of books bending with the curved wall, storeys of
different heights, tall arched windows letting moonlight slant across the
well, galleries (landings) on some storeys only, a stone spiral stair
winding up as a helix of wedge steps with its railing, and the dome with
its closed shutters at the top. The plate turns slowly in render.

Plates: c6b_bg (bg), c6b_glow (moonlight shafts and the dome's gold seams).
"""

import math
import os
import sys

import numpy as np
from numba import njit
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise, rng, seed_of  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402
import books as BK  # noqa: E402

M = MARGIN
CUT = "6b"
PW, PH = T.plate_size(CUT)
AW, AH = PW + 2 * M, PH + 2 * M

MOON = np.array(T.LIGHTS["hall"]["key"], np.float32)
AMB = np.array(T.LIGHTS["hall"]["ambient"], np.float32)
GOLD = np.array(T.LIGHTS["gold_burst"]["key"], np.float32)
WOOD = np.array([0.30, 0.20, 0.15], np.float32)
STONE = np.array([0.42, 0.40, 0.46], np.float32)

CAM = np.array([0.52, -0.42, 0.0])
TARGET = np.array([-0.30, 0.22, 9.0])
FOV = 108.0
Z_TOP = 9.2
STOREYS = [0.0, 1.35, 2.45, 3.70, 4.65, 5.75, 6.60, 7.55, 8.35, Z_TOP]
GALLERIES = [(1.35, 0.74), (3.70, 0.80), (5.75, 0.83), (7.55, 0.86)]   # (z, inner radius)
STAIR = dict(r_in=0.74, z0=-0.2, dz=0.062, dth=math.radians(15.0), th0=0.6, n=138)
WINDOWS = [(2, -2.2, 0.20), (3, 0.9, 0.18), (4, 2.6, 0.17), (5, -0.6, 0.16), (6, 1.8, 0.15), (7, -2.8, 0.14)]
MOON_TRAVEL = np.array([0.55, -0.35, -0.75])
MOON_TRAVEL = MOON_TRAVEL / np.linalg.norm(MOON_TRAVEL)


def basis():
    f = TARGET - CAM
    f = f / np.linalg.norm(f)
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    foc = (AW / 2) / math.tan(math.radians(FOV / 2))
    return f, right, up, foc


def project(p):
    f, right, up, foc = basis()
    d = p - CAM
    z = d @ f
    return AW / 2 + foc * (d @ right) / np.maximum(z, 1e-3), AH / 2 - foc * (d @ up) / np.maximum(z, 1e-3)


@njit(cache=True, fastmath=True)
def _cast(dx, dy, dz, ox, oy, oz, ztop, gz, gr, sz0, sdz, sdth, sth0, sn, srin,
          kind, hx, hy, hz, sidx):
    h, w = dx.shape
    for j in range(h):
        for i in range(w):
            ax, ay, az = dx[j, i], dy[j, i], dz[j, i]
            # cylinder (inside): far root
            a = ax * ax + ay * ay
            b = 2 * (ox * ax + oy * ay)
            c = ox * ox + oy * oy - 1.0
            disc = b * b - 4 * a * c
            if disc < 0 or a < 1e-12:
                tbest = 1e9
            else:
                tbest = (-b + math.sqrt(disc)) / (2 * a)
            k = 0
            zw = oz + tbest * az
            if zw > ztop:
                # dome: sphere centred on the axis at ztop, far root
                cx, cy, cz = ox, oy, oz - ztop
                bb = 2 * (cx * ax + cy * ay + cz * az)
                cc = cx * cx + cy * cy + cz * cz - 1.0
                d2 = bb * bb - 4 * cc
                tbest = (-bb + math.sqrt(max(d2, 0.0))) / 2
                k = 3
            si = -1
            if abs(az) > 1e-6:
                # galleries: horizontal annuli
                for g in range(gz.shape[0]):
                    t = (gz[g] - oz) / az
                    if 0 < t < tbest:
                        px = ox + t * ax
                        py = oy + t * ay
                        rr = px * px + py * py
                        if rr > gr[g] * gr[g] and rr < 1.0:
                            tbest = t
                            k = 1
                # stair steps: horizontal wedges on a helix
                for s in range(sn):
                    zs = sz0 + s * sdz
                    t = (zs - oz) / az
                    if 0 < t < tbest:
                        px = ox + t * ax
                        py = oy + t * ay
                        rr = px * px + py * py
                        if rr > srin * srin and rr < 1.0:
                            th = math.atan2(py, px)
                            tc = sth0 + s * sdth
                            dd = (th - tc + math.pi) % (2 * math.pi) - math.pi
                            if abs(dd) < sdth * 0.56:
                                tbest = t
                                k = 2
                                si = s
            kind[j, i] = k
            hx[j, i] = ox + tbest * ax
            hy[j, i] = oy + tbest * ay
            hz[j, i] = oz + tbest * az
            sidx[j, i] = si


def raycast():
    f, right, up, foc = basis()
    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float64)
    d = f[None, None] * foc + right[None, None] * (xx - AW / 2)[..., None] - up[None, None] * (yy - AH / 2)[..., None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    kind = np.zeros((AH, AW), np.int32)
    hx = np.zeros((AH, AW))
    hy = np.zeros((AH, AW))
    hz = np.zeros((AH, AW))
    sidx = np.zeros((AH, AW), np.int32)
    gz = np.array([g[0] for g in GALLERIES])
    gr = np.array([g[1] for g in GALLERIES])
    S = STAIR
    _cast(d[..., 0].copy(), d[..., 1].copy(), d[..., 2].copy(), CAM[0], CAM[1], CAM[2], Z_TOP, gz, gr,
          S["z0"], S["dz"], S["dth"], S["th0"], S["n"], S["r_in"], kind, hx, hy, hz, sidx)
    return kind, hx, hy, hz, sidx


def book_rows():
    """Per storey: shelf rows as (z bottom, z top) and, for each row, book edges in theta."""
    r = rng("6b2-rows")
    rows = []
    for s in range(len(STOREYS) - 1):
        z0, z1 = STOREYS[s], STOREYS[s + 1]
        zb = z0 + 0.14
        nrow = max(2, int((z1 - z0) / 0.36))
        hrow = (z1 - zb - 0.06) / nrow
        for k in range(nrow):
            a, b = zb + k * hrow, zb + (k + 1) * hrow - 0.03
            # books around the circle: widths vary, a few gaps and leaning ones
            ths = [-math.pi]
            while ths[-1] < math.pi:
                w = (0.010 + 0.022 * r.random()) * (1.6 if r.random() < 0.1 else 1.0)
                ths.append(ths[-1] + w)
            ths = np.array(ths)
            nb = len(ths) - 1
            hgt = 0.55 + 0.4 * r.random(nb)
            col = BK.BOOK_COLS[r.integers(0, len(BK.BOOK_COLS), nb)] * (0.8 + 0.35 * r.random(nb))[:, None]
            gap = r.random(nb) < 0.05
            rows.append((s, a, b, ths, hgt, col, gap))
    return rows


def reference():
    kind, hx, hy, hz, sidx = raycast()
    th = np.arctan2(hy, hx)
    up = np.clip(hz / Z_TOP, 0, 1)
    ref = np.zeros((AH, AW, 3), np.float32)
    # light: deep blue, lifting toward the top; windows throw moonlight onto the far wall
    L = AMB[None, None] * (1.15 + 0.7 * up)[..., None] + MOON * 0.10
    wall = kind == 0
    # books and shelves
    ref[:] = np.array([0.03, 0.03, 0.06], np.float32)
    rows = book_rows()
    for (s, a, b, ths, hgt, col, gap) in rows:
        m = wall & (hz >= a) & (hz < b)
        if not m.any():
            continue
        tt = th[m]
        idx = np.clip(np.searchsorted(ths, tt) - 1, 0, len(hgt) - 1)
        top = a + (b - a) * hgt[idx]
        zz = hz[m]
        isb = (zz < top) & ~gap[idx]
        c = col[idx]
        # a shadow edge on each spine, a lit edge on the other side
        fu = (tt - ths[idx]) / np.maximum(ths[idx + 1] - ths[idx], 1e-6)
        c = c * (0.65 + 0.45 * np.clip(fu * 1.3, 0, 1))[:, None]
        # raised bands and gilt on the spines, a label on some, the shelf's shadow at the top
        vz = (zz - a) / np.maximum(top - a, 1e-6)
        bands = (np.abs(np.mod(vz * 4 + idx * 0.37, 1.0) - 0.5) > 0.45) & (vz > 0.1) & (vz < 0.92)
        c = np.where(bands[:, None], c * 0.55 + np.array([0.20, 0.16, 0.08]) * (idx % 3 == 0)[:, None], c)
        lab = (vz > 0.58) & (vz < 0.72) & (np.abs(fu - 0.5) < 0.28) & (idx % 4 == 1)
        c = np.where(lab[:, None], np.array([0.62, 0.57, 0.46]) * 0.8, c)
        c = c * (1 - 0.5 * np.clip((zz - (b - 0.06)) / 0.06, 0, 1))[:, None]
        c = c * 0.82 + c.mean(-1, keepdims=True) * 0.18
        cc = np.where(isb[:, None], c, np.array([0.025, 0.022, 0.04], np.float32))
        ref[m] = cc
        board = m & (hz < a + 0.03)
        ref[board] = WOOD * 1.1
    # storey cornices: a stone band at every storey line
    for z in STOREYS[1:-1]:
        band = wall & (np.abs(hz - z) < 0.07)
        ref[band] = STONE * 0.8
    # windows: tall arches full of moonlit night
    for (s, tc, hw) in WINDOWS:
        z0, z1 = STOREYS[s] + 0.15, STOREYS[s + 1] - 0.12
        dth = (th - tc + math.pi) % (2 * math.pi) - math.pi
        zc = z1 - hw * 1.2
        arch = np.where(hz > zc, np.sqrt(np.maximum(1 - ((hz - zc) / (hw * 1.2)) ** 2, 0)) * hw, hw)
        win = wall & (np.abs(dth) < arch) & (hz > z0) & (hz < z1)
        frame = wall & (np.abs(dth) < arch + 0.04) & (hz > z0 - 0.05) & (hz < z1 + 0.05) & ~win
        sky = np.array([0.36, 0.44, 0.72], np.float32) / 2.3
        lead = (np.abs(np.mod(hz * 9, 1.0) - 0.5) > 0.44) | (np.abs(dth) < 0.01)
        ref[win] = np.where(lead[win][:, None], np.array([0.03, 0.03, 0.06], np.float32), sky)
        ref[frame] = STONE * 0.7
    # moonlight from each window falling across the opposite wall
    mlight = np.zeros((AH, AW), np.float32)
    for (s, tc, hw) in WINDOWS:
        to = tc + math.pi * 0.85
        dth = (th - to + math.pi) % (2 * math.pi) - math.pi
        zc = (STOREYS[s] + STOREYS[s + 1]) / 2 - 0.9
        mlight = np.maximum(mlight, np.exp(-(dth / 0.35) ** 2) * np.exp(-((hz - zc) / 0.7) ** 2))
    L = L + MOON * (mlight * 0.55 * wall)[..., None]
    # galleries: wooden landings seen from below, joists running out radially
    gal = kind == 1
    joist = np.abs(np.mod(th / (2 * math.pi) * 90, 1.0) - 0.5) < 0.12
    gcol = np.where(joist[..., None], WOOD * 0.45, WOOD * 0.7)
    ref = np.where(gal[..., None], gcol, ref)
    # stair: stone wedge undersides, each its own value, a lit nosing
    st = kind == 2
    sv = ((sidx.astype(np.int64) * 2654435761) % 4294967296) / 4294967296.0
    scol = STONE * np.array([1.0, 0.92, 0.84], np.float32) * (0.50 + 0.26 * sv)[..., None]
    scol = scol * (0.85 + 0.25 * fbm(AH, AW, 10, "6b2-steps", octaves=3))[..., None]
    S = STAIR
    tcs = S["th0"] + sidx * S["dth"]
    dd = (th - tcs + math.pi) % (2 * math.pi) - math.pi
    nose = st & (dd > S["dth"] * 0.40)
    scol = np.where(nose[..., None], STONE * np.array([1.0, 0.92, 0.84], np.float32) * 0.95, scol)
    under = st & (dd < -S["dth"] * 0.38)
    scol = np.where(under[..., None], scol * 0.55, scol)
    ref = np.where(st[..., None], scol, ref)
    # dome: ribbed copper, shutters closed with gold leaking round them
    dm = kind == 3
    lat = np.degrees(np.arcsin(np.clip(hz - Z_TOP, -1, 1)))
    rib = np.abs(np.mod(th / (2 * math.pi) * 16, 1.0) - 0.5) < 0.05
    dcol = np.array([0.24, 0.40, 0.38], np.float32) * (0.75 + 0.4 * fbm(AH, AW, 20, "6b2-dome", octaves=3))[..., None]
    dcol = np.where(rib[..., None], np.array([0.25, 0.19, 0.15], np.float32), dcol)
    ref = np.where(dm[..., None], dcol, ref)
    lit = np.clip(ref * L * 2.3, 0, 1)
    shut = dm & (lat > 40) & (lat < 62) & (np.abs(np.mod(th / (2 * math.pi) * 8, 1.0) - 0.5) > 0.47)
    lit = np.where(shut[..., None], GOLD * 0.8, lit)
    windows = np.zeros((AH, AW), bool)
    for (s, tc, hw) in WINDOWS:
        z0, z1 = STOREYS[s] + 0.15, STOREYS[s + 1] - 0.12
        dth = (th - tc + math.pi) % (2 * math.pi) - math.pi
        windows |= wall & (np.abs(dth) < hw) & (hz > z0) & (hz < z1)
    # air: the upper storeys melt into blue
    air = up ** 1.5 * (kind != 3)
    lit = lit * (1 - 0.45 * air[..., None]) + np.array([0.08, 0.10, 0.20], np.float32) * (0.45 * air)[..., None]
    return np.clip(lit, 0, 1).astype(np.float32), dict(kind=kind, th=th, hz=hz, sidx=sidx)


def paint_all():
    log("6b: reference")
    ref, info = reference()
    th, hz = info["th"], info["hz"]
    # strokes run along the books (up the wall) on the wall, around the rings elsewhere
    gy, gx = np.gradient(hz)
    ang_wall = np.arctan2(gy, gx)
    gy2, gx2 = np.gradient(np.unwrap(th, axis=1))
    ang_ring = np.arctan2(gy2, gx2) + math.pi / 2
    ang = np.where(info["kind"] == 0, ang_wall, ang_ring).astype(np.float32)
    paper = P.paper_for(AH, AW, "6b2")
    canvas = P.new_canvas(AH, AW)
    log("6b: painting")
    P.paint(canvas, ref, ang, [
        dict(radius=7, spacing=0.9, steps=(2, 6), opacity=0.96, dry=0.25, cthresh=0.12, wob=0.15,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.15),
        dict(radius=3.2, spacing=1.0, steps=(2, 5), opacity=0.93, dry=0.35, cthresh=0.09, wob=0.15,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.03),
        dict(radius=1.5, spacing=1.1, steps=(2, 4), opacity=0.9, dry=0.4, cthresh=0.07, wob=0.15,
             jitter=(0.05, 0.02, 0.04), streak=0.06, use_err=True, err_thresh=0.05),
    ], paper, "6b2")
    # railings: the galleries' inner edge and the stair's inner edge, with balusters
    st = P.Strokes(max_pts=24)
    for (gzv, gr) in GALLERIES:
        ths = np.linspace(-math.pi, math.pi, 160)
        top = np.stack([gr * np.cos(ths), gr * np.sin(ths), np.full_like(ths, gzv + 0.16)], 1)
        bot = np.stack([gr * np.cos(ths), gr * np.sin(ths), np.full_like(ths, gzv + 0.01)], 1)
        tx, ty = project(top)
        bx, by = project(bot)
        fade = max(0.25, 1.0 - gzv / 10.0)
        wsc = max(0.6, 2.2 - gzv * 0.2)
        for k in range(0, 159, 8):
            st.add(list(zip(tx[k:k + 9], ty[k:k + 9])), 1.6 * wsc, np.clip(WOOD * 0.55, 0, 1),
                   seed=int(gzv * 100) + k, opacity=0.9 * fade, dry=0.3, ragged=0.1, bristles=2, pool=0.1,
                   streak=0.1, taper0=0, taper1=0)
            st.add(list(zip(tx[k:k + 9], ty[k:k + 9] - 1.0)), 0.6 * wsc, np.clip(WOOD * 0.9 + MOON * 0.10, 0, 1),
                   seed=int(gzv * 100) + k + 3, opacity=0.6 * fade, dry=0.5, ragged=0.2, bristles=2, pool=0,
                   streak=0, taper0=0, taper1=0)
        for k in range(0, 160, 2):
            st.add([(bx[k], by[k]), (tx[k], ty[k])], 0.8 * wsc, np.clip(WOOD * 0.45, 0, 1), seed=int(gzv * 1000) + k,
                   opacity=0.8 * fade, dry=0.3, ragged=0.1, bristles=2, pool=0.1, streak=0, taper0=0, taper1=0)
    S = STAIR
    ks = np.arange(S["n"])
    ths = S["th0"] + ks * S["dth"]
    zs = S["z0"] + ks * S["dz"]
    rin = S["r_in"]
    rail = np.stack([rin * np.cos(ths), rin * np.sin(ths), zs + 0.20], 1)
    foot = np.stack([rin * np.cos(ths), rin * np.sin(ths), zs + 0.005], 1)
    rx, ry = project(rail)
    fx, fy = project(foot)
    for k in range(0, S["n"] - 6, 5):
        fade = max(0.25, 1.0 - zs[k] / 10.0)
        wsc = max(0.6, 2.6 - zs[k] * 0.25)
        st.add(list(zip(rx[k:k + 7], ry[k:k + 7])), 1.6 * wsc, np.clip(WOOD * 0.6, 0, 1), seed=5000 + k,
               opacity=0.92 * fade, dry=0.3, ragged=0.1, bristles=2, pool=0.1, streak=0.1, taper0=0, taper1=0)
        st.add(list(zip(rx[k:k + 7], ry[k:k + 7] - 1.0)), 0.6 * wsc, np.clip(WOOD + MOON * 0.12, 0, 1), seed=5500 + k,
               opacity=0.6 * fade, dry=0.5, ragged=0.2, bristles=2, pool=0, streak=0, taper0=0, taper1=0)
    for k in range(S["n"]):
        fade = max(0.25, 1.0 - zs[k] / 10.0)
        st.add([(fx[k], fy[k]), (rx[k], ry[k])], max(0.5, 1.4 - zs[k] * 0.12), np.clip(WOOD * 0.5, 0, 1),
               seed=6000 + k, opacity=0.85 * fade, dry=0.3, ragged=0.1, bristles=2, pool=0.1, streak=0, taper0=0,
               taper1=0)
    P.draw(canvas, st.arrays(), paper)
    P.finish(canvas, paper, grain=0.05)
    canvas[..., 3] = 1.0
    C.save_plate("c6b_bg", canvas)
    log("6b: shafts")
    # moonlight shafts slanting in from the windows across the well
    acc = np.zeros((AH, AW), np.float64)
    r = rng("6b2-shafts")
    for (s, tc, hw) in WINDOWS:
        z0, z1 = STOREYS[s] + 0.15, STOREYS[s + 1] - 0.12
        n = 2500
        tt = tc + (r.random(n) * 2 - 1) * hw
        zz = z0 + (z1 - z0) * r.random(n)
        p0 = np.stack([np.cos(tt), np.sin(tt), zz], 1)
        inward = -np.stack([np.cos(tc), np.sin(tc), 0.0])
        dirv = MOON_TRAVEL * 0.4 + inward * 0.6
        dirv /= np.linalg.norm(dirv)
        for t in np.linspace(0.02, 1.7, 70):
            pts = p0 + dirv[None] * t
            ok = (pts[:, 0] ** 2 + pts[:, 1] ** 2) < 1.0
            x, y = project(pts[ok])
            good = (x >= 0) & (x < AW) & (y >= 0) & (y < AH)
            np.add.at(acc, (y[good].astype(int), x[good].astype(int)), (1 - t / 1.8))
    accb = ndimage.gaussian_filter(acc, 6.0)
    inten = np.clip(accb / max(np.percentile(accb[accb > 0], 90), 1e-6), 0, 1.5) ** 0.8
    glow = (inten * (0.7 + 0.4 * value_noise(AH, AW, 14, "6b2-dust")))[..., None] * MOON * 0.16
    kind = info["kind"]
    lat = np.degrees(np.arcsin(np.clip(hz - Z_TOP, -1, 1)))
    seam = (kind == 3) & (lat > 40) & (lat < 62) & (np.abs(np.mod(th / (2 * math.pi) * 8, 1.0) - 0.5) > 0.46)
    glow += (ndimage.gaussian_filter(seam.astype(np.float32), 4.0) * 0.6)[..., None] * GOLD
    np.save(C.plate_path("c6b_glow"), glow.astype(np.float16))


if __name__ == "__main__":
    import time
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "ref":
        ref, info = reference()
        C.save_review(os.path.join(C.REVIEW, "a2-dev", "6b2_ref.png"), ref[::2, ::2])
    else:
        paint_all()
    log(f"done {time.time() - t0:.1f}s")
