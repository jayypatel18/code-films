"""Cut 1 architecture and flora: the library tower, its cliff, ivy and pines.

Everything is built in plate coordinates (anchors from timeline.py) and
returned in array coordinates (plate + MARGIN). Each builder returns a colour
reference, a coverage mask and, where a painter would sharpen, hand placed
detail strokes.
"""

import math

import numpy as np

from engine import paint as P
from engine.noise import fbm, value_noise, rng, ridged, noise1, seed_of
from engine.paint_shapes import poly_mask, wobble, ellipse_mask, soft
from engine import paint_flora as F

from common import T, MARGIN

M = MARGIN
CUT = "1"
PW, PH = T.plate_size(CUT)
AW, AH = PW + 2 * M, PH + 2 * M
A = T.ANCHORS[CUT]

CX = float(A["tower_base"][0])          # 1560
BASE_Y = float(A["tower_base"][1])      # 760
EYE = 688.0                              # eye level in plate y
FOCAL = 1600.0

LT = np.array([-0.62, -0.50, 0.60], np.float32)
LT /= np.linalg.norm(LT)

STONE_LIT = np.array([0.475, 0.465, 0.610], np.float32)
STONE_SH = np.array([0.150, 0.130, 0.270], np.float32)
STONE_RIM = np.array([0.780, 0.560, 0.600], np.float32)
COP_LIT = np.array([0.370, 0.600, 0.560], np.float32)
COP_MID = np.array([0.200, 0.390, 0.410], np.float32)
COP_SH = np.array([0.070, 0.165, 0.230], np.float32)
WIN_DARK = np.array([0.045, 0.035, 0.090], np.float32)

# window row: arc angles (deg) of the windows on the front of the drum
WIN_ANG = [-62.5, -37.5, -12.5, 12.5, 37.5, 62.5]
WIN_SILL, WIN_SPRING, WIN_W = 545.0, 506.0, 15.0


def body_radius(h):
    r = 74.0 + (h - 404.0) / 356.0 * 8.0
    return r


def _ring_y(h, R, cos_t):
    """Screen y of a horizontal ring at height h (plate y at the axis) and
    radius R, at the point whose cos(theta) is cos_t."""
    return h + (h - EYE) * (R / FOCAL) * cos_t


def _h_of(y, R, cos_t):
    c = R * cos_t / FOCAL
    return (y + EYE * c) / (1.0 + c)


def _shade_wall(sin_t, cos_t):
    ndl = LT[0] * sin_t + LT[2] * cos_t
    return ndl


def tower_reference():
    """Returns dict(rgb, alpha, windows, details, ang, parts) in array coords."""
    x0, x1 = int(CX - 110), int(CX + 110)
    y0, y1 = 200, int(BASE_Y + 30)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    hgt, wid = yy.shape
    rgb = np.zeros((hgt, wid, 3), np.float32)
    alpha = np.zeros((hgt, wid), np.float32)
    ang = np.zeros((hgt, wid), np.float32)
    dx = xx - CX

    def paint_part(mask, col, a=None):
        nonlocal rgb, alpha
        m = mask.astype(np.float32)
        rgb = rgb * (1 - m[..., None]) + col * m[..., None]
        alpha = np.maximum(alpha, m)
        if a is not None:
            ang[m > 0.5] = a[m > 0.5]

    vn = fbm(hgt, wid, 30, "tw-var", octaves=4)
    streaks = value_noise(hgt, wid, 1, "tw-st")  # placeholder replaced below
    st = fbm(hgt, wid * 1, 7, "tw-streak-x", octaves=2)
    stv = np.repeat(fbm(1, wid, 5, "tw-streak-1d", octaves=2), hgt, 0)
    streaks = np.clip((stv - 0.55) * 2.5, 0, 1) * fbm(hgt, wid, 60, "tw-streak-y", octaves=2)

    # ------------------------------------------------ body (cylinder of courses)
    def cylinder(hmin, hmax, Rf, course_h, key, base_col_lit=STONE_LIT, base_col_sh=STONE_SH, blocks=True):
        # Rf: radius as a function of h. Iterate twice to resolve h from y.
        Rg = Rf(np.clip(yy, hmin, hmax))
        u = np.clip(dx / Rg, -1, 1)
        cos_t = np.sqrt(1 - u * u)
        h = _h_of(yy, Rg, cos_t)
        Rg = Rf(np.clip(h, hmin, hmax))
        u = dx / Rg
        inside = (np.abs(u) < 1.0)
        u = np.clip(u, -0.9999, 0.9999)
        cos_t = np.sqrt(1 - u * u)
        h = _h_of(yy, Rg, cos_t)
        inside &= (h >= hmin) & (h < hmax)
        sin_t = u
        theta = np.arcsin(u)
        s = Rg * theta
        ndl = _shade_wall(sin_t, cos_t)
        tone = P.smoothstep(-0.10, 0.40, ndl)
        col = P.mix(base_col_sh, base_col_lit, tone)
        # light falls off down the tower a little and the base is in the clouds' glow
        if blocks:
            c = np.floor(h / course_h)
            ci = c.astype(np.int64)
            hs = (ci * 7919 + seed_of(key)) % 100003
            Lc = 18.0 + 20.0 * ((hs * 37 % 101) / 101.0)
            off = ((hs * 53) % 97) / 97.0 * Lc
            bpos = (s + off) / Lc
            b = np.floor(bpos)
            fu = bpos - b
            fv = h / course_h - c
            bid = (ci * 131 + b.astype(np.int64) * 977 + seed_of(key, 1)) % 1000003
            rj = (bid * 2654435761 % 4294967296) / 4294967296.0
            rj2 = (bid * 40503 % 65536) / 65536.0
            # per stone value and hue
            val = 0.86 + 0.24 * rj
            warm = (rj2 > 0.78).astype(np.float32) * 0.6
            green = ((rj2 > 0.08) & (rj2 < 0.16)).astype(np.float32) * 0.6
            col = col * val[..., None]
            col = col + (warm * tone)[..., None] * np.array([0.05, 0.02, -0.04], np.float32)
            col = col + green[..., None] * np.array([-0.03, 0.025, -0.01], np.float32)
            # mortar joints (darker), sharper and lighter-edged on the lit side
            jw = 1.1 / Lc / np.maximum(cos_t, 0.25)
            joint_v = (fu < jw)
            joint_h = (fv < 0.10)
            j = (joint_v | joint_h).astype(np.float32)
            col = col * (1 - 0.38 * j)[..., None]
            edge = ((fv >= 0.10) & (fv < 0.22)).astype(np.float32) * tone
            col = col * (1 + 0.10 * edge)[..., None]
            # chipped corners and worn faces
            wear = P.smoothstep(0.62, 0.85, vn) * (0.5 + 0.5 * rj)
            col = col * (1 - 0.12 * wear)[..., None]
        # rain streaks running down
        col = col * (1 - 0.18 * streaks)[..., None]
        # rose rim of the dawn on the far right edge
        rim = P.smoothstep(0.80, 0.97, sin_t)
        col = P.mix(col, STONE_RIM * 0.85, rim * 0.6)
        # direction of courses (ring tangent)
        k = (h - EYE) * Rg / FOCAL
        slope = -k * sin_t / (Rg * np.maximum(cos_t, 0.2))
        a = np.arctan(slope).astype(np.float32)
        return inside, col, h, theta, s, tone, a, cos_t, sin_t, Rg

    inside, col, h, theta, s, tone, a, cos_t, sin_t, Rg = cylinder(404, BASE_Y - 30, body_radius, 12.0, "body")
    # plinth
    pin, pcol, ph, *_r = cylinder(BASE_Y - 30, BASE_Y + 20, lambda q: body_radius(q) + 10.0, 15.0, "plinth")
    # string course
    scin, sccol, *_r2 = cylinder(606, 614, lambda q: body_radius(q) + 3.5, 100.0, "sc", blocks=False)
    # shadow under the string course and under the cornice
    sh1 = P.smoothstep(614, 618, h) * (1 - P.smoothstep(618, 628, h))
    sh2 = (1 - P.smoothstep(404, 420, h)) * (h >= 404)
    col = col * (1 - 0.35 * sh1 - 0.55 * sh2)[..., None]

    # windows
    windows = []
    win_mask_total = np.zeros_like(alpha)
    for wi, wa in enumerate(WIN_ANG):
        ta = math.radians(wa)
        sk = body_radius(520) * ta
        ds = s - sk
        hw = WIN_W / 2
        in_rect = (np.abs(ds) < hw) & (h <= WIN_SILL) & (h >= WIN_SPRING)
        in_arch = ((ds ** 2 + (h - WIN_SPRING) ** 2) < hw ** 2) & (h < WIN_SPRING)
        opening = (in_rect | in_arch) & inside
        # arch ring of voussoirs and the jambs
        rr = np.sqrt(ds ** 2 + np.minimum(h - WIN_SPRING, 0) ** 2)
        ring = (((rr >= hw) & (rr < hw + 4.0) & (h < WIN_SPRING + 2)) |
                ((np.abs(ds) >= hw) & (np.abs(ds) < hw + 3.0) & (h >= WIN_SPRING) & (h <= WIN_SILL))) & inside
        sill = (np.abs(ds) < hw + 4.5) & (h > WIN_SILL) & (h < WIN_SILL + 4.0) & inside
        col = np.where(ring[..., None], col * 1.10, col)
        col = np.where(sill[..., None], col * (1.18 - 0.0 * tone)[..., None], col)
        undersill = (np.abs(ds) < hw + 4.5) & (h >= WIN_SILL + 4.0) & (h < WIN_SILL + 8.0) & inside
        col = np.where(undersill[..., None], col * 0.72, col)
        # interior: dark, the right reveal catches a little light
        reveal = (ds > hw - 2.6) & opening
        icol = np.broadcast_to(WIN_DARK, col.shape).copy()
        icol[reveal] = (col[reveal] * 0.55)
        col = np.where(opening[..., None], icol, col)
        win_mask_total = np.maximum(win_mask_total, opening.astype(np.float32))
        windows.append(dict(angle=wa, mask=opening.astype(np.float32), ds=ds.astype(np.float32),
                            h=h.astype(np.float32)))
    # small slit windows lower down
    for wa, hc in [(-24.0, 662.0), (21.0, 662.0), (-48.0, 700.0)]:
        sk = body_radius(hc) * math.radians(wa)
        ds = s - sk
        op = (np.abs(ds) < 3.2) & (np.abs(h - hc) < 11) & inside
        col = np.where(op[..., None], WIN_DARK * 1.2, col)

    # weathering: dark streaks washing down from every sill and from the string
    # course, damp darkening near the base, pale lichen blooms
    stain = np.zeros_like(alpha)
    for wa in WIN_ANG:
        sk = body_radius(520) * math.radians(wa)
        ds2 = s - sk
        dh = h - (WIN_SILL + 6)
        wdt = 5.0 + 2.5 * np.clip(dh / 60.0, 0, 1)
        st1 = np.exp(-(ds2 / wdt) ** 2) * np.clip(1 - dh / (55 + 25 * math.cos(wa)), 0, 1) * (dh > 0)
        stain = np.maximum(stain, st1)
    dh2 = h - 616
    runs = np.clip((fbm(hgt, wid, 4, "runs", octaves=2) - 0.55) * 3, 0, 1)
    stain = np.maximum(stain, runs * np.clip(1 - dh2 / 70, 0, 1) * (dh2 > 0))
    stain = np.maximum(stain, runs * np.clip(1 - (h - 418) / 90, 0, 1) * (h > 418) * 0.8)
    col = col * (1 - 0.28 * stain * (0.6 + 0.4 * value_noise(hgt, wid, 3, "stainv")))[..., None]
    damp = P.smoothstep(640, BASE_Y, h)
    col = col * (1 - 0.18 * damp)[..., None] + damp[..., None] * np.array([-0.01, 0.01, 0.0], np.float32)
    lich = P.smoothstep(0.78, 0.9, fbm(hgt, wid, 12, "lichen", octaves=3)) * tone
    col = P.mix(col, col * np.array([1.10, 1.14, 1.0], np.float32), lich * 0.6)
    paint_part(inside, col, a)
    paint_part(scin, sccol * 1.08, a)
    paint_part(pin, pcol * 0.95, a)

    # ------------------------------------------------ corbels, cornice, parapet
    Rtop = body_radius(404)
    cin, ccol, chh, cth, cs, ctone, ca, *_c = cylinder(392, 404, lambda q: Rtop + 9.0, 100.0, "cornice", blocks=False)
    ccol = ccol * 1.12
    # the soffit band at the bottom of the cornice is in shadow
    ccol = np.where((chh > 400)[..., None], ccol * 0.62, ccol)
    # corbels under the cornice
    corb = np.zeros_like(alpha, bool)
    Rcb = Rtop + 4.0
    ucb = np.clip(dx / Rcb, -0.9999, 0.9999)
    ctb = np.sqrt(1 - ucb ** 2)
    hcb = _h_of(yy, Rcb, ctb)
    scb = Rcb * np.arcsin(ucb)
    period = Rcb * math.radians(13.0)
    fr = np.mod(scb + period * 0.5, period) - period * 0.5
    corb = (np.abs(fr) < 3.2) & (hcb >= 404) & (hcb < 415) & (np.abs(dx) < Rcb)
    cbcol = P.mix(STONE_SH, STONE_LIT, P.smoothstep(-0.1, 0.4, _shade_wall(ucb, ctb)))[..., :] * 0.95
    cbcol = np.where((fr > 1.2)[..., None], cbcol * 0.7, cbcol)
    paint_part(corb, cbcol)
    paint_part(cin, ccol, ca)

    pin2, pcol2, phh, pth, ps, ptone, pa, *_p = cylinder(368, 392, lambda q: Rtop + 6.0, 24.0, "parapet")
    period = (Rtop + 6) * math.radians(8.0)
    frp = np.mod(ps + period * 0.5, period) - period * 0.5
    holes = (np.abs(frp) < 1.6) & (phh > 373) & (phh < 388) & pin2
    pcol2 = np.where(holes[..., None], WIN_DARK * 1.4, pcol2)
    capline = pin2 & (phh < 371)
    pcol2 = np.where(capline[..., None], pcol2 * 1.15, pcol2)
    paint_part(pin2, pcol2, pa)

    # ------------------------------------------------ upper drum
    din, dcol, dhh, dth, ds_, dtone, da, *_d = cylinder(322, 368, lambda q: 60.0, 11.0, "drum")
    for wa in [-60.0, -30.0, 0.0, 30.0, 60.0]:
        sk = 60.0 * math.radians(wa)
        dd = ds_ - sk
        op = (((np.abs(dd) < 4.6) & (dhh > 340) & (dhh < 362)) | ((dd ** 2 + (dhh - 340) ** 2) < 4.6 ** 2)) & din
        dcol = np.where(op[..., None], WIN_DARK, dcol)
    # verdigris stains washing down from the dome
    stain = streaks * P.smoothstep(470, 330, yy)
    dcol = P.mix(dcol, dcol * np.array([0.75, 1.05, 1.0], np.float32), np.clip(stain * 1.5, 0, 0.8))
    paint_part(din, dcol, da)

    # ------------------------------------------------ dome
    lip_in, lipcol, *_l = cylinder(316, 323, lambda q: 66.0, 100.0, "lip", base_col_lit=COP_LIT * 0.9,
                                   base_col_sh=COP_SH, blocks=False)
    paint_part(lip_in, lipcol)
    a_ax, b_ax = 64.0, 66.0
    base_shift = (316 - EYE) * a_ax / FOCAL
    u = dx / a_ax
    cu = np.sqrt(np.clip(1 - u * u, 0, 1))
    ybase = 316 + base_shift * cu
    v = (yy - ybase) / b_ax
    dome = (u * u + v * v < 1.0) & (v < 0)
    nz = np.sqrt(np.clip(1 - u * u - v * v, 0.0, 1.0))
    n = np.stack([u, v * (a_ax / b_ax), nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    ndl = n @ LT
    dt = P.smoothstep(-0.05, 0.55, ndl)
    dcol2 = P.mix(COP_SH, COP_MID, P.smoothstep(-0.3, 0.2, ndl))
    dcol2 = P.mix(dcol2, COP_LIT, dt)
    lon = np.degrees(np.arctan2(u, np.maximum(nz, 1e-3)))
    rib = np.mod(lon + 9.0, 18.0) - 9.0
    ribw = 0.9 / np.maximum(np.cos(np.radians(lon)), 0.25)
    ribline = np.abs(rib) < ribw
    dcol2 = np.where(ribline[..., None], dcol2 * 1.18, dcol2)
    ribsh = (rib > ribw) & (rib < ribw * 2.6)
    dcol2 = np.where(ribsh[..., None], dcol2 * 0.8, dcol2)
    # copper sheet seams
    lat = np.degrees(np.arcsin(np.clip(-v, 0, 1)))
    seam = np.abs(np.mod(lat + 3.5, 10.0) - 5.0) < 0.35
    dcol2 = np.where(seam[..., None], dcol2 * 0.9, dcol2)
    # patina: darker green patches, a few copper brown spots, pale powder
    pat = fbm(hgt, wid, 9, "patina", octaves=3)
    pat2 = fbm(hgt, wid, 22, "patina2", octaves=3)
    dcol2 = dcol2 * (0.84 + 0.3 * pat)[..., None]
    brown = P.smoothstep(0.80, 0.92, pat2)
    dcol2 = P.mix(dcol2, dcol2 * np.array([1.35, 0.85, 0.6], np.float32), brown * 0.7)
    pale = P.smoothstep(0.75, 0.9, 1 - pat2) * dt
    dcol2 = P.mix(dcol2, np.array([0.62, 0.80, 0.74], np.float32), pale * 0.35)
    dcol2 = dcol2 * (1 - 0.25 * streaks)[..., None]
    # dawn rim on the right
    rimd = P.smoothstep(0.82, 0.98, u) * np.clip(1 + v, 0, 1)
    dcol2 = P.mix(dcol2, np.array([0.70, 0.60, 0.58], np.float32), rimd * 0.45)
    dome_ang = np.radians(90.0 + lon * 0.0) * 0 + np.arctan2(-1.0, np.where(np.abs(u) > 1e-3, -u * 0.6, 1e-3))
    paint_part(dome, dcol2, np.where(dome, (np.pi / 2 + np.arctan(u * 1.2) * 0.8), 0).astype(np.float32))

    # cracks in the dome: jagged dark lines, the biggest with a broken gap
    r = rng("dome-cracks")
    details = P.Strokes()
    crack_lines = []
    # ribs restated as painted lines: a lit thread on the light side of each, a dark seam beside it
    for lon_d in np.arange(-72, 73, 18):
        lon_r = math.radians(lon_d)
        pts_l, pts_d = [], []
        for vv in np.linspace(-0.97, -0.04, 12):
            rr = math.sqrt(max(1 - vv * vv, 0))
            uu = rr * math.sin(lon_r)
            px = CX + uu * a_ax
            py = 316 + base_shift * math.sqrt(max(1 - uu * uu, 0)) + vv * b_ax
            pts_l.append((px - x0 - 0.6, py - y0))
            pts_d.append((px - x0 + 1.0, py - y0))
        lit = max(0.0, -math.sin(lon_r) * 0.6 + 0.5)
        details.add(pts_d, 0.7, COP_SH * 0.8, seed=seed_of("ribd", lon_d), opacity=0.55, dry=0.5, ragged=0.3,
                    bristles=2, pool=0.0, streak=0.0, taper0=0.2, taper1=0.2)
        details.add(pts_l, 0.6, P.mix(COP_MID, COP_LIT * 1.15, lit), seed=seed_of("ribl", lon_d), opacity=0.6,
                    dry=0.55, ragged=0.3, bristles=2, pool=0.0, streak=0.0, taper0=0.2, taper1=0.2)
    for ci in range(3):
        uu = -0.5 + 1.0 * r.random()
        vv = -0.75 + 0.3 * r.random()
        pts = []
        for k in range(int(6 + r.integers(0, 6))):
            px = CX + uu * a_ax * math.sqrt(max(1 - vv * vv, 0.0))
            py = 316 + base_shift * math.sqrt(max(1 - uu * uu, 0)) + vv * b_ax
            pts.append((px, py))
            vv += 0.05 + 0.04 * r.random()
            uu += (r.random() - 0.5) * 0.16
            if vv > -0.05:
                break
        crack_lines.append(pts)
        w = 1.2 + 0.8 * r.random() if ci else 2.2
        details.add([(x - x0, y - y0) for x, y in pts], w, (0.04, 0.07, 0.10), seed=seed_of("crack", ci),
                    opacity=0.95, dry=0.2, ragged=0.45, bristles=2, pool=0.0, streak=0.0, taper0=0.3, taper1=0.5)
        # lit lip on the upper left of each crack
        details.add([(x - x0 - 1.2, y - y0 - 0.8) for x, y in pts], 0.7, (0.62, 0.84, 0.76), seed=seed_of("crackl", ci),
                    opacity=0.7, dry=0.5, ragged=0.4, bristles=2, pool=0.0, streak=0.0, taper0=0.3, taper1=0.5)

    # ------------------------------------------------ cupola and finial
    cup_in, cupcol, *_cu = cylinder(238, 252, lambda q: 11.0, 100.0, "cup", blocks=False)
    for wa in [-35.0, 0.0, 35.0]:
        sk = 11.0 * math.radians(wa)
        cupcol = np.where(((np.abs(_cu[2] - sk) < 2.2) & cup_in)[..., None] if False else cupcol, cupcol, cupcol)
    cupcol = np.where(((np.abs(dx) < 2.0) | (np.abs(dx - 7) < 1.5) | (np.abs(dx + 7) < 1.5))[..., None] & ((yy > 241) & (yy < 250))[..., None],
                      WIN_DARK, cupcol)
    paint_part(cup_in, cupcol)
    cap = ((dx / 12.5) ** 2 + ((yy - 239) / 11.0) ** 2 < 1) & (yy < 239)
    capn = np.clip(-(dx / 12.5) * 0.6 - ((yy - 239) / 11.0) * 0.6 + 0.3, 0, 1)
    paint_part(cap, P.mix(COP_SH, COP_LIT, capn))
    fin = (np.abs(dx) < 1.1) & (yy > 214) & (yy < 229)
    paint_part(fin, np.array([0.30, 0.42, 0.42], np.float32))
    ball = (dx ** 2 + (yy - 222) ** 2) < 2.6 ** 2
    paint_part(ball, np.array([0.45, 0.60, 0.56], np.float32))

    # ------------------------------------------------ hand placed sharpening
    def ring_pts(hh, R, a0=-85, a1=85, step=3):
        pts = []
        for ad in np.arange(a0, a1 + 0.1, step):
            t = math.radians(ad)
            pts.append((CX + R * math.sin(t) - x0, _ring_y(hh, R, math.cos(t)) - y0))
        return pts

    # cornice top edge catches the moon on the lit side; parapet cap too
    for hh, R, c, w in [(392.5, Rtop + 9.0, (0.72, 0.72, 0.86), 1.3), (368.5, Rtop + 6.0, (0.68, 0.68, 0.84), 1.1),
                        (606.5, body_radius(606) + 3.5, (0.66, 0.66, 0.82), 1.0),
                        (BASE_Y - 30.5, body_radius(BASE_Y - 30) + 10.0, (0.60, 0.60, 0.78), 1.1)]:
        pts = ring_pts(hh, R, -86, 20, 2.5)
        details.add(pts, w, c, seed=seed_of("ringhl", hh), opacity=0.8, dry=0.65, ragged=0.35, bristles=3,
                    pool=0.0, streak=0.1, taper0=0.25, taper1=0.6)
    # dark accent under the cornice soffit
    pts = ring_pts(404.5, Rtop + 9.0, -88, 88, 2.5)
    details.add(pts, 1.6, (0.08, 0.06, 0.16), seed=seed_of("soffit"), opacity=0.75, dry=0.3, ragged=0.3,
                bristles=3, pool=0.0, streak=0.0)
    # window sills: crisp light tops
    for wa in WIN_ANG:
        t = math.radians(wa)
        R = body_radius(WIN_SILL)
        if abs(wa) > 70:
            continue
        sk = R * t
        pts = []
        for dsd in np.linspace(-WIN_W / 2 - 4, WIN_W / 2 + 4, 6):
            tt = (sk + dsd) / R
            pts.append((CX + R * math.sin(tt) - x0, _ring_y(WIN_SILL + 0.5, R, math.cos(tt)) - y0))
        lit = max(0.0, _shade_wall(math.sin(t), math.cos(t)))
        c = P.mix(STONE_SH, STONE_LIT * 1.15, min(1.0, lit * 2.2 + 0.15))
        details.add(pts, 1.1, c, seed=seed_of("sill", wa), opacity=0.85, dry=0.4, ragged=0.3, bristles=3,
                    pool=0.0, streak=0.0, taper0=0.2, taper1=0.3)
    # restate each window opening with a crisp dark stroke and its arch
    for wa in WIN_ANG:
        if abs(wa) > 70:
            continue
        t = math.radians(wa)
        R = body_radius(520)
        cxw = CX + R * math.sin(t) - x0
        hw = WIN_W / 2 * math.cos(t)
        ytop = _ring_y(WIN_SPRING, R, math.cos(t)) - y0
        ybot = _ring_y(WIN_SILL, R, math.cos(t)) - y0
        details.add([(cxw, ybot - 0.5), (cxw, ytop)], hw * 0.82, WIN_DARK, seed=seed_of("win", wa), opacity=0.9,
                    dry=0.1, ragged=0.08, bristles=2, pool=0.0, streak=0.0, taper0=0.0, taper1=0.0, cap=1.0)
        details.add([(cxw, ytop + 0.5), (cxw, ytop)], hw * 0.82, WIN_DARK, seed=seed_of("arch", wa), opacity=0.9,
                    dry=0.1, ragged=0.06, bristles=2, pool=0.0, streak=0.0, taper0=0.0, taper1=0.0)
    # dawn rim line down the right edge of the body and dome
    rim_pts = [(CX + body_radius(hh) * 0.995 - x0, hh - y0) for hh in np.linspace(420, BASE_Y - 34, 16)]
    details.add(rim_pts, 1.0, (0.86, 0.62, 0.62), seed=seed_of("rim"), opacity=0.55, dry=0.75, ragged=0.4,
                bristles=3, pool=0.0, streak=0.0, taper0=0.3, taper1=0.4)

    # place into full array
    full_rgb = np.zeros((AH, AW, 3), np.float32)
    full_a = np.zeros((AH, AW), np.float32)
    full_ang = np.zeros((AH, AW), np.float32)
    sl = (slice(y0 + M, y1 + M), slice(x0 + M, x1 + M))
    full_rgb[sl] = rgb
    full_a[sl] = alpha
    full_ang[sl] = ang
    det = details.arrays()
    det["xs"] += x0 + M
    det["ys"] += y0 + M
    for wdw in windows:
        wdw["origin"] = (x0 + M, y0 + M)
    return dict(rgb=full_rgb, alpha=full_a, ang=full_ang, windows=windows, details=det,
                crop=(x0 + M, y0 + M, x1 + M, y1 + M), tone_crop=tone, theta_crop=theta, h_crop=h,
                inside_crop=inside)


# ---------------------------------------------------------------- the cliff

def cliff_reference():
    """A crag of big faceted planes, the way a painter blocks rock: lit tops
    and left faces, violet shadow faces, crisp cracks between planes, moss on
    the ledges, and mist where it sinks into the clouds."""
    from scipy.spatial import cKDTree
    top = [(1385, 800), (1400, 786), (1440, 779), (1490, 773), (1560, 770), (1630, 772), (1690, 777),
           (1735, 786), (1752, 800)]
    right = [(1770, 845), (1792, 880), (1786, 905), (1815, 950), (1850, 1010), (1868, 1060), (1905, 1130),
             (1950, 1230), (1990, 1340)]
    left = [(1130, 1340), (1180, 1230), (1215, 1140), (1236, 1075), (1262, 1020), (1240, 985), (1255, 950),
            (1300, 930), (1330, 905), (1352, 870), (1370, 835)]
    outline = np.array(top + right + left, np.float64)
    outline = wobble(outline, 3.0, 36.0, "cliff-outline", closed=True)
    outline = wobble(outline, 1.2, 8.0, "cliff-outline2", closed=True)
    outline = outline + M
    mask = poly_mask(AH, AW, outline)
    ys, xs = np.nonzero(mask > 0.01)
    r = rng("cliff-facets")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    n = int((x1 - x0) * (y1 - y0) / 2200)
    sx = r.uniform(x0 - 20, x1 + 20, n)
    sy = r.uniform(y0 - 20, y1 + 20, n)
    ky = 0.42   # facets are tall: columnar rock
    tree = cKDTree(np.stack([sx, sy * ky], 1))
    d, idx = tree.query(np.stack([xs, ys * ky], 1).astype(np.float64), k=2)
    fid = idx[:, 0]
    gap = d[:, 1] - d[:, 0]
    # facet normals: the macro form of the crag plus a random tilt; some facets are ledges facing up
    axis = CX + M + (sy - (BASE_Y + M)) * 0.12
    half = 180 + (sy - (BASE_Y + M)) * 0.55
    bnx = np.clip((sx - axis) / np.maximum(half, 60), -1, 1) * 0.95
    ledge = r.random(n) < 0.18
    bny = np.where(ledge, -0.75, 0.05 + 0.15 * r.standard_normal(n))
    bnz = np.full(n, 0.55)
    nx = bnx + 0.14 * r.standard_normal(n)
    ny = bny + 0.10 * r.standard_normal(n)
    nz = bnz + 0.15 * r.random(n)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    ndl = nx * LT[0] + ny * LT[1] + nz * LT[2]
    tone = np.clip((ndl + 0.15) / 0.9, 0, 1)
    lit = np.array([0.520, 0.490, 0.640], np.float32)
    mid = np.array([0.300, 0.270, 0.430], np.float32)
    sh = np.array([0.105, 0.085, 0.205], np.float32)
    fcol = np.where(tone[:, None] > 0.45, mid + (lit - mid) * ((tone[:, None] - 0.45) / 0.55),
                    sh + (mid - sh) * (tone[:, None] / 0.45))
    fcol = fcol * (0.9 + 0.16 * r.random(n))[:, None]
    # shadow faces get a cool bounce from the clouds below
    fcol = fcol + np.clip(ny, 0, 1)[:, None] * np.array([0.03, 0.03, 0.08])
    # moss on the ledges
    moss = ledge & (r.random(n) < 0.45)
    mcol = np.where(tone[:, None] > 0.4, np.array([0.24, 0.32, 0.32]), np.array([0.07, 0.11, 0.15]))
    fcol = np.where(moss[:, None], fcol * 0.5 + mcol * 0.5, fcol)
    col = np.zeros((AH, AW, 3), np.float32)
    col[ys, xs] = fcol[fid]
    # cracks between planes, and texture inside them
    crack = np.zeros((AH, AW), np.float32)
    crack[ys, xs] = np.clip(1.2 - gap, 0, 1) * (r.random() * 0 + 1)
    tex = fbm(AH, AW, 6, "cliff-tex", octaves=3)
    strat = np.repeat(fbm(1, AW, 3, "cliff-strat", octaves=2), AH, 0)
    col = col * (0.88 + 0.18 * tex + 0.08 * strat)[..., None]
    col = col * (1 - 0.6 * crack)[..., None]
    yy = np.mgrid[0:AH, 0:AW][0].astype(np.float32)
    # grass and moss along the top edge
    topband = P.smoothstep(BASE_Y + M + 30, BASE_Y + M + 5, yy) * (mask > 0.5)
    tmoss = topband * P.smoothstep(0.35, 0.6, fbm(AH, AW, 10, "mtop", octaves=3))
    col = P.mix(col, np.array([0.10, 0.18, 0.18], np.float32), tmoss * 0.8)
    # the crag sinks into luminous mist
    mist = P.smoothstep(BASE_Y + M + 90, BASE_Y + M + 260, yy)
    col = P.mix(col, np.array([0.34, 0.31, 0.52], np.float32), mist * 0.8)
    # and dissolves into the cloud behind it
    mask = mask * (1.0 - P.smoothstep(BASE_Y + M + 240, BASE_Y + M + 360, yy))
    ang = P.orientation(col, sigma=1.5, rho=4, base=1.45, base_weight=0.8)
    return dict(rgb=col.astype(np.float32), alpha=mask, outline=outline, ang=ang, moss=tmoss)


# ---------------------------------------------------------------- pines

PINES = [
    dict(base=(1428, 786), height=250, lean=-0.55, key="pineA", spread=1.3, wind=-1.0),
    dict(base=(1728, 790), height=170, lean=0.40, key="pineB", spread=1.2, wind=1.0),
]

BARK_LIT = np.array([0.380, 0.300, 0.380], np.float32)
BARK_SH = np.array([0.100, 0.070, 0.120], np.float32)
NEEDLE_SH = np.array([0.045, 0.100, 0.120], np.float32)
NEEDLE_MID = np.array([0.090, 0.180, 0.200], np.float32)
NEEDLE_LIT = np.array([0.230, 0.370, 0.390], np.float32)


def pines_reference():
    """Colour reference and alpha for the pines plus stroke lists:
    'trunk' strokes (bark along the trunk) and 'needles' (tufts on pads)."""
    rgb = np.zeros((AH, AW, 3), np.float32)
    alpha = np.zeros((AH, AW), np.float32)
    trunk_st = P.Strokes()
    needle_st = P.Strokes()
    pad_st = P.Strokes()
    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float32)
    for pi, pd in enumerate(PINES):
        g = F.pine(pd["base"], pd["height"], pd["lean"], pd["key"], spread=pd["spread"], wind=pd["wind"])
        r = rng("pine-paint", pi)
        # trunk and branches as stamped discs into a mask, shaded across the width
        limbs = [g["trunk"]] + g["branches"]
        for li, limb in enumerate(limbs):
            pts = np.array([(x + M, y + M, w) for x, y, w in limb])
            # dense resample
            seg = np.sqrt(((pts[1:, :2] - pts[:-1, :2]) ** 2).sum(1))
            cum = np.concatenate([[0], np.cumsum(seg)])
            ss = np.arange(0, cum[-1], 0.7)
            px = np.interp(ss, cum, pts[:, 0])
            py = np.interp(ss, cum, pts[:, 1])
            pw = np.interp(ss, cum, pts[:, 2])
            x0, x1 = int(px.min() - pw.max() - 3), int(px.max() + pw.max() + 3)
            y0, y1 = int(py.min() - pw.max() - 3), int(py.max() + pw.max() + 3)
            sy, sx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            best = np.full(sx.shape, 1e9, np.float32)
            side = np.zeros(sx.shape, np.float32)
            for k in range(0, len(px), 2):
                d = np.sqrt((sx - px[k]) ** 2 + (sy - py[k]) ** 2) / max(pw[k], 0.6)
                upd = d < best
                best = np.where(upd, d, best)
                side = np.where(upd, (sx - px[k]) / max(pw[k], 0.6), side)
            m = np.clip((1.0 - best) * max(pw.mean(), 1.0) + 0.5, 0, 1)
            # light from the left: the left side of the limb is lit, upper side for branches
            t = np.clip(-side * 0.8 + 0.25, 0, 1)
            col = P.mix(BARK_SH, BARK_LIT, P.smoothstep(0.25, 0.85, t))
            col = col * (0.8 + 0.35 * value_noise(*sx.shape, 3, ("bark", pi, li)))[..., None]
            reg = (slice(max(y0, 0), min(y1, AH)), slice(max(x0, 0), min(x1, AW)))
            cy0, cx0 = reg[0].start - y0, reg[1].start - x0
            mm = m[cy0:cy0 + reg[0].stop - reg[0].start, cx0:cx0 + reg[1].stop - reg[1].start]
            cc = col[cy0:cy0 + mm.shape[0], cx0:cx0 + mm.shape[1]]
            rgb[reg] = rgb[reg] * (1 - mm[..., None]) + cc * mm[..., None]
            alpha[reg] = np.maximum(alpha[reg], mm)
            # bark strokes along the limb
            for k in range(0, len(px) - 6, 5):
                wk = pw[k]
                for q in range(2 if wk > 3 else 1):
                    off = (r.random() - 0.5) * wk * 1.2
                    dxl = px[min(k + 6, len(px) - 1)] - px[k]
                    dyl = py[min(k + 6, len(px) - 1)] - py[k]
                    ln = math.hypot(dxl, dyl) + 1e-6
                    nxl, nyl = -dyl / ln, dxl / ln
                    a = (k, min(k + 6 + int(r.integers(0, 4)), len(px) - 1))
                    path = [(px[i] + nxl * off, py[i] + nyl * off) for i in range(a[0], a[1] + 1, 2)]
                    lit = np.clip(-(nxl * off) / max(wk, 1) * 0.8 + 0.3, 0, 1)
                    c = P.mix(BARK_SH, BARK_LIT, float(lit)) * (0.85 + 0.3 * r.random())
                    trunk_st.add(path, max(0.6, wk * 0.35), c, seed=int(r.integers(1 << 30)), opacity=0.8,
                                 dry=0.6, ragged=0.3, bristles=3, pool=0.05, streak=0.15, pickup=0.1)
        # foliage pads
        for qi, pad in enumerate(g["pads"]):
            pcx, pcy, prx, pry = pad
            pcx += M
            pcy += M
            lobes = F.pad_lobes((pcx, pcy, prx, pry), (pd["key"], qi))
            for (lx, ly, lrx, lry) in lobes:
                m = ellipse_mask(AH, AW, lx, ly, lrx, lry, ss=2) if False else None
                x0, x1 = int(lx - lrx - 2), int(lx + lrx + 3)
                y0, y1 = int(ly - lry - 2), int(ly + lry + 3)
                sy, sx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                d = ((sx - lx) / lrx) ** 2 + ((sy - ly) / lry) ** 2
                m = np.clip((1 - d) * min(lrx, lry) * 0.6 + 0.5, 0, 1)
                # lit on top, dark below; the whole pad gets lighter toward its top edge
                vt = np.clip((sy - (pcy - pry)) / (2 * pry), 0, 1)
                t = np.clip(1.0 - vt * 1.3 - (sx - pcx) / prx * 0.15, 0, 1)
                col = P.mix(NEEDLE_SH, NEEDLE_MID, P.smoothstep(0.1, 0.5, t))
                col = P.mix(col, NEEDLE_LIT, P.smoothstep(0.55, 0.95, t))
                rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - m[..., None]) + col * m[..., None]
                alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], m)
            # needle tufts: short strokes fanning up and out along the pad's top edge,
            # dark ones underneath for the ragged lower edge
            ntufts = int(prx * 1.6)
            for k in range(ntufts):
                u = r.random() * 2 - 1
                prof = math.sqrt(max(1 - u * u, 0.0))
                tx = pcx + u * prx * 0.95
                ty = pcy - pry * 0.55 * prof + (r.random() - 0.3) * pry * 0.8
                topness = np.clip(1.0 - (ty - (pcy - pry)) / (2 * pry) * 1.2, 0, 1)
                for q in range(3):
                    ang = -math.pi / 2 + u * 0.9 + (r.random() - 0.5) * 1.3
                    ln = 3.0 + r.random() * 5.0 * (0.5 + prx / 60)
                    ex, ey = tx + math.cos(ang) * ln, ty + math.sin(ang) * ln
                    c = P.mix(NEEDLE_MID, NEEDLE_LIT * 1.15, float(topness) * (0.6 + 0.5 * r.random()))
                    if r.random() < 0.25:
                        c = NEEDLE_SH * 1.1
                    needle_st.add([(tx, ty), ((tx + ex) / 2 + (r.random() - 0.5), (ty + ey) / 2), (ex, ey)],
                                  0.55 + 0.45 * r.random(), c, seed=int(r.integers(1 << 30)), opacity=0.9, dry=0.3,
                                  ragged=0.2, bristles=2, pool=0.0, streak=0.1, taper0=0.1, taper1=0.7,
                                  pickup=0.0)
            for k in range(int(prx * 0.8)):
                u = r.random() * 2 - 1
                tx = pcx + u * prx * 0.9
                ty = pcy + pry * 0.35 * math.sqrt(max(1 - u * u, 0)) + r.random() * 2
                ang = math.pi / 2 + (r.random() - 0.5) * 1.6
                ln = 2.5 + r.random() * 4
                needle_st.add([(tx, ty), (tx + math.cos(ang) * ln, ty + math.sin(ang) * ln)], 0.6, NEEDLE_SH,
                              seed=int(r.integers(1 << 30)), opacity=0.85, dry=0.3, ragged=0.2, bristles=2,
                              pool=0.0, streak=0.0, taper0=0.1, taper1=0.7, pickup=0.0)
            # clumpy dabs over the pad body
            for k in range(int(prx * pry / 6) + 8):
                u = r.random() * 2 - 1
                v = (r.random() * 2 - 1) * 0.8
                if u * u + v * v > 1:
                    continue
                tx, ty = pcx + u * prx * 0.85, pcy + v * pry * 0.8
                t = float(np.clip(1.0 - (v + 1) / 2 * 1.3, 0, 1))
                c = P.mix(NEEDLE_SH, NEEDLE_LIT, t * (0.7 + 0.5 * r.random()))
                a = (r.random() - 0.5) * 0.8
                ln = 2.5 + r.random() * 4
                a = -math.pi / 2 + u * 0.8 + (r.random() - 0.5) * 1.4
                pad_st.add([(tx - math.cos(a) * ln * 0.3, ty - math.sin(a) * ln * 0.3), (tx + math.cos(a) * ln, ty + math.sin(a) * ln)],
                           1.3 + 1.4 * r.random(), c, seed=int(r.integers(1 << 30)), opacity=0.85, dry=0.45,
                           ragged=0.45, bristles=4, pool=0.08, streak=0.12, pickup=0.1)
    return dict(rgb=rgb, alpha=alpha, trunk=trunk_st.arrays(), needles=needle_st.arrays(), pads=pad_st.arrays())


# ---------------------------------------------------------------- ivy

IVY_SH = np.array([0.040, 0.085, 0.100], np.float32)
IVY_MID = np.array([0.090, 0.170, 0.170], np.float32)
IVY_LIT = np.array([0.250, 0.400, 0.360], np.float32)


def ivy(tower):
    """Ivy climbing the tower from the cliff: a clump mass for the reference
    and leaf strokes on top. Returns dict(rgb, alpha, leaves, vines)."""
    x0, y0, x1, y1 = tower["crop"]
    inside = tower["inside_crop"]
    theta = tower["theta_crop"]
    h = tower["h_crop"]
    r = rng("ivy-paint")
    hgt, wid = inside.shape
    yy, xx = np.mgrid[0:hgt, 0:wid].astype(np.float32)
    # growth field: thick at the base, climbing the left and front, thinning upward
    thd = np.degrees(theta)
    base = P.smoothstep(640, 760, h)
    climb = P.smoothstep(-5, -45, thd) * P.smoothstep(470, 700, h)
    tongue = np.exp(-((thd - 25) / 9) ** 2) * P.smoothstep(560, 700, h)
    tongue2 = np.exp(-((thd + 62) / 9) ** 2) * P.smoothstep(420, 520, h)
    grow = np.clip(base * 0.9 + climb * 0.85 + tongue * 0.75 + tongue2 * 0.6, 0, 1)
    clump = fbm(hgt, wid, 7, "ivy-clump", octaves=4)
    dens = np.clip(grow * 1.4 - 0.65 + (clump - 0.5) * 1.1, 0, 1) * inside
    # keep window openings mostly clear
    m = P.smoothstep(0.25, 0.55, dens)
    tone = tower["tone_crop"]
    leafy = fbm(hgt, wid, 3, "ivy-leafy", octaves=2)
    col = P.mix(IVY_SH, IVY_MID, P.smoothstep(0.0, 0.6, tone * (0.6 + 0.6 * leafy)))
    col = P.mix(col, IVY_LIT, P.smoothstep(0.55, 1.0, tone * (0.5 + 0.7 * leafy)) * 0.8)
    # leaves: many small dabs where the ivy is, plus loose ones at the fringe
    leaves = P.Strokes(max_pts=3)
    ys, xs = np.nonzero(dens > 0.08)
    n = min(len(xs), 9000)
    pick = r.choice(len(xs), n, replace=False) if len(xs) > n else np.arange(len(xs))
    for i in pick:
        x, y = xs[i] + r.random(), ys[i] + r.random()
        d = dens[ys[i], xs[i]]
        if r.random() > d * 0.9 + 0.08:
            continue
        t = float(tone[ys[i], xs[i]])
        facing = r.random()
        c = P.mix(IVY_SH, IVY_MID, min(1.0, t * 1.4) * facing)
        if facing > 0.7 and t > 0.35:
            c = P.mix(c, IVY_LIT * 1.1, (facing - 0.7) / 0.3 * t)
        a = r.random() * math.pi
        ln = 1.2 + r.random() * 1.8
        leaves.add([(x - math.cos(a) * ln + x0, y - math.sin(a) * ln + y0), (x + math.cos(a) * ln + x0, y + math.sin(a) * ln + y0)],
                   1.5 + 1.3 * r.random(), c, seed=int(r.integers(1 << 30)), opacity=0.92, dry=0.2, ragged=0.35,
                   bristles=2, pool=0.12, streak=0.05, taper0=0.3, taper1=0.6, pickup=0.0)
    full_rgb = np.zeros((AH, AW, 3), np.float32)
    full_a = np.zeros((AH, AW), np.float32)
    full_rgb[y0:y1, x0:x1] = col
    full_a[y0:y1, x0:x1] = m
    # hanging strands below the cornice and trailing tendrils off the fringe
    vines = P.Strokes()
    for k in range(18):
        thd0 = -70 + r.random() * 120
        hh = 420 + r.random() * 30
        R = body_radius(hh)
        px = CX + R * math.sin(math.radians(thd0)) + M
        py = _ring_y(hh, R, math.cos(math.radians(thd0))) + M
        if r.random() < 0.55:
            continue
        ln = 25 + r.random() * 60
        pts = [(px + math.sin(q * 0.4 + k) * 1.5, py + q * ln / 8) for q in range(9)]
        vines.add(pts, 0.7, IVY_SH * 1.2, seed=seed_of("vine", k), opacity=0.85, dry=0.4, ragged=0.3, bristles=2,
                  pool=0, streak=0, pickup=0)
        for (vx, vy) in pts[1:]:
            if r.random() < 0.45:
                continue
            vy += (r.random() - 0.5) * ln / 7
            for q in range(1 + int(r.random() < 0.4)):
                a = r.random() * math.pi
                c = P.mix(IVY_SH, IVY_LIT, r.random() * 0.6)
                leaves.add([(vx + (r.random() - 0.5) * 3 - math.cos(a) * 1.4, vy - math.sin(a) * 1.4),
                            (vx + (r.random() - 0.5) * 3 + math.cos(a) * 1.4, vy + math.sin(a) * 1.4)],
                           1.3 + r.random(), c, seed=int(r.integers(1 << 30)), opacity=0.9, dry=0.2, ragged=0.35,
                           bristles=2, pool=0.1, streak=0.05, pickup=0.0)
    return dict(rgb=full_rgb, alpha=full_a, leaves=leaves.arrays(), vines=vines.arrays())
