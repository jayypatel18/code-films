"""The red panda: a parametric 2D cel drawing. Every pose is rebuilt from shapes."""

import numpy as np
from .cel import Cel, INK, tufted, fluffy
from .geom import catmull, ellipse, wobble, resample, stroke_outline, rot
from . import raster as R

# palette (cel colours)
RUSSET = (0.82, 0.40, 0.15)
RUSSET_D = (0.56, 0.22, 0.13)
DARK = (0.24, 0.13, 0.11)
DARK_D = (0.15, 0.08, 0.08)
CREAM = (0.98, 0.95, 0.87)
CREAM_D = (0.82, 0.72, 0.70)
TEAR = (0.58, 0.22, 0.11)
EAR_IN = (0.93, 0.76, 0.68)
EYE = (0.17, 0.09, 0.07)
IRIS = (0.62, 0.34, 0.12)
IRIS_L = (0.88, 0.60, 0.25)
NOSE = (0.16, 0.10, 0.10)
TONGUE = (0.85, 0.45, 0.45)
SHADOW_TINT = (0.62, 0.48, 0.62)  # multiply tint for the hard shadow


class Pose:
    """Everything that varies between drawings. Units: head radius = 1, y down."""

    def __init__(self, **kw):
        self.yaw = 0.0          # -1 left profile .. 0 front .. 1 right profile
        self.pitch = 0.0        # +1 looks down, -1 looks up
        self.eye_open = 1.0     # 0 closed .. 1 open (per eye overrides below)
        self.eye_open_l = None
        self.eye_open_r = None
        self.squeeze = 0.0      # squeezed shut (0..1), lids pressed
        self.look = (0.0, 0.0)  # pupil offset in eye units
        self.mouth = "smile"    # smile, line, o, teeth, grin, frown
        self.mouth_open = 0.0
        self.ear_l = 0.0        # extra rotation (radians), + flicks outward/back
        self.ear_r = 0.0
        self.brow = 0.0         # +1 determined (brows down), -1 worried
        self.cheek = 0.0        # puffed cheeks (wind)
        self.goggles = False
        self.goggles_up = True  # on forehead
        self.soot = 0.0
        self.wob = 0.0          # line wobble seed shift (boil)
        self.seed = 0
        for k, v in kw.items():
            setattr(self, k, v)


class Transform:
    def __init__(self, ox, oy, S, flip=False, rot=0.0):
        self.ox, self.oy, self.S, self.flip, self.rot = ox, oy, S, flip, rot

    def __call__(self, pts):
        pts = np.asarray(pts, np.float64)
        if pts.ndim == 1:
            pts = pts[None]
        x = pts[:, 0] * (-1 if self.flip else 1)
        y = pts[:, 1]
        if self.rot != 0.0:
            c, s_ = np.cos(self.rot), np.sin(self.rot)
            x, y = x * c - y * s_, x * s_ + y * c
        return np.stack([self.ox + x * self.S, self.oy + y * self.S], axis=1)

    def px(self, v):
        return v * self.S


def _ink_w(T, w):
    return max(T.px(w), 1.2)


def part(cel, T, poly_u, color, shadow_tint, light, ink_w, seed, shade=True, ink=True,
         wob=0.0, rim=None, shadow_a=0.85, inset=0.0, gap=True):
    """Fill a unit polygon, add the hard shadow crescent, ink it."""
    poly = T(poly_u)
    if wob > 0:
        poly = wobble(poly, T.px(wob), seed, closed=True)
    m = cel.fill(poly, color)
    if m is None:
        return None
    if shade:
        lx, ly = light
        d = T.px(0.16)
        mult_shade(cel, m, shadow_tint, -lx * d * (-1 if T.flip else 1), -ly * d, shadow_a, inset)
    if rim is not None:
        lx, ly = light
        d = T.px(0.07)
        cel.rim(m, rim, -lx * d * (-1 if T.flip else 1) * -1, -ly * d * -1, 0.9)
    if ink:
        lxs = light[0] * (-1 if T.flip else 1)
        cel.ink_closed(poly, _ink_w(T, ink_w), light=(lxs, light[1]), seed=seed, gap=gap)
    return m


def mult_shade(cel, m, tint, dx, dy, a=0.85, inset=0.0):
    """Hard crescent shadow as a multiply tint (keeps each local colour's hue family)."""
    from .cel import shift_sub
    if m is None:
        return
    sub, x0, y0 = m
    sh = shift_sub(sub, dx, dy)
    cres = sub * (1.0 - sh)
    if inset > 0:
        cres = cres * shift_sub(sub, -dx * inset, -dy * inset)
    cres = np.clip(cres * 1.6, 0, 1) * a  # harden the crescent edge
    h, w = sub.shape
    reg = cel.img[y0:y0 + h, x0:x0 + w]
    t = np.asarray(tint, np.float32)
    reg[:] = reg * (1 - cres[..., None]) + reg * t * cres[..., None]


# ----------------------------------------------------------------------------
# head
# ----------------------------------------------------------------------------

def head_outline(yaw, seed=0):
    """Broad head, flat crown, very fluffy cheeks, small chin; muzzle bulges with yaw."""
    pts = []
    n = 180
    tuft_angles = [0.18, 0.40, 0.62, np.pi - 0.18, np.pi - 0.40, np.pi - 0.62]  # just below horizontal: cheeks
    for i in range(n):
        a = -np.pi / 2 + 2 * np.pi * i / n
        ca, sa = np.cos(a), np.sin(a)
        r = 0.90 + 0.08 * max(sa, 0) ** 0.8 * (abs(ca) ** 0.5) + 0.12 * abs(ca) ** 2 - 0.04 * max(sa, 0) ** 3
        side = ca * np.sign(yaw) if yaw != 0 else 0
        r += 0.24 * abs(yaw) * max(side, 0) ** 2.5 * (0.25 + 0.75 * max(sa, 0) ** 0.5) * (1 - max(-sa, 0) ** 0.5) * (1 - max(sa, 0) ** 3)
        # fur bumps on the lower cheeks: sharp little triangles
        for k, ta in enumerate(tuft_angles):
            da = abs(a - ta)
            w = 0.11
            if da < w:
                r += (0.07 + 0.025 * ((k * 7) % 3)) * (1 - da / w) ** 0.6
        r *= 1 + 0.012 * (R.vnoise(i * 0.09, seed * 0.37, 5) - 0.5)
        pts.append((r * ca, r * sa * 0.92))
    return np.array(pts)


def sph(lon, lat, Y, pitch, r=1.0):
    """Project a point on the head sphere. lon: + is screen right when Y=0."""
    a = lon + Y
    x = r * np.sin(a) * np.cos(lat)
    y = -r * np.sin(lat) * 0.94 + pitch * 0.28 * np.cos(lat)
    z = np.cos(a) * np.cos(lat)
    return x, y, z


def draw_head(cel, T, cx, cy, pose, light):
    """Draw the head centred at unit (cx, cy)."""
    Y = pose.yaw * 1.25
    pitch = pose.pitch
    seed = pose.seed
    lxs = light[0] * (-1 if T.flip else 1)

    def U(pts):
        return [(cx + x, cy + y) for x, y in pts]

    # ---- ears (behind the head) ----
    ears = []
    for side in (-1, 1):
        lon = side * 1.15
        ex, ey, ez = sph(lon, 0.78, Y, pitch, 1.0)
        # ear base radius scales with facing
        extra = pose.ear_l if side < 0 else pose.ear_r
        tilt = side * 0.55 + extra + Y * 0.3
        sx = max(0.35, min(1.0, 0.55 + 0.6 * abs(np.cos(lon + Y))))
        ears.append((side, ex, ey, ez, tilt, sx))
    ears.sort(key=lambda e: e[3])  # far ear first
    for side, ex, ey, ez, tilt, sx in ears:
        base = np.array([(-0.42, 0.30), (-0.46, 0.0), (-0.30, -0.42), (-0.05, -0.62),
                         (0.18, -0.55), (0.40, -0.20), (0.44, 0.15), (0.30, 0.36)]) * np.array([sx, 1.0])
        if side < 0:
            base = base * np.array([-1, 1])
        ear = rot(catmull(base, closed=True, n=7), 0, 0, tilt)
        ear_pts = [(cx + ex + x, cy + ey + y) for x, y in ear]
        m = part(cel, T, ear_pts, RUSSET, SHADOW_TINT, light, 0.028, seed + 3 + side, gap=False)
        # white rim and inner tuft
        inner = np.array([(-0.26, 0.28), (-0.28, -0.05), (-0.14, -0.34), (0.08, -0.44),
                          (0.26, -0.14), (0.26, 0.18), (0.12, 0.30)]) * np.array([sx, 1.0])
        if side < 0:
            inner = inner * np.array([-1, 1])
        inner = rot(catmull(inner, closed=True, n=6), 0, 0, tilt)
        ip = T([(cx + ex + x, cy + ey + y) for x, y in inner])
        cel.fill(ip, CREAM)
        pink = np.array([(-0.14, 0.22), (-0.16, -0.02), (-0.04, -0.22), (0.12, -0.24),
                         (0.18, 0.0), (0.12, 0.22)]) * np.array([sx, 1.0])
        if side < 0:
            pink = pink * np.array([-1, 1])
        pink = rot(catmull(pink, closed=True, n=6), 0, 0, tilt)
        cel.fill(T([(cx + ex + x, cy + ey + y) for x, y in pink]), EAR_IN)
        # fur tuft lines in the ear
        for k in range(3):
            a0 = np.array([(-0.08 + 0.1 * k, 0.26), (-0.14 + 0.12 * k, -0.02 - 0.08 * k)]) * np.array([sx, 1])
            if side < 0:
                a0 = a0 * np.array([-1, 1])
            a0 = rot(a0, 0, 0, tilt)
            cel.ink(T([(cx + ex + x, cy + ey + y) for x, y in a0]), _ink_w(T, 0.012),
                    color=(0.75, 0.55, 0.45), seed=seed + 40 + k)

    # ---- head ----
    hp = U(head_outline(pose.yaw, seed))
    if pose.cheek > 0:
        # puffed cheeks: push the outline out low on both sides
        hp = np.asarray(hp, np.float64)
        ang = np.arctan2(hp[:, 1] - cy, hp[:, 0] - cx)
        bulge = pose.cheek * 0.16 * np.exp(-((np.abs(ang) - 0.35) / 0.35) ** 2) * (np.cos(ang) ** 2)
        hp = hp + np.stack([np.cos(ang) * bulge, np.sin(ang) * bulge], axis=1)
        hp = [tuple(p) for p in hp]
    hm = part(cel, T, hp, RUSSET, SHADOW_TINT, light, 0.034, seed + 1, shade=False, ink=False)
    head_poly = T(hp)

    # ---- white face mask ----
    # curve: dips between the eyes, wraps under the eyes, covers muzzle and chin
    mask_pts = []
    for lon, lat in [(-1.05, 0.05), (-0.80, 0.30), (-0.52, 0.40), (-0.30, 0.22), (0.0, 0.14),
                     (0.30, 0.22), (0.52, 0.40), (0.80, 0.30), (1.05, 0.05), (0.95, -0.40),
                     (0.55, -0.80), (0.0, -0.98), (-0.55, -0.80), (-0.95, -0.40)]:
        x, y, z = sph(lon, lat, Y, pitch, 0.98)
        mask_pts.append((cx + x, cy + y))
    mp = catmull(mask_pts, closed=True, n=6)
    mp = wobble(mp, 0.012, seed + 7, closed=True)
    full_head = np.zeros((cel.h, cel.w), np.float32)
    if hm is not None:
        sub, x0, y0 = hm
        full_head[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
    mm = cel.mask(T(mp))
    if mm is not None:
        mm = R.clip_mask(mm, full_head)
        cel.fill_mask(mm, CREAM)

    # ---- eyebrow spots (white) ----
    for side in (-1, 1):
        lon = side * 0.50
        x, y, z = sph(lon, 0.52, Y, pitch, 0.98)
        if z < 0.05:
            continue
        sx = max(0.3, abs(np.cos(lon + Y))) * 0.17
        spot = ellipse(cx + x, cy + y - 0.02 * pose.brow, sx, 0.12, n=14, rot=side * 0.3)
        cel.fill(T(wobble(spot, 0.008, seed + 9 + side, closed=True)), CREAM)

    # ---- tear marks ----
    for side in (-1, 1):
        lon = side * 0.46
        x1, y1, z1 = sph(lon, 0.10, Y, pitch, 0.99)
        x2, y2, z2 = sph(side * 0.70, -0.22, Y, pitch, 1.0)
        x3, y3, z3 = sph(side * 0.62, -0.48, Y, pitch, 1.0)
        if z1 < 0.0:
            continue
        path = catmull([(cx + x1, cy + y1 + 0.08), (cx + x2, cy + y2), (cx + x3, cy + y3)], n=8)
        wmax = T.px(0.14 * max(0.3, abs(np.cos(lon + Y))))
        n = len(path)
        widths = wmax * np.clip(np.sin(np.linspace(0.2, np.pi - 0.1, n)), 0.15, 1)
        po = stroke_outline(T(path), widths)
        if po is not None:
            tm = cel.mask(po)
            if tm is not None:
                tm = R.clip_mask(tm, full_head)
                cel.fill_mask(tm, TEAR)

    # ---- muzzle shadow and nose ----
    nx, ny, nz = sph(0.0, -0.22, Y, pitch, 0.99 + 0.10 * abs(np.sin(Y)))
    nsx = 0.11 * max(0.45, np.cos(Y))
    nose = ellipse(cx + nx, cy + ny, nsx, 0.075, n=14, squash=-0.2)
    nose = wobble(nose, 0.006, seed + 12, closed=True)
    cel.fill(T(nose), NOSE)
    # nose highlight
    cel.fill(T(ellipse(cx + nx - nsx * 0.35, cy + ny - 0.03, nsx * 0.3, 0.02, n=8)), (0.55, 0.45, 0.45), 0.8)
    cel.ink_closed(T(nose), _ink_w(T, 0.02), light=(lxs, light[1]), seed=seed + 13, gap=False)

    # ---- mouth ----
    mx, my, mz = sph(0.0, -0.42, Y, pitch, 0.96 + 0.08 * abs(np.sin(Y)))
    mw = 0.18 * max(0.4, np.cos(Y))
    if pose.mouth in ("smile", "teeth", "grin"):
        ml = [(cx + mx - mw, cy + my - 0.02), (cx + mx - mw * 0.5, cy + my + 0.05),
              (cx + mx, cy + my + 0.01), (cx + mx + mw * 0.5, cy + my + 0.05),
              (cx + mx + mw, cy + my - 0.02)]
        cel.ink(T(catmull(ml, n=6)), _ink_w(T, 0.022), seed=seed + 14, var=0.2)
        if pose.mouth == "teeth":
            # small opening with a tooth, used while gripping the string
            op = [(cx + mx - 0.06, cy + my + 0.03), (cx + mx + 0.06, cy + my + 0.03),
                  (cx + mx + 0.05, cy + my + 0.10), (cx + mx - 0.05, cy + my + 0.10)]
            cel.fill(T(catmull(op, closed=True, n=4)), (0.35, 0.12, 0.12))
            cel.fill(T([(cx + mx - 0.03, cy + my + 0.03), (cx + mx + 0.01, cy + my + 0.03),
                        (cx + mx - 0.01, cy + my + 0.075)]), CREAM)
    elif pose.mouth == "o":
        oo = ellipse(cx + mx, cy + my + 0.05, 0.06 * max(0.4, np.cos(Y)), 0.07 + 0.08 * pose.mouth_open, n=12)
        cel.fill(T(oo), (0.35, 0.12, 0.12))
        cel.ink_closed(T(oo), _ink_w(T, 0.018), light=(lxs, light[1]), seed=seed + 15, gap=False)
    elif pose.mouth == "frown":
        ml = [(cx + mx - mw, cy + my + 0.05), (cx + mx, cy + my - 0.01), (cx + mx + mw, cy + my + 0.05)]
        cel.ink(T(catmull(ml, n=6)), _ink_w(T, 0.022), seed=seed + 14)
    else:
        ml = [(cx + mx - mw * 0.8, cy + my + 0.01), (cx + mx, cy + my + 0.03), (cx + mx + mw * 0.8, cy + my + 0.01)]
        cel.ink(T(catmull(ml, n=6)), _ink_w(T, 0.02), seed=seed + 14)
    # philtrum
    cel.ink(T([(cx + nx, cy + ny + 0.06), (cx + mx, cy + my + 0.0)]), _ink_w(T, 0.016), seed=seed + 16, var=0.1)

    # ---- one hard shadow over the whole head, then the outline ----
    if hm is not None:
        d = T.px(0.17)
        mult_shade(cel, hm, SHADOW_TINT, -lxs * d, -light[1] * d, 0.8)
        cel.ink_closed(head_poly, _ink_w(T, 0.034), light=(lxs, light[1]), seed=seed + 1, gap=True)

    # ---- eyes ----
    for side in (-1, 1):
        lon = side * 0.50
        x, y, z = sph(lon, 0.14, Y, pitch, 0.99)
        if z < 0.02:
            continue
        op = pose.eye_open
        if side < 0 and pose.eye_open_l is not None:
            op = pose.eye_open_l
        if side > 0 and pose.eye_open_r is not None:
            op = pose.eye_open_r
        fx = max(0.32, abs(np.cos(lon + Y)))
        ew = 0.235 * fx
        eh = 0.27
        draw_eye(cel, T, cx + x, cy + y, ew, eh, op, side, pose, light, seed + 20 + side)

    # ---- goggles ----
    if pose.goggles:
        draw_goggles(cel, T, cx, cy, pose, light, Y, pitch)

    # ---- whiskers (fine coloured lines) ----
    for side in (-1, 1):
        for k in range(3):
            lon = side * 0.55
            x, y, z = sph(lon, -0.30 - 0.1 * k, Y, pitch, 1.0)
            if z < -0.1:
                continue
            dx = side * (0.36 + 0.05 * k) * max(0.25, np.cos(lon + Y) * 0.5 + 0.5)
            wk = [(cx + x, cy + y), (cx + x + dx * 0.6, cy + y + 0.02 * (k - 1)), (cx + x + dx, cy + y + 0.08 * (k - 1) + 0.02)]
            cel.ink(T(catmull(wk, n=6)), _ink_w(T, 0.010), color=(0.60, 0.40, 0.30), seed=seed + 50 + k + side * 3,
                    var=0.1, taper_in=0.05, taper_out=0.5, a=0.8)


def draw_eye(cel, T, ex, ey, ew, eh, op, side, pose, light, seed):
    """Big anime eye: dark almond, amber lower iris, two catchlights, thick upper lid."""
    lxs = light[0] * (-1 if T.flip else 1)
    sq = pose.squeeze
    if op < 0.08 or sq > 0.5:
        # closed: one curved lid line (happy arc when squeezed, soft line when asleep)
        if sq > 0.5:
            arc = [(ex - ew, ey + 0.02), (ex, ey - 0.07), (ex + ew, ey + 0.02)]
            cel.ink(T(catmull(arc, n=8)), _ink_w(T, 0.03), seed=seed, var=0.2)
            # pressed lid creases
            cel.ink(T([(ex - ew * 0.9, ey + 0.08), (ex - ew * 0.4, ey + 0.05)]), _ink_w(T, 0.014), seed=seed + 1)
            cel.ink(T([(ex + ew * 0.4, ey + 0.05), (ex + ew * 0.9, ey + 0.08)]), _ink_w(T, 0.014), seed=seed + 2)
        else:
            arc = [(ex - ew, ey - 0.02), (ex, ey + 0.05), (ex + ew, ey - 0.02)]
            cel.ink(T(catmull(arc, n=8)), _ink_w(T, 0.028), seed=seed, var=0.2)
            lash = [(ex + side * ew, ey - 0.02), (ex + side * ew * 1.25, ey - 0.06)]
            cel.ink(T(lash), _ink_w(T, 0.02), seed=seed + 3)
        return
    h = eh * (0.25 + 0.75 * op)
    brow = pose.brow
    # almond: upper lid is flatter, lower lid rounder; outer corner higher
    up = -h * 0.95
    lo = h * 0.75
    outer = side
    shape = [(ex - ew, ey + 0.02 * -outer), (ex - ew * 0.55, ey + up * 0.9 - brow * 0.03 * (1 - outer * 0.5)),
             (ex, ey + up - brow * 0.04), (ex + ew * 0.55, ey + up * 0.9 - brow * 0.03 * (1 + outer * 0.5)),
             (ex + ew, ey + 0.02 * outer), (ex + ew * 0.55, ey + lo * 0.9),
             (ex, ey + lo), (ex - ew * 0.55, ey + lo * 0.9)]
    poly = catmull(shape, closed=True, n=6)
    m = cel.fill(T(poly), EYE)
    if m is None:
        return
    # iris and lower amber light (clipped to the eye)
    lx, ly = pose.look
    pcx = ex + lx * ew * 0.35
    pcy = ey + 0.02 + ly * h * 0.3
    iris = ellipse(pcx, pcy + h * 0.25, ew * 0.75, h * 0.55, n=16)
    cel.paint_in(m, T(iris), IRIS)
    iris2 = ellipse(pcx, pcy + h * 0.45, ew * 0.5, h * 0.3, n=14)
    cel.paint_in(m, T(iris2), IRIS_L)
    pupil = ellipse(pcx, pcy, ew * 0.42, h * 0.62, n=14)
    cel.paint_in(m, T(pupil), EYE)
    # lid shadow band at the top
    band = [(ex - ew, ey + up * 0.2), (ex - ew, ey + up), (ex + ew, ey + up), (ex + ew, ey + up * 0.2),
            (ex, ey + up * 0.35)]
    cel.paint_in(m, T(catmull(band, closed=True, n=4)), (0.10, 0.05, 0.05), 0.9)
    # catchlights
    hl = ellipse(pcx - ew * 0.3 * (1 if lxs <= 0 else -1), pcy - h * 0.35, ew * 0.22, h * 0.17, n=10, rot=0.4)
    cel.paint_in(m, T(hl), (1.0, 0.98, 0.95))
    hl2 = ellipse(pcx + ew * 0.25 * (1 if lxs <= 0 else -1), pcy + h * 0.3, ew * 0.1, h * 0.08, n=8)
    cel.paint_in(m, T(hl2), (1.0, 0.95, 0.9), 0.85)
    # upper lid: thick ink, tapered, slight overshoot at outer corner
    lid = [(ex - ew * 1.02, ey + 0.02 * -outer), (ex - ew * 0.5, ey + up * 0.95 - brow * 0.03),
           (ex, ey + up * 1.03 - brow * 0.04), (ex + ew * 0.5, ey + up * 0.95 - brow * 0.03),
           (ex + ew * 1.05, ey + 0.01 * outer)]
    if side > 0:
        lid = lid
    cel.ink(T(catmull(lid, n=7)), _ink_w(T, 0.042), seed=seed + 4, var=0.15, taper_in=0.15, taper_out=0.25)
    # lower lid: thin
    low = [(ex - ew * 0.9, ey + lo * 0.5), (ex, ey + lo * 1.02), (ex + ew * 0.9, ey + lo * 0.5)]
    cel.ink(T(catmull(low, n=7)), _ink_w(T, 0.014), seed=seed + 5, var=0.2, a=0.8)
    # brow line (determined / worried)
    if abs(brow) > 0.5:
        by = ey + up - 0.26
        bw = [(ex - ew * 0.7, by + (0.05 if brow > 0 else -0.05) * -outer), (ex + ew * 0.7, by + (0.05 if brow > 0 else -0.05) * outer)]
        cel.ink(T(bw), _ink_w(T, 0.024), color=RUSSET_D, seed=seed + 6, var=0.2)


def draw_goggles(cel, T, cx, cy, pose, light, Y, pitch):
    lxs = light[0] * (-1 if T.flip else 1)
    lat = 0.62 if pose.goggles_up else 0.14
    # strap across the head
    strap = []
    for lon in np.linspace(-1.4, 1.4, 9):
        x, y, z = sph(lon, lat + 0.05, Y, pitch, 1.03)
        strap.append((cx + x, cy + y))
    cel.ink(T(catmull(strap, n=6)), _ink_w(T, 0.07), color=(0.45, 0.30, 0.18), seed=1, var=0.1, taper_in=0.02, taper_out=0.02)
    for side in (-1, 1):
        lon = side * 0.50
        x, y, z = sph(lon, lat, Y, pitch, 1.04)
        if z < 0.0:
            continue
        fx = max(0.3, abs(np.cos(lon + Y)))
        ring = ellipse(cx + x, cy + y, 0.30 * fx, 0.30, n=20)
        cel.fill(T(ring), (0.55, 0.42, 0.22))
        glass = ellipse(cx + x, cy + y, 0.23 * fx, 0.23, n=18)
        gm = cel.fill(T(glass), (0.65, 0.78, 0.82), 0.55 if not pose.goggles_up else 0.85)
        cel.paint_in(gm, T(ellipse(cx + x - 0.07 * fx, cy + y - 0.08, 0.08 * fx, 0.05, n=8, rot=-0.5)), (0.95, 0.97, 1.0), 0.8)
        cel.ink_closed(T(ring), _ink_w(T, 0.028), light=(lxs, light[1]), seed=2 + side, gap=False)
        cel.ink_closed(T(glass), _ink_w(T, 0.018), light=(lxs, light[1]), seed=4 + side, gap=False)


# ----------------------------------------------------------------------------
# tail
# ----------------------------------------------------------------------------

def draw_tail(cel, T, path_u, width, light, seed, rings=6, tip_dark=True, taper=0.55):
    """Bushy ringed tail along a unit path. width: unit thickness at the base."""
    P = catmull(path_u, n=10)
    n = len(P)
    t = np.linspace(0, 1, n)
    w = width * (0.75 + 0.45 * np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 0.6) * (1 - taper * t ** 3)
    poly = stroke_outline(T(P), T.px(w))
    poly = fluffy(poly, seed, amp=T.px(0.09), step=T.px(0.04), n=2, wob=T.px(0.012))
    m = cel.fill(poly, RUSSET)
    if m is None:
        return
    d = np.gradient(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = -d[:, 1] / L, d[:, 0] / L
    for k in range(rings):
        a = (k + 0.5) / rings
        b = (k + 1.0) / rings
        ia = int(a * (n - 1))
        ib = min(int(b * (n - 1)), n - 1)
        if ib <= ia:
            continue
        band = []
        for i in range(ia, ib + 1):
            band.append((P[i, 0] + nx[i] * w[i] * 1.4, P[i, 1] + ny[i] * w[i] * 1.4))
        for i in range(ib, ia - 1, -1):
            band.append((P[i, 0] - nx[i] * w[i] * 1.4, P[i, 1] - ny[i] * w[i] * 1.4))
        col = DARK if k % 2 == 1 else (0.72, 0.42, 0.22)
        cel.paint_in(m, T(wobble(np.array(band), 0.03, seed + k, closed=True)), col)
    lx, ly = light
    dd = T.px(0.14)
    mult_shade(cel, m, SHADOW_TINT, -lx * dd * (-1 if T.flip else 1), -ly * dd, 0.8)
    cel.ink_closed(poly, _ink_w(T, 0.03), light=(lx * (-1 if T.flip else 1), ly), seed=seed + 100, gap=True)


def draw_paw(cel, T, cx, cy, w, d, light, seed, color=DARK, grip=False):
    """A dark paw at (cx, cy) facing direction d: round pad and four toe bumps with claw ticks."""
    lx, ly = light
    d = np.asarray(d, np.float64)
    d = d / (np.hypot(*d) + 1e-9)
    n_ = np.array([-d[1], d[0]])
    pad = ellipse(cx, cy, w * 0.62, w * 0.5, n=18, rot=np.arctan2(d[1], d[0]))
    pad = wobble(pad, 0.012, seed + 3, closed=True)
    pm = cel.fill(T(pad), color)
    toes = []
    for k, off in enumerate((-0.42, -0.14, 0.14, 0.42)):
        tc = np.array([cx, cy]) + d * w * (0.55 - 0.06 * abs(off) * 2) + n_ * off * w
        tr = w * 0.19
        toe = ellipse(tc[0], tc[1], tr, tr * 0.85, n=10, rot=np.arctan2(d[1], d[0]))
        toes.append(toe)
        cel.fill(T(toe), color)
    if pm is not None:
        mult_shade(cel, pm, SHADOW_TINT, -lx * T.px(0.06) * (-1 if T.flip else 1), -ly * T.px(0.06), 0.6)
    # one outline round pad and toes together
    allp = [T(pad)] + [T(t) for t in toes]
    for q in allp:
        cel.ink_closed(q, _ink_w(T, 0.02), light=(lx * (-1 if T.flip else 1), ly), seed=seed + 1, gap=True)
    # toe separations and claw ticks
    for k, toe in enumerate(toes):
        tipc = np.array([toe[:, 0].mean(), toe[:, 1].mean()]) + d * w * 0.16
        cel.ink(T([tipc, tipc + d * w * 0.1]), _ink_w(T, 0.016), seed=seed + 10 + k, a=0.8, taper_in=0.1, taper_out=0.5)
    return pm


def draw_leg(cel, T, pts_u, w, light, seed, paw=True, color=DARK, fur=True):
    """A chunky furry leg: wide at the shoulder, narrower at the wrist, a paw with toes at the end."""
    P = catmull(pts_u, n=8)
    n = len(P)
    t = np.linspace(0, 1, n)
    widths = w * (1.3 - 0.5 * t) * (1 + 0.1 * np.array([R.vnoise(i * 0.5, 0.3, seed) - 0.5 for i in range(n)]))
    poly = stroke_outline(T(P), T.px(widths))
    if fur:
        poly = fluffy(poly, seed, amp=T.px(0.06), step=T.px(0.035), n=2, wob=T.px(0.008))
    m = cel.fill(poly, color)
    if m is None:
        return None
    lx, ly = light
    mult_shade(cel, m, SHADOW_TINT, -lx * T.px(0.10) * (-1 if T.flip else 1), -ly * T.px(0.10), 0.7)
    # fur direction: a few short strokes along the leg on the lit side
    for k in range(3):
        i = int((0.25 + 0.2 * k) * (n - 1))
        dv = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        dv = dv / (np.hypot(*dv) + 1e-9)
        nv = np.array([-dv[1], dv[0]]) * widths[i] * 0.5 * (1 if lx > 0 else -1)
        a0 = P[i] + nv * 0.55
        cel.ink(T([a0, a0 + dv * 0.2 * w]), _ink_w(T, 0.014), color=(0.38, 0.24, 0.2), seed=seed + 30 + k, a=0.8, taper_in=0.1, taper_out=0.6)
    cel.ink_closed(poly, _ink_w(T, 0.026), light=(lx * (-1 if T.flip else 1), ly), seed=seed, gap=True)
    if paw:
        end = P[-1]
        d = P[-1] - P[-3]
        draw_paw(cel, T, end[0] + d[0] * 0.05, end[1] + d[1] * 0.05, w * 0.95, d, light, seed + 2, color)
    return m


def cast(cel, receiver, caster_poly, dx, dy, a=0.75):
    """Cast shadow: multiply the receiver where the caster polygon (shifted) overlaps it."""
    if receiver is None:
        return
    poly = np.asarray(caster_poly, np.float64) + np.array([dx, dy])
    cel.mult_in(receiver, poly, SHADOW_TINT, a)


def ruff(cel, T, cx, cy, yaw, light, seed, rx=0.98, ry=0.46, dy=0.74):
    """Fur collar under the head: a crescent of hanging tufts narrower than the head, so the head
    hides its top; only the hanging edge is inked."""
    rx = min(rx, 0.86)
    pts = []
    n = 24
    for i in range(n):
        a = np.pi * i / (n - 1)
        ca, sa = np.cos(a), np.sin(a)
        r = 1.0 + 0.16 * np.sin(a * 5.0 + seed) ** 2 * (0.5 + 0.5 * sa)
        pts.append((cx + 0.1 * yaw - rx * ca * r, cy + dy + ry * sa * r))
    bottom = fluffy(pts + [(cx + 0.1 * yaw + rx * 0.6, cy + dy - 0.3), (cx + 0.1 * yaw, cy + dy - 0.4), (cx + 0.1 * yaw - rx * 0.6, cy + dy - 0.3)],
                    seed + 11, amp=0.14, step=0.05, n=3, wob=0.015)
    m = part(cel, T, bottom, RUSSET, SHADOW_TINT, light, 0.03, seed + 11, ink=False)
    # ink only the hanging edge (points below the collar line)
    edge = [p_ for p_ in bottom if p_[1] > cy + dy + 0.02]
    if len(edge) > 3:
        cel.ink(T(np.array(edge)), _ink_w(T, 0.03), seed=seed + 12, var=0.3, taper_in=0.08, taper_out=0.08)
    cast(cel, m, T(ellipse(cx + 0.1 * yaw, cy + 0.1, 1.05, 0.95, n=24)), 0, T.px(0.12), 0.85)
    return m


def body_shape(pts, seed, amp=0.11):
    return fluffy(pts, seed, amp=amp, step=0.05, n=8, wob=0.02)


def head_poly_u(cx, cy, pose, seed):
    hp = head_outline(pose.yaw, seed)
    return [(cx + x, cy + y) for x, y in hp]


def pose_sit_front(cel, T, pose, light, cx=0.0, cy=0.0, tail_curl=0.0):
    """Sitting, facing camera. Head radius 1 at (cx, cy). tail_curl 0: tail out to the right; 1: curled up round the front."""
    s = pose.seed
    c = tail_curl
    tail = [(cx + 0.7, cy + 2.05), (cx + 1.6 - 0.2 * c, cy + 2.3 + 0.1 * c), (cx + 2.2 - 0.9 * c, cy + 1.8 + 0.6 * c), (cx + 2.1 - 1.9 * c, cy + 1.1 + 1.2 * c)]
    draw_tail(cel, T, tail, 0.52, light, s + 200, rings=6)
    body = [(cx - 0.9, cy + 0.6), (cx + 0.9, cy + 0.6), (cx + 1.3, cy + 1.3), (cx + 1.45, cy + 2.1),
            (cx + 0.7, cy + 2.4), (cx - 0.7, cy + 2.4), (cx - 1.45, cy + 2.1), (cx - 1.3, cy + 1.3)]
    bm = part(cel, T, body_shape(body, s + 2), RUSSET, SHADOW_TINT, light, 0.034, s + 2)
    belly = [(cx - 0.55, cy + 1.25), (cx + 0.55, cy + 1.25), (cx + 0.9, cy + 1.8), (cx + 0.8, cy + 2.4),
             (cx - 0.8, cy + 2.4), (cx - 0.9, cy + 1.8)]
    cel.paint_in(bm, T(fluffy(belly, s + 3, amp=0.05, step=0.06, n=6)), DARK)
    # head and ruff cast onto the chest
    cast(cel, bm, T(head_poly_u(cx, cy, pose, s)), 0, T.px(0.32), 0.8)
    for side in (-1, 1):
        ft = ellipse(cx + side * 1.0, cy + 2.36, 0.42, 0.2, n=14)
        part(cel, T, fluffy(ft, s + 40 + side, amp=0.04, step=0.05, n=2), DARK, SHADOW_TINT, light, 0.026, s + 40 + side)
    for side in (-1, 1):
        lm = draw_leg(cel, T, [(cx + side * 0.55, cy + 1.3), (cx + side * 0.6, cy + 1.8), (cx + side * 0.55, cy + 2.2)],
                      0.36, light, s + 10 + side)
    ruff(cel, T, cx, cy, pose.yaw, light, s)
    draw_head(cel, T, cx, cy, pose, light)


def pose_stand(cel, T, pose, light, cx=0.0, cy=0.0, stride=0.0, head_up=0.0, crouch=0.0):
    """Standing on four short legs, facing right (flip the transform for left)."""
    s = pose.seed
    hy = cy - head_up * 0.2 + crouch * 0.2
    draw_tail(cel, T, [(cx - 2.1, cy + 1.2), (cx - 2.8, cy + 1.4), (cx - 3.5, cy + 1.1), (cx - 3.8, cy + 0.4)],
              0.55, light, s + 200, rings=6)
    draw_leg(cel, T, [(cx - 1.85, cy + 1.45), (cx - 1.95 - stride * 0.3, cy + 1.85), (cx - 1.9 - stride * 0.5, cy + 2.2)],
             0.34, light, s + 11, color=DARK_D)
    draw_leg(cel, T, [(cx - 0.45, cy + 1.4), (cx - 0.4 + stride * 0.3, cy + 1.85), (cx - 0.3 + stride * 0.5, cy + 2.2)],
             0.34, light, s + 12, color=DARK_D)
    body = [(cx - 0.3, cy + 0.45 - head_up * 0.1), (cx - 1.1, cy + 0.2), (cx - 1.9, cy + 0.3), (cx - 2.55, cy + 0.8),
            (cx - 2.55, cy + 1.5), (cx - 1.9, cy + 1.9), (cx - 1.0, cy + 1.95), (cx - 0.1, cy + 1.8), (cx + 0.25, cy + 1.2)]
    bm = part(cel, T, body_shape(body, s + 2), RUSSET, SHADOW_TINT, light, 0.034, s + 2)
    belly = [(cx + 0.1, cy + 1.3), (cx - 0.3, cy + 1.9), (cx - 1.2, cy + 2.0), (cx - 2.1, cy + 1.9),
             (cx - 2.45, cy + 1.45), (cx - 1.9, cy + 1.35), (cx - 1.0, cy + 1.3)]
    cel.paint_in(bm, T(fluffy(belly, s + 3, amp=0.05, step=0.06, n=6)), DARK)
    # form: belly under the body, head cast on the shoulder
    cel.mult_in(bm, T(catmull([(cx + 0.3, cy + 1.55), (cx - 1.0, cy + 1.7), (cx - 2.6, cy + 1.55), (cx - 2.7, cy + 2.2), (cx + 0.3, cy + 2.2)], closed=True, n=6)), SHADOW_TINT, 0.6)
    cast(cel, bm, T(head_poly_u(cx, hy, pose, s)), -T.px(0.12) * (-1 if T.flip else 1), T.px(0.3), 0.8)
    draw_leg(cel, T, [(cx - 2.1, cy + 1.4), (cx - 2.25 + stride * 0.3, cy + 1.85), (cx - 2.3 + stride * 0.5, cy + 2.25)],
             0.38, light, s + 13)
    draw_leg(cel, T, [(cx - 0.15, cy + 1.3), (cx - 0.05 - stride * 0.3, cy + 1.85), (cx + 0.05 - stride * 0.5, cy + 2.25)],
             0.38, light, s + 14)
    ruff(cel, T, cx, hy, pose.yaw, light, s, rx=0.9, ry=0.5, dy=0.62)
    draw_head(cel, T, cx, hy, pose, light)


def pose_look_back(cel, T, pose, light, cx=0.0, cy=0.0):
    """Seen from behind at three quarter, head turned up to the sky."""
    s = pose.seed
    body = [(cx - 0.7, cy + 0.55), (cx + 0.8, cy + 0.55), (cx + 1.35, cy + 1.3), (cx + 1.45, cy + 2.3),
            (cx + 0.6, cy + 2.7), (cx - 0.7, cy + 2.7), (cx - 1.45, cy + 2.3), (cx - 1.3, cy + 1.3)]
    bm = part(cel, T, body_shape(body, s + 2), RUSSET, SHADOW_TINT, light, 0.034, s + 2)
    cast(cel, bm, T(head_poly_u(cx, cy, pose, s)), 0, T.px(0.3), 0.8)
    for side in (-1, 1):
        ft = ellipse(cx + side * 1.0, cy + 2.7, 0.4, 0.2, n=14)
        part(cel, T, fluffy(ft, s + 40 + side, amp=0.04, step=0.05, n=2), DARK, SHADOW_TINT, light, 0.026, s + 40 + side)
    draw_tail(cel, T, [(cx + 0.2, cy + 2.5), (cx - 0.6, cy + 3.0), (cx - 1.6, cy + 2.9), (cx - 2.3, cy + 2.2)],
              0.52, light, s + 200, rings=6)
    ruff(cel, T, cx, cy, pose.yaw, light, s)
    draw_head(cel, T, cx, cy, pose, light)


def pose_sleep(cel, T, pose, light, cx=0.0, cy=0.0, blanket=True):
    """Curled up asleep, head at (cx, cy) resting on the paws, tail wrapped round, blanket over the back."""
    s = pose.seed
    body = [(cx + 0.3, cy - 0.3), (cx + 1.4, cy - 0.55), (cx + 2.5, cy - 0.2), (cx + 2.9, cy + 0.6),
            (cx + 2.4, cy + 1.3), (cx + 1.0, cy + 1.45), (cx - 0.3, cy + 1.3), (cx - 0.7, cy + 0.8)]
    bm = part(cel, T, body_shape(body, s + 2), RUSSET, SHADOW_TINT, light, 0.034, s + 2)
    if blanket:
        bl = [(cx + 0.9, cy - 0.25), (cx + 1.6, cy - 0.6), (cx + 2.6, cy - 0.3), (cx + 3.05, cy + 0.55), (cx + 2.5, cy + 1.25),
              (cx + 1.3, cy + 1.3), (cx + 0.75, cy + 0.7)]
        bp = catmull(bl, closed=True, n=6)
        bp = wobble(bp, 0.02, s + 77, closed=True)
        m = cel.fill(T(bp), (0.55, 0.40, 0.55))
        patches = [((cx + 1.4, cy - 0.05, 0.42, 0.3, 0.3), (0.80, 0.55, 0.30)), ((cx + 2.3, cy + 0.3, 0.4, 0.35, -0.2), (0.35, 0.50, 0.55)),
                   ((cx + 1.6, cy + 0.8, 0.45, 0.25, 0.1), (0.70, 0.35, 0.35)), ((cx + 2.5, cy + 0.95, 0.25, 0.2, 0.5), (0.85, 0.75, 0.45))]
        for (px, py, rx, ry, rr), col in patches:
            q = rot(np.array([(px - rx, py - ry), (px + rx, py - ry), (px + rx, py + ry), (px - rx, py + ry)]), px, py, rr)
            cel.paint_in(m, T(wobble(q, 0.015, s + 80, closed=True)), col)
            cel.ink_closed(T(q), _ink_w(T, 0.012), light=light, seed=s + 81, gap=False, color=(0.3, 0.2, 0.25), a=0.6)
        lx, ly = light
        mult_shade(cel, m, SHADOW_TINT, -lx * T.px(0.16), -ly * T.px(0.16), 0.8)
        cel.ink_closed(T(bp), _ink_w(T, 0.03), light=light, seed=s + 78)
        for k in range(5):
            xx = cx + 1.0 + k * 0.4
            cel.ink(T([(xx, cy + 1.1), (xx + 0.12, cy + 1.05)]), _ink_w(T, 0.012), color=CREAM, seed=s + 90 + k, a=0.7)
    draw_leg(cel, T, [(cx + 0.2, cy + 0.9), (cx - 0.5, cy + 1.05)], 0.34, light, s + 10)
    draw_head(cel, T, cx, cy, pose, light)
    draw_tail(cel, T, [(cx + 2.6, cy + 1.0), (cx + 1.9, cy + 1.7), (cx + 0.5, cy + 1.75), (cx - 0.6, cy + 1.35), (cx - 0.7, cy + 0.6)],
              0.5, light, s + 200, rings=6)
