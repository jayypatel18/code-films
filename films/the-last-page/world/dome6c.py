"""Cut 6c: the inside of the old copper dome, seen from below at an angle.

The dome is a real hemisphere on a drum, ray cast through a pinhole camera
standing off its axis, so its ribs, rings and panels fall into proper,
asymmetric perspective. Three big hinged copper shutters on the sun side
break open at shutters_break on hard held poses (closed, a snapped first
swing, wider, flung open) and broad shafts of gold light slant down through
the openings, all parallel (one sun), with dust and debris in them.

Plates: c6c_closed, c6c_open (bg), c6c_shaft (glow, open), c6c_shaft_t (how
far along its shaft each pixel lies, for the light crashing down).
"""

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise, rng, seed_of  # noqa: E402

import common as C  # noqa: E402
from common import T, MARGIN, log  # noqa: E402

M = MARGIN
CUT = "6c"
PW, PH = T.plate_size(CUT)
AW, AH = PW + 2 * M, PH + 2 * M

GOLD = np.array(T.LIGHTS["gold_burst"]["key"], np.float32)
AMB = np.array(T.LIGHTS["hall"]["ambient"], np.float32)
MOON = np.array(T.LIGHTS["hall"]["key"], np.float32)
SKY_GOLD = np.array([1.0, 0.88, 0.60], np.float32)

# camera: below the dome, off its axis, looking up and across
CAM = np.array([0.55, -0.45, -0.95])
TARGET = np.array([-0.30, 0.30, 0.55])
FOV = 100.0
SUN_TRAVEL = np.array([-0.55, 0.40, -1.0])
SUN_TRAVEL = SUN_TRAVEL / np.linalg.norm(SUN_TRAVEL)
N_RIB = 16
RINGS = [0.0, 18.0, 36.0, 54.0, 70.0]            # latitude of the horizontal ribs (degrees)
OCULUS = 76.0
# the three big shutters: (lon centre, lon half width, lat from, lat to) in degrees
SHUTTERS = [(-56.0, 10.5, 36.0, 54.0), (-22.0, 10.5, 36.0, 54.0), (13.0, 10.5, 18.0, 36.0)]


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
    """World points (..., 3) to plate array coords (x, y)."""
    f, right, up, foc = basis()
    d = p - CAM
    z = d @ f
    x = d @ right
    y = d @ up
    return AW / 2 + foc * x / np.maximum(z, 1e-3), AH / 2 - foc * y / np.maximum(z, 1e-3)


def sph(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def raycast():
    """Per pixel: hit kind (0 drum wall, 1 dome), lat, lon (deg), height on the
    drum, and the world hit point."""
    f, right, up, foc = basis()
    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float64)
    d = f[None, None] * foc + right[None, None] * (xx - AW / 2)[..., None] - up[None, None] * (yy - AH / 2)[..., None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = CAM
    # cylinder x^2 + y^2 = 1
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (o[0] * d[..., 0] + o[1] * d[..., 1])
    c = o[0] ** 2 + o[1] ** 2 - 1
    tc = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-9))
    zc = o[2] + tc * d[..., 2]
    # sphere |p| = 1, far root
    bs = 2 * (d @ o)
    cs = o @ o - 1
    ts = (-bs + np.sqrt(np.maximum(bs * bs - 4 * cs, 0))) / 2
    dome = zc >= 0
    t = np.where(dome, ts, tc)
    p = o[None, None] + t[..., None] * d
    lat = np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return dome, lat, lon, p, d


def in_shutter(lat, lon, s):
    lc, hw, la0, la1 = s
    dl = (lon - lc + 180) % 360 - 180
    return (np.abs(dl) < hw) & (lat > la0) & (lat < la1), dl


def dome_reference(state):
    dome, lat, lon, p, d = raycast()
    # light: cold blue from the stairwell below, a touch of moon; after the break, gold bounce
    n = -p                                     # inward normal
    ambient = AMB * 1.6 + MOON * 0.10
    # panels: copper, patinated green and brown, each panel its own mix; the
    # ribs are hand set, not machine straight
    lonw = lon + (value_noise(AH, AW, 90, "6c2-lonw") - 0.5) * 1.6
    rib_phase = lonw / (360.0 / N_RIB)
    panel_i = np.floor(rib_phase)
    fr = rib_phase - panel_i
    ring_i = np.searchsorted(np.array(RINGS), lat) - 1
    pid = (panel_i * 37 + ring_i * 101).astype(np.int64)
    h1 = ((pid * 2654435761) % 4294967296) / 4294967296.0
    green = np.array([0.24, 0.44, 0.40], np.float32)
    brown = np.array([0.38, 0.24, 0.16], np.float32)
    pale = np.array([0.40, 0.56, 0.50], np.float32)
    pat = fbm(AH, AW, 26, "6c2-pat", octaves=4)
    streak = fbm(AH, AW, 9, "6c2-st", octaves=3)
    mixk = np.clip(h1 * 0.8 + (pat - 0.5) * 0.5, 0, 1)
    col = green[None, None] * (1 - mixk[..., None]) + brown[None, None] * mixk[..., None]
    col = col + (P.smoothstep(0.74, 0.92, pat) * 0.35)[..., None] * (pale - col)
    col = col * (0.88 + 0.18 * streak)[..., None]
    # sheet seams inside each panel (rows of plates), rivets along ribs and seams
    sub = np.mod(lat / 6.0, 1.0)
    seam = np.abs(sub - 0.5) > 0.46
    col = np.where(seam[..., None], col * 0.88, col)
    rib_w = 0.07 + 0.05 * (lat / 90.0)
    ribv = (fr < rib_w) | (fr > 1 - rib_w * 0.3)
    ribh = np.zeros_like(ribv)
    for rl in RINGS[1:]:
        ribh |= np.abs(lat - rl) < 1.1
    ribc = np.array([0.30, 0.22, 0.17], np.float32)
    col = np.where((ribv | ribh)[..., None] & dome[..., None], col * 0.55 + ribc[None, None] * 0.45 * (0.9 + 0.2 * streak)[..., None], col)
    # the rib's lit edge (one side catches the stairwell light)
    ribedge = (fr > rib_w) & (fr < rib_w + 0.02)
    col = np.where(ribedge[..., None] & dome[..., None], col * 1.12, col)
    rivet_l = (np.abs(np.mod(lat / 2.2, 1.0) - 0.5) < 0.13) & (np.abs(fr - rib_w - 0.035) < 0.012)
    col = np.where(rivet_l[..., None] & dome[..., None], np.array([0.62, 0.50, 0.32], np.float32), col)
    # the oculus lantern at the top: dark drum with narrow slits of gold
    ocu = lat > OCULUS
    slit = (np.abs(np.mod(lon / 30.0, 1.0) - 0.5) < 0.05) & (np.mod(np.floor(lon / 30.0), 3) != 1)
    col = np.where(ocu[..., None], np.array([0.16, 0.13, 0.11], np.float32) * (0.8 + 0.3 * pat)[..., None], col)
    col = np.where((ocu & slit & (lat < 82) & (lat > OCULUS + 2))[..., None], SKY_GOLD * 0.55, col)
    # the drum below the dome: stone, courses, tall dark windows
    drum = ~dome
    zc = p[..., 2]
    stone = np.array([0.24, 0.20, 0.22], np.float32) * (0.8 + 0.3 * fbm(AH, AW, 14, "6c2-drum", octaves=3))[..., None]
    stone = stone * (0.55 + 0.45 * np.clip(1 + zc / 0.9, 0, 1))[..., None]
    course = np.abs(np.mod(zc / 0.06, 1.0) - 0.5) > 0.45
    stone = np.where(course[..., None], stone * 0.65, stone)
    win = (np.abs(np.mod((lon + 11.25) / 22.5, 1.0) - 0.5) < 0.12) & (zc > -0.55) & (zc < -0.12)
    stone = np.where(win[..., None], np.array([0.12, 0.13, 0.20], np.float32), stone)
    col = np.where(drum[..., None], stone, col)
    # shutters
    shut = np.zeros(lat.shape, bool)
    for s in SHUTTERS:
        m, dl = in_shutter(lat, lon, s)
        shut |= m & dome
    # the dome is in deep shadow before the light comes: the grid barely reads
    col = col * 0.8 + col.mean(-1, keepdims=True) * 0.2
    lit = col * ambient * 1.7
    # depth: the dome's far side is hazier, a breath of blue air
    hz = np.clip((lat - 20) / 60.0, 0, 1)
    lit = lit * (1 - 0.25 * hz[..., None]) + np.array([0.08, 0.10, 0.18], np.float32) * (0.25 * hz)[..., None]
    if state == "closed":
        # shutters closed: riveted plates with gold light leaking round their edges
        for s in SHUTTERS:
            m, dl = in_shutter(lat, lon, s)
            m &= dome
            lc, hw, la0, la1 = s
            edge = m & ((np.abs(np.abs(dl) - hw) < 0.6) | (np.abs(lat - la0) < 0.5) | (np.abs(lat - la1) < 0.5))
            plank = m & (np.abs(np.mod((dl + hw) / (2 * hw) * 4, 1.0) - 0.5) > 0.44)
            lit = np.where(plank[..., None], lit * 0.6, lit)
            lit = np.where(edge[..., None], SKY_GOLD, lit)
    else:
        # openings full of gold morning sky; the dome lights up around them
        for s in SHUTTERS:
            m, dl = in_shutter(lat, lon, s)
            m &= dome
            lit = np.where(m[..., None], SKY_GOLD * (0.95 + 0.05 * fbm(AH, AW, 40, "6c2-sky", octaves=2))[..., None], lit)
        # warm light thrown back onto the copper near the openings and where shafts land
        warm = np.zeros(lat.shape, np.float32)
        for s in SHUTTERS:
            lc, hw, la0, la1 = s
            c = sph((la0 + la1) / 2, lc)
            dd = np.linalg.norm(p - c[None, None], axis=-1)
            warm = np.maximum(warm, np.exp(-(dd / 0.45) ** 2))
        lit = lit + (warm * 0.30 * dome * ~shut)[..., None] * GOLD * col / np.maximum(col.max(-1, keepdims=True), 1e-3)
    return np.clip(lit, 0, 1).astype(np.float32), dict(dome=dome, lat=lat, lon=lon, fr=fr)


def paint_dome(state):
    ref, info = dome_reference(state)
    lat, lon = info["lat"], info["lon"]
    # strokes follow the panels: along the latitude lines, turning with the perspective
    gy, gx = np.gradient(lat)
    ang = (np.arctan2(gy, gx) + math.pi / 2).astype(np.float32)
    paper = P.paper_for(AH, AW, "6c2")
    canvas = P.new_canvas(AH, AW)
    P.paint(canvas, ref, ang, [
        dict(radius=12, spacing=0.9, steps=(2, 8), opacity=0.96, dry=0.3, cthresh=0.10, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, pickup=0.2),
        dict(radius=5, spacing=1.0, steps=(2, 7), opacity=0.92, dry=0.4, cthresh=0.08, wob=0.25,
             jitter=(0.05, 0.02, 0.05), streak=0.08, use_err=True, err_thresh=0.025),
        dict(radius=2.2, spacing=1.1, steps=(2, 5), opacity=0.9, dry=0.45, cthresh=0.06, wob=0.25,
             jitter=(0.04, 0.02, 0.04), streak=0.06, use_err=True, err_thresh=0.04),
    ], paper, ("6c2", state))
    P.finish(canvas, paper, grain=0.05)
    canvas[..., 3] = 1.0
    return canvas


def shaft_plates():
    """Glow of the light shafts through the open shutters: every point of each
    opening marched along the one sun direction, projected, splatted and
    softened. Also stores, per pixel, how far down its shaft it lies."""
    acc = np.zeros((AH, AW), np.float64)
    tac = np.zeros((AH, AW), np.float64)
    r = rng("6c2-shaft")
    for si, s in enumerate(SHUTTERS):
        lc, hw, la0, la1 = s
        la = la0 + (la1 - la0) * r.random(14000)
        lo = lc + (r.random(14000) * 2 - 1) * hw
        p0 = sph(la, lo)
        ts = np.linspace(0.0, 2.6, 140)
        dt = ts[1] - ts[0]
        for k, t in enumerate(ts):
            tj = t + (r.random(len(p0)) * dt)[:, None]
            pts = p0 + SUN_TRAVEL[None] * tj
            x, y = project(pts)
            ok = (x >= 0) & (x < AW) & (y >= 0) & (y < AH) & (pts[:, 2] > -1.2)
            xi, yi = x[ok].astype(int), y[ok].astype(int)
            w = (1.0 - t / 2.8) ** 1.2
            np.add.at(acc, (yi, xi), w)
            np.add.at(tac, (yi, xi), w * t / 2.6)
    accb = ndimage.gaussian_filter(acc, 6.0)
    tb = ndimage.gaussian_filter(tac, 6.0) / np.maximum(accb, 1e-6)
    inten = accb / np.percentile(accb[accb > 0], 85.0)
    inten = np.clip(inten, 0, 1.6) ** 0.7
    # streaks of dust along the shafts
    vn = value_noise(AH, AW, 16, "6c2-dust")
    glow = (inten * (0.75 + 0.35 * vn))[..., None] * GOLD * 0.40
    return glow.astype(np.float32), tb.astype(np.float32)


def shutter_polys(state_angle):
    """Screen polygons (plate coords) of the three shutter panels swung inward
    about their lower hinge by state_angle degrees (0 closed)."""
    polys = []
    for s in SHUTTERS:
        lc, hw, la0, la1 = s
        h0, h1 = sph(la0, lc - hw), sph(la0, lc + hw)
        t0, t1 = sph(la1, lc - hw), sph(la1, lc + hw)
        axis = h1 - h0
        axis /= np.linalg.norm(axis)
        a = math.radians(state_angle)

        def rot(v):
            # Rodrigues rotation about the hinge axis, swinging the panel inward
            return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * (axis @ v) * (1 - math.cos(a))
        c0 = h0 + rot(t0 - h0)
        c1 = h1 + rot(t1 - h1)
        # make sure the swing goes inward (toward the dome centre)
        if np.linalg.norm((c0 + c1) / 2) > np.linalg.norm((t0 + t1) / 2):
            a = -a
            c0 = h0 + rot(t0 - h0)
            c1 = h1 + rot(t1 - h1)
        quad = np.array([h0, h1, c1, c0])
        x, y = project(quad)
        polys.append(np.stack([x, y], 1))
    return polys


SWING = [(0, 0.0), (1, 38.0), (3, 76.0), (6, 112.0), (9, 104.0)]   # (frames after the break, angle): held keys


def swing_angle(k):
    a = 0.0
    for (kf, ang) in SWING:
        if k >= kf:
            a = ang
    return a


# ---------------------------------------------------------------- render

_VN = {}


def _vn(cell, key):
    """Screen sized value noise, cached (render must stay fast)."""
    k = (cell, key)
    if k not in _VN:
        _VN[k] = value_noise(T.H, T.W, cell, key)
    return _VN[k]


_GRID = []


def _grid():
    if not _GRID:
        _GRID.append(np.mgrid[0:T.H, 0:T.W].astype(np.float32))
    return _GRID[0]


def render_cut(cut, f):
    cam = T.camera(CUT, f)
    fb = int(round(T.CUES["shutters_break"] * T.FPS))
    k = f - fb
    ang = swing_angle(k) if k >= 0 else 0.0
    mask = 0.0
    if ang == 0.0:
        bg = C.sample(C.load_plate("c6c_closed"), cam)
    else:
        bg = C.sample(C.load_plate("c6c_open"), cam)
        # the swinging panels: copper plates, green patina on the inner face, a
        # warm thickness along the free edge and a hot gold rim where the light catches it
        polys = shutter_polys(ang)
        paper = P.paper_for(T.H, T.W, "screen-paper")
        masks = []
        for i, q in enumerate(polys):
            sx, sy = C.to_screen(cam, q[:, 0] - M, q[:, 1] - M)
            pts = list(zip(sx, sy))
            im = Image.new("L", (T.W * 2, T.H * 2), 0)
            dr = ImageDraw.Draw(im)
            dr.polygon([(float(x * 2), float(y * 2)) for x, y in pts], fill=255)
            mi = np.asarray(im.resize((T.W, T.H), Image.BOX), np.float32) / 255.0
            # thickness: the free edge pushed toward the light
            hx, hy = (sx[0] + sx[1]) / 2, (sy[0] + sy[1]) / 2
            fx, fy = (sx[2] + sx[3]) / 2, (sy[2] + sy[3]) / 2
            ln = math.hypot(fx - hx, fy - hy) + 1e-6
            th = max(5.0, 0.05 * ln)
            ox, oy = (fx - hx) / ln * th, (fy - hy) / ln * th
            im2 = Image.new("L", (T.W * 2, T.H * 2), 0)
            ImageDraw.Draw(im2).polygon([(float(x * 2), float(y * 2)) for x, y in
                                         [(sx[3], sy[3]), (sx[2], sy[2]), (sx[2] + ox, sy[2] + oy), (sx[3] + ox, sy[3] + oy)]],
                                        fill=255)
            mt = np.asarray(im2.resize((T.W, T.H), Image.BOX), np.float32) / 255.0
            masks.append(mi)
            ys_, xs_ = _grid()
            # position across the plate from hinge (0) to free edge (1): gold bounce grows toward the edge
            u = np.clip(((xs_ - hx) * (fx - hx) + (ys_ - hy) * (fy - hy)) / (ln * ln), 0, 1)
            pat = _vn(16, ("6c2-pat", i))
            green = np.array([0.20, 0.38, 0.32], np.float32)
            brown = np.array([0.36, 0.22, 0.14], np.float32)
            face = green * (1 - pat[..., None] * 0.6) + brown * (pat[..., None] * 0.6)
            face = face * (0.40 + 0.20 * _vn(5, ("6c2-pf", i)))[..., None]
            face = face + (u ** 2 * 0.22)[..., None] * np.array([1.0, 0.70, 0.36], np.float32)
            bg[..., :3] = bg[..., :3] * (1 - mi[..., None]) + face * mi[..., None]
            edge_col = np.array([0.95, 0.58, 0.28], np.float32)
            bg[..., :3] = bg[..., :3] * (1 - mt[..., None]) + edge_col * mt[..., None]
            st = P.Strokes(max_pts=2)
            st.add([(sx[3] + ox, sy[3] + oy), (sx[2] + ox, sy[2] + oy)], 4.0, np.array([1.0, 0.92, 0.66]), seed=i,
                   opacity=0.95, dry=0.35, ragged=0.25, bristles=2, pool=0, streak=0, cap=1.0)
            for j in range(1, 4):
                uu = j / 4
                st.add([(sx[0] + (sx[1] - sx[0]) * uu, sy[0] + (sy[1] - sy[0]) * uu),
                        (sx[3] + (sx[2] - sx[3]) * uu, sy[3] + (sy[2] - sy[3]) * uu)], 1.4,
                       np.array([0.08, 0.10, 0.08]), seed=i * 10 + j, opacity=0.55, dry=0.45, ragged=0.25,
                       bristles=2, pool=0, streak=0)
            for j in range(9):
                uu = (j + 0.5) / 9
                for (a0, a1) in ((0, 3), (1, 2)):
                    rx = sx[a0] + (sx[a1] - sx[a0]) * uu
                    ry = sy[a0] + (sy[a1] - sy[a0]) * uu
                    st.add([(rx, ry), (rx + 0.5, ry + 0.4)], 1.6, np.array([0.70, 0.52, 0.30]), seed=i * 100 + j + a0,
                           opacity=0.85, dry=0, ragged=0.1, bristles=2, pool=0, streak=0, paper_k=0)
            P.draw(bg, st, paper)
        mask = np.clip(sum(masks), 0, 1)
    bg[..., 3] = 1.0
    glow = np.zeros((T.H, T.W, 3), np.float32)
    # gold leaking round the closed shutters
    if k < 0:
        # before the break the dome is dark; only the gold cracks round the shutters burn
        bg[..., :3] *= 0.5
        return dict(bg=bg, fg=None, glow=glow)
    sh = C.sample(C.load_plate("c6c_shaft"), cam)
    tpl = C.sample(np.concatenate([C.load_plate("c6c_shaft_t")[..., None]] * 3, -1), cam)[..., 0]
    t_after = (f - fb) / T.FPS
    front = min(1.0, t_after / 0.45) ** 0.7
    open_k = min(1.0, ang / 90.0)
    reach = P.smoothstep(front + 0.08, front - 0.02, tpl)
    pm = mask if ang > 0 else 0.0
    glow += sh * (reach * open_k * (1 - 0.95 * pm))[..., None]
    # the flood: blinding gold blooms out of the openings over a large part of the
    # frame, the rest of the dome falls into shadow (a flash at the break that settles)
    cxs, cys = [], []
    for s_ in SHUTTERS:
        lc, hw, la0, la1 = s_
        px_, py_ = project(sph((la0 + la1) / 2, lc)[None])
        sxx, syy = C.to_screen(cam, float(px_[0]) - M, float(py_[0]) - M)
        cxs.append(sxx)
        cys.append(syy)
    ys_, xs_ = _grid()
    flood = np.zeros((T.H, T.W), np.float32)
    for (ox_, oy_) in zip(cxs, cys):
        d = np.sqrt((xs_ - ox_) ** 2 + (ys_ - oy_) ** 2)
        flood = np.maximum(flood, np.exp(-(d / 260.0) ** 2) * 1.0 + np.exp(-(d / 700.0) ** 2) * 0.45)
    flash = 1.0 + 0.8 * math.exp(-max(k - 2, 0) / 4.0)
    light = np.clip(flood * open_k + (sh.mean(-1) * reach) * 1.2, 0, 1)
    bg[..., :3] *= (0.42 + 0.58 * light)[..., None]
    glow += (flood * open_k * flash * (1 - 0.92 * pm))[..., None] * np.array([1.0, 0.80, 0.48], np.float32) * 0.85
    # dust and debris tumbling in the light
    r = rng("6c2-debris")
    lay = np.zeros((T.H, T.W, 4), np.float32)
    st = P.Strokes(max_pts=3)
    shl = sh.mean(-1)
    for i in range(320):
        x = r.random() * T.W
        y = r.random() * T.H
        drift = 60 * t_after * (0.5 + r.random())
        x, y = x - 0.4 * drift, y + drift
        if not (0 <= x < T.W and 0 <= y < T.H):
            continue
        b = float(shl[int(y), int(x)]) * reach[int(y), int(x)]
        if b < 0.08:
            continue
        if i < 40 and t_after < 1.0:
            sz = 2 + 5 * r.random()
            a = r.random() * 3 + t_after * 6
            st.add([(x - math.cos(a) * sz, y - math.sin(a) * sz), (x + math.cos(a) * sz, y + math.sin(a) * sz)], sz * 0.5,
                   np.array([0.22, 0.17, 0.13]) if r.random() < 0.6 else np.array([0.24, 0.36, 0.32]), seed=i,
                   opacity=0.95, dry=0.2, ragged=0.4, bristles=2, pool=0.1, streak=0, pickup=0)
        else:
            st.add([(x, y), (x + 0.4, y + 0.3)], 0.8 + 1.4 * r.random(), np.array([1.0, 0.92, 0.70]) * min(1.0, b * 1.5),
                   seed=i, opacity=min(1.0, b * 1.4), dry=0, ragged=0.05, bristles=2, pool=0, streak=0, paper_k=0,
                   pickup=0)
    P.draw(lay, st, P.paper_for(T.H, T.W, "screen-paper"))
    bg[..., :3] = bg[..., :3] * (1 - lay[..., 3:4] * (lay[..., :3].max(-1, keepdims=True) < 0.5)) + \
        lay[..., :3] * (lay[..., :3].max(-1, keepdims=True) < 0.5)
    glow += lay[..., :3] * (lay[..., :3].max(-1, keepdims=True) >= 0.5)
    return dict(bg=bg, fg=None, glow=glow)


def paint_all():
    log("6c dome closed")
    C.save_plate("c6c_closed", paint_dome("closed"))
    log("6c dome open")
    C.save_plate("c6c_open", paint_dome("open"))
    log("6c shafts")
    g, t = shaft_plates()
    np.save(C.plate_path("c6c_shaft"), g.astype(np.float16))
    np.save(C.plate_path("c6c_shaft_t"), t.astype(np.float16))


if __name__ == "__main__":
    import time
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "ref":
        ref, info = dome_reference("open")
        g, t = shaft_plates()
        C.save_review(os.path.join(C.REVIEW, "a2-dev", "6c2_ref.png"), np.clip(ref + g, 0, 1)[::2, ::2])
    else:
        paint_all()
    log(f"done {time.time() - t0:.1f}s")
