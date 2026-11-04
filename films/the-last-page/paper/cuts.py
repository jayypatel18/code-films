"""The paper cuts other than the climax: 6a, 6b, 6c, 7, 9a, 9b.

6a  every book flies open; pages tear out of A2's shelves, tumble as flat pages,
    fold on sharp drawn frames (crease, half fold, smear) and fly up as birds.
6b  straight up the stairwell: a rising vortex, near birds big and fast, far
    birds tiny, the spiral tightening toward the dome.
6c  the same vortex as the shutters break: gold light from the dome, the birds
    dark against it with a rim of gold on the edges that face the light.
7   a stream of birds pouring from the dome (above the tower, off frame) out
    into the dawn sky behind the fox; secondary to the sunrise.
9a  pages spiral down through the sunbeams; the one blank page lands at the
    fox's paws at ANCHORS["9a"]["page_land_screen"] at CUES["page_land"].
9b  a few pages still drifting far behind.

Everything is a pure function of the frame: positions are closed form in time,
drawings change on twos.
"""

import math

import numpy as np
from scipy import ndimage

import timeline as TL
from . import bird as BD
from . import drawbird as DB
from . import flock as FL
from . import pages as P_
from . import raster

W, H = TL.W, TL.H


def _dls():
    return {"back": raster.DrawList(), "front": raster.DrawList()}


def _finish(dls, glow=None):
    atlas = P_.atlas()
    back = dls["back"].render(None, atlas=atlas)
    front = dls["front"].render(None, atlas=atlas)
    if front[..., 3].max() <= 0:
        front = None
    if back[..., 3].max() <= 0:
        back = None
    return {"back": back, "front": front, "glow": glow}


def _twos(f, off=0):
    """Time on twos: the drawing holds for two frames."""
    return ((f - off) // 2 * 2 + off) / TL.FPS


# ---------------------------------------------------------------- 6a

_BOOKS = None


def hall_books():
    """Book tops in plate coords from A2's hall layout (read only)."""
    global _BOOKS
    if _BOOKS is None:
        import os
        import sys
        wdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world")
        if wdir not in sys.path:
            sys.path.insert(0, wdir)
        try:
            import cut2_hall as Hh
            bb = [b for b in Hh.all_back_books() + Hh.near_books() if b["kind"] == "stand"]
            M = Hh.M
            _BOOKS = np.array([[b["x"] - M, b["y"] - M - b["h"] * 0.8, b["h"]] for b in bb])
        except Exception:
            r = P_.rng("fallback-books")
            _BOOKS = np.stack([r.uniform(150, 2150, 800), r.uniform(80, 900, 800), r.uniform(20, 50, 800)], 1)
    return _BOOKS


def look_hall():
    L = TL.LIGHTS["hall"]
    return DB.Look(L, "hall", light_world=(0.5, 0.75, 0.35), haze=((0.10, 0.10, 0.20), 30, 70, 0.55), line=1.0,
                   trans=0.35)


ZW6A = 34.0


def render_6a(f):
    f0, f1 = TL.cut_range("6a")
    cam = FL.CutCam("6a", focal=1400.0)
    cam.set_frame(f)
    books = hall_books()
    N = 700
    r = P_.rng("6a-pages")
    bi = r.integers(0, len(books), N)
    s = BD.specs(N, "6a-birds")
    off = r.integers(0, 2, N)
    t = np.array([_twos(f, o) for o in (0, 1)])[off]
    t0 = TL.CUES["burst"]
    birth = t0 + 0.04 + 1.0 * r.uniform(0, 1, N) ** 0.8
    age = t - birth
    alive = age >= 0
    # start: the top of an open book on the wall
    px = books[bi, 0] + r.normal(0, 3, N)
    py = books[bi, 1] + r.normal(0, 2, N)
    p0 = np.array([cam.plate_to_world(px[i], py[i], ZW6A) for i in range(N)])
    page_h = books[bi, 2] * cam.z0 * ZW6A / cam.focal          # world size of the book
    scale = page_h * r.uniform(0.55, 0.8, N)
    # phase lengths: flat page flutter, then crease, half fold, smear (two frames each)
    d_flat = r.uniform(0.12, 0.55, N)
    a = np.maximum(age, 0)
    key = np.full(N, 0)
    fold_t = a - d_flat
    key = np.where(fold_t >= 0, 1, key)
    key = np.where(fold_t >= 2 / 24, 2, key)
    key = np.where(fold_t >= 4 / 24, 3, key)
    bird_t = fold_t - 6 / 24
    drawing = f // 2
    phase = r.integers(0, 6, N)
    key = np.where(bird_t >= 0, FL.flap_key(drawing, phase), key)
    # motion: burst outward from the wall, then rise and swirl toward the stairwell (up)
    v0 = np.stack([r.normal(0, 3.0, N), r.uniform(1.0, 6.0, N), r.uniform(4.0, 16.0, N)], 1)
    kdrag = 2.2
    dr = (1 - np.exp(-kdrag * a)) / kdrag
    pos = p0 + v0 * dr[:, None]
    pos[:, 1] -= 0.8 * a * a * (bird_t < 0)
    bt = np.maximum(bird_t, 0)
    swirl_c = cam.plate_to_world(1152, 300, ZW6A * 0.7)
    rel = pos - swirl_c
    ang0 = np.arctan2(rel[:, 2], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 2])
    w = r.uniform(0.8, 1.6, N) * np.sign(r.normal(0, 1, N) + 0.3)
    up = r.uniform(8.0, 15.0, N)
    ang = ang0 + w * bt
    rad2 = rad * np.exp(-0.25 * bt)
    pos_b = np.stack([swirl_c[0] + np.cos(ang) * rad2, pos[:, 1] + up * bt ** 1.15, swirl_c[2] + np.sin(ang) * rad2], 1)
    kb = np.clip(bt / 0.4, 0, 1)[:, None]
    pos = pos * (1 - kb) + pos_b * kb
    # orientation: tumbling page, then along the flight
    ax = r.normal(0, 1, (N, 3))
    spin = r.uniform(4, 11, N)
    Rt = FL.tumble(ax, spin * a + r.uniform(0, 6.28, N))
    vel = np.stack([-np.sin(ang) * w * rad2, up * np.ones(N), np.cos(ang) * w * rad2], 1) * kb + v0 * (1 - kb)
    Rb = BD.look_rotation(vel, bank=np.clip(w, -1, 1) * 0.5)
    R = np.where((key >= 10)[:, None, None], Rb, Rt)
    dls = _dls()
    FL.draw(dls, s, key, pos, R, scale, cam, look_hall(), drawing, split=ZW6A * 0.45,
            alpha=alive.astype(float), flutter=spin * a)
    return _finish(dls)


# ---------------------------------------------------------------- 6b / 6c vortex

VORTEX_ARMS = 3
VORTEX_TWIST = 0.115          # radians of spiral per unit of height
VORTEX_SPIN = 0.9             # the whole pattern turns this fast (rad/s)


def _vortex(cut, f, gold=0.0):
    """A designed vortex: three spiral bands of birds rising up the stairwell
    with near empty dark gaps between them. Each band is a helix whose radius
    shrinks with height, so on screen it is a spiral that tightens toward the
    dome; the bands turn as one while every bird rises along its band. Birds
    come in from below the frame (constant flux, no wrap), get smaller and
    lighter with height so the bands brighten toward the dome, and a few big
    near birds sweep past the lens on smear drawings."""
    cam = FL.CutCam(cut, pos=(0, 0, 0), look=(0, 1, 0), up=(0, 0, -1), focal=1150.0)
    cam.set_frame(f)
    r = P_.rng("vortex2")
    N = 1300
    s = BD.specs(N, "vortex2-birds")
    off = r.integers(0, 2, N)
    t = np.array([_twos(f, o) for o in (0, 1)])[off]
    tt = t - TL.CUES["cut_stairwell"]
    Hmax = 80.0
    vup = r.uniform(7.0, 11.0, N)
    # birth times spread so the band is always full: h = 2 + v (t - tb)
    tb = r.uniform(-Hmax / 7.0, 4.0, N)
    h = 2.0 + vup * (tt - tb)
    keep = r.random(N) < np.clip(1.35 - 1.2 * h / Hmax, 0.25, 1.0)
    R0 = 10.5
    rad = R0 * (1.0 - 0.62 * np.clip(h / Hmax, 0, 1) ** 0.75) * r.uniform(0.82, 1.06, N)
    arm = r.integers(0, VORTEX_ARMS, N)
    # band width narrows toward the dome
    wid = 0.15 * (1.0 - 0.45 * np.clip(h / Hmax, 0, 1))
    th = arm * (2 * math.pi / VORTEX_ARMS) + r.normal(0, 1, N) * wid + h * VORTEX_TWIST + VORTEX_SPIN * tt
    pos = np.stack([np.cos(th) * rad, h, np.sin(th) * rad], 1)
    # velocity along the band: up, and around with the spin plus the twist
    dth = VORTEX_SPIN + VORTEX_TWIST * vup
    vel = np.stack([-np.sin(th) * rad * dth, vup, np.cos(th) * rad * dth], 1)
    Rb = BD.look_rotation(vel, up_hint=(0, 1, 0), bank=np.full(N, 0.55))
    drawing = f // 2
    key = FL.flap_key(drawing, r.integers(0, 6, N))
    scale = r.uniform(0.6, 1.0, N)
    alpha = keep * np.clip((h - 2.0) / 2.0, 0, 1) * np.clip((Hmax * 0.95 - h) / (Hmax * 0.2), 0, 1)
    # near sweepers: big birds close to the lens, fast, on smear drawings
    M = 12
    rn = P_.rng("vortex2-near")
    sn = BD.specs(M, "vortex2-near-birds")
    hn = rn.uniform(3.2, 7.5, M)
    radn = rn.uniform(2.0, 4.2, M)
    period = rn.uniform(2.2, 3.4, M)
    phn = rn.uniform(0, 2 * math.pi, M)
    tn = _twos(f)
    thn = phn + 2 * math.pi * (tn - TL.CUES["cut_stairwell"]) / period
    hn = hn + 1.5 * np.sin(thn * 0.5)
    posn = np.stack([np.cos(thn) * radn, hn, np.sin(thn) * radn], 1)
    veln = np.stack([-np.sin(thn) * radn, np.full(M, 1.0), np.cos(thn) * radn], 1)
    Rn = BD.look_rotation(veln, up_hint=(0, 1, 0), bank=np.full(M, 0.5))
    keyn = np.where((drawing + np.arange(M)) % 3 == 0, 11, FL.flap_key(drawing, rn.integers(0, 6, M)))
    scn = rn.uniform(0.75, 1.1, M)
    if gold > 0:
        L = dict(TL.LIGHTS["gold_burst"])
        look = DB.Look(L, "gold_burst", light_world=(0.0, 1.0, 0.0), line=1.1, trans=0.25,
                       haze=((1.0, 0.86, 0.6), 18, 70, 0.85 * gold))
        dark = np.array([0.22, 0.13, 0.15])
        look.tones[:, 1] = look.tones[:, 1] * (1 - 0.8 * gold) + dark * 0.8 * gold
        look.tones[:, 2] = look.tones[:, 1]
        look.tones[:, 3] = look.tones[:, 3] * (1 - 0.82 * gold) + np.array([0.40, 0.22, 0.20]) * 0.82 * gold
    else:
        L = dict(TL.LIGHTS["hall"])
        L["key"] = (0.80, 0.86, 1.0)
        look = DB.Look(L, "hall", light_world=(0.2, 1.0, 0.2), line=1.0, trans=0.55,
                       haze=((0.78, 0.84, 1.0), 14, 70, 0.6))
        # moonlit paper, lighter than the dark shelves so the bands pop
        look.tones = np.clip(look.tones * 1.25 + 0.04, 0, 1)
    dls = _dls()
    FL.draw(dls, s, key, pos, Rb, scale, cam, look, drawing, alpha=alpha)
    FL.draw(dls, sn, keyn, posn, Rn, scn, cam, look, drawing + 7, lod_px=40.0)
    out = _finish(dls)
    if gold > 0 and out["back"] is not None:
        _radial_rim(out["back"], gold)
    return out


def _radial_rim(lay, amount, centre=(W / 2, H / 2 - 20)):
    """Gold rim on the edges of the birds that face the dome light at the centre."""
    a = lay[..., 3]
    ab = ndimage.gaussian_filter(a, 1.0)
    gy, gx = np.gradient(ab)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = centre[0] - xs, centre[1] - ys
    n = np.hypot(dx, dy) + 1e-3
    # alpha falls toward the light: the edge faces it
    rim = np.clip(-(gx * dx + gy * dy) / n * 5.0, 0, 1) * a * amount
    c = np.array([1.0, 0.86, 0.55], np.float32)
    lay[..., :3] = lay[..., :3] * (1 - rim[..., None]) + c * rim[..., None]
    return lay


def render_6b(f):
    return _vortex("6b", f)


def render_6c(f):
    t = f / TL.FPS
    gold = float(np.clip((t - TL.CUES["shutters_break"]) / 0.25, 0, 1))
    out = _vortex("6c", f, gold=gold)
    if gold > 0:
        # light crashing down the shaft: warm glow at the centre of the frame
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot((xs - W / 2) / 700, (ys - H / 2 + 20) / 600)
        g = np.exp(-d * d * 2.5)[..., None] * np.array([1.0, 0.8, 0.5], np.float32) * 0.35 * gold
        out["glow"] = g.astype(np.float32)
    return out


# ---------------------------------------------------------------- 7

def render_7(f):
    cam = FL.CutCam("7", focal=1400.0)
    cam.set_frame(f)
    t = f / TL.FPS
    N = 560
    r = P_.rng("stream7")
    s = BD.specs(N, "stream7-birds")
    off = r.integers(0, 2, N)
    tt = np.array([_twos(f, o) for o in (0, 1)])[off] - TL.CUES["cut_sunrise"]
    # path in plate coords with depth, from above the tower (off frame top left) out toward the sun
    path = np.array([[300, -360, 22], [560, 40, 34], [1050, 300, 62], [1650, 360, 130], [2200, 260, 260]], float)
    u0 = r.uniform(-1.2, 1.0, N)
    sp = r.uniform(0.16, 0.24, N)
    u = u0 + sp * tt
    vis = (u >= 0) & (u <= 1)
    uu = np.clip(u, 0, 1)
    seg = np.clip((uu * (len(path) - 1)).astype(int), 0, len(path) - 2)
    fr = uu * (len(path) - 1) - seg
    P = path[seg] * (1 - fr)[:, None] + path[seg + 1] * fr[:, None]
    spread = (34 + 110 * uu) * r.normal(0, 1, N)
    Z = P[:, 2] * r.uniform(0.85, 1.2, N)
    pos = np.array([cam.plate_to_world(P[i, 0] + spread[i] * 0.4, P[i, 1] + spread[i], Z[i]) for i in range(N)])
    d = path[seg + 1] - path[seg]
    vel = np.array([cam.plate_to_world(P[i, 0] + d[i, 0] * 0.05, P[i, 1] + d[i, 1] * 0.05, Z[i] + d[i, 2] * 0.05)
                    for i in range(N)]) - pos
    Rb = BD.look_rotation(vel, bank=r.normal(0, 0.3, N))
    drawing = f // 2
    key = FL.flap_key(drawing, r.integers(0, 6, N))
    scale = r.uniform(1.8, 3.0, N)
    L = TL.LIGHTS["sunrise"]
    look = DB.Look(L, "sunrise", light_world=(0.85, 0.15, -0.5), line=0.9, trans=0.6,
                   haze=((1.0, 0.82, 0.62), 60, 400, 0.65))
    dls = _dls()
    FL.draw(dls, s, key, pos, Rb, scale, cam, look, drawing, alpha=vis.astype(float))
    return _finish(dls)


# ---------------------------------------------------------------- 9a / 9b

def look_morning(name="morning"):
    L = TL.LIGHTS[name]
    return DB.Look(L, name, light_world=(0.75, 0.55, -0.35), line=1.0, trans=0.55,
                   haze=((1.0, 0.92, 0.80), 10, 60, 0.5))


def _cam9a():
    return FL.CutCam("9a", pos=(0.0, 2.2, 9.0), look=(0.0, 0.9, 0.0), focal=1400.0)


def landing_point():
    cam = _cam9a()
    f_land = TL.sec(TL.CUES["page_land"])
    sx, sy = TL.ANCHORS["9a"]["page_land_screen"]
    x0, y0 = cam.screen_to_screen0(f_land, sx, sy)
    b = cam.base
    d = b.R.T @ np.array([(x0 - b.cx) / b.focal, (y0 - b.cy) / b.focal, 1.0])
    k = -b.pos[1] / d[1]
    return b.pos + d * k


def _falling_pages(f, cam, N, key, t_ref, area, rail_sy=None, land=True, size=(0.35, 0.7)):
    """Pages spiralling down like leaves. With land=True they settle flat on the
    floor (area must lie in front of the balustrade); with rail_sy they fade
    out before crossing the painted rail's top edge on screen (they are behind it)."""
    r = P_.rng("fall", key)
    s = BD.specs(N, ("fall-pages", key))
    t = _twos(f) - t_ref
    x0 = r.uniform(area[0], area[1], N)
    z0 = r.uniform(area[2], area[3], N)
    y0 = r.uniform(area[4], area[5], N)
    vy = r.uniform(0.55, 1.1, N)
    rad = r.uniform(0.2, 0.8, N)
    om = r.uniform(1.5, 3.2, N) * np.sign(r.normal(0, 1, N))
    ph = r.uniform(0, 6.28, N)
    y = y0 - vy * t
    landed = (y <= 0.004) & land
    y = np.maximum(y, 0.004) if land else y
    tl = np.where(landed, (y0 - 0.004) / vy, t)
    x = x0 + np.cos(ph + om * tl) * rad + 0.25 * tl
    z = z0 + np.sin(ph + om * tl) * rad
    pos = np.stack([x, y, z], 1)
    ax = np.stack([np.cos(ph), np.full(N, 0.4), np.sin(ph)], 1)
    rock = 1.1 * np.sin(om * tl * 1.3 + ph) + om * tl * 0.5
    Rt = FL.tumble(ax, rock)
    flat = BD.orient(ph, np.zeros(N), np.zeros(N))
    R = np.where(landed[:, None, None], flat, Rt)
    scale = r.uniform(size[0], size[1], N)
    alpha = np.ones(N)
    if rail_sy is not None:
        sy = cam.project(pos)[:, 1]
        alpha = np.clip((rail_sy - sy) / 40.0, 0, 1)
    flut = np.where(landed, 0.0, om * tl * 2.0)
    s["curl"] = np.where(landed, s["curl"] * 0.4, s["curl"])
    return s, pos, R, scale, alpha, flut


def render_9a(f):
    cam = _cam9a()
    cam.set_frame(f)
    t = _twos(f)
    dls = _dls()
    look = look_morning()
    t_ref = TL.CUES["cut_last_page"] - 1.5
    # near pages land on the floor in front of the balustrade (ground z > -0.6);
    # far pages drift down beyond it and fade before the rail top (screen y 470)
    s, pos, R, scale, alpha, flut = _falling_pages(f, cam, 16, "9a-near", t_ref, (-6, 6, -0.4, 4.0, 1.5, 7.0),
                                                   size=(0.25, 0.5))
    FL.draw(dls, s, np.zeros(s["B"], int), pos, R, scale, cam, look, f // 2, split=7.0, alpha=alpha, flutter=flut)
    s, pos, R, scale, alpha, flut = _falling_pages(f, cam, 22, "9a-far", t_ref, (-14, 14, -16, -4, 3.0, 9.0),
                                                   rail_sy=cam.project(np.array([[0.0, 0.0, -0.75]]))[0, 1] - 190,
                                                   land=False, size=(0.4, 0.8))
    FL.draw(dls, s, np.zeros(s["B"], int), pos, R, scale, cam, look, f // 2, alpha=alpha, flutter=flut)
    # the blank page: spirals down and lands at the fox's paws
    hs = BD.specs(1, "blank-page")
    hs["page"] = np.array([P_.BLANK])
    hs["page_back"] = np.array([P_.BLANK])
    hs["tint"] = np.array([1])
    hs["curl"] = np.array([0.5 if t < TL.CUES["page_land"] else 0.22])
    land = landing_point()
    tl = TL.CUES["page_land"]
    ts = TL.CUES["cut_last_page"]
    u = np.clip((t - ts) / (tl - ts), 0, 1)
    fall = 1 - u
    ang = 4.2 * fall
    p = land + np.array([np.cos(ang) * 1.1 * fall + 0.4 * fall, 3.6 * fall ** 1.3 + 0.004, np.sin(ang) * 0.9 * fall])
    if t >= tl:
        # settle: a small slide and the corner lifting once
        k = min(1.0, (t - tl) / 0.4)
        p = land + np.array([0.05 * (1 - (1 - k) ** 2), 0.004, 0.0])
    heading = math.radians(-8) + 0.6 * fall
    Rf = BD.orient(np.array([heading]), np.array([0.0]), np.array([0.0]))[0]
    if t < tl:
        Rf = FL.tumble(np.array([[0.4, 0.3, 1.0]]), np.array([1.3 * math.sin(ang * 1.4) * fall]))[0] @ Rf
    hscale = 0.85
    hk = np.array([0])
    flut_h = np.array([ang * 2.0 if t < tl else 0.0])
    FL.draw(dls, hs, hk, p[None], Rf[None], np.array([hscale]), cam, look, f // 2, split=7.0, flutter=flut_h)
    return _finish(dls)


def render_9b(f):
    cam = FL.CutCam("9b", pos=(0.0, 3.0, 9.0), look=(0.0, 1.8, 0.0), focal=1400.0)
    cam.set_frame(f)
    dls = _dls()
    s, pos, R, scale, alpha, flut = _falling_pages(f, cam, 9, "9b", TL.CUES["cut_title"] - 2.0,
                                                   (-14, 14, -40, -25, 7.0, 13.0), rail_sy=230.0, land=False)
    key = np.zeros(s["B"], int)
    FL.draw(dls, s, key, pos, R, scale * 1.4, cam, look_morning(), f // 2, alpha=alpha, flutter=flut)
    return _finish(dls)


RENDER = {"6a": render_6a, "6b": render_6b, "6c": render_6c, "7": render_7, "9a": render_9a, "9b": render_9b}
