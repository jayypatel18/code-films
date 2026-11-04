"""Cut 8, the climax, second pass: the great bird (great2), its gathering
flock, the wingbeat with its painted blast ring and the shock across the cloud
tops, a cloud wisp crossing a wingtip, pages always shedding from the wing
edges, and the arc in side or three quarter view with the ribbon of light.

Layers: 'back' (blast ring, shock, shed pages and flock birds behind the bird,
the bird with its golden rim), 'front' (pages and birds nearer than the bird,
the cloud wisp), 'glow' (the ribbon of light, the shock's bright crest, the
warm bloom around the bird).
"""

import math

import numpy as np
from scipy import ndimage

import timeline as TL
from engine import comp
from engine import raster as eraster
from engine.noise import fbm
from . import bird as BD
from . import drawbird as DB
from . import flock as FL
from . import great2 as G
from . import pages as P_
from . import raster
from . import wisp as WS
from .scene8 import _catmull, arc_table, rim_light

W, H = TL.W, TL.H
FOCAL = 1350.0
CAM = BD.Camera(pos=(0, 0, 0), look=(0, 0.16, -1), focal=FOCAL)
CAM.k = 1.0
CAM.depth = lambda X: (np.asarray(X) - CAM.pos) @ CAM.R[2]
SUN_WORLD = (0.85, 0.22, -0.28)
HAZE = (0.99, 0.84, 0.70)
SCALE = 16.0
ARC_Z = [70.0, 92.0, 115.0, 128.0, 220.0]


def look():
    return G.GreatLook(TL.LIGHTS["sunrise"], "sunrise", sun_world=SUN_WORLD, haze_col=HAZE, haze_z=(60, 240, 0.5))


def small_look():
    return DB.Look(TL.LIGHTS["sunrise"], "sunrise", light_world=SUN_WORLD, haze=(HAZE, 40, 260, 0.6), line=0.9,
                   trans=0.55)


def unproject(sx, sy, Z):
    d = CAM.R.T @ np.array([(sx - CAM.cx) / FOCAL, (sy - CAM.cy) / FOCAL, 1.0])
    return CAM.pos + d * (Z / (CAM.R[2] @ d))


def _orient_visible(fwd, pos, bank, min_side=0.55, min_wing=0.32, up_toward=0.35):
    """Body rotation that keeps the bird in side or three quarter view with both
    wings visible: the heading is pushed off the line of sight, the up vector
    leans toward the camera, and the bank is clamped so the wing plane never
    goes edge on."""
    v = CAM.pos - pos
    v = v / np.linalg.norm(v)
    f = fwd / (np.linalg.norm(fwd) + 1e-12)
    d = float(f @ v)
    c = math.sqrt(1 - min_side ** 2)
    if abs(d) > c:
        f = f - v * d + v * math.copysign(c, d)
        f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0]) * (1 - up_toward) + v * up_toward
    best = None
    for b in np.linspace(-1.2, 1.2, 49):
        R = BD.look_rotation(f[None], up_hint=up, bank=np.array([b]))[0]
        wing_n = R[:, 2]
        ok = abs(float(wing_n @ v)) >= min_wing
        cost = abs(b - bank) + (0 if ok else 10)
        if best is None or cost < best[0]:
            best = (cost, R)
    return best[1]


def _twos_t(f):
    return (f // 2 * 2) / TL.FPS


def sun_screen(f):
    """Screen position of A2's sun in cut 8 (plate (2300, 900) on the 0.5 parallax camera)."""
    f0 = TL.cut_range("8")[0]
    z0, x0, y0 = TL.camera("8", f0)
    z, x, y = TL.camera("8", f)
    zz, cx, cy = z0 * (z / z0) ** 0.5, x0 + (x - x0) * 0.5, y0 + (y - y0) * 0.5
    return (2300 - cx) * zz + W / 2, (900 - cy) * zz + H / 2


def body_state(t):
    a = TL.ANCHORS["8"]
    f = t * TL.FPS
    if t < TL.CUES["arc_start"]:
        px, py = a["bird_form_centre"]
        lift = 30 * G.smooth(46.5, 47.3, t) - 12 * G.smooth(44.0, 46.3, t)
        drift = 60 * G.smooth(47.0, 48.0, t)
        sx, sy = TL.plate_to_screen("8", f, px + drift, py - lift)
        Z = ARC_Z[0] + 4.0 * (1 - G.smooth(44.0, 46.3, t))
        pos = unproject(sx - 40, sy + 120, Z)
        k1 = float(G.smooth(46.25, 46.52, t))
        k2 = float(G.smooth(47.1, 48.0, t))
        fwd = np.array([0.50, 0.40, 0.77]) * (1 - k1) + np.array([0.42, 0.42, 0.80]) * k1
        up = np.array([0.0, 1.0, -0.30]) * (1 - k1) + np.array([0.0, 1.0, -0.35]) * k1
        fwd = fwd * (1 - k2) + np.array([0.95, 0.22, -0.2]) * k2
        up = up * (1 - k2) + np.array([0.0, 1.0, 0.45]) * k2
        R = BD.look_rotation(fwd[None], up_hint=up)[0]
        return pos, R, SCALE, (sx, sy)
    pts, seg, dn = arc_table()
    x = np.clip((t - TL.CUES["arc_start"]) / (TL.CUES["arc_end"] - TL.CUES["arc_start"]), 0, 1)
    u = 0.3 * x + 0.7 * x * x * (3 - 2 * x)

    def at(uu):
        uu = float(np.clip(uu, 0, 1))
        px = np.interp(uu, dn, pts[:, 0])
        py = np.interp(uu, dn, pts[:, 1])
        sg = np.interp(uu, dn, seg)
        Z = np.interp(sg, np.arange(len(ARC_Z)), ARC_Z)
        sx, sy = TL.plate_to_screen("8", f, px, py)
        # keep the whole bird inside the frame through the turn at the right edge
        lim = 1480.0
        if sx > lim:
            sx = lim + (sx - lim) * 0.35
        return unproject(sx - 20, sy + 110, Z), (sx, sy)

    pos, sxy = at(u)
    # heading from the smoothed screen velocity: a reversal passes through a three
    # quarter view toward the camera, banked hard so both wings stay visible
    vs = []
    for du in (-0.03, -0.015, 0.015, 0.03):
        vs.append(np.array(at(u + du)[1]))
    vel = (vs[2] + vs[3] - vs[0] - vs[1]) / 2.0
    sp = np.linalg.norm(vel) + 1e-6
    vx, vy = vel[0] / sp, vel[1] / sp
    toward = CAM.pos - pos
    toward = toward / np.linalg.norm(toward)
    right = CAM.R[0]
    upw = np.array([0.0, 1.0, 0.0])
    th = math.acos(float(np.clip(vx, -1, 1)))           # 0 rightward, pi leftward, pi/2 toward us
    th = 0.5 * math.pi + (th - 0.5 * math.pi) * 0.9 if abs(vx) < 0.999 else th
    heading = right * math.cos(th) + toward * math.sin(th) * 0.75
    fwd = heading - upw * vy * 0.22
    turning = 1.0 - abs(vx)                               # 1 at the reversal
    bank = math.radians(20) * turning
    R = BD.look_rotation(fwd[None], up_hint=upw * 0.5 + toward * 0.5, bank=np.array([bank]))[0]
    k = float(G.smooth(48.0, 48.5, t))
    if k < 1:
        pos_b, R_b, _, sxy_b = body_state(TL.CUES["arc_start"] - 1e-3)
        pos = pos_b * (1 - k) + pos * k
        M = R_b * (1 - k) + R * k
        uu, _, vt = np.linalg.svd(M)
        R = uu @ vt
        sxy = (sxy_b[0] * (1 - k) + sxy[0] * k, sxy_b[1] * (1 - k) + sxy[1] * k)
    return pos, R, SCALE, sxy


def wing_world(t, x, y, side, h=0.0):
    pose = G.pose_at(t)
    pos, R, S, _ = body_state(t)
    P = G.wing_point(np.asarray(x), np.asarray(y), np.full(np.shape(x), h), pose["wing"], side)
    return pos + S * (P @ R.T)


def key_points(t):
    pose = G.pose_at(t)
    pos, R, S, _ = body_state(t)
    wp = pose["wing"]
    tl = G.wing_point(np.array([G.le_x(1.7) - 0.1]), np.array([1.85]), np.array([0.0]), wp, +1)[0]
    tr = tl * np.array([1, -1, 1])
    tail = np.array([-0.9, 0.0, 0.0])
    loc = np.stack([tail, tl, tr])
    return pos[None] + S * (loc @ R.T)


# ---------------------------------------------------------------- the ribbon of light (kept from v1)

def trail_glow(f):
    t = f / TL.FPS
    t_start = TL.CUES["arc_start"]
    if t <= t_start + 0.05:
        return None
    ts = np.arange(t_start, t, 1.0 / 48.0)
    if len(ts) < 3:
        return None
    pts = np.array([CAM.project(body_state(tt)[0][None])[0] for tt in ts])
    sc = 4
    h, w = H // sc, W // sc
    core = np.zeros((h, w), np.float32)
    halo = np.zeros((h, w), np.float32)
    age = (t - ts) / max(1e-3, t - t_start)
    fade = np.clip(1.0 - age, 0, 1) ** 1.2
    near = np.clip((t - ts - 0.08) / 0.4, 0, 1)
    P = pts[:, :2] / sc
    eraster.stroke(core, P, (0.6 + 4.0 * fade) * near)
    eraster.stroke(halo, P, (3.0 + 20.0 * fade) * near)
    nz = fbm(h, w, 22, ("trail", f // 2), octaves=3)
    core *= 0.55 + 0.7 * nz
    halo *= 0.6 + 0.6 * nz
    core = ndimage.gaussian_filter(core, 1.6)
    halo = ndimage.gaussian_filter(halo, 10.0)
    g = core[..., None] * np.array([1.0, 0.82, 0.55], np.float32) * 0.32 + \
        halo[..., None] * np.array([1.0, 0.70, 0.40], np.float32) * 0.22
    up = np.repeat(np.repeat(g, sc, 0), sc, 1)[:H, :W]
    return ndimage.uniform_filter(up, size=(sc, sc, 1)).astype(np.float32)


# ---------------------------------------------------------------- the wingbeat blast

def _brushed_ring(cov, cx, cy, rx, ry, width, key, a0=0.0, a1=2 * math.pi, n=96, strength=1.0):
    """A painted ring: several broken brush strokes along an ellipse."""
    r = np.random.default_rng(abs(hash(key)) % (2 ** 31))
    for k in range(6):
        off = r.normal(0, width * 0.35)
        th0 = a0 + r.uniform(0, 0.4)
        th1 = a1 - r.uniform(0, 0.4)
        th = np.linspace(th0, th1, n)
        x = cx + np.cos(th) * (rx + off)
        y = cy + np.sin(th) * (ry + off * ry / max(rx, 1))
        prof = np.sin(np.linspace(0, math.pi, n)) ** 0.5 * r.uniform(0.5, 1.0)
        prof *= 0.6 + 0.4 * np.sin(th * r.uniform(3, 7) + r.uniform(0, 6))
        eraster.stroke(cov, np.stack([x, y], 1), np.maximum(prof * width * r.uniform(0.35, 0.7), 0.0) * strength,
                       accumulate=True)


def _ring_field(h, w, cx, cy, rx, ry, thick, lobes, amp, phase):
    """Painted air ring on a (h, w) grid: returns (coverage, lit, crest, theta).
    The outer edge is crisp and scalloped (billow lobes), the inner edge soft."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xs - cx) / max(rx, 1e-3), (ys - cy) / max(ry, 1e-3)
    rho = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    scal = amp * np.abs(np.sin(lobes * 0.5 * th + phase)) ** 0.55
    outer = 1.0 + scal
    inner = outer - thick * (0.75 + 0.5 * np.abs(np.sin(lobes * 0.5 * th + phase + 0.6)))
    px = 1.0 / max(rx, 1.0)
    cov_out = np.clip((outer - rho) / (1.5 * px), 0, 1)
    cov_in = np.clip((rho - inner) / (thick * 0.55 + 1e-4), 0, 1) ** 1.6
    cov = cov_out * cov_in
    # cel tones: the lobe tops toward the sun are lit, the undersides in shade
    lit = ((rho - inner) / (outer - inner + 1e-4) > 0.45) & (dy < 0.25 + 0.35 * np.sin(th * 2 + phase))
    crest = np.clip(1.0 - np.abs(outer - 0.012 - rho) / (2.0 * px), 0, 1) * cov_out
    return cov, lit.astype(np.float32), crest, th


def blast(f):
    """The wingbeat: a designed, cel painted ring of displaced air around the
    bird (far half behind it, near half in front) and a shock arc rolling out
    across the cloud tops. Returns (back layer, front layer, glow)."""
    t = f / TL.FPS
    b0, b1 = 46.55, 48.7
    if not (b0 <= t <= b1):
        return None, None, None
    sc = 2
    h, w = H // sc, W // sc
    pos, R, S, sxy = body_state(min(t, 47.9))
    u = float(G.smooth(b0, 48.0, t))
    life = float(1 - G.smooth(47.5, b1, t))
    # wisps break off as it fades: a noise threshold that rises with age
    nz = fbm(h, w, 22, ("ringn", f // 2), octaves=3)
    brk = np.clip((nz - (1.0 - life) * 0.75) * 6.0, 0, 1)
    cx, cy = (sxy[0] - 30) / sc, (sxy[1] + 250 + 110 * u) / sc
    rx = (300 + 1050 * u) / sc
    ry = rx * 0.28
    thick = 0.11 * (1.0 - 0.5 * u) + 0.03
    cov, lit, crest, th = _ring_field(h, w, cx, cy, rx, ry, thick, 16, 0.085 * (1 - 0.4 * u),
                                      (f // 2) * 0.07)
    # an echo ring trailing inside the first, thinner and fainter
    cov2, lit2, crest2, _ = _ring_field(h, w, cx, cy - 6 / sc, rx * 0.72, ry * 0.72, thick * 0.6, 12,
                                        0.06, (f // 2) * 0.05 + 1.3)
    cov = np.maximum(cov, cov2 * 0.55)
    lit = np.where(cov2 * 0.55 > cov * 0.99, lit2, lit)
    crest = np.maximum(crest, crest2 * 0.5)
    front_m = (th > 0).astype(np.float32)
    # the shock across the cloud tops: a thin crisp arc, light line over a shadow line
    us = float(G.smooth(46.62, 48.4, t))
    scx, scy = (sxy[0] + 40) / sc, 770 / sc
    srx = (180 + 1800 * us) / sc
    sry = srx * 0.10
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rho_s = np.sqrt(((xs - scx) / srx) ** 2 + ((ys - scy) / sry) ** 2)
    pxs = 1.0 / max(srx, 1.0)
    wl = (10.0 - 4.0 * us) / sc / max(srx, 1.0) + 0.01
    band_l = np.clip((1.0 - rho_s) / (1.5 * pxs), 0, 1) * np.clip((rho_s - (1.0 - wl)) / (1.5 * pxs), 0, 1)
    band_d = np.clip((1.0 - wl - rho_s) / (1.5 * pxs), 0, 1) * np.clip((rho_s - (1.0 - 2.6 * wl)) / (wl * 1.2), 0, 1)
    lower = (ys > scy).astype(np.float32)
    shock_l = band_l * lower
    shock_d = band_d * lower * 0.7
    shock_life = float(1 - G.smooth(48.0, b1, t))
    nzs = fbm(h, w, 90, ("shockn", f // 3), octaves=2)
    brk_s = np.clip((nzs - 0.25) * 3.0, 0.15, 1)
    cream = np.array([1.0, 0.95, 0.86], np.float32)
    lilac = np.array([0.80, 0.72, 0.86], np.float32)
    shade_ring = np.array([0.86, 0.74, 0.84], np.float32)

    def layer(m, with_shock):
        a = np.clip(cov * m * brk * 0.7 * life, 0, 1)
        col = shade_ring[None, None] * (1 - lit[..., None]) + cream[None, None] * lit[..., None]
        lay = np.zeros((h, w, 4), np.float32)
        ma = mist_a * m
        lay[..., :3] = cream * ma[..., None]
        lay[..., 3] = ma
        lay[..., :3] = lay[..., :3] * (1 - a[..., None]) + col * a[..., None]
        lay[..., 3] = lay[..., 3] * (1 - a) + a
        if with_shock:
            sl = np.clip(shock_l * brk_s * 0.8 * shock_life, 0, 1)
            sd = np.clip(shock_d * brk_s * 0.5 * shock_life, 0, 1)
            for aa, cc in ((sd, lilac), (sl, cream)):
                lay[..., :3] = lay[..., :3] * (1 - aa[..., None]) + cc * aa[..., None]
                lay[..., 3] = lay[..., 3] * (1 - aa) + aa
        return np.repeat(np.repeat(lay, sc, 0), sc, 1)[:H, :W]

    # the soft downdraft under the wings (approved in the first pass): broken puffs of vapour
    dd = np.zeros((h, w), np.float32)
    ud = float(G.smooth(46.55, 47.2, t)) * float(1 - G.smooth(47.3, 48.3, t))
    if ud > 0:
        _brushed_ring(dd, cx, cy - 20 / sc, rx * 0.55, ry * 0.6, 120 / sc, ("dd", f // 2), strength=1.0)
        nzd = fbm(h, w, 18, ("ddn", f // 2), octaves=4)
        dd = ndimage.gaussian_filter(dd, 4.0) * np.clip((nzd - 0.3) * 2.0, 0, 1) * ud
    mist_a = np.clip(dd * 0.55, 0, 0.6)
    g = np.clip(crest * brk * life, 0, 1) * (0.6 + 0.4 * front_m) + np.clip(shock_l * brk_s * shock_life, 0, 1) * 0.6
    g = np.repeat(np.repeat(g, sc, 0), sc, 1)[:H, :W]
    glow = g[..., None] * np.array([1.0, 0.88, 0.66], np.float32) * 0.28
    return layer(1.0 - front_m, False), layer(front_m, True), glow


# ---------------------------------------------------------------- the cloud wisp

def wisp_layer(f):
    t = f / TL.FPS
    sp = WS.load()
    if sp is None or t > 48.4:
        return None
    u = (t - 44.0) / 4.4
    cx = 560 - 360 * u
    cy = 245 - 50 * u
    s = 0.72
    hh, ww = int(sp.shape[0] * s), int(sp.shape[1] * s)
    from PIL import Image
    im = np.stack([np.asarray(Image.fromarray((sp[..., c] * 65535).astype(np.uint16)).resize((ww, hh), Image.BILINEAR),
                              np.float32) / 65535 for c in range(4)], -1)
    a = float(G.smooth(44.0, 44.6, t)) * float(1 - G.smooth(47.9, 48.4, t)) * 0.92
    lay = np.zeros((H, W, 4), np.float32)
    comp.place(lay, im * a, int(cx - ww / 2), int(cy - hh / 2))
    return lay


# ---------------------------------------------------------------- shed pages

def shed_pages(dls, f, look_s, split):
    """Pages always peeling off the wing edges and tips, more on the beat; real
    paper (curl, a fold, flutter), varied sizes, tumbling as they fall."""
    t = f / TL.FPS
    r = P_.rng("shed8")
    N = 74
    birth = np.concatenate([r.uniform(45.6, 52.0, 52), r.uniform(46.55, 46.95, 22)])
    birth = np.round(birth * 24) / 24
    alive = birth <= t
    if not alive.any():
        return 0
    idx = np.flatnonzero(alive)
    s = BD.take(BD.specs(N, "shed8-pages"), idx)
    side = np.where(r.random(N) < 0.5, 1, -1)[idx]
    yy = r.uniform(0.4, 1.9, N)[idx]
    xx = (G.te_x(np.minimum(yy, 1.3)) + 0.05)
    xx = np.where(yy > 1.3, G.le_x(yy) - 0.25, xx)
    sz = (r.uniform(0.0, 1.0, N) ** 2.5 * 2.2 + 0.6)[idx]
    phase = r.uniform(0, 6.28, N)[idx]
    spin_ax = r.normal(0, 1, (N, 3))[idx]
    spin = r.uniform(2.0, 6.0, N)[idx]
    drift = r.normal(0, 1, (N, 3))[idx] * np.array([3.0, 1.0, 3.0])
    pos = np.zeros((len(idx), 3))
    for tb in np.unique(birth[idx]):
        m = birth[idx] == tb
        P0 = np.stack([wing_world(tb, xx[m][i:i + 1], yy[m][i:i + 1], side[m][i])[0] for i in range(m.sum())])
        pa, _, _, _ = body_state(tb)
        pb, _, _, _ = body_state(tb + 0.05)
        v = (pb - pa) / 0.05 * 0.45
        age = t - tb
        dr = (1 - np.exp(-1.6 * age)) / 1.6
        pos[m] = P0 + v[None] * dr + drift[m] * age + np.array([0, -4.0, 0]) * age * age * 0.5
    age = t - birth[idx]
    sway = np.stack([np.sin(phase + age * 2.3), np.zeros_like(age), np.cos(phase + age * 1.9)], 1) * 1.5
    pos = pos + sway
    Rt = FL.tumble(spin_ax, spin * age + phase)
    key = np.where(r.random(N)[idx] < 0.6, 0, 1)
    alpha = np.clip(age / 0.08, 0, 1) * np.clip((3.2 - age) / 0.8, 0, 1)
    return FL.draw(dls, s, key, pos, Rt, sz, CAM, look_s, f // 2, split=split, alpha=alpha, flutter=spin * age * 1.4)


# ---------------------------------------------------------------- the flock joining

def flock(dls, f, look_s, split):
    """Birds of the flock streaming in: the first ones each fly to one page's
    place and become it at that page's arrival time (the assembly), the rest
    keep joining the rim of the finished bird until 48.6 s."""
    t = _twos_t(f) if f % 2 else f / TL.FPS
    dz = G.design()
    N1 = 520
    N2 = 160
    r = P_.rng("flock8c")
    pj = r.choice(dz.n, N1, replace=False)
    arrive = np.concatenate([dz.arrive[pj], r.uniform(46.6, 48.6, N2)])
    N = N1 + N2
    vis = t < arrive
    if not vis.any():
        return 0
    idx = np.flatnonzero(vis)
    s = BD.take(BD.specs(N, "flock8c-birds"), idx)
    n = len(idx)
    t_eval = min(t, 48.6)
    pose = G.pose_at(max(t_eval, 46.3) if t_eval < 46.3 else t_eval)
    pos_b, R, S, sxy = body_state(t_eval)
    # targets: page centres (formed pose) for the assembly birds, rim points for the late ones
    side = np.where(r.random(N) < 0.5, 1, -1)
    ty = r.uniform(0.2, 1.9, N)
    tx = np.where(r.random(N) < 0.5, G.le_x(ty), G.te_x(np.minimum(ty, 1.3)))
    rim = G.wing_point(tx, ty, np.zeros(N), pose["wing"], 1) * np.where(side[:, None] > 0, 1, np.array([1, -1, 1]))
    loc = np.concatenate([dz.centre_formed[pj], rim[N1:]], 0)[idx]
    tgt = pos_b[None] + S * (loc @ R.T)
    ang = r.uniform(0, 2 * math.pi, N)[idx]
    start_r = r.uniform(800, 1500, N)[idx]
    zst = r.uniform(40, 150, N)[idx]
    sx0 = sxy[0] + np.cos(ang) * start_r
    sy0 = sxy[1] + np.sin(ang) * start_r * 0.6 + 220
    p0 = np.array([unproject(sx0[i], sy0[i], zst[i]) for i in range(n)])
    t_go = arrive[idx] - r.uniform(1.4, 2.4, N)[idx]
    u = np.clip((t - t_go) / np.maximum(1e-3, arrive[idx] - t_go), 0, 1)
    u = u * u * (3 - 2 * u)
    P = p0 + (tgt - p0) * u[:, None]
    P[:, 1] += np.sin(u * math.pi) * r.uniform(2, 10, N)[idx]
    Rb = BD.look_rotation(tgt - p0, bank=r.normal(0, 0.3, N)[idx])
    drawing = f // 2
    key = FL.flap_key(drawing, r.integers(0, 6, N)[idx])
    sc = (1.0 - 0.5 * u ** 3) * r.uniform(1.2, 2.2, N)[idx]
    alpha = np.clip((1 - u) / 0.06, 0, 1)
    return FL.draw(dls, s, key, P, Rb, sc, CAM, look_s, drawing, split=split, alpha=alpha)


# ---------------------------------------------------------------- speed lines

def speed_lines(dlb, t, R, pos, S, n_per=9, N=17):
    dt = 0.14
    p_now = G.pose_at(t)["wing"]
    p_then = G.pose_at(t - dt)["wing"]
    ys = np.linspace(0.8, G.YTIP, n_per)
    xs = G.le_x(ys) - np.linspace(0.45, 0.12, n_per)
    pts_all, cols = [], []
    pos0, R0, S0, _ = body_state(t - dt)
    for side in (1, -1):
        A = G.wing_point(xs, ys, np.zeros(n_per), p_then, side)
        B = G.wing_point(xs, ys, np.zeros(n_per), p_now, side)
        for i in range(n_per):
            u = np.linspace(0, 1, N)[:, None]
            Aw = pos0 + S0 * (R0 @ A[i])
            Bw = pos + S * (R @ B[i])
            mid = (Aw + Bw) / 2 + (pos - Aw) * 0.08
            P = (1 - u) ** 2 * Aw + 2 * (1 - u) * u * mid + u ** 2 * Bw
            Sx = CAM.project(P)
            rad = (1.0 + 2.2 * (i / n_per)) * np.sin(u[:, 0] * math.pi) ** 0.6 * (0.3 + 0.7 * u[:, 0])
            pts_all.append(np.concatenate([Sx[:, :2], np.full((N, 1), 1e5), rad[:, None]], -1))
            cols.append([1.0, 0.95, 0.85, 0.85])
    dlb.add(str_pts=np.array(pts_all, np.float32)[None], str_col=np.array(cols, np.float32)[None], ss=2, depth=1e5)


_PAR = None


def balcony_alpha(f):
    """Alpha of A2's balcony layer for this frame (read only), so paper stays behind it."""
    global _PAR
    try:
        import os
        import sys
        wdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world")
        if wdir not in sys.path:
            sys.path.insert(0, wdir)
        import common as WC
        if _PAR is None:
            _PAR = WC.load_plate("c8_par")
        return WC.sample(_PAR, TL.camera("8", f))[..., 3]
    except Exception:
        return None


# ---------------------------------------------------------------- the cut

def render(f, with_flock=True):
    t = f / TL.FPS
    lg = look()
    ls = small_look()
    pose = G.pose_at(t)
    pos, R, S, sxy = body_state(t)
    z_bird = float(CAM.depth(pos))
    dls = {"back": raster.DrawList(), "front": raster.DrawList()}
    dlb = raster.DrawList()
    smear_dir, smear_amt = None, 0.0
    if pose["smear"]:
        smear_dir = -(R @ np.array([0.0, 0.0, 1.0]))
        smear_amt = 0.5 * math.sin(math.pi * (t - 46.55) / 0.15)
        speed_lines(dls["back"], t, R, pos, S)
    eyes = G.eye_strokes(CAM, R, pos, S, 0.95)
    bx, by = CAM.project(pos[None])[0, :2]
    sx_, sy_ = sun_screen(f)
    sil = float(G.smooth(700, 180, math.hypot(bx - sx_, by - sy_)))
    tq = _twos_t(f)
    if tq < 46.3:
        ea = float(G.smooth(45.7, 46.0, tq))
        eyes = ([p for p in eyes[0]], [np.array([*c[:3], c[3] * ea]) for c in eyes[1]])
    G.bird_draw(dlb, G.design(), pose, lg, CAM, R, pos, S, drawing=f // 2, smear_dir=smear_dir,
                smear_amt=smear_amt, extra=eyes, backlit=sil, assemble=tq if tq < 46.3 else None)
    shed_pages(dls, f, ls, z_bird - 2)
    if with_flock:
        flock(dls, f, ls, z_bird - 1)
    atlas = P_.atlas()
    mist, mist_near, mglow = blast(f)
    # everything in this cut is farther than the balcony: one back layer in depth order
    back = mist if mist is not None else np.zeros((H, W, 4), np.float32)
    dls["back"].render(back, atlas=atlas)
    bl = dlb.render(None, atlas=atlas)
    rim_light(bl, width=3.0 + 3.0 * sil, amount=min(1.0, 0.85 + 0.5 * sil))
    comp.over(back, bl)
    dls["front"].render(back, atlas=atlas)
    if mist_near is not None:
        comp.over(back, mist_near)
    wl = wisp_layer(f)
    if wl is not None:
        comp.over(back, wl)
    occ = balcony_alpha(f)
    if occ is not None:
        back *= (1.0 - occ)[..., None]
    front = None
    glow = trail_glow(f)
    if glow is None:
        glow = np.zeros((H, W, 3), np.float32)
    if mglow is not None:
        glow += mglow
    small = ndimage.gaussian_filter(bl[::4, ::4, 3], 14)
    halo = np.repeat(np.repeat(small, 4, 0), 4, 1)[:H, :W]
    glow += halo[..., None] * np.array([1.0, 0.76, 0.48], np.float32) * 0.14
    if occ is not None:
        glow *= (1.0 - occ)[..., None]
    return {"back": back, "front": front, "glow": glow}
