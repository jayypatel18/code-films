"""Cut 8, the climax: the flock gathers into the great bird, one wingbeat, the arc.

The camera stands at the balcony looking up into the golden sky. The bird's
screen position comes from the timeline anchors (bird_form_centre, bird_arc,
plate coordinates through timeline.camera), its depth from a key list, so it
swells as it swings close in the turn and shrinks as it leaves.

Layers: 'back' (mist, loose pages and small birds behind the bird, the bird
itself), 'front' (pages and birds nearer than the bird), 'glow' (the trail of
light and the warm bloom around the bird, additive).
"""

import math

import numpy as np
from scipy import ndimage

import timeline as TL
from engine import raster as eraster
from engine.noise import fbm
from . import bird as BD
from . import drawbird as DB
from . import great as G
from . import pages as P_
from . import raster

W, H = 1920, 1080
FOCAL = 1350.0
CAM = BD.Camera(pos=(0, 0, 0), look=(0, 0.16, -1), focal=FOCAL)
SUN_WORLD = (0.85, 0.22, -0.28)
HAZE = (0.99, 0.84, 0.70)
SCALE = 17.0
F0, F1 = TL.cut_range("8")
ARC_Z = [70.0, 88.0, 82.0, 125.0, 230.0]


def look():
    return G.GreatLook(TL.LIGHTS["sunrise"], "sunrise", sun_world=SUN_WORLD, haze_col=HAZE,
                       haze_z=(60, 240, 0.55))


def small_look():
    L = TL.LIGHTS["sunrise"]
    return DB.Look(L, "sunrise", light_world=SUN_WORLD, haze=(HAZE, 40, 260, 0.6), line=0.9, trans=0.55)


def unproject(sx, sy, Z):
    d = CAM.R.T @ np.array([(sx - CAM.cx) / FOCAL, (sy - CAM.cy) / FOCAL, 1.0])
    return CAM.pos + d * (Z / (CAM.R[2] @ d))


def _catmull(P, n=60):
    P = np.asarray(P, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, seg = [], []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            seg.append(i - 1 + t)
    out.append(P[-2])
    seg.append(len(P) - 3)
    return np.array(out), np.array(seg)


_ARC = None


def arc_table():
    global _ARC
    if _ARC is None:
        pts, seg = _catmull(TL.ANCHORS["8"]["bird_arc"])
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        _ARC = (pts, seg, d / d[-1])
    return _ARC


def body_state(t):
    """World position, rotation and scale of the great bird at time t."""
    a = TL.ANCHORS["8"]
    f = t * TL.FPS
    if t < TL.CUES["arc_start"]:
        px, py = a["bird_form_centre"]
        # it rises a little on the beat, then drifts toward the arc
        lift = 30 * G.smooth(46.5, 47.3, t) - 12 * G.smooth(44.0, 46.3, t)
        drift = 60 * G.smooth(47.0, 48.0, t)
        sx, sy = TL.plate_to_screen("8", f, px + drift, py - lift)
        Z = ARC_Z[0] + 4.0 * (1 - G.smooth(44.0, 46.3, t))
        # the anchor marks the bird's visual centre (wings and neck rise above the body)
        pos = unproject(sx - 40, sy + 120, Z)
        # right above us, belly toward the camera, head up and right toward the sun,
        # wings spread across the sky; after the beat it rolls level into the arc
        # formed: three quarters, wings raised; it rears round to face us for the
        # beat (the stroke then sweeps down across the screen), then banks away
        k1 = float(G.smooth(46.25, 46.52, t))
        k2 = float(G.smooth(47.1, 48.0, t))
        fwd = np.array([0.90, 0.30, 0.30]) * (1 - k1) + np.array([0.42, 0.42, 0.80]) * k1
        up = np.array([0.0, 1.0, 0.70]) * (1 - k1) + np.array([0.0, 1.0, -0.35]) * k1
        fwd = fwd * (1 - k2) + np.array([0.95, 0.22, -0.2]) * k2
        up = up * (1 - k2) + np.array([0.0, 1.0, 0.2]) * k2
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
        return unproject(sx - 20, sy + 110, Z), (sx, sy)

    pos, sxy = at(u)
    p1, s1 = at(u + 0.01)
    p0, s0 = at(u - 0.01)
    fwd = p1 - p0
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([1.0, 0, 0])
    # present the side to the camera: keep most of the heading in the screen plane
    fwd = fwd / np.linalg.norm(fwd)
    fwd[2] *= 0.3
    # bank into the turn: screen curvature
    v0 = np.array(sxy) - np.array(s0)
    v1 = np.array(s1) - np.array(sxy)
    cross = v0[0] * v1[1] - v0[1] * v1[0]
    turn = cross / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-6)
    bank = float(np.clip(-turn * 9.0, -1.0, 1.0)) * math.radians(40)
    R = BD.look_rotation(fwd[None], bank=np.array([bank]))[0]
    # blend in from the pre arc pose over the first frames
    k = float(G.smooth(48.0, 48.5, t))
    if k < 1:
        pos_b, R_b, _, sxy_b = body_state(TL.CUES["arc_start"] - 1e-3)
        pos = pos_b * (1 - k) + pos * k
        R = _slerp_rot(R_b, R, k)
        sxy = (sxy_b[0] * (1 - k) + sxy[0] * k, sxy_b[1] * (1 - k) + sxy[1] * k)
    return pos, R, SCALE, sxy


def _slerp_rot(A, B, k):
    M = A * (1 - k) + B * k
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def key_points(t):
    """World positions of the tail base and both wing tips (for trails)."""
    pose = G.pose_at(t)
    pos, R, S, _ = body_state(t)
    wp = pose["wing"]
    tip = G.wing_map(np.array([G.le_x(G.YTIP) - 0.05]), np.array([G.YTIP]), np.array([0.0]), wp)[0]
    tl = tip + np.array([0, 0.13, 0.06])
    tr = tl * np.array([1, -1, 1])
    tail = np.array([-1.6, 0.0, 0.0])
    loc = np.stack([tail, tl, tr])
    return pos[None] + S * (loc @ R.T)


# ---------------------------------------------------------------- trail of light

def trail_glow(f, out=None):
    """The curving trail of light behind the bird (additive RGB, full frame)."""
    t = f / TL.FPS
    t_start = TL.CUES["arc_start"]
    if t <= t_start + 0.05:
        return out
    ts = np.arange(t_start, t, 1.0 / 48.0)
    if len(ts) < 3:
        return out
    pts = np.array([CAM.project(body_state(tt)[0][None])[0] for tt in ts])
    sc = 4
    h, w = H // sc, W // sc
    core = np.zeros((h, w), np.float32)
    halo = np.zeros((h, w), np.float32)
    age = (t - ts) / max(1e-3, t - t_start)
    fade = np.clip(1.0 - age, 0, 1) ** 1.2
    # thin at the old end, swelling toward the bird, which leaves it a little behind
    near = np.clip((t - ts) / 0.25, 0, 1)
    P = pts[:, :2] / sc
    eraster.stroke(core, P, (0.6 + 4.0 * fade) * near, accumulate=False)
    eraster.stroke(halo, P, (3.0 + 20.0 * fade) * near, accumulate=False)
    nz = fbm(h, w, 22, ("trail", f // 2), octaves=3)
    core *= 0.55 + 0.7 * nz
    halo *= 0.6 + 0.6 * nz
    core = ndimage.gaussian_filter(core, 1.6)
    halo = ndimage.gaussian_filter(halo, 10.0)
    g = core[..., None] * np.array([1.0, 0.82, 0.55], np.float32) * 0.32 + \
        halo[..., None] * np.array([1.0, 0.70, 0.40], np.float32) * 0.22
    up = np.repeat(np.repeat(g, sc, 0), sc, 1)[:H, :W]
    up = ndimage.uniform_filter(up, size=(sc, sc, 1))
    if out is None:
        return up.astype(np.float32)
    out += up
    return out


def mist_layer(f):
    """Soft vapour along the trail and blasted from the wingbeat (premultiplied RGBA)."""
    t = f / TL.FPS
    sc = 2
    h, w = H // sc, W // sc
    cov = np.zeros((h, w), np.float32)
    # wingbeat blast: a ring of mist pushed down and out from under the wings
    b = float(G.smooth(46.52, 46.72, t)) * float(1 - G.smooth(47.5, 48.8, t))
    if b > 0:
        pos, R, S, sxy = body_state(t)
        rr = 260 + 700 * G.smooth(46.52, 47.6, t)
        cx, cy = (sxy[0] - 40) / sc, (sxy[1] + 260 + 140 * G.smooth(46.52, 47.6, t)) / sc
        th = np.linspace(math.pi * 0.05, math.pi * 0.95, 40)
        ring = np.stack([cx + np.cos(th) * rr / sc, cy + np.sin(th) * rr * 0.35 / sc], -1)
        eraster.stroke(cov, ring, np.full(40, 34.0 / sc) * (0.3 + 0.7 * np.sin(np.linspace(0, math.pi, 40))), accumulate=True)
        # the downdraft under the wings: a broad sheet of vapour pushed down
        sheet = np.stack([np.linspace(cx - rr * 0.8 / sc, cx + rr * 0.8 / sc, 20), np.full(20, cy - 30 / sc)], -1)
        eraster.stroke(cov, sheet, np.full(20, 60.0 / sc) * np.sin(np.linspace(0, math.pi, 20)), accumulate=True)
        kp = CAM.project(key_points(t))
        grow = G.smooth(46.52, 47.4, t)
        for j in (1, 2):
            px, py = kp[j, 0] / sc, kp[j, 1] / sc
            out = np.sign(kp[j, 0] - sxy[0])
            arc = np.stack([px + out * np.linspace(0, 220, 16) * grow / sc,
                            py + np.linspace(0, 160, 16) * grow / sc + np.sin(np.linspace(0, 3, 16)) * 20 / sc], -1)
            eraster.stroke(cov, arc, np.linspace(30, 70, 16) * (0.4 + grow) / sc, accumulate=True)
        cov *= b
    t_start = TL.CUES["arc_start"]
    if t > t_start + 0.05:
        ts = np.arange(t_start, t, 1.0 / 24.0)
        if len(ts) >= 3:
            pts = np.array([CAM.project(body_state(tt)[0][None])[0] for tt in ts])
            age = (t - ts) / max(1e-3, t - t_start)
            rad = (10 + 60 * age) / sc
            P = pts[:, :2] / sc
            P = P + np.stack([np.zeros_like(age), age * 50 / sc], -1)   # vapour sinks as it ages
            eraster.stroke(cov, P, rad * (0.35 + 0.65 * np.clip(1 - age * 0.6, 0, 1)), accumulate=True)
    if cov.max() <= 0:
        return None
    nz = fbm(h, w, 40, ("mist", f // 3), octaves=4)
    cov = ndimage.gaussian_filter(cov, 7.0) * (0.45 + 0.9 * nz)
    a = np.clip(cov * 0.5, 0, 0.62)
    col = np.array([1.0, 0.93, 0.86], np.float32)
    up = np.repeat(np.repeat(a, sc, 0), sc, 1)[:H, :W]
    lay = np.zeros((H, W, 4), np.float32)
    lay[..., :3] = up[..., None] * col
    lay[..., 3] = up
    return lay


# ---------------------------------------------------------------- loose pages

def loose_pages(dl, f, look_g, bird_depth, depth_split):
    """Pages shed along the trail and by the wingbeat, tumbling as they fall."""
    t = f / TL.FPS
    r = P_.rng("loose8")
    N = 420
    birth = np.concatenate([r.uniform(46.5, 46.9, 140), r.uniform(47.6, 52.0, N - 140)])
    src_kind = np.concatenate([np.zeros(140), np.ones(N - 140)])
    alive = birth <= t
    if not alive.any():
        return 0
    idx = np.flatnonzero(alive)
    n = len(idx)
    page = r.integers(1, P_.NPAGES, N)[idx]
    tint = r.choice(len(BD.TINTS), N, p=BD.TINT_P)[idx]
    spin = r.normal(0, 1, (N, 3))[idx]
    phase = r.uniform(0, 6.28, N)[idx]
    side = r.uniform(-1, 1, N)[idx]
    sz = r.uniform(0.05, 0.11, N)[idx]
    age = t - birth[idx]
    pos = np.zeros((n, 3))
    for i in range(n):
        tb = birth[idx[i]]
        kp = key_points(tb)
        bpos, R, S, _ = body_state(tb)
        if src_kind[idx[i]] == 0:
            # blasted from under the wings, outward and down
            src = kp[1] if side[i] > 0 else kp[2]
            src = bpos + (src - bpos) * abs(side[i])
            v = (src - bpos) * 0.9 + np.array([0, -28.0, 0])
        else:
            src = kp[0] * 0.6 + (kp[1] if side[i] > 0 else kp[2]) * 0.4
            v = np.array([0, -6.0, 0]) + r.normal(0, 3, 3) * 0 + np.array([side[i] * 4, 0, 0])
        a = age[i]
        drag = (1 - np.exp(-a * 1.5)) / 1.5
        pos[i] = src + v * drag + np.array([math.sin(phase[i] + a * 2.2) * 2.5, -3.0 * a, 0])
    ang = spin * age[:, None] * 2.6 + phase[:, None]
    Rs = BD._ry(ang[:, 1]) @ BD._rx(ang[:, 0])
    # a page: 6 vertices, bent
    L = sz * SCALE
    wdt = L / 1.38
    loc = np.zeros((n, 6, 3))
    bend = 0.18 * np.sin(phase + age * 5.0)
    for k, (u, v) in enumerate([(0, -0.5), (0, 0.5), (0.5, -0.5), (0.5, 0.5), (1, -0.5), (1, 0.5)]):
        loc[:, k, 0] = (u - 0.5) * L
        loc[:, k, 1] = v * wdt
        loc[:, k, 2] = bend * L * (u - 0.5) ** 2 * 4 - bend * L * 0.5
    Xw = pos[:, None] + np.einsum("nij,nkj->nki", Rs, loc)
    S_ = CAM.project(Xw)
    nrm = np.cross(Xw[:, 1] - Xw[:, 0], Xw[:, 4] - Xw[:, 0])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    vd = CAM.view_dir(Xw.mean(1))
    front = (nrm * vd).sum(-1) > 0
    nv = np.where(front[:, None], nrm, -nrm)
    ndl = (nv * look_g.sun).sum(-1)
    tone = np.where(ndl > 0.05, 0, np.where(-ndl > 0.15, 2, 1))
    col = look_g.tones[tint, tone]
    z = S_[..., 2].mean(1)
    z0, z1, amax = look_g.haze_z
    hz = np.clip((z - z0) / (z1 - z0), 0, 1) * amax
    col = col * (1 - hz[:, None]) + look_g.haze_col * hz[:, None]
    tv = S_[:, G.TRIS6]
    tcol = np.zeros((n, 4, 4), np.float32)
    tcol[..., :3] = col[:, None]
    tcol[..., 3] = 1.0
    PW, PH = P_.PW - 1, P_.PH - 1
    uv6 = np.array([[0, 0], [PW, 0], [0, PH / 2], [PW, PH / 2], [0, PH], [PW, PH]], np.float64)
    tuv = np.broadcast_to(uv6[G.TRIS6].reshape(4, 6), (n, 4, 6))
    tk = np.full((n, 4), 0.8) * (1 - hz[:, None])
    tpage = np.repeat(page[:, None], 4, 1)
    loop = S_[:, G.LOOP6]
    Nl = 15
    tt = np.linspace(0, 6, Nl)
    i0 = np.floor(tt).astype(int).clip(0, 5)
    fr = (tt - i0)[None, :, None]
    P = loop[:, i0] * (1 - fr) + loop[:, i0 + 1] * fr
    ext = np.ptp(S_[..., 0], axis=1)
    rad = np.clip(ext * 0.012, 0.3, 0.9)[:, None] * np.ones(Nl)[None] * (1 - 0.6 * hz[:, None])
    pts = np.concatenate([P, rad[..., None]], -1)
    lc = DB.INK[None] * 0.5 + col * 0.5 * 0.55
    scol = np.concatenate([lc, np.full((n, 1), 0.9)], -1)
    front_m = z < depth_split
    for which, m in (("back", ~front_m), ("front", front_m)):
        if not m.any():
            continue
        dl[which].add(tri_xyz=tv[m].astype(np.float32), tri_uv=np.ascontiguousarray(tuv[m]).astype(np.float32),
                      tri_col=tcol[m], tri_k=tk[m].astype(np.float32), tri_page=tpage[m].astype(np.int64),
                      str_pts=pts[m][:, None].astype(np.float32), str_col=scol[m][:, None].astype(np.float32),
                      ss=2, depth=z[m], bias=0.01)
    return n


# ---------------------------------------------------------------- the gathering flock

def flock(dl, f, look_s, depth_split, T=None):
    """Small paper birds streaming in to become the great bird (and a few trailing it)."""
    t = f / TL.FPS
    T = T or BD.topo(4)
    N = 520
    s = BD.specs(N, "flock8")
    r = P_.rng("flock8-path")
    arrive = r.uniform(45.0, 46.6, N)
    trailing = r.random(N) < 0.12
    arrive = np.where(trailing, 99.0, arrive)
    vis = (t < arrive)
    if not vis.any():
        return 0
    idx = np.flatnonzero(vis)
    s = BD.take(s, idx)
    n = len(idx)
    # start far out around the frame (and from the tower below left), end on the bird
    ang = r.uniform(0, 2 * math.pi, N)[idx]
    start_r = r.uniform(900, 1600, N)[idx]
    zst = r.uniform(30, 140, N)[idx]
    tgt_off = r.normal(0, 1, (N, 3))[idx] * np.array([1.4, 0.6, 0.5])
    pos, R, S, sxy = body_state(min(t, 47.9))
    tgt = pos[None] + S * (tgt_off @ R.T)
    sx0 = sxy[0] + np.cos(ang) * start_r
    sy0 = sxy[1] + np.sin(ang) * start_r * 0.6 + 250
    p0 = np.array([unproject(sx0[i], sy0[i], zst[i]) for i in range(n)])
    t_go = arrive[idx] - r.uniform(1.4, 2.4, N)[idx]
    u = np.clip((t - t_go) / np.maximum(1e-3, arrive[idx] - t_go), 0, 1)
    u = u * u * (3 - 2 * u)
    tr = trailing[idx]
    # trailing birds follow behind the great bird during the arc
    trail_pos = np.array([key_points(max(44.0, t - 0.25 - 0.6 * k))[0] for k in r.random(N)[idx]])
    P = p0 + (tgt - p0) * u[:, None]
    # arc in: curve upward a little on the way
    P[:, 1] += np.sin(u * math.pi) * r.uniform(2, 10, N)[idx]
    P = np.where(tr[:, None], trail_pos + r.normal(0, 4, (n, 3)), P)
    # heading: toward the target
    fwd = np.where(tr[:, None], np.broadcast_to(R[:, 0], (n, 3)), tgt - p0)
    Rb = BD.look_rotation(fwd, bank=r.normal(0, 0.3, N)[idx])
    # flap on twos with per bird phase; shrink into a page as they arrive
    drawing = f // 2
    ph = r.integers(0, 6, N)[idx]
    key_i = (drawing + ph) % 6
    sc = np.where(tr, 1.0, 1.0 - 0.5 * u ** 4) * r.uniform(1.4, 2.4, N)[idx]
    z = np.array([CAM.project(P[i:i + 1])[0, 2] for i in range(n)])
    count = 0
    for k in range(6):
        m = key_i == k
        if not m.any():
            continue
        sk = BD.take(s, np.flatnonzero(m))
        st = BD.bird_state(sk, T, BD.FLAP_KEYS[k])
        X, UV = BD.build(sk, st, st, T)
        for which, mm in (("back", z[m] >= depth_split), ("front", z[m] < depth_split)):
            if not mm.any():
                continue
            sel = np.flatnonzero(mm)
            s2 = BD.take(sk, sel)
            DB.bird_batch(dl[which], s2, X[sel], UV[sel], st, look_s, CAM, Rb[m][sel], P[m][sel], sc[m][sel],
                          T=T, drawing=drawing + k, ss=2, N=24)
            count += len(sel)
    return count


def speed_lines(dlb, t, R, pos, S, n_per=9, N=24):
    """Dry brush speed lines trailing the wing tips and trailing edges along the
    stroke, on the smear drawings only."""
    dt = 0.14
    p_now = G.pose_at(t)["wing"]
    p_then = G.pose_at(t - dt)["wing"]
    ys = np.linspace(0.7, G.YTIP, n_per)
    xs = G.le_x(ys) - np.linspace(0.45, 0.12, n_per)
    pts_all, cols = [], []
    pos0, R0, S0, _ = body_state(t - dt)
    for side in (1, -1):
        A = G.wing_map(xs, ys, np.zeros(n_per), p_then) + np.array([0, 0.13, 0.06])
        B = G.wing_map(xs, ys, np.zeros(n_per), p_now) + np.array([0, 0.13, 0.06])
        if side < 0:
            A = A * np.array([1, -1, 1])
            B = B * np.array([1, -1, 1])
        for i in range(n_per):
            u = np.linspace(0, 1, N)[:, None]
            # interpolate the wing between the two drawings so the line arcs with the stroke
            pw = G.wing_map(np.array([xs[i]]), np.array([ys[i]]), np.array([0.0]), p_then)[0]
            Aw = pos0 + S0 * (R0 @ A[i])
            Bw = pos + S * (R @ B[i])
            mid = (Aw + Bw) / 2 + (Bw - Aw) * 0 + (pos - Aw) * 0.08
            P = (1 - u) ** 2 * Aw + 2 * (1 - u) * u * mid + u ** 2 * Bw
            Sx = CAM.project(P)
            rad = (1.0 + 2.2 * (i / n_per)) * np.sin(u[:, 0] * math.pi) ** 0.6 * (0.3 + 0.7 * u[:, 0])
            pts = np.concatenate([Sx[:, :2], np.full((N, 1), 1e5), rad[:, None]], -1)
            pts_all.append(pts)
            cols.append([1.0, 0.95, 0.85, 0.85])
    dlb.add(str_pts=np.array(pts_all, np.float32)[None], str_col=np.array(cols, np.float32)[None], ss=2,
            depth=1e5)


def rim_light(bl, width=3.0, col=(1.0, 0.90, 0.64), amount=0.9):
    """Golden rim on the sunward silhouette of the bird layer (in place): where the
    neighbour toward the sun is sky, the edge catches the light."""
    a = bl[..., 3]
    d = np.asarray(TL.LIGHTS["sunrise"]["key_dir"], np.float64)
    d = d / np.linalg.norm(d)
    sh = ndimage.shift(a, (-d[1] * width, -d[0] * width), order=1, mode="constant")
    rim = np.clip(a - sh, 0, 1)
    rim = ndimage.gaussian_filter(rim, 0.6) * amount
    c = np.asarray(col, np.float32)
    bl[..., :3] = bl[..., :3] * (1 - rim[..., None]) + c * rim[..., None] * a[..., None]
    return bl


# ---------------------------------------------------------------- the cut

def render(f, with_flock=True):
    t = f / TL.FPS
    lg = look()
    ls = small_look()
    pose = G.pose_at(t)
    pos, R, S, sxy = body_state(t)
    z_bird = CAM.project(pos[None])[0, 2]
    dl = {"back": raster.DrawList(), "front": raster.DrawList(), "bird": raster.DrawList()}
    smear_dir = None
    smear_amt = 0.0
    if pose["smear"]:
        # the wings sweep down: smear back up along the stroke (world up of the bird)
        smear_dir = -(R @ np.array([0.0, 0.0, 1.0]))
        smear_amt = 0.5 * math.sin(math.pi * (t - 46.55) / 0.15)
    dz = G.design()
    eyes = G.eye_strokes(dz, pose, CAM, R, pos, S, 0.9)
    G.bird_draw(dl["bird"], dz, pose, lg, CAM, R, pos, S, drawing=f // 2, smear_dir=smear_dir,
                smear_amt=smear_amt, extra=eyes)
    if pose["smear"]:
        speed_lines(dl["back"], t, R, pos, S)
    loose_pages(dl, f, lg, z_bird, z_bird - 4)
    if with_flock:
        flock(dl, f, ls, z_bird)
    atlas = P_.atlas()
    back = np.zeros((H, W, 4), np.float32)
    mist = mist_layer(f)
    if mist is not None:
        back = mist
    dl["back"].render(back, atlas=atlas)
    bl = dl["bird"].render(None, atlas=atlas)
    rim_light(bl)
    from engine import comp
    comp.over(back, bl)
    front = dl["front"].render(None, atlas=atlas)
    glow = trail_glow(f)
    # warm bloom around the bird: the sun through a hundred thousand pages
    if glow is None:
        glow = np.zeros((H, W, 3), np.float32)
    a = back[..., 3]
    small = ndimage.gaussian_filter(a[::4, ::4], 14)
    halo = np.repeat(np.repeat(small, 4, 0), 4, 1)[:H, :W]
    glow += halo[..., None] * np.array([1.0, 0.76, 0.48], np.float32) * 0.16
    return {"back": back, "front": front, "glow": glow}
