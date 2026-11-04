"""Review sheets for the paper birds (Stage 1): 20 unique birds, the fold
sequence from page to bird, and the flap cycle.

    python3 -m paper.sheets            (run from films/the-last-page)

Writes build/review/a4-stage1/birds-sheet.png and fold-flap-sheet.png.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FILM, "..", ".."))
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.noise import value_noise, paper_grain  # noqa: E402
from paper import pages, raster, bird as BD, drawbird as DB  # noqa: E402

OUT = os.path.join(FILM, "build", "review", "a4-stage1")
W, H = 1920, 1080
SHEET_LIGHT = dict(key=(1.0, 0.95, 0.86), key_dir=(-0.8, -0.6), ambient=(0.40, 0.44, 0.64), rim=None, warm=None)


def card(h=H, w=W, key="card", base=(0.27, 0.30, 0.38)):
    """A painted model sheet card: mottled slate with paper tooth."""
    rgb = np.empty((h, w, 3), np.float32)
    rgb[:] = base
    m = value_noise(h, w, 260, (key, "m")) - 0.5
    m2 = value_noise(h, w, 60, (key, "m2")) - 0.5
    g = paper_grain(h, w, (key, "g")) - 0.5
    rgb *= (1 + m[..., None] * 0.16 + m2[..., None] * 0.05 + g[..., None] * 0.06)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xs - w / 2) / (w * 0.62), (ys - h / 2) / (h * 0.62))
    rgb *= (1 - 0.25 * np.clip(d - 0.5, 0, 1))[..., None]
    return rgb


def pencil_lines(dl, segs, col=(0.62, 0.64, 0.70), width=0.45, alpha=0.35):
    """Faint pencil construction lines (list of (x0, y0, x1, y1))."""
    n = len(segs)
    if n == 0:
        return
    N = 12
    pts = np.zeros((1, n, N, 4), np.float32)
    rr = np.random.default_rng(11)
    for i, (x0, y0, x1, y1) in enumerate(segs):
        t = np.linspace(0, 1, N)
        pts[0, i, :, 0] = x0 + (x1 - x0) * t + rr.normal(0, 0.3, N)
        pts[0, i, :, 1] = y0 + (y1 - y0) * t + rr.normal(0, 0.3, N)
        pts[0, i, :, 2] = 1e6
        pts[0, i, :, 3] = width * (0.5 + 0.5 * np.sin(np.pi * t) ** 0.3)
    col4 = np.zeros((1, n, 4), np.float32)
    col4[0, :, :3] = col
    col4[0, :, 3] = alpha
    dl.add(str_pts=pts, str_col=col4, ss=2, depth=1e6, order=-1)


def sheet_look():
    return DB.Look(SHEET_LIGHT, "sunrise", light_world=(-0.88, 0.32, 0.35), line=1.15)


def place(cx, cy, cam_tilt=32.0, heading=0.0, pitch=0.0, bank=0.0, yaw_view=0.0):
    """Rotation for a bird drawn in a sheet cell, seen from above at cam_tilt degrees."""
    R = BD.orient(np.array([math.radians(heading)]), np.array([math.radians(pitch)]), np.array([math.radians(bank)]))
    ax = math.radians(cam_tilt)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ay = math.radians(yaw_view)
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    return Rx[None] @ Ry[None] @ R


def speed_lines(S_from, S_to, B, n=3, N=40, width=1.2, col=(0.86, 0.84, 0.78), alpha=0.75, seed=0):
    """Dry brush speed lines from screen points S_from to S_to (B, k, 3): curved,
    tapered, a little ragged. Returns (list of pts arrays, list of colour arrays)."""
    r = np.random.default_rng(seed)
    pts_l, col_l = [], []
    k = S_from.shape[1]
    for i in range(min(n, k)):
        a = S_from[:, i]
        b = S_to[:, i]
        t = np.linspace(0, 1, N)[None, :, None]
        mid = (a + b) / 2
        d = b - a
        nrm = np.stack([-d[:, 1], d[:, 0], np.zeros(B)], -1) * r.uniform(-0.15, 0.15)
        P = (1 - t) ** 2 * a[:, None] + 2 * (1 - t) * t * (mid + nrm)[:, None] + t ** 2 * b[:, None]
        rad = width * np.sin(np.linspace(0, 1, N) * math.pi) ** 0.7 * np.linspace(0.3, 1.0, N)
        pts = np.concatenate([P, np.broadcast_to(rad, (B, N))[..., None]], -1)
        pts[..., 2] = 1e5
        pts_l.append(pts)
        c = np.zeros((B, 4))
        c[:, :3] = col
        c[:, 3] = alpha * r.uniform(0.6, 1.0)
        col_l.append(c)
    return pts_l, col_l


def smear_mix(Xa, Xb, T, front_from_b=True):
    """Smear drawing: the leading side of the wing at drawing b, the trailing side
    still at drawing a, so the paper stretches across the stroke."""
    X = Xb.copy()
    back = np.array(T.Wb + T.Ri + T.Ro + T.Go)
    X[:, :, back] = 0.35 * Xb[:, :, back] + 0.65 * Xa[:, :, back]
    return X


def birds_sheet(path=None):
    atlas = pages.atlas()
    T = BD.topo()
    look = sheet_look()
    dl = raster.DrawList()
    s = BD.specs(20, "sheet20")
    s["page"] = np.random.default_rng(5).choice(np.arange(1, pages.NPAGES), 20, replace=False)
    poses = [BD.GLIDE, BD.FLAP_KEYS[0], BD.GLIDE, BD.FLAP_KEYS[5], BD.GLIDE]
    cols, rows = 5, 4
    cw, ch = W / cols, H / rows
    segs = []
    for c in range(1, cols):
        segs.append((c * cw, 14, c * cw, H - 14))
    for rr in range(1, rows):
        segs.append((14, rr * ch, W - 14, rr * ch))
    pencil_lines(dl, segs)
    rng = np.random.default_rng(3)
    for i in range(20):
        si = BD.take(s, [i])
        pose = poses[i % len(poses)]
        st = BD.bird_state(si, T, pose)
        X, UV = BD.build(si, st, st, T)
        r_, c_ = divmod(i, cols)
        cx, cy = (c_ + 0.5) * cw, (r_ + 0.5) * ch + 8
        face_left = (i // cols + i) % 2 == 1
        heading = (-rng.uniform(18, 40)) if not face_left else (180 + rng.uniform(18, 40))
        tilt = rng.uniform(46, 60) if pose is BD.GLIDE else rng.uniform(18, 30)
        R = place(cx, cy, cam_tilt=tilt, heading=heading, pitch=rng.uniform(-4, 10),
                  bank=rng.uniform(-8, 8) * (-1 if face_left else 1))
        cam = BD.Camera(pos=(0, 0, 50), look=(0, 0, 0), focal=118, cx=cx, cy=cy, ortho=True)
        DB.bird_batch(dl, si, X, UV, st, look, cam, R, np.zeros((1, 3)), np.array([1.55]), T=T, drawing=i)
    lay = dl.render(atlas=atlas)
    rgb = lay[..., :3] + card() * (1 - lay[..., 3:4])
    path = path or os.path.join(OUT, "birds-sheet.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
    return path


def fold_flap_sheet(path=None, key="foldbird", which=3):
    atlas = pages.atlas()
    T = BD.topo()
    look = sheet_look()
    dl = raster.DrawList()
    s = BD.take(BD.specs(8, key), [which])
    s["page"] = np.array([7])
    s["curl"] = np.array([0.7])
    pencil_lines(dl, [(14, H * 0.5, W - 14, H * 0.5)])
    # fold sequence: flat page, first crease, half fold, smear, bird
    states = [BD.fold_state(s, T, 0), BD.fold_state(s, T, 1), BD.fold_state(s, T, 2), None,
              BD.bird_state(s, T, BD.GLIDE)]
    cw = W / 5
    Xs = []
    for st in states:
        if st is None:
            Xs.append(None)
            continue
        X, UV = BD.build(s, st, st, T)
        Xs.append(X)
    for i, st in enumerate(states):
        cx, cy = (i + 0.5) * cw, H * 0.25
        cam = BD.Camera(pos=(0, 0, 50), look=(0, 0, 0), focal=150, cx=cx, cy=cy, ortho=True)
        R = place(cx, cy, cam_tilt=40, heading=-25, pitch=6, bank=-8)
        if st is None:
            # smear between half fold and bird
            st = dict(states[4])
            st["stage"] = 3
            X = smear_mix(Xs[2], Xs[4], T)
            Sa = cam.project(np.einsum("bij,bhnj->bhni", R, Xs[2] * 1.5))
            Sb = cam.project(np.einsum("bij,bhnj->bhni", R, Xs[4] * 1.5))
            tips = [T.Wf[-1], T.Wb[-1], T.Wf[T.K // 2]]
            ep, ec = [], []
            for h in range(2):
                p_, c_ = speed_lines(Sa[:, h, tips], Sb[:, h, tips], 1, n=3, seed=h)
                ep += p_
                ec += c_
            DB.bird_batch(dl, s, X, UV, st, look, cam, R, np.zeros((1, 3)), np.array([1.5]), T=T,
                          drawing=40 + i, extra_strokes=(ep, ec))
            continue
        DB.bird_batch(dl, s, Xs[i], UV, st, look, cam, R, np.zeros((1, 3)), np.array([1.5]), T=T, drawing=40 + i)
    # flap cycle: six drawings, on twos
    cw2 = W / 6
    Xprev = None
    keys = BD.FLAP_KEYS
    allX = []
    for k in keys:
        st = BD.bird_state(s, T, k)
        X, UV = BD.build(s, st, st, T)
        allX.append((st, X))
    for i, (st, X) in enumerate(allX):
        cx, cy = (i + 0.5) * cw2, H * 0.74
        cam = BD.Camera(pos=(0, 0, 50), look=(0, 0, 0), focal=125, cx=cx, cy=cy, ortho=True)
        R = place(cx, cy, cam_tilt=16, heading=-38, pitch=4, bank=0)
        extra = None
        Xd = X
        if st.get("smear"):
            Xa = allX[i - 1][1]
            Xd = smear_mix(Xa, X, T)
            Sa = cam.project(np.einsum("bij,bhnj->bhni", R, Xa * 1.5))
            Sb = cam.project(np.einsum("bij,bhnj->bhni", R, X * 1.5))
            tips = [T.Wf[-1], T.Wb[-1], T.Wf[T.K * 2 // 3]]
            ep, ec = [], []
            for h in range(2):
                p_, c_ = speed_lines(Sa[:, h, tips], Sb[:, h, tips], 1, n=3, seed=10 + h)
                ep += p_
                ec += c_
            extra = (ep, ec)
        bob = st.get("bob", 0.0)
        DB.bird_batch(dl, s, Xd, UV, st, look, cam, R, np.array([[0.0, bob * 1.5, 0.0]]), np.array([1.5]),
                      T=T, drawing=60 + i, extra_strokes=extra)
    lay = dl.render(atlas=atlas)
    rgb = lay[..., :3] + card(key="card2") * (1 - lay[..., 3:4])
    path = path or os.path.join(OUT, "fold-flap-sheet.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "birds"):
        print(birds_sheet())
    if what in ("all", "fold"):
        print(fold_flap_sheet())
