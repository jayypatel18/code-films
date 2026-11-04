"""Paper birds to draw batches: cel tones, translucency, page ink, pressure lines.

Tones per face follow the house cel style: one lit tone, one hard shadow tone
hue shifted toward the ambient (from engine.light.tones, the same colour
script the fox uses), a warm bounce on faces turned down, and a translucent
glow when the light is behind the paper (the hidden side is lit, so the page
glows and its ink shows through softer). Lines are the fox's warm umber,
thicker on the shadow side, thinner and sometimes broken on the lit side,
tapered at every lift, and they boil a fraction of a pixel per drawing.
"""

import math
import zlib

import numpy as np

from engine import light as elight
from . import bird as BD
from . import pages as P_

INK = elight.hexrgb("#3B2618")


class Look:
    """Scene light and line settings for one shot."""

    def __init__(self, L, name, light_world, haze=None, line=1.0, ink=1.0, trans=0.45,
                 thr=0.08, bounce=0.35, rim=None, crease=1.0):
        self.L, self.name = L, name
        lw = np.asarray(light_world, np.float64)
        self.lw = lw / np.linalg.norm(lw)
        self.haze = haze                 # (rgb, near_depth, far_depth, max_amount) or None
        self.line = line
        self.ink = ink
        self.trans = trans
        self.thr = thr
        self.bounce = bounce
        self.rim = rim
        self.crease = crease
        tab = np.zeros((len(BD.TINTS), 5, 3), np.float32)
        for i, alb in enumerate(BD.TINTS):
            t = elight.tones(alb, L, name, shade_depth=0.62)
            tab[i, 0] = t["lit"]
            tab[i, 1] = t["shade"]
            tab[i, 2] = t["warm_shade"] if L.get("warm") is not None else np.clip(t["shade"] * 1.12 + alb * 0.05, 0, 1)
            tab[i, 3] = np.clip(t["shade"] + (t["lit"] - t["shade"]) * trans + alb * np.asarray(L["key"]) * 0.10, 0, 1)
            tab[i, 4] = t["rim"]
        self.tones = tab


# chain styles per fold stage: weight per chain point segment, by chain name
def chain_weights(T, name, n, stage, st):
    """Per point weight (n,) for a chain at this drawing. stage 0..2 fold keys, 3 bird."""
    w = np.ones(n)
    K = T.K
    if name == "outline":
        # H0 H1 H2 H4 | Wf0..WfK | WbK..Wb0
        if stage == 3:
            w[:] = 1.0
        else:
            w[:] = [0.0, 0.25, 0.6][stage]
            w[0:4] = [0.0, 0.25, 0.65][stage]
            if stage < 2:
                # the wing tip edge is the page edge while the page is open
                w[4 + K] = 0.0
                w[4 + K + 1] = 0.0
        return w, (stage < 2)
    if name == "page":
        # N0 N1 Eo0 | Eo0..EoK | LoK WfK WbK RiK RoK | GoK..Go0 | F1 F2 T4 T5
        if stage < 3:
            w[:] = 1.0
            return w, False
        w[:] = 0.0
        w[0:2] = 0.75
        i = 3 + K + 1
        w[i] = 0.55            # LoK (folded strip end, seen from below)
        w[i + 3] = 0.55        # RiK
        w[i + 4] = 0.55        # RoK
        w[-4:] = [0.6, 0.6, 1.0, 1.0]
        return w, False
    if name == "tail":
        w[:] = [0.0, 0.3, 0.6, 1.0][stage]
        return w, stage < 2
    if name in ("lstrip", "rstrip"):
        w[:] = [0.0, 0.0, 0.35, 0.7][stage]
        return w, stage < 3
    if name in ("root", "neck"):
        w[:] = [0.0, 0.45, 0.6, 0.55][stage]
        return w, True
    if name == "flex":
        # the paper's bend across the wing, drawn as a soft crease that fades out
        w[:] = [0.0, 0.0, 0.3, 0.55][stage]
        return w, True
    if name == "spine":
        w[:] = [0.12, 0.6, 0.75, 0.6][stage]
        return w, True
    return w, True


def _densify(P, W, N):
    """Subdivide a chain (B, n, k) with point weights (n,) into exactly N points.

    Every segment gets one weight, min of its two end weights, and junction
    points are doubled, so a hidden part of the chain never leaves a tapered
    sliver reaching into it. Returns points (B, N, k) and weights (N,)."""
    B, n, k = P.shape
    segw = np.minimum(W[:-1], W[1:])
    vis = np.flatnonzero(segw > 0)
    if len(vis) == 0:
        return np.repeat(P[:, :1], N, 1), np.zeros(N)
    # points per visible segment, at least 2 (both ends)
    budget = N - 1
    reps = max(1, budget // max(1, len(vis)) - 1)
    pts, ws = [], []
    for i in vis:
        t = np.linspace(0, 1, reps + 1)
        pts.append(P[:, i:i + 1] * (1 - t)[None, :, None] + P[:, i + 1:i + 2] * t[None, :, None])
        ws.append(np.full(reps + 1, segw[i]))
        if i + 1 not in vis:
            # end of a visible run: a zero point stops the stroke
            pts.append(P[:, i + 1:i + 2])
            ws.append(np.zeros(1))
    Q = np.concatenate(pts, 1)
    Wd = np.concatenate(ws)
    m = Q.shape[1]
    if m < N:
        Q = np.concatenate([Q, np.repeat(Q[:, -1:], N - m, 1)], 1)
        Wd = np.concatenate([Wd, np.zeros(N - m)])
    elif m > N:
        idx = np.linspace(0, m - 1, N).round().astype(int)
        Q = Q[:, idx]
        Wd = Wd[idx]
    return Q, Wd


def _run_taper(Pd, Wd, land=0.16, lift=0.2, floor=0.12):
    """Taper every visible run at both ends, like a pen landing and lifting."""
    B, N = Pd.shape[:2]
    seg = np.linalg.norm(np.diff(Pd[..., :2], axis=1), axis=-1)
    sl = np.concatenate([np.zeros((B, 1)), np.cumsum(seg, 1)], 1)
    out = np.zeros((B, N))
    on = Wd > 0
    i = 0
    while i < N:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < N and on[j]:
            j += 1
        j2 = min(j, N - 1)
        s0 = sl[:, i:i + 1]
        s1 = sl[:, j2:j2 + 1]
        L = np.maximum(s1 - s0, 1e-6)
        u = (sl[:, i:j2 + 1] - s0) / L
        a = np.clip(u / land, 0, 1)
        b = np.clip((1 - u) / lift, 0, 1)
        t = np.minimum(np.sin(a * math.pi / 2) ** 0.8, np.sin(b * math.pi / 2) ** 0.9)
        out[:, i:j2 + 1] = np.maximum(out[:, i:j2 + 1], floor + (1 - floor) * t)
        i = j
    return out, sl


def bird_batch(dl, s, X, UV, st, look, cam, R, pos, scale, T=None, alpha=1.0, ss=3,
               line_px=None, drawing=0, N=40, order=0, depth_bias=None, extra_strokes=None,
               st_r=None, smear_from=None):
    """Add B birds to a raster.DrawList.

    X (B, 2, NV, 3) bird local verts (left, right halves); UV (B, 2, NV, 2).
    st: the state dict used to fold them (for alphas and crease weights).
    R (B, 3, 3), pos (B, 3), scale (B,): local to world.
    line_px: outline radius in px per bird (default from the projected size).
    drawing: drawing number (twos), for line boil and pressure noise.
    """
    T = T or BD.topo()
    B = s["B"]
    st_r = st_r or st
    bob = np.broadcast_to(np.asarray(st.get("bob", 0.0), np.float64), (B,))
    Xl = X.copy()
    Xl[..., 2] += bob[:, None, None]
    Xw = pos[:, None, None, :] + scale[:, None, None, None] * np.einsum("bij,bhnj->bhni", R, Xl)
    S = cam.project(Xw)
    # boil: a fraction of a pixel per drawing
    rb = np.random.default_rng((int(drawing) * 7919 + int(s["seed"][0])) & 0x7FFFFFFF)
    S[..., :2] += rb.normal(0, 0.18, S[..., :2].shape)
    tri = T.tris
    NT = T.NT
    # gather per half
    TX, TUV, TCOL, TK, TPAGE = [], [], [], [], []
    stage = st.get("stage", 3)
    for h in range(2):
        sth = st if h == 0 else st_r
        v = Xw[:, h][:, tri]            # (B, NT, 3, 3)
        sv = S[:, h][:, tri]
        uv = UV[:, h][:, tri]
        n = np.cross(v[:, :, 1] - v[:, :, 0], v[:, :, 2] - v[:, :, 0])
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        n = n + n[:, T.tpair]
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        n *= _side_sign(T, h)
        cen = v.mean(2)
        vd = cam.view_dir(cen)
        front = (n * vd).sum(-1) > 0
        nv = np.where(front[..., None], n, -n)
        ndl = (nv * look.lw).sum(-1)
        lit = ndl > look.thr
        back_lit = (-ndl) > 0.2
        down = -nv[..., 1] > look.bounce
        tone = np.where(lit, 0, np.where(back_lit, 3, np.where(down, 2, 1)))
        tab = look.tones[s["tint"]]                       # (B, 5, 3)
        col = np.take_along_axis(tab, tone[..., None].repeat(3, -1), axis=1)
        col = col + s["tintj"][:, None, :] * np.where(lit, 1.0, 0.6)[..., None]
        a = np.full((B, NT), 1.0, np.float32)
        g = T.tgrp[None, :]
        a = np.where(g == T.groups["ecorner"], sth.get("alpha_e", 0.0), a)
        a = np.where(g == T.groups["gcorner"], sth.get("alpha_g", 0.0), a)
        k = s["ink"][:, None] * look.ink * np.where(lit, 1.0, np.where(tone == 3, 0.62, 0.85))
        page = np.where(front, s["page"][:, None], s["page_back"][:, None])
        if look.haze is not None:
            hc, z0, z1, amax = look.haze
            hz = np.clip((sv[..., 2].mean(-1) - z0) / max(1e-6, z1 - z0), 0, 1) * amax
            col = col * (1 - hz[..., None]) + np.asarray(hc)[None, None] * hz[..., None]
            k = k * (1 - hz)
        TX.append(sv)
        TUV.append(uv)
        TCOL.append(np.concatenate([np.clip(col, 0, 1), a[..., None]], -1))
        TK.append(k)
        TPAGE.append(page)
    tri_xyz = np.concatenate(TX, 1)
    tri_uv = np.concatenate(TUV, 1)
    tri_col = np.concatenate(TCOL, 1).astype(np.float32)
    tri_col[..., 3] *= alpha if np.ndim(alpha) == 0 else np.asarray(alpha)[:, None]
    tri_k = np.concatenate(TK, 1)
    tri_page = np.concatenate(TPAGE, 1)
    # ---------------------------------------------------------- lines
    cen2 = S[..., :2].reshape(B, -1, 2).mean(1)
    ext = np.ptp(S[..., 0].reshape(B, -1), axis=1)
    # tiny far birds: lines thin out so hundreds of them never pile into an ink blot
    line_fade = np.clip((ext - 6.0) / 30.0, 0.15, 1.0)
    if line_px is None:
        line_px = np.clip(ext * 0.0052, 0.34, 1.9) * look.line
    line_px = np.broadcast_to(np.asarray(line_px, np.float64), (B,))
    ldir = np.asarray(look.L["key_dir"], np.float64)
    ldir = ldir / np.linalg.norm(ldir)
    pts_all, col_all = [], []
    tone_tab = look.tones[s["tint"]]
    shade_c = tone_tab[:, 1]
    ink_out = INK[None] * 0.88 + shade_c * 0.12
    ink_cr = INK[None] * 0.42 + shade_c * 0.58
    for h in range(2):
        for name, idx in T.chains:
            if name == "spine" and h == 1:
                continue
            idx = np.asarray(idx)
            w, is_crease = chain_weights(T, name, len(idx), stage, st)
            P = S[:, h][:, idx]
            Pd, Wd = _densify(P, w, N)
            taper, sl = _run_taper(Pd, Wd)
            u = sl / (sl[:, -1:] + 1e-6)
            # light side: outward normal vs light direction
            tg = np.gradient(Pd[..., :2], axis=1)
            nrm = np.stack([tg[..., 1], -tg[..., 0]], -1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
            out = Pd[..., :2] - cen2[:, None]
            nrm *= np.sign((nrm * out).sum(-1, keepdims=True) + 1e-9)
            ndl = (nrm * ldir).sum(-1)
            lw_ = 1.0 + 0.55 * np.clip(-ndl, 0, 1) - 0.45 * np.clip(ndl, 0, 1)
            pr = np.random.default_rng((int(drawing) * 31 + zlib.crc32(name.encode()) % 9973 + h * 17 + int(s["seed"][0])) & 0x7FFFFFFF)
            press = 0.82 + 0.3 * pr.random((B, 1)) + 0.1 * np.sin(u * pr.uniform(4, 9) + pr.uniform(0, 6, (B, 1)))
            rad = line_px[:, None] * Wd[None] * taper * lw_ * press
            if is_crease:
                rad *= 0.55 * look.crease
            pts = np.concatenate([Pd, rad[..., None]], -1)
            pts_all.append(pts)
            if is_crease:
                c = np.concatenate([ink_cr, np.full((B, 1), 0.85)], -1)
            else:
                c = np.concatenate([ink_out, np.ones((B, 1))], -1)
            if look.haze is not None:
                hc, z0, z1, amax = look.haze
                hz = np.clip((P[..., 2].mean(-1) - z0) / max(1e-6, z1 - z0), 0, 1) * amax
                c[:, :3] = c[:, :3] * (1 - hz[:, None]) + np.asarray(hc)[None] * hz[:, None]
            c[:, 3] *= alpha if np.ndim(alpha) == 0 else np.asarray(alpha)
            c[:, 3] *= line_fade
            col_all.append(c)
    if extra_strokes is not None:
        ep, ec = extra_strokes
        pts_all.extend(ep)
        col_all.extend(ec)
    str_pts = np.stack(pts_all, 1).astype(np.float32)
    str_col = np.stack(col_all, 1).astype(np.float32)
    depth = S[..., 2].reshape(B, -1).mean(1)
    zext = np.ptp(S[..., 2].reshape(B, -1), axis=1)
    # lines sit on their own faces; the folded strips lie 0.02 page units under
    # the wing, so the bias must stay well below that
    bias = (np.asarray(scale, np.float64) * 0.006 + 1e-5) if depth_bias is None else depth_bias
    dl.add(tri_xyz=tri_xyz.astype(np.float32), tri_uv=tri_uv.astype(np.float32), tri_col=tri_col,
           tri_k=tri_k.astype(np.float32), tri_page=tri_page.astype(np.int64),
           str_pts=str_pts, str_col=str_col, ss=ss, depth=depth, alpha=1.0, bias=bias, order=order)
    return S


_SIGN = {}


def _side_sign(T, h):
    """Sign that makes triangle normals point out of the printed side of the page."""
    key = (id(T), h)
    if key not in _SIGN:
        s = BD.specs(1, "sign")
        V2 = BD.flat(s, T)
        X = BD.to_local(np.concatenate([V2, np.zeros(V2.shape[:2] + (1,))], -1), +1 if h == 0 else -1)
        v = X[0][T.tris]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        _SIGN[key] = 1.0 if np.median(n[:, 2]) > 0 else -1.0
    return _SIGN[key]
