"""The paper bird: one torn page, folded.

Every bird is its own page. The page lies in page space (p across the page
width, front edge p = -0.5, back edge p = +0.5; q along the page height,
-Q..Q, which becomes the wingspan; z up off the printed side). Fold lines cut
the half page (q >= 0) into regions with a fixed topology:

    H  head strip      (a crane neck: two hinges, the nose corner NF folded under)
    B  body strip      (the spine ridge runs along q = 0)
    T  tail            (its outer back corner TF folds under; the halves splay)
    W  wing            (between the curved leading fold and the trailing fold,
                        cut into spanwise bands so the paper bends smoothly)
    L  leading strip   (folds under the wing along the leading edge)
    R  trailing strip  (folds under the wing along the trailing edge)
    E  front corner    (rides on L; tucked away in the finished bird)
    G  back corner     (rides on R; tucked away in the finished bird)

Folding is done like paper: rotations about the fold lines in order (each
strip vertex about the local tangent of its fold), then the wing bends about
spanwise hinges (the flap and the flex), then the spine folds the two halves
into a ridge. Because the page is one sheet, its handwriting runs across the
whole bird and the fold sequence from flat page to bird is real folding.

All arrays are vectorized over B birds: positions (B, NV, 3).
"""

import math

import numpy as np

from . import pages as P_

PW, PH = P_.PW, P_.PH
P0, P1 = -0.5, 0.5


# ---------------------------------------------------------------- topology

class Topo:
    """Vertex indices, triangles and ink chains for one half with K wing bands."""

    def __init__(self, K=8):
        self.K = K
        names = []

        def add(prefix, n):
            start = len(names)
            names.extend([f"{prefix}{i}" for i in range(n)])
            return list(range(start, start + n))

        self.H = add("H", 6)
        self.N = add("N", 3)
        self.Bd = add("B", 4)
        self.NR = 6                       # torn binding edge points on the tail
        self.T = add("T", 6 + self.NR)    # T0..T4, torn points, then the spine end (P1, 0)
        self.F = add("F", 3)
        self.Wf = add("Wf", K + 1)
        self.Wb = add("Wb", K + 1)
        self.Lo = add("Lo", K + 1)
        self.Ri = add("Ri", K + 1)
        self.Ro = add("Ro", K + 1)
        self.Eo = add("Eo", K + 1)
        self.Go = add("Go", K + 1)
        self.NV = len(names)
        tris, grp = [], []
        G_HEAD, G_NOSE, G_BODY, G_TAIL, G_TFLAP, G_WING, G_L, G_R, G_E, G_G = range(10)
        self.groups = dict(head=G_HEAD, nose=G_NOSE, body=G_BODY, tail=G_TAIL, tflap=G_TFLAP,
                           wing=G_WING, lstrip=G_L, rstrip=G_R, ecorner=G_E, gcorner=G_G)

        def fan(idx, g):
            for i in range(1, len(idx) - 1):
                tris.append((idx[0], idx[i], idx[i + 1]))
                grp.append(g)

        pairs = []

        def quad_strip(a, b, g):
            for k in range(len(a) - 1):
                n0 = len(tris)
                tris.append((a[k], b[k], b[k + 1]))
                tris.append((a[k], b[k + 1], a[k + 1]))
                grp.extend([g, g])
                pairs.append((n0, n0 + 1))

        H = self.H
        tris += [(H[0], H[1], H[2]), (H[0], H[2], H[3]), (H[3], H[2], H[4]), (H[3], H[4], H[5])]
        grp += [G_HEAD] * 4
        fan(self.N, G_NOSE)
        fan(self.Bd, G_BODY)
        fan(self.T, G_TAIL)
        fan(self.F, G_TFLAP)
        quad_strip(self.Wf, self.Wb, G_WING)
        quad_strip(self.Wf, self.Lo, G_L)      # leading strip shares the fold vertices
        quad_strip(self.Ri, self.Ro, G_R)
        quad_strip(self.Lo, self.Eo, G_E)
        quad_strip(self.Ro, self.Go, G_G)
        self.tris = np.array(tris, np.int64)
        self.tgrp = np.array(grp, np.int64)
        self.NT = len(tris)
        # the other triangle of the same quad (itself for fans): quads take one tone
        self.tpair = np.arange(self.NT)
        for a_, b_ in pairs:
            self.tpair[a_] = b_
            self.tpair[b_] = a_
        # Leading strip triangles reference wing front vertices Wf (the fold line).
        # Those must not move with the strip, which is true: they lie on the fold.
        vg = np.zeros(self.NV, np.int64)
        for idx, g in ((self.H, G_HEAD), (self.N, G_NOSE), (self.Bd, G_BODY), (self.T, G_TAIL),
                       (self.F, G_TFLAP), (self.Wf + self.Wb, G_WING), (self.Lo, G_L),
                       (self.Ri + self.Ro, G_R), (self.Eo, G_E), (self.Go, G_G)):
            vg[idx] = g
        self.vgrp = vg
        # ink chains: (name, indices)
        self.chains = [
            ("outline", [H[0], H[1], H[2], H[4]] + self.Wf + self.Wb[::-1]),
            ("page", [self.N[0], self.N[1], self.Eo[0]] + self.Eo + [self.Lo[-1], self.Wf[-1], self.Wb[-1],
                      self.Ri[-1], self.Ro[-1]] + self.Go[::-1] + [self.F[1], self.F[2]] + self.T[4:]),
            ("tail", [self.T[2], self.T[3], self.T[4]]),
            ("lstrip", self.Lo[:]),
            ("rstrip", self.Ro[:]),
            ("root", [self.Bd[1], self.Bd[2]]),
            ("neck", [H[3], H[2]]),
            ("flex", [self.Wf[K // 2], self.Wb[K // 2]]),
            ("spine", [H[0], H[3], H[5], self.Bd[3], self.T[0], self.T[-1]]),
        ]
        self.page_n = len(self.chains[1][1])


TOPO = {}


def topo(K=8):
    if K not in TOPO:
        TOPO[K] = Topo(K)
    return TOPO[K]


# ---------------------------------------------------------------- specs

TINTS = np.array([
    [0.965, 0.925, 0.830],   # cream
    [0.975, 0.955, 0.895],   # ivory
    [0.905, 0.805, 0.640],   # tea stained
    [0.935, 0.865, 0.705],   # foxed yellow
    [0.925, 0.850, 0.725],   # parchment
    [0.985, 0.975, 0.950],   # white rag
    [0.810, 0.845, 0.905],   # blue grey
    [0.955, 0.845, 0.825],   # rose
    [0.850, 0.875, 0.805],   # green grey
], np.float32)
TINT_P = np.array([0.24, 0.19, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.03])


def specs(B, key, page_pool=None):
    """Per bird design parameters: every bird a different page and a different fold."""
    r = P_.rng("birdspec", key)
    s = {}
    s["Q"] = r.uniform(0.66, 0.84, B)
    s["b"] = r.uniform(0.045, 0.08, B)
    s["t"] = s["b"] + r.uniform(0.03, 0.15, B)
    s["a0"] = r.uniform(-0.18, -0.07, B)                 # leading edge at the root
    s["c0"] = s["a0"] + r.uniform(0.17, 0.34, B)         # trailing edge at the root
    s["wrist"] = r.uniform(0.0, 0.10, B)                 # leading edge juts forward at the wrist
    s["wristu"] = r.uniform(0.25, 0.5, B)
    s["sweep"] = r.uniform(0.0, 0.30, B)                 # tip set back from the root
    s["tipc"] = r.uniform(0.03, 0.15, B)                 # chord at the tip
    s["tegam"] = r.uniform(0.6, 1.6, B)                  # trailing edge curve
    s["n"] = r.uniform(0.07, 0.18, B)                    # nose fold length
    s["k1"] = r.uniform(0.0, 0.24, B) * (r.random(B) < 0.75)
    s["k2"] = np.minimum(r.uniform(0.0, 1.0, B) * s["t"] * 0.9, s["t"] - 0.01) * (s["k1"] > 0)
    s["fl"] = r.uniform(0.5, 0.9, B)                     # leading strip width fraction
    s["fr"] = r.uniform(0.4, 0.85, B)
    s["spread"] = np.radians(r.uniform(0, 28, B)) * (r.random(B) < 0.7)
    s["head_up"] = np.radians(r.uniform(14, 42, B))
    s["head_bend"] = np.radians(r.uniform(10, 45, B)) * (r.random(B) < 0.8)
    s["tail_up"] = np.radians(r.uniform(-12, 22, B))
    s["ridge"] = np.radians(r.uniform(8, 26, B))
    s["curl"] = r.uniform(0.0, 1.0, B) * (r.random(B) < 0.6)
    s["curl_side"] = r.integers(0, 2, B)
    s["flex"] = r.uniform(0.7, 1.3, B)
    if page_pool is None:
        s["page"] = r.integers(1, P_.NPAGES, B)
    else:
        s["page"] = r.choice(np.asarray(page_pool), B)
    s["page_back"] = (s["page"] + r.integers(5, P_.NPAGES - 5, B)) % P_.NPAGES
    s["page_back"] = np.where(s["page_back"] == P_.BLANK, 1, s["page_back"])
    s["flipx"] = r.random(B) < 0.5
    s["flipy"] = r.random(B) < 0.5
    s["tint"] = r.choice(len(TINTS), B, p=TINT_P)
    s["tintj"] = r.normal(0, 0.018, (B, 3)).astype(np.float32)
    s["ink"] = r.uniform(0.55, 1.0, B)
    s["seed"] = r.integers(0, 2 ** 30, B)
    s["B"] = B
    return s


def hash01(seed, j):
    """Per bird random numbers in [0, 1) from integer seeds (B,) and salts j: a
    bird's shape never depends on which other birds share its batch."""
    h = (np.asarray(seed, np.uint64)[..., None] if np.ndim(j) else np.asarray(seed, np.uint64))
    h = (h * np.uint64(2654435761) + np.asarray(j, np.uint64) * np.uint64(40503)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(13)
    return h.astype(np.float64) / 4294967296.0


def take(s, idx):
    """Specs for a subset of birds."""
    idx = np.asarray(idx)
    out = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    out["B"] = len(idx)
    return out


# ---------------------------------------------------------------- flat page

def edges(s, q):
    """Leading and trailing fold positions (p) at span positions q (B, n)."""
    Q, b = s["Q"][:, None], s["b"][:, None]
    u = np.clip((q - b) / (Q - b), 0, 1)
    a0 = s["a0"][:, None]
    wr = s["wrist"][:, None]
    wu = s["wristu"][:, None]
    bump = np.where(u < wu, np.sin(0.5 * math.pi * u / wu), np.cos(0.5 * math.pi * (u - wu) / (1 - wu)) ** 1.5)
    front = a0 - wr * bump * (1 - u * 0.3) + s["sweep"][:, None] * u ** 1.7
    tipb = front[:, -1:] if front.shape[1] else front
    ftip = a0 + s["sweep"][:, None]
    back_tip = ftip + s["tipc"][:, None]
    c0 = s["c0"][:, None]
    back = c0 + (back_tip - c0) * u ** s["tegam"][:, None]
    back = np.maximum(back, front + 0.03)
    return front, back


def flat(s, T):
    """Flat page coordinates (B, NV, 2) of the half page q >= 0."""
    B = s["B"]
    K = T.K
    Q, b, t = s["Q"], s["b"], s["t"]
    a0, c0 = s["a0"], s["c0"]
    V = np.zeros((B, T.NV, 2))

    def put(i, p, q):
        V[:, i, 0] = p
        V[:, i, 1] = q

    qk = b[:, None] + (Q - b)[:, None] * np.linspace(0, 1, K + 1)[None]
    fr, bk = edges(s, qk)
    tet = edges(s, t[:, None])[1][:, 0]
    ph = P0 + s["n"] + (a0 - P0 - s["n"]) * 0.42
    put(T.H[0], P0, 0.0)
    put(T.H[1], P0 + s["n"], b)
    put(T.H[2], ph, b)
    put(T.H[3], ph, 0.0)
    put(T.H[4], a0, b)
    put(T.H[5], a0, 0.0)
    put(T.N[0], P0, 0.0)
    put(T.N[1], P0, b)
    put(T.N[2], P0 + s["n"], b)
    put(T.Bd[0], a0, 0.0)
    put(T.Bd[1], a0, b)
    put(T.Bd[2], c0, b)
    put(T.Bd[3], c0, 0.0)
    put(T.T[0], c0, 0.0)
    put(T.T[1], c0, b)
    put(T.T[2], tet, t)
    put(T.T[3], P1 - s["k1"], t)
    put(T.T[4], P1, t - s["k2"])
    # the binding edge: torn, never straight
    tear = hash01(s["seed"], np.arange(T.NR)[None] + 101) * 1.6
    for j in range(T.NR):
        u = (j + 1) / (T.NR + 1)
        put(T.T[5 + j], P1 - 0.006 - 0.016 * tear[:, j], (t - s["k2"]) * (1 - u))
    put(T.T[-1], P1, 0.0)
    put(T.F[0], P1 - s["k1"], t)
    put(T.F[1], P1, t)
    put(T.F[2], P1, t - s["k2"])
    chord = bk - fr
    wl = np.minimum(fr - P0 - 0.012, 0.85 * chord) * s["fl"][:, None]
    for k in range(K + 1):
        put(T.Wf[k], fr[:, k], qk[:, k])
        put(T.Wb[k], bk[:, k], qk[:, k])
        put(T.Lo[k], fr[:, k] - wl[:, k], qk[:, k])
        put(T.Eo[k], P0, qk[:, k])
    # trailing strip from q = t to Q
    sk = t[:, None] + (Q - t)[:, None] * np.linspace(0, 1, K + 1)[None]
    fr2, bk2 = edges(s, sk)
    wr = np.minimum(P1 - bk2 - 0.012, 0.85 * (bk2 - fr2)) * s["fr"][:, None]
    wr = np.maximum(wr, 0.004)
    for k in range(K + 1):
        put(T.Ri[k], bk2[:, k], sk[:, k])
        put(T.Ro[k], bk2[:, k] + wr[:, k], sk[:, k])
        put(T.Go[k], P1, sk[:, k])
    return V


def texcoords(s, T, V2, side):
    """Texel coordinates (B, NV, 2) for one half (side +1 left, -1 right)."""
    p = V2[..., 0]
    q = V2[..., 1] * side
    Q = s["Q"][:, None]
    tx = (p + 0.5) * (PW - 1)
    ty = (q + Q) / (2 * Q) * (PH - 1)
    tx = np.where(s["flipx"][:, None], PW - 1 - tx, tx)
    ty = np.where(s["flipy"][:, None], PH - 1 - ty, ty)
    return np.stack([tx, ty], -1)


# ---------------------------------------------------------------- folding

def _rodrigues(v, a, theta):
    """Rotate vectors v (..., 3) about unit axes a (..., 3) by theta (...)."""
    c = np.cos(theta)[..., None]
    sn = np.sin(theta)[..., None]
    cross = np.cross(a, v)
    dot = (v * a).sum(-1, keepdims=True)
    return v * c + cross * sn + a * dot * (1 - c)


def _fold_about(X, P, d, theta, flap_side):
    """Rotate points X (B, n, 3) about lines through P (B, n, 3) with in-plane
    directions d (B, n, 3) so the flap (on side flap_side (B, n, 3), in plane)
    swings down first."""
    cz = d[..., 0] * flap_side[..., 1] - d[..., 1] * flap_side[..., 0]
    a = d * np.where(cz < 0, 1.0, -1.0)[..., None]
    th = np.broadcast_to(theta[:, None], X.shape[:2])
    return P + _rodrigues(X - P, a, th)


def _tangents(V2, idx, theta=None):
    """Fold axes along a curved fold line: the local tangent at each level when
    the strip is tucked flat, blended toward the strip's chord for partial folds
    so a half folded strip turns as one piece instead of twisting."""
    P = V2[:, idx]
    d = np.gradient(P, axis=1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
    if theta is not None:
        ch = P[:, -1] - P[:, 0]
        ch /= np.linalg.norm(ch, axis=-1, keepdims=True) + 1e-12
        w = 1.0 - np.clip((theta - math.radians(150)) / math.radians(28), 0, 1)
        d = d * (1 - w[:, None, None]) + ch[:, None, :] * w[:, None, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
    return P, d


def _z0(a):
    return np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], -1)


def fold(s, T, V2, st, eps=0.02):
    """Fold the half page into 3D. st: dict of angle arrays (B,) or scalars:
    th_l, th_r (strip folds), th_n, th_f (corner folds), head, neck, tail,
    spread, ridge, bend (B, K) hinge angles (bend[:, 0] is the root flap), curl.
    Returns (B, NV, 3) in page space (p, q, z)."""
    B = s["B"]
    K = T.K
    g = lambda k, d=0.0: np.broadcast_to(np.asarray(st.get(k, d), np.float64), (B,)).astype(np.float64)
    X = _z0(V2)
    # leading strip (and the front corner riding on it): each level folds about
    # the local tangent of the leading edge at that level
    th = g("th_l")
    Pf, df = _tangents(V2, T.Wf, th)
    for grp_idx in (T.Lo, T.Eo):
        idx = np.array(grp_idx)
        side = _z0(V2[:, idx] - Pf)
        Xs = _fold_about(X[:, idx], _z0(Pf), _z0(df), th, side)
        Xs[..., 2] -= eps * np.sin(th / 2)[:, None] ** 2
        X[:, idx] = Xs
    th = g("th_r")
    Pr, dr = _tangents(V2, T.Ri, th)
    for grp_idx in (T.Ro, T.Go):
        idx = np.array(grp_idx)
        side = _z0(V2[:, idx] - Pr)
        Xs = _fold_about(X[:, idx], _z0(Pr), _z0(dr), th, side)
        Xs[..., 2] -= eps * 1.5 * np.sin(th / 2)[:, None] ** 2
        X[:, idx] = Xs
    # trailing strip inner edge sits on the trailing fold: tuck it a hair under too
    X[:, np.array(T.Ri), 2] -= eps * 0.5 * np.sin(th / 2)[:, None] ** 2
    # nose and tail corner flaps
    for idx_l, a_i, b_i, name in ((T.N, T.N[0], T.N[2], "th_n"), (T.F, T.F[0], T.F[2], "th_f")):
        idx = np.array(idx_l)
        P = _z0(V2[:, a_i])[:, None].repeat(len(idx), 1)
        d = V2[:, b_i] - V2[:, a_i]
        d = _z0(d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12))[:, None].repeat(len(idx), 1)
        side = _z0(V2[:, idx] - V2[:, a_i][:, None])
        th = g(name)
        Xs = _fold_about(X[:, idx], P, d, th, side)
        Xs[..., 2] -= eps * np.sin(th / 2)[:, None] ** 2
        X[:, idx] = Xs
    # head and tail strips take a shallow ridge while still flat (a ridge applied
    # after the tilt would push the raised halves apart)
    for idx_l, fac in ((T.H + T.N, 0.3), (T.T + T.F, 0.55)):
        idx = np.array(idx_l)
        ang = -g("ridge")[:, None] * fac
        q = X[:, idx, 1].copy()
        z = X[:, idx, 2].copy()
        X[:, idx, 1] = q * np.cos(ang) - z * np.sin(ang)
        X[:, idx, 2] = q * np.sin(ang) + z * np.cos(ang)
    # crane neck: the front of the head bends down at ph, then the whole head lifts at a0
    ph = V2[:, T.H[3], 0][:, None]
    hv = np.array([T.H[0], T.H[1]] + T.N)
    ang = -g("neck")[:, None]
    dp = X[:, hv, 0] - ph
    z = X[:, hv, 2]
    X[:, hv, 0] = ph + np.cos(ang) * dp + np.sin(ang) * z
    X[:, hv, 2] = -np.sin(ang) * dp + np.cos(ang) * z
    a0 = s["a0"][:, None]
    hv = np.array(T.H + T.N)
    ang = g("head")[:, None]
    dp = X[:, hv, 0] - a0
    z = X[:, hv, 2]
    X[:, hv, 0] = a0 + np.cos(ang) * dp + np.sin(ang) * z
    X[:, hv, 2] = -np.sin(ang) * dp + np.cos(ang) * z
    # tail splay about (c0, 0) in plane, then tilt about p = c0
    c0 = s["c0"][:, None]
    tv = np.array(T.T + T.F)
    sp = g("spread")[:, None]
    dp = X[:, tv, 0] - c0
    q = X[:, tv, 1]
    X[:, tv, 0] = c0 + dp * np.cos(sp) - q * np.sin(sp)
    X[:, tv, 1] = dp * np.sin(sp) + q * np.cos(sp)
    ang = g("tail")[:, None]
    dp = X[:, tv, 0] - c0
    z = X[:, tv, 2]
    X[:, tv, 0] = c0 + np.cos(ang) * dp - np.sin(ang) * z
    X[:, tv, 2] = np.sin(ang) * dp + np.cos(ang) * z
    # curl a wing tip corner up (paper memory), from where the paper is now
    wv = np.array(T.Wf + T.Wb + T.Lo + T.Ri + T.Ro + T.Eo + T.Go)
    curl = g("curl")
    if np.any(curl != 0):
        Q = s["Q"][:, None]
        front_tip = V2[:, T.Wf[-1], 0][:, None]
        back_tip = V2[:, T.Wb[-1], 0][:, None]
        cp = np.where(s["curl_side"][:, None] == 0, front_tip, back_tip)
        pq = X[:, wv, :2]
        dd = np.hypot((pq[..., 0] - cp) / 0.3, (pq[..., 1] - Q) / 0.28)
        w = np.clip(1 - dd, 0, 1) ** 2
        X[:, wv, 2] += curl[:, None] * 0.12 * w
    # wing bending: hinges parallel to the body axis at q = h_j
    bend = np.asarray(st.get("bend", np.zeros((B, K))), np.float64)
    if bend.ndim == 1:
        bend = np.broadcast_to(bend, (B, K))
    qf = X[:, wv, 1].copy()
    Q = s["Q"][:, None]
    b = s["b"][:, None]
    for j in range(K - 1, -1, -1):
        h = b + (Q - b) * j / K
        a = bend[:, j][:, None]
        m = (qf > h + 1e-9) if j > 0 else np.ones_like(qf, bool)
        dq = X[:, wv, 1] - h
        z = X[:, wv, 2]
        nq = h + dq * np.cos(a) - z * np.sin(a)
        nz = dq * np.sin(a) + z * np.cos(a)
        X[:, wv, 1] = np.where(m, nq, X[:, wv, 1])
        X[:, wv, 2] = np.where(m, nz, X[:, wv, 2])
    # spine ridge: the half rotates down about the body axis (the head and tail
    # strips are flatter, so the tilted neck does not split into two prongs)
    fac = np.ones(T.NV)
    fac[np.array(T.H + T.N + T.T + T.F)] = 0.0
    ang = -g("ridge")[:, None] * fac[None]
    q = X[..., 1].copy()
    z = X[..., 2].copy()
    X[..., 1] = q * np.cos(ang) - z * np.sin(ang)
    X[..., 2] = q * np.sin(ang) + z * np.cos(ang)
    return X


def to_local(X, side):
    """Page space (p, q, z) of one half to bird local (x forward, y left, z up)."""
    return np.stack([-X[..., 0], X[..., 1] * side, X[..., 2]], -1)


# ---------------------------------------------------------------- states

def wing_bend(T, B, root, flex, tip):
    """Hinge angles (B, K): root flap at the body, flex spread along the span,
    an extra flick in the outer bands."""
    K = T.K
    root = np.broadcast_to(np.asarray(root, np.float64), (B,))
    flex = np.broadcast_to(np.asarray(flex, np.float64), (B,))
    tip = np.broadcast_to(np.asarray(tip, np.float64), (B,))
    a = np.zeros((B, K))
    a[:, 0] = root
    if K > 1:
        w = np.linspace(0.6, 1.4, K - 1)
        w = w / w.sum()
        a[:, 1:] = flex[:, None] * w[None]
        if K > 2:
            a[:, -2] += tip * 0.35
            a[:, -1] += tip * 0.65
        else:
            a[:, -1] += tip
    return a


D = math.radians

# fold sequence key drawings (page to bird)
FOLD_KEYS = {
    0: dict(th_l=0, th_r=0, th_n=0, th_f=0, head=0, neck=0, tail=0, spread=0, ridge=D(-4),
            root=D(-3), flex=D(12), tip=D(8), alpha_e=1, alpha_g=1),
    1: dict(th_l=D(44), th_r=D(40), th_n=D(30), th_f=D(25), head=D(-6), neck=0, tail=D(-4),
            spread=0, ridge=D(28), root=D(6), flex=D(18), tip=D(8), alpha_e=1, alpha_g=1),
    2: dict(th_l=D(130), th_r=D(124), th_n=D(125), th_f=D(115), head=D(12), neck=D(8), tail=D(6),
            spread=D(5), ridge=D(50), root=D(24), flex=D(14), tip=D(10), alpha_e=1, alpha_g=1),
}

BIRD_FOLDS = dict(th_l=D(178), th_r=D(178), th_n=D(178), th_f=D(178))

# flap cycle key drawings (root, flex, tip in degrees; body bob; pitch)
FLAP_KEYS = [
    dict(name="up", root=48, flex=14, tip=-4, bob=-0.05, pitch=-4),
    dict(name="down_smear", root=6, flex=-8, tip=22, bob=0.0, pitch=0, smear=True),
    dict(name="down", root=-44, flex=-14, tip=22, bob=0.05, pitch=3),
    dict(name="low", root=-32, flex=-16, tip=-4, bob=0.06, pitch=4),
    dict(name="rise", root=-6, flex=-24, tip=-16, bob=0.02, pitch=1),
    dict(name="high", root=30, flex=-6, tip=-16, bob=-0.03, pitch=-2),
]
GLIDE = dict(name="glide", root=10, flex=8, tip=5, bob=0.0, pitch=0)


def bird_state(s, T, key):
    """Angles for the finished bird at a flap key drawing (dict from FLAP_KEYS)."""
    B = s["B"]
    st = dict(BIRD_FOLDS)
    fl = s["flex"]
    # the root angle is measured from horizontal: undo the body ridge
    st["bend"] = wing_bend(T, B, D(key["root"]) + s["ridge"], D(key["flex"]) * fl, D(key["tip"]) * fl)
    st["head"] = s["head_up"] - D(key.get("pitch", 0)) * 0.5
    st["neck"] = s["head_bend"]
    st["tail"] = s["tail_up"] + D(key.get("pitch", 0)) * 0.6
    st["spread"] = s["spread"]
    st["ridge"] = s["ridge"]
    st["curl"] = s["curl"]
    st["alpha_e"] = 0.0
    st["alpha_g"] = 0.0
    st["bob"] = key.get("bob", 0.0)
    st["smear"] = key.get("smear", False)
    st["stage"] = 3
    return st


def fold_state(s, T, k):
    B = s["B"]
    fk = FOLD_KEYS[k]
    st = {kk: v for kk, v in fk.items() if kk.startswith("th_") or kk in ("head", "neck", "tail", "spread", "ridge")}
    st["bend"] = wing_bend(T, B, fk["root"], fk["flex"], fk["tip"])
    st["curl"] = s["curl"] * (0.6 if k == 0 else 0.8)
    st["alpha_e"] = fk["alpha_e"]
    st["alpha_g"] = fk["alpha_g"]
    st["bob"] = 0.0
    st["stage"] = k
    return st


def build(s, st_l, st_r=None, T=None):
    """Both halves in bird local coords: verts (B, 2, NV, 3), texcoords (B, 2, NV, 2)."""
    T = T or topo()
    st_r = st_r or st_l
    V2 = flat(s, T)
    XL = to_local(fold(s, T, V2, st_l), +1)
    XR = to_local(fold(s, T, V2, st_r), -1)
    UVL = texcoords(s, T, V2, +1)
    UVR = texcoords(s, T, V2, -1)
    return np.stack([XL, XR], 1), np.stack([UVL, UVR], 1)


# ---------------------------------------------------------------- camera

class Camera:
    """Pinhole camera. World: x right, y up, z toward the viewer. Screen y down."""

    def __init__(self, pos, look, up=(0, 1, 0), focal=1400.0, cx=960.0, cy=540.0, ortho=False):
        self.pos = np.asarray(pos, np.float64)
        f = np.asarray(look, np.float64) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, np.float64))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, -u, f])   # camera x right, y down, z forward
        self.focal, self.cx, self.cy = focal, cx, cy
        self.ortho = ortho

    def project(self, Xw):
        c = (Xw - self.pos) @ self.R.T
        z = c[..., 2]
        if self.ortho:
            x = c[..., 0] * self.focal + self.cx
            y = c[..., 1] * self.focal + self.cy
        else:
            zz = np.maximum(z, 1e-4)
            x = c[..., 0] / zz * self.focal + self.cx
            y = c[..., 1] / zz * self.focal + self.cy
        return np.stack([x, y, z], -1)

    def view_dir(self, Xw):
        """Unit vectors from points toward the camera."""
        if self.ortho:
            return np.broadcast_to(-self.R[2], Xw.shape)
        v = self.pos - Xw
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (3, 3))
    M[..., 0, 0] = 1
    M[..., 1, 1] = c
    M[..., 1, 2] = -s
    M[..., 2, 1] = s
    M[..., 2, 2] = c
    return M


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (3, 3))
    M[..., 1, 1] = 1
    M[..., 0, 0] = c
    M[..., 0, 2] = s
    M[..., 2, 0] = -s
    M[..., 2, 2] = c
    return M


# bird local (x fwd, y left, z up) to world (x right, y up, z to viewer) for heading 0
_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def orient(heading, pitch, bank):
    """Rotations (B, 3, 3) from bird local to world. heading: angle in the world
    xz plane from +x toward -z (away from the viewer); pitch: nose up; bank:
    roll about the body axis (left wing up > 0)."""
    heading = np.asarray(heading, np.float64)
    pitch = np.asarray(pitch, np.float64)
    bank = np.asarray(bank, np.float64)
    Lr = _ry(-pitch) @ _rx(bank)
    return _ry(heading) @ (_C @ Lr)


def look_rotation(fwd, up_hint=(0.0, 1.0, 0.0), bank=0.0):
    """Rotations (B, 3, 3) whose local x points along fwd (B, 3), local z near up,
    then rolled by bank (B,) about the forward axis."""
    f = np.asarray(fwd, np.float64)
    f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-12)
    u = np.broadcast_to(np.asarray(up_hint, np.float64), f.shape)
    left = np.cross(u, f)
    left /= np.linalg.norm(left, axis=-1, keepdims=True) + 1e-12
    up = np.cross(f, left)
    M = np.stack([f, left, up], -1)            # columns: local x, y, z in world
    bank = np.broadcast_to(np.asarray(bank, np.float64), f.shape[:-1])
    return M @ _rx(bank)
