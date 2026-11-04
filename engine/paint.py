"""A stroke based gouache painter.

The method is the one a background artist uses, done in code:

1. Compose a reference image (layout, values, light, shapes) with plain numpy.
2. Paint over it in passes from big brushes to small ones. Every stroke is a
   curved ribbon that follows an orientation field (cloud contours, stone
   courses, book spines), takes its colour from the reference with a little
   jitter in hue and value, and is rendered as a textured, mostly opaque mark:
   separate bristle lanes, ragged edges, dry brush streaks near the end where
   the brush runs out of paint, paint catching only on the paper's high points,
   slightly darker edges where pigment pools, and pick up of the colour
   underneath (gouache reactivates when wet paint goes over it).
3. Place detail strokes by hand where a painter would sharpen.

Canvases are float32 (H, W, 4) premultiplied RGBA in display space. A layer may
be partly transparent (a tower painted on its own for parallax), so the same
kernel paints both opaque plates and cut out layers.

Stroke data is a dict of arrays (see `Strokes`); the numba kernel `_draw` renders
them in order. Everything is seeded, so plates repaint identically.

Usage (the pattern every Film A plate follows):

    from engine import paint as P
    paper = P.paper_for(h, w, "my-plate")            # cached paper grain
    ref = ...                                        # (h, w, 3) float32 colour reference
    ang = P.orientation(ref, sigma=2, rho=6, base=0.0, base_weight=0.5)   # stroke directions
    canvas = P.new_canvas(h, w)                      # premultiplied RGBA, transparent
    P.paint(canvas, ref, ang, [                      # passes, big brush to small
        dict(radius=16, steps=(2, 8), opacity=0.95, dry=0.3, cthresh=0.12),
        dict(radius=7, steps=(2, 8), use_err=True, err_thresh=0.03),      # only where needed
        dict(radius=3, steps=(2, 6), use_err=True, err_thresh=0.05),
    ], paper, "my-plate", clip=alpha_or_None)
    st = P.Strokes(); st.add([(x0, y0), (x1, y1)], width, rgb, seed=1, cap=1.0)   # hand placed
    P.draw(canvas, st, paper)
    P.finish(canvas, paper)                          # paper shows through a little

Brush keywords (DEFAULTS): opacity, bristles (lanes), dry (runs out toward the
end), ragged (edge wobble), pool (darker edges), streak (bristle streaks),
paper_k (catches only on paper tooth), taper0/taper1 (start and end taper),
pickup (drags the colour underneath), cap=1.0 for square ends (book spines,
edges). Passes also take mask (where strokes may start), cthresh (stop when
the colour changes by this much), wob and curve (path wobble and bend).
Paint plates with margins once, offline, through tools/heavy.py; never per frame.
"""

import math

import numpy as np
from numba import njit
from scipy import ndimage

from engine.noise import noise1, rng, seed_of, paper_grain

# parameter columns of the per stroke parameter array
P_WIDTH, P_OPAC, P_BRIST, P_DRY, P_RAG, P_POOL, P_STREAK, P_PAPER, P_T0, P_T1, P_PICK, P_ALPHA, P_CAP = range(13)
NPARAM = 13

DEFAULTS = dict(opacity=0.92, bristles=9.0, dry=0.25, ragged=0.18, pool=0.10, streak=0.10,
                paper_k=0.25, taper0=0.12, taper1=0.35, pickup=0.15, alpha=1.0, cap=0.0)


# ---------------------------------------------------------------- the kernel

@njit(cache=True, fastmath=True, inline="always")
def _smooth(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def _draw(canvas, clip, use_clip, paper, xs, ys, ns, cols, prm, seeds):
    H = canvas.shape[0]
    W = canvas.shape[1]
    S = ns.shape[0]
    cum = np.zeros(xs.shape[1], np.float64)
    for s in range(S):
        n = ns[s]
        if n <= 0:
            continue
        width = prm[s, 0]
        opac = prm[s, 1]
        brist = prm[s, 2]
        dry = prm[s, 3]
        ragged = prm[s, 4]
        pool = prm[s, 5]
        streak = prm[s, 6]
        paper_k = prm[s, 7]
        t0 = prm[s, 8]
        t1 = prm[s, 9]
        pick = prm[s, 10]
        palpha = prm[s, 11]
        flatcap = prm[s, 12] > 0.5
        seed = seeds[s]
        cr = cols[s, 0]
        cg = cols[s, 1]
        cb = cols[s, 2]
        if width < 0.3 or opac <= 0.0:
            continue
        cum[0] = 0.0
        for k in range(1, n):
            cum[k] = cum[k - 1] + math.sqrt((xs[s, k] - xs[s, k - 1]) ** 2 + (ys[s, k] - ys[s, k - 1]) ** 2)
        total = cum[n - 1]
        if total < 1e-3:
            total = 1e-3
        pad = width * (1.0 + ragged) + 2.0
        bx0 = 1e30
        by0 = 1e30
        bx1 = -1e30
        by1 = -1e30
        for k in range(n):
            if xs[s, k] < bx0:
                bx0 = xs[s, k]
            if xs[s, k] > bx1:
                bx1 = xs[s, k]
            if ys[s, k] < by0:
                by0 = ys[s, k]
            if ys[s, k] > by1:
                by1 = ys[s, k]
        x0 = int(max(0.0, math.floor(bx0 - pad)))
        x1 = int(min(W - 1.0, math.ceil(bx1 + pad)))
        y0 = int(max(0.0, math.floor(by0 - pad)))
        y1 = int(min(H - 1.0, math.ceil(by1 + pad)))
        if x0 > x1 or y0 > y1:
            continue
        inv_w = 1.0 / max(width, 0.5)
        fr1 = 1.0 / max(0.7 * width, 1.5)
        fr2 = 1.0 / max(0.22 * width, 0.8)
        drift = noise1(seed * 0.37, 91)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                best = 1e30
                bs = 0.0
                bsd = 1.0
                beyond = 0.0
                if n == 1:
                    best = (x - xs[s, 0]) ** 2 + (y - ys[s, 0]) ** 2
                    bsd = (x - xs[s, 0]) - (y - ys[s, 0])
                else:
                    for k in range(n - 1):
                        ax = xs[s, k]
                        ay = ys[s, k]
                        dx = xs[s, k + 1] - ax
                        dy = ys[s, k + 1] - ay
                        l2 = dx * dx + dy * dy
                        bey = 0.0
                        if l2 < 1e-9:
                            tt = 0.0
                        else:
                            tt = ((x - ax) * dx + (y - ay) * dy) / l2
                            if tt < 0.0:
                                if k == 0:
                                    bey = -tt * math.sqrt(l2)
                                tt = 0.0
                            elif tt > 1.0:
                                if k == n - 2:
                                    bey = (tt - 1.0) * math.sqrt(l2)
                                tt = 1.0
                        qx = ax + tt * dx
                        qy = ay + tt * dy
                        d2 = (x - qx) ** 2 + (y - qy) ** 2
                        if d2 < best:
                            best = d2
                            bs = cum[k] + tt * math.sqrt(l2)
                            bsd = dx * (y - ay) - dy * (x - ax)
                            beyond = bey
                d = math.sqrt(best)
                if d > pad:
                    continue
                u = bs / total
                # width profile: blunt loaded start, tapering lift at the end
                prof = 1.0
                if u < t0:
                    prof = 0.55 + 0.45 * math.sqrt(u / t0)
                if u > 1.0 - t1:
                    v = (1.0 - u) / t1
                    prof *= 0.2 + 0.8 * math.sqrt(max(v, 0.0))
                side = 1.0
                sseed = seed * 2 + 1
                if bsd < 0.0:
                    side = -1.0
                    sseed = seed * 2 + 2
                rag = (noise1(bs * fr1, sseed) - 0.5) * 2.0 + 0.45 * (noise1(bs * fr2, sseed + 7) - 0.5) * 2.0
                wcur = width * prof * (1.0 + ragged * rag)
                if wcur < 0.3:
                    continue
                if d > wcur + 1.0:
                    continue
                if flatcap:
                    # square ends: measure the edge across the stroke only and cut at the ends
                    dperp = math.sqrt(max(best - beyond * beyond, 0.0))
                    aedge = wcur - dperp + 0.5
                    ce = 0.5 - beyond + 0.5
                    if ce < aedge:
                        aedge = ce
                    d = dperp
                else:
                    aedge = wcur - d + 0.5
                if aedge > 1.0:
                    aedge = 1.0
                if aedge <= 0.0:
                    continue
                tn = d / max(width * prof, 0.5)
                if tn > 1.0:
                    tn = 1.0
                lane = (side * tn * 0.5 + 0.5) * brist + drift * 13.0
                b = 0.62 * noise1(lane, seed + 3) + 0.38 * noise1(lane * 2.9 + bs * 0.004, seed + 4)
                load = 1.18 - dry * u ** 1.25
                g = paper[y, x]
                resist = b * 0.62 + (1.0 - g) * paper_k
                cov = _smooth(resist - 0.16, resist + 0.16, load)
                a = opac * aedge * cov
                if use_clip:
                    a *= clip[y, x]
                if a <= 0.002:
                    continue
                # colour modulation: bristle streaks, slow drift along the stroke, pooled edges
                m = 1.0 + streak * (noise1(lane * 1.7, seed + 5) - 0.5) * 1.4
                m += streak * 0.6 * (noise1(bs * inv_w * 0.33, seed + 6) - 0.5)
                m -= pool * _smooth(0.55, 1.0, tn)
                m += 0.05 * paper_k * (g - 0.5)
                rr = cr * m
                gg = cg * m
                bb = cb * m
                if pick > 0.0:
                    ca = canvas[y, x, 3]
                    if ca > 0.02:
                        pk = pick * u
                        rr = rr * (1.0 - pk) + canvas[y, x, 0] / ca * pk
                        gg = gg * (1.0 - pk) + canvas[y, x, 1] / ca * pk
                        bb = bb * (1.0 - pk) + canvas[y, x, 2] / ca * pk
                ia = 1.0 - a
                canvas[y, x, 0] = canvas[y, x, 0] * ia + rr * a * palpha
                canvas[y, x, 1] = canvas[y, x, 1] * ia + gg * a * palpha
                canvas[y, x, 2] = canvas[y, x, 2] * ia + bb * a * palpha
                canvas[y, x, 3] = canvas[y, x, 3] * ia + a * palpha


@njit(cache=True, fastmath=True)
def _trace(ref, ang, mask, sxs, sys_, dirs, cols, step, max_steps, min_steps, cthresh, wob, seeds,
           curve, out_x, out_y, out_n):
    H = ref.shape[0]
    W = ref.shape[1]
    S = sxs.shape[0]
    for s in range(S):
        x = sxs[s]
        y = sys_[s]
        xi = int(x)
        yi = int(y)
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            out_n[s] = 0
            continue
        a = ang[yi, xi]
        pdx = math.cos(a) * dirs[s]
        pdy = math.sin(a) * dirs[s]
        out_x[s, 0] = x
        out_y[s, 0] = y
        cnt = 1
        for k in range(1, max_steps):
            xi = int(x)
            yi = int(y)
            a = ang[yi, xi] + wob * (noise1(k * 0.61, seeds[s]) - 0.5)
            dx = math.cos(a)
            dy = math.sin(a)
            if dx * pdx + dy * pdy < 0.0:
                dx = -dx
                dy = -dy
            dx = (1.0 - curve) * pdx + curve * dx
            dy = (1.0 - curve) * pdy + curve * dy
            ln = math.sqrt(dx * dx + dy * dy) + 1e-9
            dx /= ln
            dy /= ln
            nx = x + dx * step
            ny = y + dy * step
            nxi = int(nx)
            nyi = int(ny)
            if nxi < 0 or nyi < 0 or nxi >= W or nyi >= H:
                break
            if mask[nyi, nxi] < 0.5:
                break
            diff = abs(ref[nyi, nxi, 0] - cols[s, 0]) + abs(ref[nyi, nxi, 1] - cols[s, 1]) + abs(ref[nyi, nxi, 2] - cols[s, 2])
            if cnt >= min_steps and diff > cthresh:
                break
            x = nx
            y = ny
            pdx = dx
            pdy = dy
            out_x[s, cnt] = x
            out_y[s, cnt] = y
            cnt += 1
        out_n[s] = cnt


# ---------------------------------------------------------------- stroke lists

class Strokes:
    """A growable list of strokes. Each stroke is a polyline with a width, a
    colour and brush parameters (see DEFAULTS)."""

    def __init__(self, max_pts=24):
        self.max_pts = max_pts
        self._paths = []
        self._cols = []
        self._prm = []
        self._seeds = []

    def add(self, path, width, color, seed=0, **kw):
        p = dict(DEFAULTS)
        p.update(kw)
        path = np.asarray(path, np.float32).reshape(-1, 2)
        if len(path) > self.max_pts:
            idx = np.linspace(0, len(path) - 1, self.max_pts).round().astype(int)
            path = path[idx]
        self._paths.append(path)
        self._cols.append(np.asarray(color, np.float32)[:3])
        self._prm.append([width, p["opacity"], p["bristles"], p["dry"], p["ragged"], p["pool"],
                          p["streak"], p["paper_k"], p["taper0"], p["taper1"], p["pickup"], p["alpha"], p["cap"]])
        self._seeds.append(int(seed) & 0x3FFFFFFF)

    def __len__(self):
        return len(self._paths)

    def arrays(self):
        S = len(self._paths)
        P = max([len(p) for p in self._paths] + [1])
        xs = np.zeros((S, P), np.float32)
        ys = np.zeros((S, P), np.float32)
        ns = np.zeros(S, np.int32)
        for i, p in enumerate(self._paths):
            xs[i, :len(p)] = p[:, 0]
            ys[i, :len(p)] = p[:, 1]
            ns[i] = len(p)
        return dict(xs=xs, ys=ys, ns=ns, cols=np.array(self._cols, np.float32).reshape(S, 3),
                    prm=np.array(self._prm, np.float32).reshape(S, NPARAM),
                    seeds=np.array(self._seeds, np.int64))


_ONE = np.ones((1, 1), np.float32)


def draw(canvas, strokes, paper, clip=None):
    """Render strokes (a Strokes or its arrays dict) onto canvas in order."""
    a = strokes.arrays() if isinstance(strokes, Strokes) else strokes
    if len(a["ns"]) == 0:
        return canvas
    use_clip = clip is not None
    _draw(canvas, clip.astype(np.float32) if use_clip else _ONE, use_clip, paper,
          a["xs"], a["ys"], a["ns"], a["cols"], a["prm"], a["seeds"])
    return canvas


def concat(*arrs):
    """Join several stroke array dicts into one (paths padded to the longest)."""
    arrs = [a for a in arrs if a is not None and len(a["ns"])]
    if not arrs:
        return None
    P = max(a["xs"].shape[1] for a in arrs)

    def pad(m):
        if m.shape[1] == P:
            return m
        out = np.zeros((m.shape[0], P), m.dtype)
        out[:, :m.shape[1]] = m
        return out
    return dict(xs=np.concatenate([pad(a["xs"]) for a in arrs]),
                ys=np.concatenate([pad(a["ys"]) for a in arrs]),
                ns=np.concatenate([a["ns"] for a in arrs]),
                cols=np.concatenate([a["cols"] for a in arrs]),
                prm=np.concatenate([a["prm"] for a in arrs]),
                seeds=np.concatenate([a["seeds"] for a in arrs]))


def shuffle(a, key):
    if a is None:
        return a
    p = rng("shuffle", key).permutation(len(a["ns"]))
    return {k: v[p] for k, v in a.items()}


# ---------------------------------------------------------------- colour helpers

_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
_RGB = np.linalg.inv(_YIQ).astype(np.float32)


def jitter_colors(cols, r, value=0.05, hue=0.05, sat=0.08):
    """Jitter (S, 3) colours in value (multiplicative), hue (radians/2pi) and chroma."""
    yiq = cols @ _YIQ.T
    S = len(cols)
    v = 1.0 + value * r.standard_normal(S).astype(np.float32)
    h = hue * 2 * np.pi * r.standard_normal(S).astype(np.float32)
    c = 1.0 + sat * r.standard_normal(S).astype(np.float32)
    i, q = yiq[:, 1], yiq[:, 2]
    ch, sh = np.cos(h), np.sin(h)
    i2 = (i * ch - q * sh) * c
    q2 = (i * sh + q * ch) * c
    out = np.stack([yiq[:, 0] * v, i2 * v, q2 * v], 1) @ _RGB.T
    return np.clip(out, 0.0, 1.2).astype(np.float32)


def lum(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def blur(img, sigma):
    if sigma <= 0:
        return img
    if img.ndim == 2:
        return ndimage.gaussian_filter(img, sigma).astype(np.float32)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma)
    return out


def mix(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2 and (np.ndim(a) in (1, 3) or np.ndim(b) in (1, 3)):
        t = t[..., None]
    return (a * (1.0 - t) + b * t).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def over(dst, src):
    """Premultiplied src over dst, in place on dst."""
    a = src[..., 3:4]
    dst *= (1.0 - a)
    dst += src
    return dst


# ---------------------------------------------------------------- orientation

def orientation(ref, sigma=1.5, rho=6.0, base=None, base_weight=0.35, edge_gain=1.0):
    """Stroke direction field in radians (mod pi).

    Strokes run along edges (perpendicular to the gradient) where the reference
    has structure, and along `base` (a scalar angle or an (H, W) angle field)
    where it is flat. The structure tensor keeps directions coherent.
    """
    L = lum(ref) if ref.ndim == 3 else ref
    Ls = ndimage.gaussian_filter(L, sigma)
    gx = ndimage.sobel(Ls, axis=1)
    gy = ndimage.sobel(Ls, axis=0)
    j11 = ndimage.gaussian_filter(gx * gx, rho)
    j22 = ndimage.gaussian_filter(gy * gy, rho)
    j12 = ndimage.gaussian_filter(gx * gy, rho)
    vx = -(j11 - j22)
    vy = -(2.0 * j12)
    mag = np.sqrt(vx * vx + vy * vy)
    scale = np.percentile(mag, 97) + 1e-8
    k = np.clip(mag / scale * edge_gain, 0, 1)
    ex = vx / (mag + 1e-12) * k
    ey = vy / (mag + 1e-12) * k
    if base is None:
        base = 0.0
    base = np.broadcast_to(np.asarray(base, np.float32), L.shape)
    bx = np.cos(2 * base) * base_weight
    by = np.sin(2 * base) * base_weight
    wx = ex + bx * (1.0 - k * 0.7)
    wy = ey + by * (1.0 - k * 0.7)
    return (0.5 * np.arctan2(wy, wx)).astype(np.float32)


def blend_angles(a, b, t):
    """Blend two angle fields (mod pi) with weight t."""
    t = np.asarray(t, np.float32)
    x = np.cos(2 * a) * (1 - t) + np.cos(2 * b) * t
    y = np.sin(2 * a) * (1 - t) + np.sin(2 * b) * t
    return (0.5 * np.arctan2(y, x)).astype(np.float32)


# ---------------------------------------------------------------- passes

def make_pass(ref, ang, key, radius, spacing=1.0, steps=(2, 8), step=None, mask=None,
              canvas=None, err_thresh=None, cthresh=0.25, wob=0.35, curve=0.55,
              jitter=(0.05, 0.03, 0.08), width_jitter=0.25, density=1.0, ref_sigma=None,
              **brush):
    """Generate one pass of strokes over `ref` (H, W, 3).

    radius: brush half width in px. spacing: grid spacing as a multiple of radius.
    steps: (min, max) number of path steps; step length defaults to radius.
    mask: (H, W) where strokes may start and continue (> 0.5).
    canvas + err_thresh: only place strokes where the canvas differs from ref.
    density: (H, W) or scalar probability of keeping a seed.
    brush: overrides for DEFAULTS (opacity, dry, bristles, ...).
    Returns a stroke arrays dict (unshuffled).
    """
    H, W = ref.shape[:2]
    r = rng("pass", key)
    sig = radius * 0.6 if ref_sigma is None else ref_sigma
    refs = blur(ref, sig) if sig > 0.3 else ref
    sp = max(radius * spacing, 1.0)
    gy, gx = np.mgrid[0:H:sp, 0:W:sp]
    gx = gx.ravel().astype(np.float32) + (r.random(gx.size).astype(np.float32) - 0.5) * sp * 1.3
    gy = gy.ravel().astype(np.float32) + (r.random(gy.size).astype(np.float32) - 0.5) * sp * 1.3
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    gx, gy = gx[ok], gy[ok]
    xi, yi = gx.astype(int), gy.astype(int)
    keep = np.ones(len(gx), bool)
    if mask is not None:
        keep &= mask[yi, xi] > 0.5
    dens = np.broadcast_to(np.asarray(density, np.float32), (H, W)) if np.ndim(density) else None
    if dens is not None:
        keep &= r.random(len(gx)) < dens[yi, xi]
    elif density < 1.0:
        keep &= r.random(len(gx)) < density
    if canvas is not None and err_thresh is not None:
        a = canvas[..., 3:4]
        cb = blur(canvas[..., :3], radius * 0.8)
        ab = blur(a[..., 0], radius * 0.8)[..., None]
        err = np.abs(cb - refs * ab).sum(-1) + (1.0 - ab[..., 0]) * 3.0
        keep &= err[yi, xi] > err_thresh
    gx, gy = gx[keep], gy[keep]
    S = len(gx)
    if S == 0:
        return None
    cols = refs[gy.astype(int), gx.astype(int)]
    cols = jitter_colors(cols, r, *jitter)
    max_steps = int(steps[1]) + 1
    out_x = np.zeros((S, max_steps), np.float32)
    out_y = np.zeros((S, max_steps), np.float32)
    out_n = np.zeros(S, np.int32)
    dirs = np.where(r.random(S) < 0.5, -1.0, 1.0).astype(np.float32)
    seeds = r.integers(0, 2 ** 30, S).astype(np.int64)
    m = mask.astype(np.float32) if mask is not None else np.ones((H, W), np.float32)
    _trace(refs, ang, m, gx, gy, dirs, cols, float(step or radius), max_steps, int(steps[0]),
           float(cthresh), float(wob), seeds, float(curve), out_x, out_y, out_n)
    p = dict(DEFAULTS)
    p.update(brush)
    widths = radius * (1.0 + width_jitter * (r.random(S).astype(np.float32) - 0.5) * 2)
    prm = np.zeros((S, NPARAM), np.float32)
    prm[:, P_WIDTH] = widths
    prm[:, P_OPAC] = p["opacity"] * (1.0 - 0.15 * r.random(S))
    prm[:, P_BRIST] = np.minimum(p["bristles"], np.maximum(2.0, widths * 0.8)) * (0.8 + 0.4 * r.random(S))
    prm[:, P_DRY] = p["dry"] * (0.6 + 0.8 * r.random(S))
    prm[:, P_RAG] = p["ragged"]
    prm[:, P_POOL] = p["pool"]
    prm[:, P_STREAK] = p["streak"]
    prm[:, P_PAPER] = p["paper_k"]
    prm[:, P_T0] = p["taper0"]
    prm[:, P_T1] = p["taper1"]
    prm[:, P_PICK] = p["pickup"]
    prm[:, P_ALPHA] = p["alpha"]
    prm[:, P_CAP] = p["cap"]
    return dict(xs=out_x, ys=out_y, ns=out_n, cols=cols, prm=prm, seeds=seeds)


def paint(canvas, ref, ang, passes, paper, key, clip=None, mask=None):
    """Run a list of pass dicts (keyword args for make_pass) over canvas."""
    for i, ps in enumerate(passes):
        ps = dict(ps)
        pclip = ps.pop("clip", clip)
        pmask = ps.pop("mask", mask)
        use_err = ps.pop("use_err", False)
        st = make_pass(ref, ang, (key, i), mask=pmask, canvas=canvas if use_err else None, **ps)
        if st is None:
            continue
        draw(canvas, shuffle(st, (key, i)), paper, pclip)
    return canvas


_PAPER = {}


def paper_for(h, w, key="paper"):
    k = (h, w, key)
    if k not in _PAPER:
        _PAPER[k] = paper_grain(h, w, key)
    return _PAPER[k]


def finish(canvas, paper, grain=0.035, tooth=0.02):
    """Let the paper show through the paint a little: value modulation and a
    faint tooth in the lights. Works on premultiplied canvases."""
    g = (paper - 0.5)[..., None]
    canvas[..., :3] *= (1.0 + grain * g * 2.0)
    canvas[..., :3] += tooth * g * canvas[..., 3:4]
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return canvas


def new_canvas(h, w, color=None):
    c = np.zeros((h, w, 4), np.float32)
    if color is not None:
        c[..., :3] = np.asarray(color, np.float32)
        c[..., 3] = 1.0
    return c


def seed(*keys):
    return seed_of(*keys)
