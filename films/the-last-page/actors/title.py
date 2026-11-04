"""The last page: the blank page, the brushed title "Opus 5.5", the 寓話 hanko and
the line "Every story begins with a blank page."

Every letter is authored here as brush strokes (centre lines; the brush gives
the pressure), laid out in page space, written on stroke by stroke in order,
then mapped onto the page lying on the balcony floor with a homography. No
font is used anywhere.
"""

import math
import os

import numpy as np
from scipy import ndimage

import timeline as T
from engine import brush, comp, geom, raster, vnoise
from engine import noise as wnoise

PW, PH = 1600, 1000           # page space (the blank page, flat)
INK_DARK = np.array([0.10, 0.07, 0.06], np.float32)
INK_LIGHT = np.array([0.30, 0.24, 0.20], np.float32)
SEAL = np.array([0.76, 0.15, 0.12], np.float32)

# ---------------------------------------------------------------- glyphs
# em units: baseline y = 0, x height 0.5, caps 0.72, y up. Each glyph: (advance, [strokes]).


def _oval(cx, cy, rx, ry, a0=70, a1=430, n=14):
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in a]


GLYPHS = {
    "O": (0.70, [_oval(0.34, 0.36, 0.29, 0.37, 80, 445, 18)]),
    "p": (0.50, [[(0.09, 0.49), (0.08, 0.20), (0.07, -0.26)],
                 [(0.08, 0.34), (0.16, 0.47), (0.30, 0.50), (0.42, 0.38), (0.42, 0.14), (0.28, 0.0), (0.10, 0.05)]]),
    "u": (0.52, [[(0.06, 0.49), (0.06, 0.16), (0.12, 0.02), (0.25, -0.01), (0.37, 0.10), (0.40, 0.49)],
                 [(0.41, 0.48), (0.40, 0.10), (0.47, 0.0)]]),
    "s": (0.42, [[(0.35, 0.43), (0.25, 0.50), (0.11, 0.47), (0.08, 0.37), (0.20, 0.27), (0.33, 0.18), (0.33, 0.07),
                  (0.21, -0.01), (0.04, 0.04)]]),
    "5": (0.52, [[(0.42, 0.71), (0.27, 0.71), (0.13, 0.70)],
                 [(0.14, 0.70), (0.11, 0.42), (0.23, 0.47), (0.37, 0.43), (0.44, 0.27), (0.38, 0.08), (0.23, -0.01),
                  (0.06, 0.05)]]),
    ".": (0.20, [[(0.07, 0.04), (0.10, 0.01)]]),
    " ": (0.30, []),
    "E": (0.48, [[(0.09, 0.72), (0.08, 0.36), (0.07, 0.0)], [(0.08, 0.72), (0.25, 0.73), (0.42, 0.72)],
                 [(0.08, 0.38), (0.22, 0.38), (0.34, 0.39)], [(0.07, 0.0), (0.26, 0.0), (0.44, 0.01)]]),
    "v": (0.46, [[(0.03, 0.49), (0.12, 0.24), (0.21, 0.0), (0.31, 0.26), (0.42, 0.50)]]),
    "e": (0.46, [[(0.06, 0.25), (0.22, 0.26), (0.38, 0.29), (0.36, 0.44), (0.22, 0.50), (0.09, 0.41), (0.05, 0.21),
                  (0.14, 0.03), (0.28, -0.01), (0.40, 0.08)]]),
    "r": (0.38, [[(0.07, 0.49), (0.07, 0.24), (0.06, 0.0)], [(0.07, 0.30), (0.14, 0.44), (0.26, 0.50), (0.35, 0.46)]]),
    "y": (0.46, [[(0.03, 0.49), (0.12, 0.27), (0.21, 0.06)],
                 [(0.41, 0.49), (0.32, 0.22), (0.24, 0.0), (0.15, -0.19), (0.04, -0.24)]]),
    "t": (0.36, [[(0.15, 0.67), (0.14, 0.30), (0.14, 0.08), (0.21, 0.0), (0.31, 0.03)], [(0.02, 0.46), (0.17, 0.47), (0.31, 0.47)]]),
    "o": (0.48, [_oval(0.23, 0.25, 0.19, 0.25, 80, 445, 14)]),
    "b": (0.49, [[(0.08, 0.77), (0.07, 0.38), (0.06, 0.0)],
                 [(0.07, 0.30), (0.18, 0.46), (0.32, 0.49), (0.42, 0.33), (0.40, 0.12), (0.26, 0.0), (0.08, 0.04)]]),
    "g": (0.48, [[(0.38, 0.41), (0.26, 0.50), (0.11, 0.44), (0.05, 0.26), (0.12, 0.08), (0.26, 0.04), (0.38, 0.14)],
                 [(0.40, 0.49), (0.39, 0.20), (0.38, -0.12), (0.28, -0.24), (0.12, -0.24), (0.04, -0.18)]]),
    "i": (0.20, [[(0.09, 0.47), (0.08, 0.20), (0.08, 0.0)], [(0.09, 0.65), (0.11, 0.62)]]),
    "n": (0.47, [[(0.07, 0.49), (0.07, 0.24), (0.06, 0.0)],
                 [(0.07, 0.32), (0.16, 0.46), (0.28, 0.50), (0.37, 0.42), (0.38, 0.20), (0.39, 0.0)]]),
    "w": (0.58, [[(0.03, 0.49), (0.10, 0.22), (0.15, 0.0), (0.21, 0.20), (0.27, 0.40), (0.33, 0.20), (0.39, 0.0),
                  (0.46, 0.25), (0.53, 0.50)]]),
    "h": (0.47, [[(0.08, 0.77), (0.07, 0.38), (0.06, 0.0)],
                 [(0.07, 0.32), (0.16, 0.46), (0.28, 0.50), (0.37, 0.42), (0.38, 0.20), (0.39, 0.0)]]),
    "a": (0.49, [[(0.37, 0.41), (0.26, 0.50), (0.10, 0.44), (0.05, 0.24), (0.12, 0.05), (0.26, 0.02), (0.37, 0.14)],
                 [(0.39, 0.49), (0.38, 0.24), (0.38, 0.08), (0.45, 0.0)]]),
    "l": (0.23, [[(0.10, 0.78), (0.09, 0.40), (0.09, 0.08), (0.14, 0.0), (0.21, 0.02)]]),
    "k": (0.43, [[(0.08, 0.78), (0.07, 0.38), (0.06, 0.0)], [(0.37, 0.49), (0.22, 0.34), (0.08, 0.20)],
                 [(0.15, 0.27), (0.27, 0.13), (0.40, 0.0)]]),
}

TITLE = "Opus 5.5"
LINE = "Every story begins with a blank page."
TITLE_EM = 330.0
LINE_EM = 100.0
TITLE_BASE = 420.0
LINE_BASE = 830.0
SEAL_BOX = (1380.0, 470.0, 112.0, 214.0)     # x, y, w, h in page space


def layout(text, em, base_y, centre_x, key, width_scale=1.0, pressure_scale=1.0):
    """Strokes in page space for a line of text: list of (points (n,2), radii (n,))."""
    r = vnoise.rng("layout", key)
    adv = sum(GLYPHS[c][0] for c in text) * em * width_scale
    x = centre_x - adv / 2
    out = []
    for i, c in enumerate(text):
        a, strokes = GLYPHS[c]
        dy = r.normal(0, 0.012) * em          # a hand-lettered baseline, not a ruler
        slant = 0.10                          # brush lettering leans a little
        sc = 1.0 + r.normal(0, 0.025)
        for st in strokes:
            P = np.array(st, np.float64)
            P = geom.catmull_rom(P, 8) if len(P) > 2 else P
            if len(P) == 2:
                P = np.stack([np.linspace(P[0, 0], P[1, 0], 6), np.linspace(P[0, 1], P[1, 1], 6)], 1)
            X = x + (P[:, 0] + P[:, 1] * slant) * em * width_scale * sc
            Y = base_y - P[:, 1] * em * sc + dy
            pts = np.stack([X, Y], 1)
            s = geom.arclen(pts)
            u = s / max(s[-1], 1e-6)
            thick = em * 0.055 * pressure_scale
            if c == "." or (c == "i" and len(st) == 2 and st[0][1] > 0.6):
                rad = np.full(len(u), thick * 1.25)
            else:
                rad = thick * brush.pressure(u, land=0.10, lift=0.35, swell=0.15)
                # downstrokes are heavier than upstrokes, the way a brush loads
                dyy = np.gradient(Y)
                rad *= 0.85 + 0.30 * np.clip(dyy / (np.abs(np.gradient(X)) + np.abs(dyy) + 1e-6), 0, 1)
            out.append((pts, rad, c))
        x += a * em * width_scale
    return out


def _seal_strokes():
    """寓 over 話 in a blocky seal style, unit box per character (x right, y down)."""
    yu = [  # 寓
        [(0.50, 0.02), (0.50, 0.10)], [(0.12, 0.24), (0.12, 0.13), (0.88, 0.13), (0.88, 0.24)],
        [(0.28, 0.30), (0.72, 0.30)], [(0.28, 0.30), (0.28, 0.52)], [(0.72, 0.30), (0.72, 0.52)],
        [(0.28, 0.41), (0.72, 0.41)], [(0.28, 0.52), (0.72, 0.52)], [(0.50, 0.30), (0.50, 0.80)],
        [(0.15, 0.95), (0.15, 0.61), (0.85, 0.61), (0.85, 0.95), (0.78, 0.92)],
        [(0.38, 0.70), (0.30, 0.86), (0.66, 0.84)], [(0.60, 0.74), (0.68, 0.86)],
    ]
    hua = [  # 話
        [(0.21, 0.02), (0.23, 0.10)], [(0.03, 0.17), (0.39, 0.17)], [(0.09, 0.29), (0.33, 0.29)],
        [(0.09, 0.40), (0.33, 0.40)], [(0.09, 0.52), (0.09, 0.84)], [(0.09, 0.52), (0.33, 0.52), (0.33, 0.84)],
        [(0.09, 0.82), (0.33, 0.82)],
        [(0.90, 0.05), (0.58, 0.15)], [(0.48, 0.31), (0.97, 0.31)], [(0.73, 0.13), (0.73, 0.53)],
        [(0.56, 0.58), (0.56, 0.92)], [(0.56, 0.58), (0.91, 0.58), (0.91, 0.92)], [(0.56, 0.90), (0.91, 0.90)],
    ]
    return yu, hua


def seal_layer(alpha_scale=1.0, squash=1.0):
    """The hanko as (coverage, None) in page space: red with white carved characters."""
    x, y, w, h = SEAL_BOX
    cov = np.zeros((PH, PW), np.float32)
    r = vnoise.rng("seal")
    # rough red field
    n = 64
    edge = []
    for i in range(n):
        tt = i / n
        if tt < 0.25:
            px, py = x + w * tt / 0.25, y
        elif tt < 0.5:
            px, py = x + w, y + h * (tt - 0.25) / 0.25
        elif tt < 0.75:
            px, py = x + w * (1 - (tt - 0.5) / 0.25), y + h
        else:
            px, py = x, y + h * (1 - (tt - 0.75) / 0.25)
        edge.append((px + r.normal(0, 1.4), py + r.normal(0, 1.4)))
    E = np.array(edge)
    cxy = np.array([x + w / 2, y + h / 2])
    E = cxy + (E - cxy) * np.array([1.0 / squash ** 0.5, squash])
    field = raster.poly_mask(E, PH, PW).astype(np.float32)
    # carved characters
    carve = np.zeros((PH, PW), np.float32)
    dummy = np.zeros((PH, PW), np.float32)
    yu, hua = _seal_strokes()
    m = 0.12 * w
    cw = w - 2 * m
    ch = (h - 3 * m) / 2
    for k, strokes in enumerate((yu, hua)):
        ox = x + m
        oy = y + m + k * (ch + m)
        for st in strokes:
            P = np.array(st, np.float64)
            pts = np.stack([ox + P[:, 0] * cw, oy + P[:, 1] * ch], 1)
            pts = cxy + (pts - cxy) * np.array([1.0 / squash ** 0.5, squash])
            rad = np.full(len(pts), 0.052 * cw)
            brush.stroke(carve, dummy, pts, rad, 1.0, seed=k * 31 + len(st), dry0=1.0, pool=0.0)
    # stamp texture: uneven pressure, specks of paper showing through
    tex = wnoise.fbm(PH, PW, 18, "seal-tex", octaves=3)
    speck = (wnoise.value_noise(PH, PW, 3, "seal-speck") > 0.80).astype(np.float32)
    cov = field * (1 - carve) * (0.82 + 0.25 * tex) * (1 - 0.6 * speck)
    cov = ndimage.gaussian_filter(cov, 0.6)
    return np.clip(cov * alpha_scale, 0, 1)


# ---------------------------------------------------------------- timing

def _schedule():
    """Stroke list with (start, end) times: the title, then the line."""
    c = T.CUES
    # the paw rests on the page's left part, so the lettering sits right of it
    title = layout(TITLE, TITLE_EM, TITLE_BASE, PW * 0.55, "title", width_scale=1.0, pressure_scale=1.15)
    line = layout(LINE, LINE_EM, LINE_BASE, PW * 0.56, "line", width_scale=0.92, pressure_scale=1.05)
    out = []
    for strokes, t0, t1 in ((title, c["title_ink_start"], c["title_ink_end"]), (line, c["line_ink_start"], c["line_ink_end"])):
        lens = [geom.arclen(p)[-1] for p, _, _ in strokes]
        gap = 0.25 * np.mean(lens)       # the brush lifts between strokes
        total = sum(lens) + gap * (len(lens) - 1)
        acc = 0.0
        for (p, rad, ch), ln in zip(strokes, lens):
            a = t0 + (t1 - t0) * acc / total
            b = t0 + (t1 - t0) * (acc + ln) / total
            out.append((p, rad, a, b, ch))
            acc += ln + gap
    return out


_SCHED = None
_CACHE = {}


def schedule():
    global _SCHED
    if _SCHED is None:
        _SCHED = _schedule()
    return _SCHED


def ink_page(t):
    """(coverage, density, seal coverage) in page space at time t."""
    key = round(t * T.FPS)
    sched = schedule()
    done = all(t >= b for (_, _, a, b, _) in sched)
    if done and "final" in _CACHE:
        cov, den = _CACHE["final"]
    else:
        cov = np.zeros((PH, PW), np.float32)
        den = np.zeros((PH, PW), np.float32)
        for i, (p, rad, a, b, ch) in enumerate(sched):
            if t <= a:
                continue
            u = min(1.0, (t - a) / max(b - a, 1e-6))
            u = 1 - (1 - u) ** 1.6     # quick in the middle, easing as the brush lifts
            brush.stroke(cov, den, p, rad, u, seed=i * 13 + 5, dry0=0.70, dryk=0.85, pool=0.40,
                         lanes=7.0 if rad.max() > 12 else 5.0)
        fibre = wnoise.paper_grain(PH, PW, "title-fibre") if "fibre" not in _CACHE else _CACHE["fibre"]
        _CACHE["fibre"] = fibre
        cov = brush.bleed(cov, fibre, 0.8, 0.8)
        if done:
            _CACHE["final"] = (cov, den)
    hk = T.CUES["hanko"]
    seal = None
    if t >= hk:
        u = min(1.0, (t - hk) / 0.12)
        if u >= 1.0:
            if "seal" not in _CACHE:
                _CACHE["seal"] = seal_layer(1.0, 1.0)
            seal = _CACHE["seal"]
        else:
            seal = seal_layer(alpha_scale=u ** 0.5, squash=1.0 + 0.10 * (1 - u))
    return cov, den, seal


# ---------------------------------------------------------------- the page on the floor

def page_quad(cut_id, f):
    if cut_id == "9b":
        return np.array(T.ANCHORS["9b"]["page_quad_screen"], np.float64)
    # 9a: the page settles at the fox's paws (smaller, further away)
    cx, cy = T.ANCHORS["9a"]["page_land_screen"]
    w, h = 520, 210
    q = np.array([(-w / 2, -h / 2), (w / 2, -h / 2 - 12), (w / 2 + 30, h / 2), (-w / 2 - 20, h / 2 + 8)]) + [cx, cy]
    t = f / T.FPS
    tl = T.CUES["page_land"]
    if t < tl:
        # falling and rocking down to the floor
        u = max(0.0, min(1.0, (t - (tl - 1.6)) / 1.6))
        drop = (1 - u) ** 1.4 * 900
        sway = math.sin(u * 7.0) * (1 - u) * 160
        rot = math.sin(u * 6.0 + 0.5) * (1 - u) * 0.6
        tilt = 0.25 + 0.75 * u
        c = q.mean(axis=0)
        qq = (q - c) * np.array([1.0, tilt])
        ca, sa = math.cos(rot), math.sin(rot)
        qq = np.stack([qq[:, 0] * ca - qq[:, 1] * sa, qq[:, 0] * sa + qq[:, 1] * ca], 1)
        q = qq + c + np.array([sway, -drop])
    return q


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.array(A))
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


CURL = dict(R=330.0, lift=34.0)        # the near right corner curls up off the floor (page px, screen px)


def _edge_noise(n, key, amp):
    r = vnoise.rng("pageedge", key)
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    for k, a in ((3, 1.0), (7, 0.5), (17, 0.25)):
        out += a * np.sin(2 * math.pi * (k * x + r.uniform(0, 1)))
    return amp * (0.6 + 0.4 * out / 1.75)


_EDGES = {}


def _edges():
    if "e" not in _EDGES:
        _EDGES["e"] = (_edge_noise(PH, "l", 5.0), _edge_noise(PH, "r", 5.0), _edge_noise(PW, "t", 5.0), _edge_noise(PW, "b", 5.0))
    return _EDGES["e"]


def curl_lift(u, v):
    k = 1.0 - ((PW - u) + (PH - v)) / CURL["R"]
    return CURL["lift"] * np.clip(k, 0, 1) ** 1.8


def warp_to_screen(img, quad, H=T.H, W=T.W, curl=True):
    """Map a page-space image (PH, PW[, c]) onto a screen quad, with the corner curl and
    slightly irregular paper edges. Returns ((image, inside), bbox)."""
    src = np.array([(0, 0), (PW, 0), (PW, PH), (0, PH)], np.float64)
    Hm = homography(np.asarray(quad), src)   # screen -> page
    x0 = int(max(0, math.floor(quad[:, 0].min()) - 2))
    x1 = int(min(W, math.ceil(quad[:, 0].max()) + 2))
    y0 = int(max(0, math.floor(quad[:, 1].min()) - CURL["lift"] - 4))
    y1 = int(min(H, math.ceil(quad[:, 1].max()) + 2))
    if x1 <= x0 or y1 <= y0:
        return None, None
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)

    def uv(yy):
        X = Hm[0, 0] * xs + Hm[0, 1] * yy + Hm[0, 2]
        Y = Hm[1, 0] * xs + Hm[1, 1] * yy + Hm[1, 2]
        Z = Hm[2, 0] * xs + Hm[2, 1] * yy + Hm[2, 2]
        return X / Z, Y / Z
    u, v = uv(ys)
    lift = np.zeros_like(u)
    if curl:
        for _ in range(3):
            lift = curl_lift(np.clip(u, 0, PW), np.clip(v, 0, PH))
            u, v = uv(ys + lift)
    el, er, et, eb = _edges()
    vi = np.clip(v, 0, PH - 1).astype(int)
    ui = np.clip(u, 0, PW - 1).astype(int)
    inside = (u >= el[vi]) & (u < PW - er[vi]) & (v >= et[ui]) & (v < PH - eb[ui])
    if img.ndim == 2:
        out = ndimage.map_coordinates(img, [v, u], order=1, mode="nearest")
    else:
        out = np.stack([ndimage.map_coordinates(img[..., k], [v, u], order=1, mode="nearest")
                        for k in range(img.shape[2])], -1)
    return (out, inside, lift), (y0, y1, x0, x1)


_PAGE = {}


def _page_image():
    if "img" not in _PAGE:
        import placeholder
        tex = placeholder.page_texture()
        # a page-space crop of the paper texture, a little lighter at the centre
        img = tex[40:40 + 1000, 160:160 + 1600].copy() * 1.03
        ys, xs = np.mgrid[0:PH, 0:PW].astype(np.float32)
        curl = 1.0 - 0.06 * ((xs / PW - 0.5) ** 2 * 4) - 0.05 * np.clip(1 - ys / 80, 0, 1)
        img *= curl[..., None]
        _PAGE["img"] = np.clip(img, 0, 1).astype(np.float32)
    return _PAGE["img"]


def draw_page(layer, cut_id, f):
    """The blank page lying on the balcony (9a: landing at the fox's paws; 9b: close).
    Paper, not a card: a corner curling off the floor, soft edges, a contact shadow."""
    q = page_quad(cut_id, f)
    landed = cut_id == "9b" or f / T.FPS >= T.CUES["page_land"]
    m = raster.poly_mask(q, T.H, T.W).astype(np.float32)
    if landed:
        tight = ndimage.gaussian_filter(raster.shift(m > 0.5, 2, 3).astype(np.float32), 2.5) * 0.30
        broad = ndimage.gaussian_filter(raster.shift(m > 0.5, 9, 13).astype(np.float32), 13) * 0.20
        sh = np.clip(tight + broad, 0, 0.45)
    else:
        sh = ndimage.gaussian_filter(raster.shift(m > 0.5, 30, 60).astype(np.float32), 22) * 0.18
    layer[..., :3] *= (1 - sh[..., None])
    layer[..., :3] += sh[..., None] * np.array([0.30, 0.20, 0.28], np.float32)
    layer[..., 3] = layer[..., 3] * (1 - sh) + sh
    res, bb = warp_to_screen(_page_image(), q, curl=landed)
    if res is None:
        return
    (img, inside, lift), (y0, y1, x0, x1) = res, bb
    a = ndimage.uniform_filter(inside.astype(np.float32), 2)
    # the lifted corner catches a little more sky; a thin shade where it leaves the floor
    k = np.clip(lift / CURL["lift"], 0, 1)[..., None]
    img = img * (1 + 0.07 * k) - 0.05 * (k * (1 - k) * 4)
    sub = layer[y0:y1, x0:x1]
    sub *= (1 - a[..., None])
    sub[..., :3] += np.clip(img, 0, 1) * a[..., None]
    sub[..., 3] += a


def draw_title(layer, f):
    t = f / T.FPS
    if t < T.CUES["title_ink_start"]:
        return
    cov, den, seal = ink_page(t)
    q = page_quad("9b", f)
    col = INK_LIGHT[None, None, :] * (1 - np.clip(den, 0, 1)[..., None]) + INK_DARK[None, None, :] * np.clip(den, 0, 1)[..., None]
    img = np.concatenate([col * cov[..., None], cov[..., None]], -1)
    if seal is not None:
        sa = seal[..., None]
        img = img * (1 - sa) + np.concatenate([SEAL * sa, sa], -1)
    res, bb = warp_to_screen(img, q)
    if res is None:
        return
    (wimg, inside, _lift), (y0, y1, x0, x1) = res, bb
    wimg *= inside[..., None]
    sub = layer[y0:y1, x0:x1]
    a = np.clip(wimg[..., 3:4], 0, 1)
    sub *= (1 - a)
    sub += wimg


def events():
    sched = schedule()
    return {"title_strokes": [[round(a, 3), round(b, 3)] for (_, _, a, b, ch) in sched
                              if a < T.CUES["title_ink_end"] + 0.01 and ch != " "][:40],
            "line_strokes_start": round(min(a for (_, _, a, b, ch) in sched if a >= T.CUES["line_ink_start"] - 0.01), 3),
            "hanko": T.CUES["hanko"]}
