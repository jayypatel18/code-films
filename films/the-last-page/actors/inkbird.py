"""The ink bird (cut 4b): one line of ink runs out from under the paw and draws a
tiny sketched bird in one confident brushstroke. The bird shivers, peels off the
page (a little paper cut-out still carrying its ink) and flaps away.
"""

import math

import numpy as np
from scipy import ndimage

import timeline as T
from engine import brush, geom, raster, vnoise
from engine import noise as wnoise

INK_DARK = np.array([0.09, 0.06, 0.05], np.float32)
INK_LIGHT = np.array([0.26, 0.20, 0.16], np.float32)
PAPER = np.array([0.90, 0.84, 0.70], np.float32)

# the bird in local units (width about 1), facing right, y down
BODY = [(-0.55, 0.12), (-0.64, 0.04), (-0.47, 0.10), (-0.30, 0.17), (-0.05, 0.21), (0.20, 0.13), (0.35, 0.01),
        (0.42, -0.11), (0.58, -0.12), (0.45, -0.17), (0.35, -0.22), (0.24, -0.17), (0.14, -0.10)]
WING = [(0.05, -0.17), (-0.06, -0.50), (-0.14, -0.31), (-0.09, -0.11)]
TAIL = [(-0.30, -0.04), (-0.55, 0.02), (-0.68, -0.05)]
SHOULDER = np.array([0.05, -0.12])
EYE = (0.33, -0.13)
# a few quick sketch flicks after the main stroke: two feathers on the wing, one in the tail
EXTRA_WING = [[(-0.02, -0.33), (-0.06, -0.19)], [(0.035, -0.29), (0.005, -0.17)]]
EXTRA_TAIL = [[(-0.48, 0.05), (-0.60, 0.13)]]


def _rot_wing(pts, wing_angle):
    q = np.array(pts) - SHOULDER
    ca, sa = math.cos(wing_angle), math.sin(wing_angle)
    return np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1) + SHOULDER


def _extras(w, place, flap=0.0):
    out = []
    for st in EXTRA_WING:
        out.append(place(_rot_wing(st, flap)))
    for st in EXTRA_TAIL:
        out.append(place(np.array(st)))
    return out


def _bird_points(wing_angle=0.0):
    wing = np.array(WING) - SHOULDER
    ca, sa = math.cos(wing_angle), math.sin(wing_angle)
    wing = np.stack([wing[:, 0] * ca - wing[:, 1] * sa, wing[:, 0] * sa + wing[:, 1] * ca], 1) + SHOULDER
    return np.vstack([np.array(BODY), wing, np.array(TAIL)])


def _lead_in():
    a = T.ANCHORS["4b"]
    px, py = a["paw_press_screen"]
    bx, by = a["ink_bird_centre_screen"]
    w = a["ink_bird_width_screen"]
    tail = np.array([bx, by]) + np.array(BODY[0]) * w
    return np.array([(px + 20, py + 18), (px + 120, py + 10), (px + 230, py - 20),
                     ((px + tail[0]) / 2 + 40, (py + tail[1]) / 2 - 10), tuple(tail)])


def _path(t):
    """Screen polyline of the whole stroke (lead-in plus bird) and where the bird part starts."""
    a = T.ANCHORS["4b"]
    bx, by = a["ink_bird_centre_screen"]
    w = a["ink_bird_width_screen"]
    lead = geom.catmull_rom(_lead_in(), 8)
    bird = geom.catmull_rom(np.array([bx, by]) + _bird_points() * w, 6)
    return np.vstack([lead, bird[1:]]), len(lead)


def _radius(P, k_bird, w):
    s = geom.arclen(P)
    u = s / s[-1]
    r = w * 0.022 * brush.pressure(u, land=0.04, lift=0.12, swell=0.25)
    # a confident line: thicker as it turns into the bird, light on the lead-in
    r[:k_bird] *= np.linspace(0.75, 1.0, k_bird)
    return r


def _bird_transform(t, w):
    """Screen centre, scale, rotation and wing angle of the free bird at time t."""
    a = T.ANCHORS["4b"]
    c0 = np.array(a["ink_bird_centre_screen"], np.float64)
    ex = np.array(a["bird_exit_screen"], np.float64)
    c = T.CUES
    if t < c["bird_peel"]:
        return c0, 1.0, 0.0, 0.0, 0.0
    if t < c["bird_flap_start"]:
        u = (t - c["bird_peel"]) / (c["bird_flap_start"] - c["bird_peel"])
        return c0 + np.array([-4, -10]) * u, 1.0 + 0.08 * u, -0.08 * u, -0.25 * u, u
    u = min(1.0, (t - c["bird_flap_start"]) / 1.0)
    e = u * u * (1.6 - 0.6 * u)
    mid = c0 + np.array([260, -60])
    pos = (1 - e) ** 2 * (c0 + np.array([-4, -10])) + 2 * (1 - e) * e * mid + e * e * ex
    flap = 0.95 * math.sin(2 * math.pi * 3.6 * (t - c["bird_flap_start"]))
    return pos, 1.08 + 0.35 * e, -0.08 - 0.35 * e, flap, 1.0


def draw(layer, f):
    t = f / T.FPS
    c = T.CUES
    if t < c["ink_start"]:
        return
    a = T.ANCHORS["4b"]
    w = float(a["ink_bird_width_screen"])
    H, W = layer.shape[:2]
    cov = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    P, kb = _path(t)
    R = _radius(P, kb, w)
    if t < c["bird_peel"]:
        u = min(1.0, (t - c["ink_start"]) / (c["ink_end"] - c["ink_start"]))
        u = u * u * (3 - 2 * u) if u < 1 else 1.0
        u = 0.15 * u + 0.85 * (1 - (1 - u) ** 1.8)
        Pd = P
        if t >= c["bird_shiver"]:
            # the drawing shivers before it comes alive (on twos)
            fh = f - f % 2
            r = vnoise.rng("shiver", fh)
            jit = r.normal(0, 2.2, P.shape)
            jit[:kb] = 0
            Pd = P + jit
        brush.stroke(cov, den, Pd, R, u, seed=7, dry0=0.92, dryk=0.9, pool=0.45, lanes=6.0)
        if u >= 1.0:
            c0 = np.array(a["ink_bird_centre_screen"], np.float64)
            ex, ey = c0 + np.array(EYE) * w
            brush.stroke(cov, den, [(ex, ey), (ex + 2.0, ey + 2.0)], [w * 0.036, w * 0.032], 1.0, seed=9, dry0=1.0)
            ue = min(1.0, max(0.0, (t - c["ink_end"]) / 0.12))
            for k, st in enumerate(_extras(w, lambda q: c0 + q * w)):
                if ue > 0:
                    brush.stroke(cov, den, st, [w * 0.016, w * 0.008], ue, seed=20 + k, dry0=0.6, pool=0.3, lanes=4.0)
    else:
        # the lead-in stays on the page; the bird peels off as a little paper cut-out
        brush.stroke(cov, den, P[:kb + 2], R[:kb + 2], 1.0, seed=7, dry0=0.92, pool=0.45, lanes=6.0)
        pos, sc, rot, flap, lift = _bird_transform(t, w)
        fh = f - f % 2
        pos, sc, rot, flap, lift = _bird_transform(fh / T.FPS, w)
        pts = _bird_points(flap)
        ca, sa = math.cos(rot), math.sin(rot)
        pts = np.stack([pts[:, 0] * ca - pts[:, 1] * sa, pts[:, 0] * sa + pts[:, 1] * ca], 1)
        bird = pos + pts * w * sc
        loop = bird[:len(BODY)]
        wing = bird[len(BODY):len(BODY) + len(WING)]
        # shadow on the page, then the paper body, then its ink
        sh = raster.poly_mask(loop + np.array([12, 18]) * lift, H, W).astype(np.float32)
        sh = np.maximum(sh, raster.poly_mask(np.vstack([wing, loop[-1:]]) + np.array([12, 18]) * lift, H, W))
        sh = ndimage.gaussian_filter(sh, 6) * 0.30 * lift * (1.0 if t < c["bird_flap_start"] + 0.3 else
                                                              max(0.0, 1 - (t - c["bird_flap_start"] - 0.3) * 2))
        layer[..., :3] *= (1 - sh[..., None])
        layer[..., :3] += sh[..., None] * np.array([0.22, 0.14, 0.12], np.float32)
        layer[..., 3] = layer[..., 3] * (1 - sh) + sh
        pm = raster.poly_mask(loop, H, W).astype(np.float32)
        pm = np.maximum(pm, raster.poly_mask(np.vstack([wing, (wing[0] + wing[-1]) / 2]), H, W).astype(np.float32))
        pm = ndimage.uniform_filter(pm, 2) * lift
        layer[..., :3] = layer[..., :3] * (1 - pm[..., None]) + PAPER * pm[..., None]
        layer[..., 3] = layer[..., 3] * (1 - pm) + pm
        Bd = geom.catmull_rom(bird, 6)
        Rb = w * sc * 0.022 * brush.pressure(np.linspace(0, 1, len(Bd)), land=0.05, lift=0.12, swell=0.25)
        brush.stroke(cov, den, Bd, Rb, 1.0, seed=8, dry0=0.92, pool=0.4, lanes=6.0)
        ex, ey = pos + np.array([EYE[0] * ca - EYE[1] * sa, EYE[0] * sa + EYE[1] * ca]) * w * sc
        brush.stroke(cov, den, [(ex, ey), (ex + 2.0, ey + 2.0)], [w * sc * 0.036, w * sc * 0.032], 1.0, seed=9, dry0=1.0)

        def place(q):
            q = np.asarray(q, np.float64)
            return pos + np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1) * w * sc
        for k, st in enumerate(_extras(w, place, flap)):
            brush.stroke(cov, den, st, [w * sc * 0.016, w * sc * 0.008], 1.0, seed=20 + k, dry0=0.6, pool=0.3, lanes=4.0)
    fibre = _fibre(H, W)
    cov = brush.bleed(cov, fibre, 0.7, 0.7)
    d = np.clip(den, 0, 1)[..., None]
    col = INK_LIGHT * (1 - d) + INK_DARK * d
    layer[..., :3] = layer[..., :3] * (1 - cov[..., None]) + col * cov[..., None]
    layer[..., 3] = layer[..., 3] * (1 - cov) + cov


_F = {}


def _fibre(H, W):
    if "f" not in _F:
        _F["f"] = wnoise.paper_grain(H, W, "inkbird-fibre")
    return _F["f"]


def events():
    c = T.CUES
    return {"ink_start": c["ink_start"], "ink_end": c["ink_end"], "bird_shiver": c["bird_shiver"],
            "bird_peel": c["bird_peel"], "bird_flap_start": c["bird_flap_start"],
            "bird_wingbeats": [round(c["bird_flap_start"] + k / 3.6, 3) for k in range(4)]}
