"""Lead's placeholder world, used for any cut the world department has not painted yet.

Where it fits, the placeholders reuse the world department's painted hall plate
(build/plates/hall_bg.npy) through another camera, so the rough cut already
sits in the right library. Everything else is a simple painted stand in: sky
gradients, a stone ledge, a lectern and an open book, a page close up. Each
returns the same dict as world.render: bg (premultiplied RGBA), fg, glow.
"""

import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE, os.path.join(HERE, "world")):
    if p not in sys.path:
        sys.path.insert(0, p)

import timeline as T  # noqa: E402
from engine import noise as wnoise  # noqa: E402

W, H = T.W, T.H
_MEMO = {}


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def _grid():
    if "grid" not in _MEMO:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        _MEMO["grid"] = (xs, ys)
    return _MEMO["grid"]


def _paper():
    if "paper" not in _MEMO:
        _MEMO["paper"] = wnoise.paper_grain(H, W, "a1-placeholder-paper")
    return _MEMO["paper"]


def _opaque(rgb):
    out = np.ones((H, W, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0, 1)
    return out


def _hall_plate():
    try:
        import common as WC  # world department helpers
        return WC, WC.load_plate("hall_bg")
    except Exception:
        return None, None


def _hall_view(cam, blur=0.0, dim=1.0):
    WC, plate = _hall_plate()
    if plate is None:
        return _opaque(np.zeros((H, W, 3), np.float32) + np.array([0.06, 0.055, 0.11], np.float32))
    out = WC.sample(plate, cam)
    if blur > 0:
        out[..., :3] = ndimage.gaussian_filter(out[..., :3], (blur, blur, 0))
    out[..., :3] *= dim
    out[..., 3] = 1.0
    return out


def _warm_spill(x, y, r, strength=0.22, rgb=(1.0, 0.62, 0.30)):
    xs, ys = _grid()
    d2 = ((xs - x) ** 2 + (ys - y) ** 2) / (r * r)
    g = np.exp(-d2 * 1.2) * 0.7 + np.exp(-d2 * 0.25) * 0.3
    return ((g * strength)[..., None] * np.asarray(rgb, np.float32)).astype(np.float32)


# ---------------------------------------------------------------- cuts

def cut3(f):
    """Close-up: the bookcases behind the fox at its eye height, a longer lens, soft."""
    if "c3" not in _MEMO:
        _MEMO["c3"] = _hall_view((1.45, 1240.0, 600.0), blur=2.6, dim=1.2)
    lx, ly = T.ANCHORS["3"]["lantern_glow_from_screen"]
    return {"bg": _MEMO["c3"].copy(), "fg": None, "glow": _warm_spill(lx, ly, 560, 0.18)}


def _lectern_layer():
    """Placeholder stone lectern and open blank book in cut 4a plate space (with margin)."""
    if "lect" in _MEMO:
        return _MEMO["lect"]
    from engine import raster
    a = T.ANCHORS["4a"]
    pw, ph = T.plate_size("4a")
    M = 96
    canvas = np.zeros((ph + 2 * M, pw + 2 * M, 4), np.float32)
    lx = a["lectern_centre_x"]
    top = a["book_top_y"]
    floor = a["floor_y"]
    x0, x1 = a["book_span_x"]

    def poly(pts, col):
        P = np.array(pts, np.float64) + M
        m = raster.poly_mask(P, canvas.shape[0], canvas.shape[1]).astype(np.float32)
        m = ndimage.uniform_filter(m, 2)
        canvas[..., :3] = canvas[..., :3] * (1 - m[..., None]) + np.asarray(col, np.float32) * m[..., None]
        canvas[..., 3] = canvas[..., 3] * (1 - m) + m
    stone = _hex("#3A3650")
    lit = _hex("#56526E")
    poly([(lx - 330, floor), (lx + 330, floor), (lx + 300, floor - 70), (lx - 300, floor - 70)], stone * 0.9)
    poly([(lx - 250, floor - 70), (lx + 250, floor - 70), (lx + 230, floor - 140), (lx - 230, floor - 140)], stone)
    poly([(lx - 70, floor - 140), (lx + 70, floor - 140), (lx + 50, top + 120), (lx - 50, top + 120)], stone)
    poly([(lx + 20, floor - 140), (lx + 70, floor - 140), (lx + 50, top + 120), (lx + 15, top + 120)], lit)
    poly([(x0 - 30, top + 60), (x1 + 30, top + 60), (x1 + 10, top + 130), (x0 - 10, top + 130)], lit * 0.9)
    cov = _hex("#5A2A2E")
    poly([(x0 - 14, top + 46), (x1 + 14, top + 46), (x1 + 8, top + 74), (x0 - 8, top + 74)], cov)
    gx = 0.5 * (x0 + x1)
    tt = np.linspace(0, 1, 18)
    left = [(x0 + (gx - x0) * t, top + 8 - 22 * math.sin(math.pi * t) + 30 * t) for t in tt]
    right = [(gx + (x1 - gx) * t, top + 38 - 22 * math.sin(math.pi * t) - 30 * t) for t in tt]
    poly([(x0, top + 50)] + left + [(gx, top + 54)], _hex("#D9CBA6"))
    poly([(gx, top + 54)] + right + [(x1, top + 50)], _hex("#CDBE98"))
    _MEMO["lect"] = canvas
    return canvas


def cut4a(f):
    cam = T.camera("4a", f)
    bg = _hall_view(cam)
    WC, _ = _hall_plate()
    if WC is not None:
        layer = WC.sample(_lectern_layer(), cam)
        bg[..., :3] = bg[..., :3] * (1 - layer[..., 3:4]) + layer[..., :3]
    lx, ly = T.plate_to_screen("4a", f, *T.ANCHORS["4a"]["lantern"])
    return {"bg": bg, "fg": None, "glow": _warm_spill(lx, ly - 30, 260, 0.20)}


def page_texture():
    """A close view of yellowed, blank paper (used by the 4b placeholder)."""
    if "page" not in _MEMO:
        xs, ys = _grid()
        paper = _paper()
        rgb = np.zeros((H, W, 3), np.float32) + _hex("#E2D3AE")
        st = wnoise.fbm(H, W, 220, "page-stain", octaves=4)
        rgb *= (0.94 + 0.10 * st)[..., None]
        rgb = rgb * (0.97 + 0.06 * (paper[..., None] - 0.5))
        _MEMO["page"] = rgb
    return _MEMO["page"]


def cut4b(f):
    if "c4b" not in _MEMO:
        xs, ys = _grid()
        rgb = page_texture().copy()
        g = np.clip((xs - 40) / 260.0, 0, 1)
        rgb *= (0.55 + 0.45 * g ** 0.6)[..., None]
        warm = np.exp(-(((xs - 200) / 1300) ** 2 + ((ys - 1200) / 900) ** 2))
        rgb = rgb * (0.70 + 0.35 * warm[..., None]) * np.array([1.0, 0.96, 0.90], np.float32)
        rgb += np.exp(-(((xs - 1900) / 700) ** 2 + ((ys + 100) / 500) ** 2))[..., None] * np.array([0.02, 0.03, 0.08])
        _MEMO["c4b"] = _opaque(rgb)
    return {"bg": _MEMO["c4b"].copy(), "fg": None, "glow": None}


def cut5ab(cut_id, f):
    key = "c" + cut_id
    if key not in _MEMO:
        cam = (2.4, 1060.0, 520.0) if cut_id == "5a" else (1.9, 1340.0, 560.0)
        _MEMO[key] = _hall_view(cam, blur=0.6, dim=1.3)
    return {"bg": _MEMO[key].copy(), "fg": None, "glow": None}


def cut5c(f):
    if "c5c" not in _MEMO:
        _MEMO["c5c"] = _hall_view((1.35, 1000.0, 760.0), blur=2.2, dim=1.1)
    lx, ly = T.ANCHORS["5c"]["lantern_screen"]
    return {"bg": _MEMO["c5c"].copy(), "fg": None, "glow": _warm_spill(lx, ly, 380, 0.26)}


def cut6(cut_id, f):
    xs, ys = _grid()
    if cut_id == "6a":
        return {"bg": _hall_view(T.camera("6a", f)), "fg": None, "glow": None}
    if cut_id == "6b":
        if "c6b" not in _MEMO:
            r = np.hypot(xs - W / 2, ys - H / 2 + 60)
            t = np.clip(r / 900.0, 0, 1)
            rgb = (1 - t)[..., None] ** 2.2 * np.array([0.30, 0.30, 0.45]) + np.array([0.04, 0.035, 0.07])
            rgb *= (0.85 + 0.15 * (0.5 + 0.5 * np.cos(r / 28.0)))[..., None]
            _MEMO["c6b"] = _opaque(rgb)
        return {"bg": _MEMO["c6b"].copy(), "fg": None, "glow": None}
    t = (f / T.FPS - T.CUES["shutters_break"])
    k = float(np.clip(t / 0.4, 0, 1))
    r = np.hypot(xs - W / 2, ys - 80)
    rgb = np.array([0.05, 0.05, 0.09]) + np.clip(1 - r / 1100, 0, 1)[..., None] ** 2 * np.array([0.16, 0.15, 0.22])
    shaft = np.exp(-((xs - W / 2) / (220 + 200 * k)) ** 2) * np.clip(1.1 - ys / H, 0, 1)
    glow = (shaft * k * 0.9)[..., None] * np.array([1.0, 0.80, 0.45], np.float32)
    return {"bg": _opaque(rgb), "fg": None, "glow": glow.astype(np.float32)}


def _sky(f, cut_id):
    zoom, cx, cy = T.camera(cut_id, f)
    xs, ys = _grid()
    py = (ys - H / 2) / zoom + cy
    px = (xs - W / 2) / zoom + cx
    sea = T.ANCHORS["7"]["cloud_sea_top_y"] if cut_id == "7" else 820
    u = np.clip(py / sea, 0, 1.0)
    top, mid, low = _hex("#6E8FC8"), _hex("#F2B98A"), _hex("#FFE2B0")
    rgb = top * (1 - u[..., None]) ** 1.5 + mid * (u[..., None] * (1 - u[..., None])) * 2 + low * u[..., None] ** 3
    t = (f / T.FPS - 39.0) / 5.0
    if cut_id == "7":
        sx, sy = 1560, 860 - 140 * min(max(t, 0), 1)
    else:
        sx, sy = 1700, 640
    d = np.hypot(px - sx, py - sy)
    rgb += (np.exp(-d / 90.0) * 0.6 + np.exp(-d / 420.0) * 0.25)[..., None] * np.array([1.0, 0.85, 0.6])
    if "skyn" not in _MEMO:
        _MEMO["skyn"] = wnoise.fbm(H, W, 160, "sky-cloud", octaves=4)
    n = _MEMO["skyn"]
    edge = sea + 30 * np.sin(px / 140.0) + 80 * (n - 0.5)
    cloud = np.clip((py - edge) / 20.0, 0, 1)
    ccol = _hex("#F6D2B4") * (0.9 + 0.15 * n[..., None]) * (1 - 0.25 * np.clip((py - edge) / 400, 0, 1))[..., None]
    rgb = rgb * (1 - cloud[..., None]) + ccol * cloud[..., None]
    return rgb, px, py


def cut7(f):
    rgb, px, py = _sky(f, "7")
    a = T.ANCHORS["7"]
    lx0, lx1 = a["ledge_x"]
    ly = a["ledge_top_y"]
    ledge = (px > lx0 - 300) & (px < lx1) & (py > ly)
    rgb = np.where(ledge[..., None], np.where((py < ly + 26)[..., None], _hex("#9C8496"), _hex("#6A5468")), rgb)
    return {"bg": _opaque(rgb), "fg": None, "glow": None}


def cut8(f):
    rgb, px, py = _sky(f, "8")
    bal = (px < 520 - (py - 1240) * 0.4) & (py > 1240)
    rgb = np.where(bal[..., None], _hex("#7A6478"), rgb)
    return {"bg": _opaque(rgb), "fg": None, "glow": None}


def cut9(cut_id, f):
    """The balcony in morning light: warm stone floor, a low wall, sky beyond."""
    key = "c9"
    if key not in _MEMO:
        xs, ys = _grid()
        rgb = np.zeros((H, W, 3), np.float32) + _hex("#F5D7B2") * (1 - ys[..., None] / H * 0.3)
        n = wnoise.fbm(H, W, 90, "balcony", octaves=4)
        wall = (ys > 330) & (ys < 470)
        rgb = np.where(wall[..., None], _hex("#B89A9E") * (0.92 + 0.12 * n[..., None]), rgb)
        floor = ys >= 470
        u = (xs - W / 2) / (ys - 300 + 1e-3)
        joints = (np.abs(((u * 3.0) % 1.0) - 0.5) < 0.02) | \
                 (np.abs(((np.log(np.maximum(ys - 300, 1)) * 6.0) % 1.0) - 0.5) < 0.03)
        fl = _hex("#D7B9A8") * (0.90 + 0.12 * n[..., None])
        fl = np.where(joints[..., None], fl * 0.86, fl)
        rgb = np.where(floor[..., None], fl, rgb)
        rgb *= (0.92 + 0.12 * np.clip(xs / W, 0, 1))[..., None]
        _MEMO[key] = _opaque(rgb)
    return {"bg": _MEMO[key].copy(), "fg": None, "glow": None}


def layers(cut_id, f):
    if cut_id == "3":
        return cut3(f)
    if cut_id == "4a":
        return cut4a(f)
    if cut_id == "4b":
        return cut4b(f)
    if cut_id in ("5a", "5b"):
        return cut5ab(cut_id, f)
    if cut_id == "5c":
        return cut5c(f)
    if cut_id in ("6a", "6b", "6c"):
        return cut6(cut_id, f)
    if cut_id == "7":
        return cut7(f)
    if cut_id == "8":
        return cut8(f)
    if cut_id in ("9a", "9b"):
        return cut9(cut_id, f)
    return {"bg": _opaque(np.zeros((H, W, 3), np.float32) + 0.08), "fg": None, "glow": None}


def plate_cut3(lantern_xy=None, refresh=False):
    """Kept for the Stage 1 composite command."""
    return cut3(0)["bg"]
