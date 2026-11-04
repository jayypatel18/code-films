"""The paper lantern: bamboo ribs, paper globe, wooden caps, a warm flame inside.

paint(cv, ...) draws it into a canvas the fox is drawing (held in the mouth);
draw(...) renders it alone as a tile (set down on the floor). Both return a glow
spec (x, y, radius, rgb) in screen px for the compositor.
"""

import math

import numpy as np

from engine import geom, ink, raster
from engine.canvas import Canvas
from engine.light import hexrgb

INK = hexrgb("#3B2618")
PAPER_OUT = hexrgb("#F2A24C")
PAPER_IN = hexrgb("#FFD48A")
CORE = hexrgb("#FFF4D6")
RIB = hexrgb("#C0612A")
CAP = hexrgb("#4A3426")


def _oval(c, rx, ry, ang=0.0, n=48):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ca, sa = math.cos(ang), math.sin(ang)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    return np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], 1)


def body_points(C, rx, ry, swing=0.0, n=72):
    """Globe outline around centre C, tilted by swing (radians) about its top."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([rx * np.cos(t), ry * np.sin(t) * (1 - 0.08 * np.cos(t) ** 2)], 1)
    ca, sa = math.cos(swing), math.sin(swing)
    pivot = np.array([0.0, -ry * 1.6])
    q = pts - pivot
    q = np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1) + pivot
    return q + np.asarray(C[:2])


def paint(cv, C, rx, ry, lw, stick=None, cord=None, swing=0.0, flame_lean=0.0, flicker=0.0):
    """Draw the lantern into canvas cv (canvas coords). C is the globe centre."""
    C = np.asarray(C[:2], dtype=np.float64)
    body = body_points(C, rx, ry, swing)
    ca, sa = math.cos(swing), math.sin(swing)

    def rot(p):
        p = np.asarray(p, dtype=np.float64) - C - np.array([0.0, -ry * 1.6])
        return np.array([p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca]) + C + np.array([0.0, -ry * 1.6])

    allp = [body]
    if stick is not None:
        allp.append(np.asarray(stick)[:, :2])
    if cord is not None:
        allp.append(np.asarray(cord)[:, :2])
    box = cv.bbox(allp, pad=int(lw * 4) + 4)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    if stick is not None:
        sw = max(lw * 1.8, 0.035 * rx * 2.5)
        buf = cv.stroke_buffer(box)
        ink.ink_designed(buf, np.asarray(stick)[:, :2], sw, profile=lambda u: np.ones_like(u) * 0.95, ox=x0, oy=y0)
        cv.paint(box, buf, INK)
        buf = cv.stroke_buffer(box)
        ink.ink_designed(buf, np.asarray(stick)[:, :2], sw * 0.55, profile=lambda u: np.ones_like(u) * 0.9, ox=x0, oy=y0)
        cv.paint(box, buf, hexrgb("#9C8450"))
    if cord is not None:
        buf = cv.stroke_buffer(box)
        cd = np.asarray(cord)[:, :2].copy()
        cd[-1] = rot(C + np.array([0.0, -ry * 1.02]))
        ink.ink_designed(buf, cd, lw * 0.45, profile=lambda u: np.ones_like(u), ox=x0, oy=y0)
        cv.paint(box, buf, INK)
    _, m = cv.mask(body, box)
    cv.paint(box, m.astype(np.float32), PAPER_OUT)
    inner = body_points(C + np.array([0.04 * rx, 0.05 * ry]), rx * 0.80, ry * 0.80, swing)
    _, mi = cv.mask(inner, box)
    cv.paint(box, (m & mi).astype(np.float32), PAPER_IN)
    # the flame shows through the paper as a hot core; it leans when the air moves
    fc = rot(C + np.array([flame_lean * rx * 0.35, 0.18 * ry]))
    core = _oval(fc, rx * (0.30 + 0.03 * flicker), ry * (0.36 + 0.04 * flicker), swing + flame_lean * 0.6)
    _, mc = cv.mask(core, box)
    cv.paint(box, (m & mc).astype(np.float32), CORE)
    buf = cv.stroke_buffer(box)
    for k in range(1, 6):
        yy = -1 + 2 * k / 6.0
        w = math.sqrt(max(0.0, 1 - yy * yy))
        xs = np.linspace(-w, w, 24)
        pts = np.stack([C[0] + xs * rx, C[1] + yy * ry + (1 - xs ** 2 / max(w * w, 1e-6)) * 0.06 * ry * w], 1)
        pts = np.array([rot(p) for p in pts])
        ink.ink_designed(buf, pts, lw * 0.35, profile=lambda u: np.sin(np.clip(u, 0, 1) * math.pi) ** 0.5, ox=x0, oy=y0)
    cv.paint(box, buf * m * 0.55, RIB)
    pen = ink.Pen(width=lw * 0.5, min_len=900, max_len=2000, break_lit=0, wobble=0)
    for sgn in (-1, 1):
        cap = _oval(rot(C + np.array([0, sgn * ry * 0.98])), rx * 0.42, ry * 0.11, swing, 32)
        _, mcap = cv.mask(cap, box)
        cv.paint(box, mcap.astype(np.float32), CAP)
        buf = cv.stroke_buffer(box)
        ink.ink_polyline(buf, cap, pen, ox=x0, oy=y0, closed=True)
        cv.paint(box, buf, INK)
    buf = cv.stroke_buffer(box)
    ink.ink_polyline(buf, body, ink.Pen(width=lw * 0.7, break_lit=0.0, min_len=200, max_len=600), ox=x0, oy=y0, closed=True)
    cv.paint(box, buf, hexrgb("#6A3418"))
    sx = C[0] / cv.ss + cv.x0
    sy = C[1] / cv.ss + cv.y0
    return (sx, sy, rx / cv.ss, (1.0, 0.72, 0.40))


def draw(x, y, size, swing=0.0, flame_lean=0.0, flicker=0.0, ss=3, line=None):
    """Lantern standing alone (set down). (x, y) is where its bottom touches the floor,
    size is the globe height in screen px. Returns (tile, x0, y0, glow)."""
    ry = size * 0.5
    rx = ry * 0.84
    cx, cy = x, y - ry * 1.1
    pad = int(size * 1.6) + 8
    cv = Canvas(int(cx - pad), int(cy - pad * 1.4), 2 * pad, int(pad * 2.4), ss)
    C = cv.to_c(np.array([[cx, cy]]))[0]
    lw = (line if line else 0.22 * math.sqrt(size / 0.9) + 0.3) * ss
    # a short bamboo loop handle standing up from the top cap
    top = C + np.array([0.0, -ry * ss * 1.0])
    handle = np.array([top + [-rx * ss * 0.25, 0], top + [-rx * ss * 0.22, -ry * ss * 0.55],
                       top + [rx * ss * 0.22, -ry * ss * 0.55], top + [rx * ss * 0.25, 0]])
    handle = geom.catmull_rom(handle, 8)
    glow = paint(cv, C, rx * ss, ry * ss, lw, stick=handle, swing=swing, flame_lean=flame_lean, flicker=flicker)
    return cv.resolve(), cv.x0, cv.y0, glow
