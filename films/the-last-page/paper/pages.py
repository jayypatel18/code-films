"""The page atlas: every page of every book, drawn in code.

Each page is a multiplicative map over the paper colour (white = bare paper):
faded ink handwriting made of pen strokes (a looping cursive oscillator,
thick on the downstrokes like a nib), printed text blocks (rows of tiny
letter strokes with ascenders and descenders), headings and drop caps in red
rubric, small old illustrations drawn as brush and pen marks (moons, trees,
ships, mountains, birds, fish, botanical studies, towers, star charts, maps,
music staves), and the age of the paper (browned edges, foxing spots, tea
rings, mottling, fibres). No fonts anywhere: every mark is a stroke.

Pages are drawn at twice the atlas size and filtered down, then a mip pyramid
is built for the rasterizer. The atlas is cached in build/paper/.
"""

import math
import os

import numpy as np
from scipy import ndimage

from engine import raster as eraster
from engine.noise import rng, value_noise, paper_grain

PW, PH = 320, 448          # atlas page size (texels): width along the text lines
NPAGES = 72
BLANK = 0                  # page 0 is the blank page (aged paper, no ink)
VERSION = 3

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "build", "paper")

# multiplicative ink colours over paper (1 = no change)
INKS = {
    "sepia": np.array([0.46, 0.33, 0.24], np.float32),
    "gall": np.array([0.30, 0.25, 0.22], np.float32),
    "blue": np.array([0.33, 0.37, 0.52], np.float32),
    "print": np.array([0.24, 0.22, 0.21], np.float32),
    "red": np.array([0.78, 0.36, 0.28], np.float32),
    "faded": np.array([0.62, 0.52, 0.44], np.float32),
}


class Pen:
    """Coverage buffers per ink, drawn at 2x."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.bufs = {}
        self.wash = np.zeros((h, w, 3), np.float32)   # watercolour tint (multiplicative darkening)
        self.washa = np.zeros((h, w), np.float32)

    def buf(self, ink):
        if ink not in self.bufs:
            self.bufs[ink] = np.zeros((self.h, self.w), np.float32)
        return self.bufs[ink]

    def line(self, ink, pts, width, profile=None):
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        n = len(pts)
        if profile is None:
            u = np.linspace(0, 1, n)
            profile = 0.35 + 0.65 * np.sin(np.clip(u, 0, 1) * math.pi) ** 0.5
        rad = 0.5 * width * np.asarray(profile, np.float64)
        eraster.stroke(self.buf(ink), pts, rad)

    def dot(self, ink, x, y, r):
        eraster.stroke(self.buf(ink), np.array([[x, y], [x + 0.01, y]]), np.array([r, r]))

    def wash_blob(self, cx, cy, rx, ry, col, amount, r, wob=0.25):
        """A soft watercolour patch with a darker pooled edge."""
        h, w = self.h, self.w
        x0, x1 = int(max(0, cx - rx * 1.6)), int(min(w, cx + rx * 1.6))
        y0, y1 = int(max(0, cy - ry * 1.6)), int(min(h, cy + ry * 1.6))
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ang = np.arctan2(ys - cy, xs - cx)
        ph = r.uniform(0, 6.28, 3)
        rr = 1.0 + wob * (0.5 * np.sin(3 * ang + ph[0]) + 0.3 * np.sin(5 * ang + ph[1]) + 0.2 * np.sin(8 * ang + ph[2]))
        d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) / rr
        inside = np.clip((1.0 - d) * 6.0, 0, 1)
        edge = np.exp(-((d - 0.93) / 0.06) ** 2) * 0.6
        a = (inside * 0.75 + edge) * amount
        sl = (slice(y0, y1), slice(x0, x1))
        tint = (1.0 - np.asarray(col, np.float32))
        self.wash[sl] = np.maximum(self.wash[sl], a[..., None] * tint)


# ---------------------------------------------------------------- writing

def cursive_word(pen, ink, x, y, xh, n, r, width, slant=0.25):
    """One handwritten word: a looping pen line, heavier on the downstrokes."""
    per = 9
    m = n * per + 1
    s = np.linspace(0, n, m)
    adv = xh * r.uniform(0.75, 1.05)
    hts = np.ones(n)
    loopy = np.zeros(n)
    for i in range(n):
        k = r.random()
        if k < 0.13:
            hts[i] = r.uniform(1.9, 2.5)       # ascender loop (l, h, b)
            loopy[i] = 1.0
        elif k < 0.22:
            hts[i] = -r.uniform(1.0, 1.5)      # descender loop (g, y)
            loopy[i] = 1.0
        elif k < 0.40:
            loopy[i] = 1.0                     # e
        else:
            hts[i] = r.uniform(0.85, 1.1)      # n, m, u
            loopy[i] = r.uniform(0.0, 0.5)
    idx = np.minimum(s.astype(int), n - 1)
    # smooth the per letter parameters a little so letters join
    h = np.interp(s, np.arange(n) + 0.5, hts)
    h = np.where(np.abs(hts[idx]) > 1.3, hts[idx], h)
    lp = loopy[idx]
    ph = 2 * math.pi * s
    ax = adv * (0.12 + 0.30 * lp)
    px = x + adv * s - ax * np.sin(ph)
    up = 0.5 - 0.5 * np.cos(ph)
    py = y - h * xh * up
    py += xh * 0.06 * np.sin(s * 1.7 + r.uniform(0, 6))
    px += slant * (y - py)
    # lead in and lead out ticks
    pts = np.stack([px, py], 1)
    d = np.diff(pts, axis=0, prepend=pts[:1])
    ln = np.hypot(d[:, 0], d[:, 1]) + 1e-6
    down = np.clip(d[:, 1] / ln, 0, 1)
    prof = 0.35 + 0.85 * down
    prof[:3] *= np.array([0.3, 0.6, 0.85])
    prof[-3:] *= np.array([0.85, 0.6, 0.3])
    pen.line(ink, pts, width, prof)
    # dots and crossings
    for i in range(n):
        if hts[i] > 1.8 and r.random() < 0.35:
            cx = x + adv * (i + 0.5)
            pen.line(ink, [[cx - xh * 0.7, y - xh * 1.25], [cx + xh * 0.8, y - xh * 1.35]], width * 0.8)
        elif 0.8 < hts[i] < 1.2 and r.random() < 0.08:
            pen.dot(ink, x + adv * (i + 0.6) + slant * xh * 1.6, y - xh * 1.7, width * 0.55)
    return x + adv * n + slant * 0


def handwriting_block(pen, ink, x0, y0, x1, y1, xh, lead, r, width, slant=0.25, indent=True, ragged=0.15):
    y = y0 + xh * 2.2
    line_i = 0
    while y < y1:
        x = x0 + (xh * 3 if (indent and line_i % 7 == 0) else 0)
        right = x1 - r.uniform(0, ragged) * (x1 - x0)
        if r.random() < 0.06:
            right = x0 + (x1 - x0) * r.uniform(0.3, 0.7)    # paragraph end
        wob = r.uniform(-0.5, 0.5) * xh * 0.4
        while True:
            n = int(r.integers(2, 9))
            wlen = n * xh * 0.95
            if x + wlen > right:
                break
            cursive_word(pen, ink, x, y + wob, xh, n, r, width, slant)
            x += wlen + xh * r.uniform(0.7, 1.5)
        y += lead * r.uniform(0.97, 1.03)
        line_i += 1


def print_word(pen, ink, x, y, xh, n, r, width):
    for i in range(n):
        k = r.random()
        top = y - xh
        bot = y
        if k < 0.18:
            top = y - xh * 1.55
        elif k < 0.26:
            bot = y + xh * 0.5
        cx = x + i * xh * 0.62
        if r.random() < 0.55:
            pen.line(ink, [[cx, top], [cx, bot]], width, profile=np.array([0.9, 1.0]))
            if r.random() < 0.5:
                pen.line(ink, [[cx, y - xh * 0.95], [cx + xh * 0.35, y - xh * 0.7], [cx + xh * 0.35, y]],
                         width * 0.85, profile=np.array([0.8, 1.0, 0.9]))
        else:
            a = np.linspace(0, 2 * math.pi, 7)
            pen.line(ink, np.stack([cx + xh * 0.2 + np.cos(a) * xh * 0.22, y - xh * 0.5 + np.sin(a) * xh * 0.45], 1),
                     width * 0.9, profile=np.ones(7))
    return x + n * xh * 0.62


def print_block(pen, ink, x0, y0, x1, y1, xh, lead, r, width, justify=True):
    y = y0 + xh * 1.6
    li = 0
    while y < y1:
        x = x0 + (xh * 2.0 if li % 9 == 0 else 0)
        right = x1
        last = r.random() < 0.08
        if last:
            right = x0 + (x1 - x0) * r.uniform(0.25, 0.8)
        words = []
        while True:
            n = int(r.integers(1, 10))
            wl = n * xh * 0.62
            if x + wl > right:
                break
            words.append((x, n))
            x += wl + xh * 0.55
        # justify: spread the gap
        if justify and words and not last:
            slack = right - x
            gap = slack / max(1, len(words) - 1)
            words = [(wx + gap * i, n) for i, (wx, n) in enumerate(words)]
        for wx, n in words:
            print_word(pen, ink, wx, y, xh, n, r, width)
        y += lead
        li += 1


def heading(pen, ink, cx, y, xh, r, width, n_words=None):
    n_words = n_words or int(r.integers(1, 4))
    lens = [int(r.integers(3, 8)) for _ in range(n_words)]
    total = sum(l * xh * 0.75 for l in lens) + (n_words - 1) * xh
    x = cx - total / 2
    for l in lens:
        for i in range(l):
            lx = x + i * xh * 0.75
            pen.line(ink, [[lx, y - xh], [lx + xh * 0.1, y]], width, profile=np.array([1.0, 0.9]))
            if r.random() < 0.6:
                pen.line(ink, [[lx, y - xh], [lx + xh * 0.45, y - xh * 0.9], [lx + xh * 0.45, y - xh * 0.1]],
                         width * 0.8, profile=np.array([0.7, 1.0, 0.8]))
        x += l * xh * 0.75 + xh


def drop_cap(pen, x, y, s, r, width):
    ink = "red" if r.random() < 0.7 else "print"
    pen.line("print", [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]], width * 0.6, profile=np.ones(5))
    # a flourished letter shape: a stem and two bowls, then vine curls
    pen.line(ink, [[x + s * 0.3, y + s * 0.18], [x + s * 0.32, y + s * 0.85]], width * 2.2, profile=np.array([0.7, 1.0]))
    t = np.linspace(-1.4, 1.4, 12)
    pen.line(ink, np.stack([x + s * 0.32 + np.cos(t) * s * 0.32, y + s * 0.36 + np.sin(t) * s * 0.18], 1), width * 1.6)
    pen.line(ink, np.stack([x + s * 0.32 + np.cos(t) * s * 0.38, y + s * 0.66 + np.sin(t) * s * 0.2], 1), width * 1.6)
    for _ in range(3):
        a0 = r.uniform(0, 6.28)
        tt = np.linspace(0, 4.5, 14)
        rr = s * 0.12 * (1 - tt / 6)
        cx, cy = x + r.uniform(0.15, 0.85) * s, y + r.uniform(0.15, 0.85) * s
        pen.line("print", np.stack([cx + np.cos(a0 + tt) * rr, cy + np.sin(a0 + tt) * rr], 1), width * 0.5)


# ---------------------------------------------------------------- illustrations

def _jit(r, pts, a):
    pts = np.asarray(pts, np.float64)
    return pts + r.normal(0, a, pts.shape)


def frame_box(pen, ink, x0, y0, x1, y1, width, r, double=True):
    pen.line(ink, _jit(r, [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], 0.4), width, profile=np.ones(5) * 0.9)
    if double:
        d = width * 2.2
        pen.line(ink, _jit(r, [[x0 + d, y0 + d], [x1 - d, y0 + d], [x1 - d, y1 - d], [x0 + d, y1 - d], [x0 + d, y0 + d]], 0.3),
                 width * 0.5, profile=np.ones(5))


def hatch(pen, ink, x0, y0, x1, y1, step, ang, width, r, mask=None):
    """Parallel pen hatching inside a box (optionally a mask function)."""
    c, s = math.cos(ang), math.sin(ang)
    L = math.hypot(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = -L / 2
    while k < L / 2:
        p0 = np.array([cx + (-s) * k - c * L / 2, cy + c * k - s * L / 2])
        p1 = np.array([cx + (-s) * k + c * L / 2, cy + c * k + s * L / 2])
        n = 12
        pts = p0[None] + (p1 - p0)[None] * np.linspace(0, 1, n)[:, None]
        ok = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if mask is not None:
            ok &= mask(pts[:, 0], pts[:, 1])
        # draw runs
        i = 0
        while i < n:
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= 2:
                pen.line(ink, _jit(r, pts[i:j], 0.3), width)
            i = j
        k += step * r.uniform(0.8, 1.2)


def ill_moon(pen, ink, x0, y0, w, h, r, lw):
    cx, cy, R = x0 + w * 0.5, y0 + h * 0.45, min(w, h) * 0.28
    t = np.linspace(0.35 * math.pi, 1.65 * math.pi, 24)
    outer = np.stack([cx + np.cos(t) * R, cy + np.sin(t) * R], 1)
    inner = np.stack([cx + R * 0.45 + np.cos(t[::-1]) * R * 0.85, cy + np.sin(t[::-1]) * R * 0.85], 1)
    pen.line(ink, _jit(r, outer, 0.4), lw * 1.3)
    pen.line(ink, _jit(r, inner, 0.4), lw)
    pen.wash_blob(cx - R * 0.3, cy, R * 0.6, R * 0.9, (0.95, 0.85, 0.55), 0.35, r)
    for _ in range(int(r.integers(5, 11))):
        sx, sy = x0 + r.uniform(0.08, 0.92) * w, y0 + r.uniform(0.08, 0.9) * h
        if math.hypot(sx - cx, sy - cy) < R * 1.25:
            continue
        a = r.uniform(1.5, 3.2)
        pen.line(ink, [[sx - a, sy], [sx + a, sy]], lw * 0.6)
        pen.line(ink, [[sx, sy - a], [sx, sy + a]], lw * 0.6)


def ill_tree(pen, ink, x0, y0, w, h, r, lw):
    bx, by = x0 + w * 0.5, y0 + h * 0.92
    top = y0 + h * 0.42
    trunk = np.array([[bx, by], [bx + r.uniform(-3, 3), (by + top) / 2], [bx + r.uniform(-5, 5), top]])
    pen.line(ink, _jit(r, trunk, 0.5), lw * 2.2, profile=np.array([1.0, 0.8, 0.5]))
    for _ in range(int(r.integers(3, 6))):
        a = r.uniform(-2.6, -0.5)
        L = r.uniform(0.15, 0.3) * w
        sx, sy = trunk[2] + r.normal(0, 3, 2)
        pen.line(ink, [[sx, sy], [sx + math.cos(a) * L, sy + math.sin(a) * L]], lw, profile=np.array([1.0, 0.3]))
    # canopy scribble of looping clumps
    cx, cy = bx, y0 + h * 0.32
    for _ in range(int(r.integers(6, 11))):
        ox, oy = r.normal(0, w * 0.17), r.normal(0, h * 0.09)
        t = np.linspace(0, 2 * math.pi * r.uniform(0.8, 1.1), 16)
        rr = w * r.uniform(0.06, 0.11)
        pen.line(ink, _jit(r, np.stack([cx + ox + np.cos(t) * rr, cy + oy + np.sin(t) * rr * 0.8], 1), 0.6), lw * 0.8)
    pen.wash_blob(cx, cy, w * 0.3, h * 0.2, (0.62, 0.72, 0.5), 0.4, r)
    pen.line(ink, _jit(r, [[x0 + w * 0.1, by], [x0 + w * 0.9, by]], 0.5), lw * 0.7)


def ill_ship(pen, ink, x0, y0, w, h, r, lw):
    cx, wy = x0 + w * 0.5, y0 + h * 0.7
    hull = np.array([[cx - w * 0.3, wy - h * 0.08], [cx - w * 0.22, wy], [cx + w * 0.22, wy], [cx + w * 0.32, wy - h * 0.1]])
    pen.line(ink, _jit(r, hull, 0.4), lw * 1.5)
    pen.line(ink, [[cx, wy - h * 0.06], [cx, wy - h * 0.6]], lw * 1.2)
    sail = np.array([[cx + 1, wy - h * 0.58], [cx + w * 0.22, wy - h * 0.2], [cx + 1, wy - h * 0.14]])
    pen.line(ink, _jit(r, sail, 0.4), lw)
    sail2 = np.array([[cx - 1, wy - h * 0.5], [cx - w * 0.17, wy - h * 0.16], [cx - 1, wy - h * 0.12]])
    pen.line(ink, _jit(r, sail2, 0.4), lw)
    pen.wash_blob(cx, wy + h * 0.1, w * 0.42, h * 0.1, (0.55, 0.68, 0.8), 0.45, r)
    for k in range(4):
        y = wy + h * (0.04 + 0.06 * k)
        xs = np.linspace(x0 + w * 0.08, x0 + w * 0.92, 18)
        ys = y + np.sin(xs * 0.35 + k) * h * 0.012
        pen.line(ink, _jit(r, np.stack([xs, ys], 1), 0.3), lw * 0.6)


def ill_mountains(pen, ink, x0, y0, w, h, r, lw):
    base = y0 + h * 0.8
    xs = np.linspace(x0 + w * 0.05, x0 + w * 0.95, 9)
    ys = base - h * np.array([0.05, 0.35, 0.15, 0.55, 0.25, 0.45, 0.12, 0.3, 0.05]) * r.uniform(0.8, 1.2, 9)
    pen.line(ink, _jit(r, np.stack([xs, ys], 1), 0.6), lw * 1.3)
    # sun
    cx, cy, R = x0 + w * r.uniform(0.25, 0.75), y0 + h * 0.22, w * 0.08
    t = np.linspace(0, 2 * math.pi, 16)
    pen.line(ink, np.stack([cx + np.cos(t) * R, cy + np.sin(t) * R], 1), lw)
    for a in np.linspace(0, 2 * math.pi, 9)[:-1]:
        pen.line(ink, [[cx + math.cos(a) * R * 1.4, cy + math.sin(a) * R * 1.4],
                       [cx + math.cos(a) * R * 2.0, cy + math.sin(a) * R * 2.0]], lw * 0.6)
    pen.wash_blob(cx, cy, R * 1.6, R * 1.6, (0.98, 0.8, 0.55), 0.35, r)
    hatch(pen, ink, x0 + w * 0.05, base - h * 0.4, x0 + w * 0.95, base, h * 0.05, 0.9, lw * 0.5, r,
          mask=lambda px, py: py > np.interp(px, xs, ys) + h * 0.05)


def ill_bird(pen, ink, x0, y0, w, h, r, lw):
    for _ in range(int(r.integers(1, 4))):
        cx, cy = x0 + w * r.uniform(0.25, 0.75), y0 + h * r.uniform(0.3, 0.7)
        s = w * r.uniform(0.12, 0.3)
        t = np.linspace(0, 1, 10)
        left = np.stack([cx - s * t[::-1], cy - s * 0.5 * np.sin(t[::-1] * math.pi) * 0.7 - s * 0.1 * t[::-1]], 1)
        right = np.stack([cx + s * t, cy - s * 0.5 * np.sin(t * math.pi) * 0.7 - s * 0.1 * t], 1)
        pen.line(ink, _jit(r, np.vstack([left, right]), 0.3), lw * 1.1)


def ill_fish(pen, ink, x0, y0, w, h, r, lw):
    cx, cy, L = x0 + w * 0.5, y0 + h * 0.5, w * 0.36
    t = np.linspace(0, math.pi, 12)
    top = np.stack([cx - L + 2 * L * t / math.pi, cy - np.sin(t) * h * 0.14], 1)
    bot = np.stack([cx - L + 2 * L * t / math.pi, cy + np.sin(t) * h * 0.12], 1)
    pen.line(ink, _jit(r, top, 0.3), lw * 1.2)
    pen.line(ink, _jit(r, bot, 0.3), lw * 1.2)
    pen.line(ink, [[cx + L, cy], [cx + L * 1.35, cy - h * 0.12], [cx + L * 1.3, cy + h * 0.12], [cx + L, cy]], lw)
    pen.dot(ink, cx - L * 0.7, cy - h * 0.03, lw * 0.9)
    for k in range(4):
        xx = cx - L * 0.3 + k * L * 0.35
        tt = np.linspace(-0.8, 0.8, 6)
        pen.line(ink, np.stack([xx + np.cos(tt) * 4 - 4, cy + np.sin(tt) * h * 0.09], 1), lw * 0.5)
    pen.wash_blob(cx, cy, L * 0.9, h * 0.11, (0.7, 0.78, 0.82), 0.35, r)


def ill_botanical(pen, ink, x0, y0, w, h, r, lw):
    bx, by = x0 + w * 0.5, y0 + h * 0.95
    stem = np.stack([bx + np.sin(np.linspace(0, 2, 12)) * w * 0.05, np.linspace(by, y0 + h * 0.1, 12)], 1)
    pen.line(ink, _jit(r, stem, 0.3), lw * 1.2)
    for i in range(int(r.integers(4, 8))):
        k = int(r.integers(2, 11))
        sx, sy = stem[k]
        side = -1 if i % 2 else 1
        L = w * r.uniform(0.15, 0.3)
        a = r.uniform(-0.6, 0.2) * side
        tip = np.array([sx + side * L * math.cos(a), sy - L * 0.5 + L * math.sin(a) * 0.3])
        mid = (np.array([sx, sy]) + tip) / 2
        nrm = np.array([-(tip - [sx, sy])[1], (tip - [sx, sy])[0]]) * 0.25
        t = np.linspace(0, 1, 9)[:, None]
        e1 = (1 - t) ** 2 * [sx, sy] + 2 * (1 - t) * t * (mid + nrm) + t ** 2 * tip
        e2 = (1 - t) ** 2 * [sx, sy] + 2 * (1 - t) * t * (mid - nrm) + t ** 2 * tip
        pen.line(ink, e1, lw * 0.9)
        pen.line(ink, e2, lw * 0.9)
        pen.line(ink, [[sx, sy], tip], lw * 0.4)
        pen.wash_blob(mid[0], mid[1], L * 0.35, L * 0.2, (0.66, 0.74, 0.52), 0.3, r)
    # flower head
    fx, fy = stem[-1]
    for a in np.linspace(0, 2 * math.pi, int(r.integers(5, 9)), endpoint=False):
        tt = np.linspace(-0.4, 0.4, 7)
        R = w * 0.1
        pen.line(ink, np.stack([fx + np.cos(a + tt) * R * np.cos(tt * 3), fy + np.sin(a + tt) * R * np.cos(tt * 3)], 1), lw * 0.8)
    pen.wash_blob(fx, fy, w * 0.1, w * 0.1, (0.9, 0.62, 0.6), 0.4, r)


def ill_tower(pen, ink, x0, y0, w, h, r, lw):
    cx, by = x0 + w * 0.5, y0 + h * 0.9
    tw = w * 0.16
    top = y0 + h * 0.3
    pen.line(ink, _jit(r, [[cx - tw, by], [cx - tw * 0.9, top]], 0.3), lw * 1.2)
    pen.line(ink, _jit(r, [[cx + tw, by], [cx + tw * 0.9, top]], 0.3), lw * 1.2)
    t = np.linspace(math.pi, 2 * math.pi, 12)
    pen.line(ink, np.stack([cx + np.cos(t) * tw * 1.0, top + np.sin(t) * h * 0.12], 1), lw * 1.2)
    pen.line(ink, [[cx, top - h * 0.12], [cx, top - h * 0.18]], lw)
    for k in range(3):
        wy = top + (by - top) * (0.2 + 0.25 * k)
        pen.line(ink, [[cx - tw * 0.2, wy], [cx - tw * 0.2, wy - h * 0.05], [cx + tw * 0.2, wy - h * 0.05], [cx + tw * 0.2, wy]], lw * 0.6)
    pen.wash_blob(cx, top - h * 0.05, tw * 1.1, h * 0.08, (0.6, 0.8, 0.72), 0.45, r)
    hatch(pen, ink, cx - tw, top, cx, by, h * 0.04, 0.3, lw * 0.4, r)
    pen.line(ink, _jit(r, [[x0 + w * 0.05, by], [x0 + w * 0.95, by + r.uniform(-3, 3)]], 0.5), lw * 0.7)


def ill_stars(pen, ink, x0, y0, w, h, r, lw):
    cx, cy, R = x0 + w * 0.5, y0 + h * 0.5, min(w, h) * 0.42
    t = np.linspace(0, 2 * math.pi, 40)
    pen.line(ink, np.stack([cx + np.cos(t) * R, cy + np.sin(t) * R], 1), lw, profile=np.ones(40))
    pen.line(ink, np.stack([cx + np.cos(t) * R * 0.93, cy + np.sin(t) * R * 0.93], 1), lw * 0.4, profile=np.ones(40))
    pts = []
    for _ in range(int(r.integers(7, 13))):
        a, d = r.uniform(0, 6.28), R * math.sqrt(r.uniform(0.02, 0.8))
        pts.append((cx + math.cos(a) * d, cy + math.sin(a) * d))
    pts = np.array(pts)
    order = np.argsort(pts[:, 0])
    pen.line(ink, pts[order[: max(3, len(order) // 2)]], lw * 0.5, profile=np.ones(max(3, len(order) // 2)))
    for p in pts:
        pen.dot(ink, p[0], p[1], lw * r.uniform(0.8, 1.6))
    for a in np.linspace(0, 2 * math.pi, 13)[:-1]:
        pen.line(ink, [[cx + math.cos(a) * R * 0.93, cy + math.sin(a) * R * 0.93], [cx + math.cos(a) * R, cy + math.sin(a) * R]], lw * 0.5)
    pen.wash_blob(cx, cy, R * 0.8, R * 0.8, (0.62, 0.66, 0.84), 0.25, r)


def ill_map(pen, ink, x0, y0, w, h, r, lw):
    cx, cy = x0 + w * 0.48, y0 + h * 0.5
    t = np.linspace(0, 2 * math.pi, 60)
    rad = 1 + 0.25 * np.sin(3 * t + r.uniform(0, 6)) + 0.15 * np.sin(7 * t + r.uniform(0, 6)) + 0.08 * np.sin(13 * t)
    coast = np.stack([cx + np.cos(t) * rad * w * 0.25, cy + np.sin(t) * rad * h * 0.25], 1)
    pen.line(ink, _jit(r, coast, 0.4), lw * 1.1, profile=np.ones(60) * 0.9)
    pen.wash_blob(cx, cy, w * 0.26, h * 0.26, (0.85, 0.78, 0.55), 0.3, r, wob=0.1)
    route = np.stack([np.linspace(x0 + w * 0.1, cx, 14), cy + np.sin(np.linspace(0, 3, 14)) * h * 0.15 + h * 0.2], 1)
    for i in range(0, 13, 2):
        pen.line("red", route[i:i + 2], lw * 0.6)
    rx, ry, R = x0 + w * 0.82, y0 + h * 0.2, w * 0.09
    for a in np.linspace(0, 2 * math.pi, 9)[:-1]:
        L = R if int(round(a / (math.pi / 4))) % 2 == 0 else R * 0.55
        pen.line(ink, [[rx, ry], [rx + math.cos(a) * L, ry + math.sin(a) * L]], lw * 0.7, profile=np.array([1.0, 0.2]))
    t = np.linspace(0, 2 * math.pi, 16)
    pen.line(ink, np.stack([rx + np.cos(t) * R * 0.4, ry + np.sin(t) * R * 0.4], 1), lw * 0.4)


ILLS = [ill_moon, ill_tree, ill_ship, ill_mountains, ill_bird, ill_fish, ill_botanical, ill_tower, ill_stars, ill_map]


def music(pen, ink, x0, y0, x1, y1, r, lw, s):
    y = y0 + 10 * s
    while y + 20 * s < y1:
        gap = 3.2 * s
        for k in range(5):
            xs = np.linspace(x0, x1, 10)
            pen.line(ink, np.stack([xs, y + k * gap + np.sin(xs * 0.05) * 0.4], 1), lw * 0.35, profile=np.ones(10))
        x = x0 + 8 * s
        while x < x1 - 6 * s:
            if r.random() < 0.12:
                pen.line(ink, [[x, y], [x, y + 4 * gap]], lw * 0.5)
                x += 3 * s
                continue
            ny = y + gap * r.integers(-1, 9) * 0.5
            pen.dot(ink, x, ny, gap * 0.55)
            up = r.random() < 0.5
            pen.line(ink, [[x + gap * 0.5 * (1 if up else -1), ny], [x + gap * 0.5 * (1 if up else -1), ny - 3.3 * gap * (1 if up else -1)]], lw * 0.45)
            x += r.uniform(4, 8) * s
        y += 24 * s


def ledger(pen, x0, y0, x1, y1, r, lw, s):
    y = y0
    while y < y1:
        xs = np.linspace(x0 - 20 * s, x1 + 20 * s, 8)
        pen.line("blue", np.stack([xs, np.full(8, y)], 1), lw * 0.3, profile=np.ones(8) * 0.6)
        y += 9 * s
    pen.line("red", [[x0 + 18 * s, y0 - 30 * s], [x0 + 18 * s, y1 + 30 * s]], lw * 0.5, profile=np.ones(2) * 0.8)
    pen.line("red", [[x1 - 40 * s, y0 - 30 * s], [x1 - 40 * s, y1 + 30 * s]], lw * 0.4, profile=np.ones(2) * 0.6)
    y = y0 - 2 * s
    while y < y1:
        if r.random() < 0.8:
            x = x0 + 22 * s
            for _ in range(int(r.integers(1, 4))):
                n = int(r.integers(2, 6))
                cursive_word(pen, "gall", x, y, 2.6 * s, n, r, lw * 0.7, 0.2)
                x += n * 2.6 * s + 5 * s
            cursive_word(pen, "gall", x1 - 36 * s, y, 2.6 * s, int(r.integers(2, 5)), r, lw * 0.7, 0.05)
        y += 9 * s


# ---------------------------------------------------------------- the page

def age(page_rgb, r, key):
    """Brown the edges, add foxing, mottling, a tea ring, fibres."""
    h, w = page_rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
    n = value_noise(h, w, 40, (key, "edge"))
    edge = np.exp(-d / (10 + 18 * n)) * r.uniform(0.35, 0.8)
    brown = np.array([0.10, 0.22, 0.42], np.float32)
    mott = value_noise(h, w, 90, (key, "mott")) - 0.5
    m2 = value_noise(h, w, 24, (key, "mott2")) - 0.5
    darken = edge[..., None] * brown + (mott * 0.06 + m2 * 0.03)[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
    page_rgb *= np.clip(1.0 - darken, 0, 1)
    # foxing spots
    for _ in range(int(r.integers(0, 28))):
        cx, cy = r.uniform(0, w), r.uniform(0, h)
        rr = r.uniform(1.0, 4.5) * (2.0 if r.random() < 0.1 else 1.0)
        dd = np.hypot(xs - cx, ys - cy)
        a = np.clip(1.2 - dd / rr, 0, 1) * r.uniform(0.15, 0.45) + np.exp(-dd / (rr * 2.5)) * 0.06
        page_rgb *= 1.0 - a[..., None] * np.array([0.25, 0.42, 0.62], np.float32)
    # tea ring
    if r.random() < 0.28:
        cx, cy = r.uniform(0.15, 0.85) * w, r.uniform(0.15, 0.85) * h
        R = r.uniform(0.12, 0.3) * w
        dd = np.hypot(xs - cx, ys - cy)
        ring = np.exp(-((dd - R) / (R * 0.03)) ** 2) * 0.35 + np.clip(1 - dd / R, 0, 1) ** 0.3 * 0.08
        ring *= 0.6 + 0.8 * value_noise(h, w, 20, (key, "ring"))
        page_rgb *= 1.0 - ring[..., None] * np.array([0.18, 0.32, 0.55], np.float32)
    # a faint old fold
    if r.random() < 0.3:
        fy = r.uniform(0.3, 0.7) * h
        crease = np.exp(-((ys - fy) / 1.6) ** 2) * 0.07
        page_rgb *= 1.0 - crease[..., None]
    g = paper_grain(h, w, (key, "grain"))
    page_rgb *= (1.0 + (g - 0.5) * 0.05)[..., None]
    return page_rgb


def draw_page(i):
    """One page at 2x, returned as float32 (2*PH, 2*PW, 3) multiplicative map."""
    s = 2.0
    h, w = int(PH * s), int(PW * s)
    r = rng("page", i, VERSION)
    pen = Pen(h, w)
    rgb = np.ones((h, w, 3), np.float32)
    rgb = age(rgb, r, ("page", i))
    if i == BLANK:
        return rgb
    kind = r.choice(["manuscript", "print", "illustrated", "plate", "music", "map", "ledger", "letter"],
                    p=[0.22, 0.22, 0.22, 0.10, 0.06, 0.05, 0.05, 0.08])
    mx = w * r.uniform(0.09, 0.14)
    my = h * r.uniform(0.07, 0.11)
    x0, x1, y0, y1 = mx, w - mx * r.uniform(0.8, 1.2), my, h - my
    lw = s * r.uniform(0.75, 1.05)
    hand = r.choice(["sepia", "gall", "blue", "faded"], p=[0.4, 0.3, 0.15, 0.15])
    slant = r.uniform(0.05, 0.4)
    if kind == "manuscript":
        if r.random() < 0.5:
            heading(pen, "red" if r.random() < 0.5 else hand, w / 2, y0 + 10 * s, 6 * s, r, lw * 1.2)
            y0 += 22 * s
        xh = r.uniform(3.2, 4.6) * s
        handwriting_block(pen, hand, x0, y0, x1, y1, xh, xh * r.uniform(3.0, 3.8), r, lw * 0.9, slant)
    elif kind == "letter":
        xh = r.uniform(4.6, 6.0) * s
        handwriting_block(pen, hand, x0, y0 + 10 * s, x1, y1 - 50 * s, xh, xh * 3.6, r, lw * 1.1, slant, ragged=0.3)
        n = int(r.integers(4, 8))
        cursive_word(pen, hand, x1 - n * xh * 1.3, y1 - 26 * s, xh * 1.3, n, r, lw * 1.3, slant)
        t = np.linspace(0, 1, 20)
        fx = x1 - n * xh * 1.3 + t * n * xh * 1.4
        pen.line(hand, np.stack([fx, y1 - 18 * s + np.sin(t * 9) * 3 * s * (1 - t)], 1), lw * 0.9)
    elif kind == "print":
        xh = r.uniform(2.6, 3.4) * s
        pen.line("print", [[x0, y0 - 4 * s], [x1, y0 - 4 * s]], lw * 0.35, profile=np.ones(2))
        heading(pen, "print", w / 2, y0 - 8 * s, 2.6 * s, r, lw * 0.6, n_words=2)
        yy = y0 + 4 * s
        if r.random() < 0.45:
            heading(pen, "red" if r.random() < 0.4 else "print", w / 2, yy + 16 * s, 6 * s, r, lw * 1.3)
            yy += 30 * s
        if r.random() < 0.35:
            drop_cap(pen, x0, yy + 2 * s, 30 * s, r, lw)
            print_block(pen, "print", x0 + 36 * s, yy, x1, yy + 34 * s, xh, xh * 2.6, r, lw * 0.75)
            yy += 36 * s
        print_block(pen, "print", x0, yy, x1, y1 - 10 * s, xh, xh * 2.6, r, lw * 0.75)
        cursive_word(pen, "print", w / 2 - 4 * s, h - my * 0.45, 2.5 * s, 2, r, lw * 0.6, 0.0)
    elif kind in ("illustrated", "plate"):
        ink = r.choice(["sepia", "gall", "print"])
        f = ILLS[int(r.integers(0, len(ILLS)))]
        if kind == "plate":
            fx0, fy0, fx1, fy1 = x0, y0 + 10 * s, x1, y1 - 40 * s
            frame_box(pen, ink, fx0, fy0, fx1, fy1, lw * 0.9, r)
            f(pen, ink, fx0 + 8 * s, fy0 + 8 * s, fx1 - fx0 - 16 * s, fy1 - fy0 - 16 * s, r, lw * 2.0)
            g = ILLS[int(r.integers(0, len(ILLS)))]
            if g is not f and r.random() < 0.6:
                g(pen, ink, fx0 + 8 * s, fy0 + 8 * s, (fx1 - fx0) * 0.45, (fy1 - fy0) * 0.35, r, lw * 1.4)
            heading(pen, ink, w / 2, y1 - 18 * s, 3.4 * s, r, lw * 0.7, n_words=3)
        else:
            top = r.random() < 0.5
            ih = (y1 - y0) * r.uniform(0.32, 0.5)
            if top:
                iy0, iy1 = y0, y0 + ih
                ty0, ty1 = iy1 + 10 * s, y1
            else:
                ty0, ty1 = y0, y1 - ih - 10 * s
                iy0, iy1 = ty1 + 10 * s, y1
            if r.random() < 0.6:
                frame_box(pen, ink, x0, iy0, x1, iy1, lw * 0.8, r, double=r.random() < 0.5)
            f(pen, ink, x0 + 6 * s, iy0 + 6 * s, x1 - x0 - 12 * s, iy1 - iy0 - 12 * s, r, lw * 1.7)
            if r.random() < 0.5:
                xh = r.uniform(2.6, 3.4) * s
                print_block(pen, "print", x0, ty0, x1, ty1, xh, xh * 2.6, r, lw * 0.75)
            else:
                xh = r.uniform(3.2, 4.4) * s
                handwriting_block(pen, hand, x0, ty0, x1, ty1, xh, xh * 3.4, r, lw * 0.9, slant)
    elif kind == "music":
        heading(pen, "print", w / 2, y0 + 6 * s, 4 * s, r, lw * 0.9)
        music(pen, "print", x0, y0 + 16 * s, x1, y1, r, lw * 1.2, s)
    elif kind == "map":
        frame_box(pen, "sepia", x0, y0, x1, y1 - 30 * s, lw, r)
        ill_map(pen, "sepia", x0 + 6 * s, y0 + 6 * s, x1 - x0 - 12 * s, y1 - y0 - 42 * s, r, lw * 1.1)
        hatch(pen, "sepia", x0 + 2 * s, y0 + 2 * s, x1 - 2 * s, y1 - 32 * s, 9 * s, 0.0, lw * 0.25, r,
              mask=lambda px, py: (np.hypot((px - w * 0.48) / (w * 0.4), (py - (y0 + y1 - 30 * s) / 2) / (h * 0.36)) > 0.9))
        handwriting_block(pen, "sepia", x0, y1 - 26 * s, x1, y1, 3.6 * s, 12 * s, r, lw * 0.8, slant)
    elif kind == "ledger":
        heading(pen, "gall", w / 2, y0 + 4 * s, 4.5 * s, r, lw)
        ledger(pen, x0 - 10 * s, y0 + 16 * s, x1 + 4 * s, y1, r, lw, s)
    # marginalia: a pressed flower, a doodle, a correction
    if r.random() < 0.2:
        ill_bird(pen, hand, w - mx * 1.05, h * r.uniform(0.2, 0.8), mx * 0.9, mx * 0.7, r, lw * 0.7)
    # ink: fade unevenly, a little bleed
    fade = 0.55 + 0.45 * value_noise(h, w, 60, ("fade", i))
    for name, b in pen.bufs.items():
        a = ndimage.gaussian_filter(b, 0.5) * 0.85 + b * 0.15
        a = np.clip(a * fade * r.uniform(0.75, 1.0), 0, 1)
        ink = INKS[name]
        rgb *= 1.0 - a[..., None] * (1.0 - ink)
    if pen.washa is not None:
        rgb *= 1.0 - pen.wash * (0.85 + 0.15 * fade[..., None])
    return np.clip(rgb, 0, 1)


def _down2(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def build(n=NPAGES):
    pages = np.zeros((n, PH, PW, 3), np.float32)
    for i in range(n):
        pages[i] = _down2(draw_page(i))
    return pages


def pyramid(pages, nlev=6):
    lv = [np.clip(pages * 255 + 0.5, 0, 255).astype(np.uint8)]
    cur = pages
    for _ in range(nlev - 1):
        cur = np.stack([_down2(p) for p in cur])
        lv.append(np.clip(cur * 255 + 0.5, 0, 255).astype(np.uint8))
    return lv


_ATLAS = None


def atlas():
    """The cached texture pyramid as a raster.Atlas."""
    global _ATLAS
    if _ATLAS is not None:
        return _ATLAS
    from . import raster
    path = os.path.join(CACHE, f"atlas_v{VERSION}.npz")
    if os.path.exists(path):
        z = np.load(path)
        levels = [z[f"l{k}"] for k in range(len(z.files))]
    else:
        os.makedirs(CACHE, exist_ok=True)
        levels = pyramid(build())
        np.savez(path, **{f"l{k}": L for k, L in enumerate(levels)})
    _ATLAS = raster.Atlas(levels)
    return _ATLAS
