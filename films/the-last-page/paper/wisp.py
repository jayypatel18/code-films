"""A painted cloud wisp sprite (premultiplied RGBA) that drifts in front of the
great bird's wingtip in cut 8, for scale. Composed as a reference (overlapping
puffs, lit from the sun at the right, lilac underside), painted with A2's
gouache stroke painter onto a transparent canvas, cached in build/paper/.

    python3 tools/heavy.py -- python3 -m paper.wisp      (from films/the-last-page)
"""

import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FILM, "..", ".."))
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import paint as P  # noqa: E402
from engine.noise import fbm  # noqa: E402

WW, WH = 1100, 420
PATH = os.path.join(FILM, "build", "paper", "wisp8.npy")


def reference():
    ys, xs = np.mgrid[0:WH, 0:WW].astype(np.float32)
    r = np.random.default_rng(7)
    cov = np.zeros((WH, WW), np.float32)
    lit = np.zeros((WH, WW), np.float32)
    s3 = np.array([0.65, -0.45, 0.6])
    s3 /= np.linalg.norm(s3)
    puffs = []
    for i in range(60):
        u = r.uniform(0, 1)
        cx = 120 + u * 860 + r.normal(0, 25)
        top = 250 - 90 * math.sin(math.pi * u) ** 1.2
        cy = top + r.uniform(0, 60)
        rr = (30 + 55 * math.sin(math.pi * u) ** 0.8) * r.uniform(0.7, 1.2)
        puffs.append((cx, cy, rr))
    puffs.sort(key=lambda p: p[1] + p[2] * 0.3)
    for (cx, cy, rr) in puffs:
        nx, ny = (xs - cx) / rr, (ys - cy) / (rr * 0.82)
        d2 = nx * nx + ny * ny
        inside = d2 < 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        l = np.clip(nx * s3[0] + ny * s3[1] + nz * s3[2], 0, 1)
        lit = np.where(inside, l, lit)
        cov = np.maximum(cov, np.clip((1 - np.sqrt(d2)) * 8, 0, 1))
    # flat, torn underside: clouds are cut off by the layer they sit in
    base = 300 + 25 * np.sin(xs * 0.012) + 12 * np.sin(xs * 0.031)
    cov *= np.clip((base - ys) / 25.0, 0, 1)
    n = fbm(WH, WW, 40, "wisp", octaves=4)
    cov *= np.clip(0.75 + (n - 0.5) * 1.2, 0, 1)
    c_lit = np.array([1.0, 0.89, 0.72], np.float32)
    c_mid = np.array([0.95, 0.76, 0.72], np.float32)
    c_sh = np.array([0.66, 0.56, 0.72], np.float32)
    lit = ndimage.gaussian_filter(lit, 9.0) * 0.6 + np.clip(1.0 - (ys - 120) / 220.0, 0, 1) * 0.4
    col = np.where((lit > 0.55)[..., None], c_lit, np.where((lit > 0.35)[..., None], c_mid, c_sh))
    col = ndimage.gaussian_filter(col, (4.0, 4.0, 0))
    return col.astype(np.float32), cov


def paint():
    col, cov = reference()
    canvas = P.new_canvas(WH, WW)
    paper = P.paper_for(WH, WW, "wisp8")
    ang = P.orientation(col, 1.5, 5.0, base=0.15, base_weight=0.3, edge_gain=1.3)
    mask = (cov > 0.35).astype(np.float32)
    passes = [
        dict(radius=14, spacing=0.9, steps=(3, 9), opacity=0.95, dry=0.3, cthresh=0.1, wob=0.25, mask=mask,
             jitter=(0.03, 0.015, 0.05), pickup=0.2, curve=0.4),
        dict(radius=6, spacing=1.0, steps=(2, 7), opacity=0.9, dry=0.45, cthresh=0.07, wob=0.3, mask=mask,
             jitter=(0.03, 0.015, 0.05), pickup=0.25, curve=0.4),
        dict(radius=3, spacing=1.1, steps=(2, 6), opacity=0.88, dry=0.5, cthresh=0.06, wob=0.3, mask=mask,
             jitter=(0.03, 0.015, 0.05), pickup=0.25, curve=0.4),
    ]
    P.paint(canvas, col, ang, passes, paper, "wisp8")
    P.finish(canvas, paper, grain=0.03, tooth=0.015)
    # soften the outer edge into the sky
    a = np.clip(canvas[..., 3] * ndimage.gaussian_filter(cov, 2.0) * 1.1, 0, 1)
    out = np.zeros((WH, WW, 4), np.float32)
    straight = canvas[..., :3] / np.maximum(canvas[..., 3:4], 1e-4)
    out[..., :3] = np.clip(straight, 0, 1) * a[..., None]
    out[..., 3] = a
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.save(PATH, out)
    return out


_W = None


def load():
    global _W
    if _W is None:
        _W = np.load(PATH) if os.path.exists(PATH) else None
    return _W


if __name__ == "__main__":
    out = paint()
    from PIL import Image
    bg = np.ones((WH, WW, 3), np.float32) * np.array([0.5, 0.55, 0.75], np.float32)
    rgb = out[..., :3] + bg * (1 - out[..., 3:4])
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(FILM, "build", "review", "a4-stage2", "wisp8.png"))
    print("wisp painted")
