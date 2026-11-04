"""A painted placeholder sky for reviewing cut 8 before A2's plate exists.

Composed as a reference (values, light, cloud masses) and painted over with
A2's gouache stroke painter (engine.paint), like every background. Cached in
build/paper/sky8_review.npy. Review only: the film uses A2's plate.

    python3 tools/heavy.py -- python3 -m paper.sky      (from films/the-last-page)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FILM, "..", ".."))
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import paint as P  # noqa: E402
from engine.noise import fbm, value_noise  # noqa: E402

W, H = 1920, 1080
PATH = os.path.join(FILM, "build", "paper", "sky8_review.npy")
SUN = (1760.0, 1010.0)


def _billows(mask_lit, ys, xs, puffs, sun):
    """Puffs drawn back to front (later ones in front), each shaded as a sphere
    lit from the sun's direction, so billows read as round lit caps over
    shaded bellies."""
    cov = np.zeros(ys.shape, np.float32)
    lit = np.zeros(ys.shape, np.float32)
    s3 = np.array([0.55, -0.55, 0.62])
    s3 /= np.linalg.norm(s3)
    for (cx, cy, r) in puffs:
        x0, x1 = int(max(0, cx - r * 1.1)), int(min(W, cx + r * 1.1))
        y0, y1 = int(max(0, cy - r * 1.1)), int(min(H, cy + r * 1.1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = ys[y0:y1, x0:x1], xs[y0:y1, x0:x1]
        nx, ny = (xx - cx) / r, (yy - cy) / (r * 0.85)
        d2 = nx * nx + ny * ny
        inside = d2 < 1.0
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        l = np.clip(nx * s3[0] + ny * s3[1] + nz * s3[2], 0, 1)
        c = np.clip((1.0 - np.sqrt(d2)) * 8.0, 0, 1)
        lit[y0:y1, x0:x1] = np.where(inside, l, lit[y0:y1, x0:x1])
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], c)
    return cov, lit


def reference():
    from scipy import ndimage
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u = ys / H
    top = np.array([0.30, 0.38, 0.64], np.float32)
    mid = np.array([0.68, 0.62, 0.76], np.float32)
    low = np.array([1.00, 0.82, 0.58], np.float32)
    ds = np.hypot((xs - SUN[0]) / 1500, (ys - SUN[1]) / 1100)
    g = np.clip(1.0 - ds, 0, 1)
    k1 = np.clip(u * 1.3 + g * 0.6 - 0.12, 0, 1)
    ref = top * (1 - k1[..., None]) + mid * k1[..., None]
    k2 = np.clip(g * 1.5 - 0.35, 0, 1) ** 1.2
    ref = ref * (1 - k2[..., None]) + low * k2[..., None]
    ref = ref + np.exp(-(ds * 5.0) ** 2)[..., None] * np.array([0.25, 0.2, 0.12], np.float32)
    r = np.random.default_rng(81)
    puffs = []
    # towering cumulus, right, catching the sun
    for i in range(70):
        a = r.uniform(0, 1)
        cx = 1500 + r.normal(0, 150) + a * 180
        cy = 980 - a * 520 * r.uniform(0.6, 1.0) + r.normal(0, 30)
        puffs.append((cx, cy, r.uniform(50, 130) * (1.2 - a * 0.5)))
    # a lower mass at left behind the balcony
    for i in range(40):
        puffs.append((r.uniform(-50, 700), 860 + r.normal(0, 50), r.uniform(60, 140)))
    # the cloud sea: rows of billows getting smaller toward the horizon
    for row in range(6):
        yb = 1080 - row * 34
        rr = 110 * (0.78 ** row)
        for x in np.arange(-100, 2100, rr * 1.1):
            puffs.append((x + r.normal(0, rr * 0.3), yb + r.normal(0, rr * 0.15), rr * r.uniform(0.8, 1.2)))
    # small drifting clouds high up
    for i in range(6):
        cx, cy = r.uniform(100, 1800), r.uniform(80, 420)
        for j in range(8):
            puffs.append((cx + r.normal(0, 40), cy + r.normal(0, 10), r.uniform(14, 34)))
    puffs.sort(key=lambda p: p[1] + p[2] * 0.3)
    cov, lit = _billows(None, ys, xs, puffs, SUN)
    cov = ndimage.gaussian_filter(cov, 1.2)
    cl_lit = np.array([1.0, 0.88, 0.68], np.float32)
    cl_mid = np.array([0.93, 0.72, 0.68], np.float32)
    cl_sh = np.array([0.62, 0.53, 0.70], np.float32)
    far = np.clip((u - 0.5) * 2, 0, 1)[..., None]
    cc = np.where((lit > 0.62)[..., None], cl_lit, np.where((lit > 0.4)[..., None], cl_mid, cl_sh))
    cc = ndimage.gaussian_filter(cc, (1.5, 1.5, 0))
    cc = cc * (1 - 0.25 * (1 - far)) + ref * 0.25 * (1 - far)
    ref = ref * (1 - cov[..., None]) + cc * cov[..., None]
    return np.clip(ref, 0, 1).astype(np.float32), cov


def paint():
    ref, cl = reference()
    canvas = P.new_canvas(H, W, ref.mean((0, 1)))
    paper = P.paper_for(H, W, "sky8")
    ang = P.orientation(ref, 1.5, 6.0, base=0.1, base_weight=0.25, edge_gain=1.4)
    passes = [
        dict(radius=30, spacing=0.9, steps=(4, 12), opacity=0.96, dry=0.25, cthresh=0.12, wob=0.15,
             jitter=(0.025, 0.015, 0.04), streak=0.05, ragged=0.15, pickup=0.25, curve=0.3),
        dict(radius=14, spacing=1.0, steps=(3, 10), opacity=0.9, dry=0.45, cthresh=0.08, wob=0.2,
             jitter=(0.03, 0.015, 0.05), streak=0.05, pickup=0.3, curve=0.35, use_err=True, err_thresh=0.03),
        dict(radius=6, spacing=1.1, steps=(2, 8), opacity=0.88, dry=0.5, cthresh=0.06, wob=0.25,
             jitter=(0.03, 0.015, 0.05), streak=0.05, pickup=0.3, curve=0.35, use_err=True, err_thresh=0.04),
    ]
    P.paint(canvas, ref, ang, passes, paper, "sky8-review")
    P.finish(canvas, paper, grain=0.03, tooth=0.015)
    rgb = canvas[..., :3] + ref * (1 - canvas[..., 3:4])
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.save(PATH, rgb.astype(np.float32))
    return rgb


def load():
    if os.path.exists(PATH):
        return np.load(PATH)
    return reference()[0]


if __name__ == "__main__":
    import time
    t = time.time()
    rgb = paint()
    from PIL import Image
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(FILM, "build", "review", "a4-dev", "sky8.png"))
    print("sky painted", time.time() - t)
