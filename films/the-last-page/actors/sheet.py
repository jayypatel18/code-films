"""Fox model sheet: the poses every cut needs, drawn by the same code as the film.

    python3 tools/heavy.py -- python3 films/the-last-page/actors/sheet.py [out.png]

Row 1: busts (three quarter neutral, ears swivelled, blink closed, ears flat).
Row 2: side walk keys with the lantern, seated three quarter, back three quarter
on the ledge looking up. Row 3: the blink in five drawings and a head turn.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(FILM, "..", ".."))
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

import timeline as T  # noqa: E402
from engine import comp, post, vnoise  # noqa: E402
from engine import noise as wnoise  # noqa: E402
from actors import fox  # noqa: E402

W, H = 3840, 2160


def paper(h, w):
    """Warm sheet paper with a faint grain."""
    bg = np.zeros((h, w, 4), np.float32)
    base = np.array([0.935, 0.905, 0.845], np.float32)
    g = wnoise.paper_grain(h, w, "sheetpaper")
    bg[..., :3] = base * (0.97 + 0.05 * (g[..., None] - 0.5))
    bg[..., 3] = 1.0
    return bg


def panel_border(sheet, x0, y0, w, h, key):
    """A thin hand-ruled frame line around a panel."""
    from engine import raster
    buf = np.zeros(sheet.shape[:2], np.float32)
    r = vnoise.rng("frame", key)
    corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]
    for (ax, ay), (bx, by) in zip(corners[:-1], corners[1:]):
        n = 24
        t = np.linspace(0, 1, n)
        pts = np.stack([ax + (bx - ax) * t, ay + (by - ay) * t], 1)
        pts += r.normal(0, 0.25, pts.shape)
        rad = 1.1 + 0.25 * np.sin(t * 3.1 + r.uniform(0, 6))
        raster.stroke(buf, pts, rad)
    ink = np.array([0.23, 0.15, 0.10], np.float32)
    sheet[..., :3] = sheet[..., :3] * (1 - buf[..., None] * 0.8) + ink * buf[..., None] * 0.8


def poses():
    """(pose, light, panel or None). Panels crop busts like a shot would."""
    P = fox.FoxPose
    lt = "morning"
    out = []
    # row 1: four close-up panels
    pw, ph, py = 900, 780, 70
    xs = [60 + i * (pw + 40) for i in range(4)]
    busts = [
        dict(yaw=0.50, head_pitch=0.08, gaze=(0.3, -0.2), worry=0.35),
        dict(yaw=0.45, head_pitch=0.05, head_roll=0.06, ear_swivel=(-0.55, 0.85), gaze=(0.6, -0.1), worry=0.25),
        dict(yaw=0.50, head_pitch=0.02, eye_open=0.0, worry=0.2),
        dict(yaw=0.40, head_pitch=-0.12, head_roll=-0.04, ear_flat=(1.0, 1.0), worry=0.9, gaze=(0.2, 0.25)),
    ]
    for i, kw in enumerate(busts):
        out.append((P(x=xs[i] + pw * 0.48, y=py + ph * 0.56, scale=205, seed=1 + i, **kw), lt, (xs[i], py, pw, ph)))
    # row 2: full body, root = ground point
    y2 = 1500
    out.append((P(x=520, y=y2, scale=68, yaw=1.5708, body="full", walk=0.08, stride=1.35, lantern=True, tail=(0.15, 0.0, 0.2), seed=5,
                  head_pitch=-0.05), lt, None))
    out.append((P(x=1320, y=y2, scale=68, yaw=1.5708, body="full", walk=0.58, stride=1.35, lantern=True, tail=(0.25, 0.0, 0.1),
                  body_bob=-0.08, seed=6, head_pitch=-0.03), lt, None))
    out.append((P(x=2330, y=y2, scale=72, yaw=0.62, body="full", stance="sit", head_yaw=0.12, cam_tilt=0.22,
                  gaze=(0.2, -0.1), seed=7), lt, None))
    out.append((P(x=3250, y=y2, scale=70, yaw=2.45, body="full", head_pitch=0.55, mouth_open=0.35, tail=(0.6, 0.8, 0.3),
                  cam_tilt=0.12, seed=8), "sunrise", None))
    # row 3: the blink in five drawings, then a head turn, as small head panels
    qw, qh, qy = 330, 470, 1630
    blink = [1.0, 0.5, 0.0, 0.5, 1.0]
    for i, o in enumerate(blink):
        x0 = 60 + i * (qw + 20)
        out.append((P(x=x0 + qw * 0.47, y=qy + qh * 0.55, scale=92, yaw=0.5, eye_open=o, seed=20 + i, gaze=(0.3, -0.1)),
                    lt, (x0, qy, qw, qh)))
    for i, yw in enumerate([0.0, 0.8, 1.45, 3.14159]):
        x0 = 1880 + i * (qw + 160 if i == 0 else qw + 160)
        x0 = 1880 + i * 490
        out.append((P(x=x0 + qw * 0.5, y=qy + qh * 0.55, scale=92, yaw=yw, seed=30 + i), lt, (x0, qy, qw + 120, qh)))
    return out


def main():
    outp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FILM, "build", "review", "a1-stage1", "fox-model-sheet.png")
    sheet = paper(H, W)
    for k, (pose, lt, panel) in enumerate(poses()):
        L = T.LIGHTS[lt]
        tile, x0, y0, fd = fox.draw(pose, lt, L, ss=3, frame_wh=(W, H))
        layer = comp.empty(H, W)
        comp.place(layer, tile, x0, y0)
        if panel is not None:
            px, py, pw, ph = panel
            clip = np.zeros((H, W, 1), np.float32)
            clip[py:py + ph, px:px + pw] = 1.0
            # a slightly lighter panel ground
            sheet[py:py + ph, px:px + pw, :3] = sheet[py:py + ph, px:px + pw, :3] * 0.6 + np.array([0.97, 0.95, 0.91]) * 0.4
            layer *= clip
        comp.over(sheet, layer)
        for (gx, gy, gr, gc) in fd.glow:
            sheet[..., :3] += post.radial_glow(H, W, gx, gy, gr * 2.2, gc, 0.25)
        if panel is not None:
            panel_border(sheet, *panel, key=k)
    rgb = post.to8(np.clip(sheet[..., :3], 0, 1), 0)
    from PIL import Image
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    Image.fromarray(rgb).save(outp)
    print(outp)


if __name__ == "__main__":
    main()
