"""Stage 1 review stills of the great bird (cut 8): formed, mid wingbeat, mid arc.

    python3 -m paper.stage1 [frames]      (run from films/the-last-page)

Composited over A2's cut 8 plate if it exists, else over the painted review
sky (paper/sky.py), with a placeholder balcony and A1's fox for scale, then
finished with engine.post like the lead's compositor.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FILM, "..", ".."))
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

import timeline as TL  # noqa: E402
from engine import comp, post  # noqa: E402
from engine.noise import fbm  # noqa: E402
from paper import scene8b as scene8, sky  # noqa: E402

OUT = os.path.join(FILM, "build", "review", os.environ.get("A4_REVIEW", "a4-stage1"))
W, H = 1920, 1080
STILLS = {"formed": int(round(46.3 * 24)), "wingbeat": int(round(46.62 * 24)), "blast": 1130, "arc": 1205,
          "arc-late": 1235}


def backdrop(f):
    try:
        import world
        layers = world.render("8", f)
        rgb = comp.flatten(layers["bg"])
        if layers.get("glow") is not None:
            rgb = rgb + layers["glow"]
        return rgb, layers
    except Exception:
        pass
    return sky.load().copy(), None


def balcony(f):
    """Placeholder stone balcony corner, bottom left, in shadow with a warm rim."""
    lay = np.zeros((H, W, 4), np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    z, cx, cy = TL.camera("8", f)
    top = (1240 - cy) * z + 540
    x_end = (760 - cx) * z + 960
    edge = top + (xs - x_end).clip(0) * 0.9
    m = (ys > edge) & (xs < x_end + 140)
    # balusters
    bal = np.zeros_like(m)
    for k in range(9):
        bx = (40 + k * 78 - cx) * z + 960
        bal |= (np.abs(xs - bx) < 9 + 3 * np.cos((ys - top) * 0.08)) & (ys > top - 70 * z) & (ys < top)
    rail = (ys > top - 86 * z) & (ys < top - 70 * z) & (xs < x_end + 20)
    mm = m | bal | rail
    n = fbm(H, W, 30, "balc", octaves=3)
    col = np.array([0.30, 0.22, 0.30], np.float32) * (0.85 + 0.3 * n[..., None])
    rim = (mm & ~np.roll(mm, 3, axis=1)) | (mm & ~np.roll(mm, 3, axis=0))
    col = np.where(rim[..., None], np.array([0.95, 0.70, 0.45], np.float32), col)
    lay[..., :3] = col * mm[..., None]
    lay[..., 3] = mm
    return lay


def fox_layer(f):
    try:
        from actors import fox
        a = TL.ANCHORS["8"]
        sx, sy = TL.plate_to_screen("8", f, *a["fox_paws"])
        z = TL.camera("8", f)[0]
        hpx = a["fox_height"] * z
        pose = fox.FoxPose(x=sx, y=sy, scale=hpx / 4.6, yaw=0.6, body="full", stance="sit",
                           head_pitch=0.55, gaze=(0.3, -0.8), mouth_open=0.3, tail_wind=0.6,
                           wind_dir=(-1.0, 0.0), fur_wind=0.5, seed=f // 2)
        tile, x0, y0, _ = fox.draw(pose, "sunrise", TL.LIGHTS["sunrise"], ss=3)
        lay = np.zeros((H, W, 4), np.float32)
        comp.place(lay, tile, x0, y0)
        return lay
    except Exception as e:  # the fox is A1's; for scale only
        print("fox unavailable:", type(e).__name__, e)
        return None


def compose(f, layers=None):
    bg, wl = backdrop(f)
    _ = "placeholder" if wl is None else "a2"
    pl = layers or scene8.render(f)
    frame = np.concatenate([bg, np.ones((H, W, 1), np.float32)], -1)
    if pl["back"] is not None:
        comp.over(frame, pl["back"])
    if _ == "placeholder":
        comp.over(frame, balcony(f))
    fx = fox_layer(f)
    if fx is not None:
        comp.over(frame, fx)
    if pl["front"] is not None:
        comp.over(frame, pl["front"])
    rgb = comp.flatten(frame)
    if pl["glow"] is not None:
        rgb = comp.add(rgb, pl["glow"])
    rgb = post.bloom(rgb, threshold=0.86, strength=0.24)
    rgb = post.grade(rgb, lift=(0.03, 0.02, 0.02), gamma=(1.03, 1.0, 0.98), gain=(1.02, 1.0, 0.97), sat=1.04,
                     contrast=1.02)
    rgb = post.vignette(rgb, amount=0.24)
    rgb = post.grain(rgb, f, amount=0.009)
    return post.to8(rgb, f)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(STILLS)
    for name in names:
        f = STILLS[name] if name in STILLS else int(name)
        img = compose(f)
        path = os.path.join(OUT, f"great-{name}-f{f:04d}.png")
        Image.fromarray(img).save(path)
        print(path)


if __name__ == "__main__":
    main()
