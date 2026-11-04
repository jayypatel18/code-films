"""Contact sheets per paper cut: paper over A2's world (bg, fg, glow), 8 frames
spread over the cut, quarter size, frame numbers burned in as tick marks.

    python3 tools/heavy.py -- python3 -m paper.contact [cuts]     (from films/the-last-page)
"""

import os
import sys
import time

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
import paper  # noqa: E402

OUT = os.path.join(FILM, "build", "review", "a4-stage2")


def frame(cut, f):
    pl = paper.render(cut, f)
    try:
        import world
        w = world.render(cut, f)
    except Exception:
        w = {"bg": None}
    if w.get("bg") is None:
        bg = np.zeros((TL.H, TL.W, 4), np.float32)
        bg[..., :3] = (0.1, 0.1, 0.16)
        bg[..., 3] = 1
        w = {"bg": bg}
    fr = w["bg"].copy()
    if pl["back"] is not None:
        comp.over(fr, pl["back"])
    if w.get("fg") is not None:
        comp.over(fr, w["fg"])
    if pl["front"] is not None:
        comp.over(fr, pl["front"])
    rgb = comp.flatten(fr)
    if w.get("glow") is not None:
        rgb = rgb + w["glow"]
    if pl.get("glow") is not None:
        rgb = rgb + pl["glow"]
    rgb = post.bloom(rgb, threshold=0.88, strength=0.2)
    return np.clip(rgb, 0, 1)


def sheet(cut, n=8):
    f0, f1 = TL.cut_range(cut)
    fs = np.linspace(f0, f1 - 1, n).round().astype(int)
    tiles = []
    t = time.time()
    for f in fs:
        im = frame(cut, int(f))[::4, ::4]
        tiles.append(im)
    cols = 4
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    g = np.full((rows * (h + 6) + 6, cols * (w + 6) + 6, 3), 0.12, np.float32)
    for k, im in enumerate(tiles):
        r_, c_ = divmod(k, cols)
        g[6 + r_ * (h + 6): 6 + r_ * (h + 6) + h, 6 + c_ * (w + 6): 6 + c_ * (w + 6) + w] = im
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"contact-{cut}.png")
    Image.fromarray((g * 255).astype(np.uint8)).save(path)
    print(path, "frames", list(fs), f"{(time.time() - t) / n:.1f} s a frame", flush=True)
    return path


if __name__ == "__main__":
    cuts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["6a", "6b", "6c", "7", "8", "9a", "9b"]
    for c in cuts:
        sheet(c)
