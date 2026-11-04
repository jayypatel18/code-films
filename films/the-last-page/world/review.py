"""Render review stills of the world through world.render().

    python3 world/review.py OUTDIR CUT:FRAME [CUT:FRAME ...] [--crop x0,y0,x1,y1:name]
Frames are global frame numbers. Stills are flattened over black (no fox).
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import world  # noqa: E402
from world import common as C  # noqa: E402


def crops(out):
    """Brushwork crops straight from the cached plates (native plate pixels,
    then enlarged with nearest neighbour so single strokes are visible)."""
    from PIL import Image
    M = C.MARGIN

    def save(a, name, scale=1):
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        im.save(os.path.join(out, name))
    st = C.load_plate("cut1_stage")
    sky = C.load_plate("cut1_sky")
    comp = sky[..., :3] * (1 - st[..., 3:4]) + st[..., :3]
    save(comp[360 + M:620 + M, 1430 + M:1700 + M], "crop_cut1_tower_native.png")
    save(comp[440 + M:600 + M, 1470 + M:1650 + M], "crop_cut1_tower_3x.png", 3)
    save(comp[200 + M:420 + M, 1470 + M:1660 + M], "crop_cut1_dome_3x.png", 3)
    save(comp[780 + M:1080 + M, 700 + M:1200 + M], "crop_cut1_clouds_native.png")
    save(comp[480 + M:700 + M, 1080 + M:1440 + M], "crop_cut1_pine_2x.png", 2)
    hb = C.load_plate("hall_bg")
    save(hb[600 + M:800 + M, 700 + M:1100 + M, :3], "crop_cut2_books_native.png")
    save(hb[620 + M:720 + M, 800 + M:1000 + M, :3], "crop_cut2_books_3x.png", 3)
    save(hb[620 + M:900 + M, 1600 + M:2060 + M, :3], "crop_cut2_table_native.png")


def sheet(out, cut, n=6, cols=3, tw=640):
    """Contact sheet: n frames spread across the cut, flattened over black."""
    from PIL import Image
    f0, f1 = C.T.cut_range(cut)
    frames = [int(round(f0 + (f1 - 1 - f0) * i / max(n - 1, 1))) for i in range(n)]
    th = tw * 9 // 16
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * th + (rows - 1) * 6), (12, 12, 12))
    for i, f in enumerate(frames):
        img = C.flatten(world.render(cut, f))
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
        canvas.paste(im, ((i % cols) * (tw + 6), (i // cols) * (th + 6)))
    path = os.path.join(out, f"sheet_cut{cut}.png")
    canvas.save(path)
    C.log(f"{path} frames {frames}")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if "--crops" in sys.argv:
        crops(out)
    if "--sheet" in sys.argv:
        for cut in sys.argv[sys.argv.index("--sheet") + 1:]:
            sheet(out, cut)
        return
    for arg in sys.argv[2:]:
        if arg.startswith("--"):
            continue
        cut, f = arg.split(":")
        f = int(f)
        t0 = time.time()
        layers = world.render(cut, f)
        dt = time.time() - t0
        img = C.flatten(layers)
        path = os.path.join(out, f"cut{cut}_f{f:04d}.png")
        C.save_review(path, img)
        C.log(f"{path} in {dt:.2f}s")


if __name__ == "__main__":
    main()
