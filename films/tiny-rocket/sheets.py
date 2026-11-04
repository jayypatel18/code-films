"""Contact sheets per shot from a rendered video (no re-render).

python3 sheets.py <video.mp4> <out_dir>
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tr import shots
from tr.film import FPS

video, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
tmp = os.path.join(out_dir, "_frames")
os.makedirs(tmp, exist_ok=True)
# one small frame every 6 frames
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", "select=not(mod(n\\,6)),scale=480:270", "-vsync", "vfr",
                os.path.join(tmp, "f%05d.png")], check=True)
film = shots.build()
files = sorted(os.listdir(tmp))
for sh in film.shots:
    a, b = sh.frame_range()
    idx = [i for i in range(a, b, 6)]
    tiles = []
    for i in idx:
        k = i // 6
        if k < len(files):
            tiles.append(Image.open(os.path.join(tmp, files[k])))
    if not tiles:
        continue
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 270), (16, 16, 16))
    for j, im in enumerate(tiles):
        sheet.paste(im, ((j % cols) * 480, (j // cols) * 270))
    sheet.save(os.path.join(out_dir, f"sheet-{sh.name}.png"))
    print("wrote", sh.name, len(tiles))
for f in files:
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
