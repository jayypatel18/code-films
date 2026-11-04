"""Assemble the master from the two rendered halves, cut the three stills and the X cut.

python3 finalize.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tr import film as F
from tr import shots

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = f"{ROOT}/out"
PARTS = f"{ROOT}/build/parts"
os.makedirs(OUT, exist_ok=True)

# 1. master: already rendered in one pass by render.py encode
master = f"{OUT}/tiny-rocket-master.mp4"

# 2. stills, rendered fresh at full quality from the same frame functions
film = shots.build()
F.still(film, 104, f"{OUT}/still-closeup.png")   # 4.3 s: pleased, string released
F.still(film, 1176, f"{OUT}/still-climax.png")   # 49 s: the star over the valley
F.still(film, 1518, f"{OUT}/still-title.png")    # 63.25 s: title, hanko, name

# 3. X cut: cold open on the knot tug, then the film from the start
cues = json.load(open(f"{ROOT}/cues.json"))
a, b = cues["cold_open"]
xcut = f"{OUT}/tiny-rocket-x.mp4"
fc = (f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[c];"
      f"[0:v]setpts=PTS-STARTPTS[m];[c][m]concat=n=2:v=1:a=0[v]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", master, "-filter_complex", fc, "-map", "[v]", "-an",
                "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", xcut], check=True)
print("done")
