"""Render CLI.

python3 render.py still <frame> <out.png>
python3 render.py sheet <shot_name> <out.png>
python3 render.py encode <out.mp4> [f0 f1]
python3 render.py bg <shot_name>      (paint and cache a background only)
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tr import film as F
from tr import shots

film = shots.build()
cmd = sys.argv[1]
t0 = time.time()
if cmd == "still":
    F.still(film, int(sys.argv[2]), sys.argv[3])
elif cmd == "sheet":
    sh = [s for s in film.shots if s.name == sys.argv[2]][0]
    F.contact_sheet(film, sh, sys.argv[3])
elif cmd == "encode":
    f0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    f1 = int(sys.argv[4]) if len(sys.argv) > 4 else None
    F.encode(film, sys.argv[2], f0, f1, log=lambda m: print(m, flush=True))
elif cmd == "frames":
    # a few stills from a shot for inspection
    sh = [s for s in film.shots if s.name == sys.argv[2]][0]
    a, b = sh.frame_range()
    for f in range(a, b, int(sys.argv[3])):
        F.still(film, f, f"{sys.argv[4]}/{sh.name}_{f:05d}.png")
print(f"done in {time.time() - t0:.0f} s", flush=True)
