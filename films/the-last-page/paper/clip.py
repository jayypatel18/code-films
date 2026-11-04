"""Motion review for paper cuts: a short clip, consecutive frame strips, and a
pop detector.

    python3 tools/heavy.py -- python3 -m paper.clip CUT [F0 F1]     (from films/the-last-page)

Writes build/review/a4-stage2/clip-<cut>.mp4 (half size, 24 fps, streamed to
ffmpeg, no PNG sequence), strip-<cut>-<f>.png (8 consecutive frames) at a few
points, and prints the frame to frame change of the paper layer alone: a
spike far above its neighbours means something popped.
"""

import os
import subprocess
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
import paper  # noqa: E402
from paper.contact import frame  # noqa: E402

OUT = os.path.join(FILM, "build", "review", "a4-stage2")


def paper_only(cut, f):
    pl = paper.render(cut, f)
    acc = np.zeros((TL.H // 4, TL.W // 4), np.float32)
    for k in ("back", "front"):
        if pl.get(k) is not None:
            acc += pl[k][::4, ::4, 3]
    return acc


def main():
    cut = sys.argv[1]
    f0, f1 = TL.cut_range(cut)
    if len(sys.argv) > 3:
        f0, f1 = int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"clip-{cut}.mp4")
    w, h = TL.W // 2, TL.H // 2
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "18", path], stdin=subprocess.PIPE)
    strips_at = set(int(x) for x in np.linspace(f0, max(f0, f1 - 8), 4).round())
    strip = None
    strip_f = None
    prev = None
    diffs = []
    for f in range(f0, f1):
        rgb = frame(cut, f)
        small = rgb[::2, ::2]
        ff.stdin.write((np.clip(small, 0, 1) * 255).astype(np.uint8).tobytes())
        a = paper_only(cut, f)
        if prev is not None:
            diffs.append((f, float(np.abs(a - prev).mean())))
        prev = a
        if f in strips_at:
            strip, strip_f = [], f
        if strip is not None:
            strip.append(rgb[::4, ::4])
            if len(strip) == 8:
                row = np.concatenate([np.concatenate(strip[:4], 1), np.concatenate(strip[4:], 1)], 0)
                Image.fromarray((np.clip(row, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(OUT, f"strip-{cut}-{strip_f}.png"))
                strip = None
    ff.stdin.close()
    ff.wait()
    d = np.array([x[1] for x in diffs])
    if len(d):
        med = np.median(d) + 1e-6
        print(f"{cut}: {len(d) + 1} frames, paper change median {med:.4f}")
        # paper on twos changes every other frame: compare against the local max of neighbours
        for i, (f, v) in enumerate(diffs):
            # compare with frames of the same parity (paper is drawn on twos)
            nb = [d[j] for j in (i - 6, i - 4, i - 2, i + 2, i + 4, i + 6) if 0 <= j < len(d)]
            ref = (np.median(nb) if nb else v) + 1e-6
            if v > 3.0 * ref and v > 0.004:
                print(f"  possible pop at f{f}: {v:.4f} vs local {ref:.4f}")
    print(path)


if __name__ == "__main__":
    main()
