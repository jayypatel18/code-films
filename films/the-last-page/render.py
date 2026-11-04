"""THE LAST PAGE: compositor and renderer (lead).

Order per frame: world bg, paper back, actors (fox, lantern, paw, ink bird,
lettering), world fg, paper front, world glow, paper glow and actor glows, then
post (grade per cut, bloom, shake, vignette, grain, dither) and the fades.

    python3 render.py --stills 340,400          # single frames -> build/review/stills/
    python3 render.py --contact 3               # contact sheet of a cut -> build/review/contact/
    python3 render.py --range 0:720 --out a.mp4 # a stretch of the film streamed to ffmpeg
    python3 render.py --film                    # the whole film -> out/the-last-page-master.mp4
    python3 render.py --concat a.mp4 b.mp4      # join segments into the master
    python3 render.py --events                  # build/events.json for sound
    python3 render.py --cut3-composite          # the review frame of the close-up

Every frame is a pure function of its number, so any frame renders alone.
Run heavy work through tools/heavy.py.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import timeline as T  # noqa: E402
from engine import comp, post  # noqa: E402
from actors import anim, fox  # noqa: E402
from actors import lantern as LT  # noqa: E402

REVIEW = os.path.join(HERE, "build", "review")
OUT = os.path.join(HERE, "out")
PAPER_RGB = np.array([0.93, 0.90, 0.84], np.float32)

# one grade per cut light: lift, gamma, gain, saturation, contrast
GRADES = {
    "hall": dict(lift=(0.012, 0.010, 0.030), gamma=(1.0, 1.0, 1.04), gain=(1.0, 0.99, 1.02), sat=1.02, contrast=1.04),
    "night_out": dict(lift=(0.010, 0.010, 0.035), gamma=(1.0, 1.0, 1.05), gain=(0.98, 0.98, 1.04), sat=1.0, contrast=1.03),
    "gold_burst": dict(lift=(0.03, 0.02, 0.01), gamma=(1.04, 1.0, 0.97), gain=(1.03, 1.0, 0.95), sat=1.05, contrast=1.02),
    "sunrise": dict(lift=(0.03, 0.02, 0.02), gamma=(1.03, 1.0, 0.98), gain=(1.02, 1.0, 0.97), sat=1.04, contrast=1.02),
    "morning": dict(lift=(0.02, 0.02, 0.02), gamma=(1.02, 1.0, 1.0), gain=(1.0, 1.0, 0.99), sat=1.02, contrast=1.02),
}

_WORLD_OK = {}


def world_layers(cut_id, f):
    """A2's painted world where it exists, else the lead's placeholder."""
    if _WORLD_OK.get(cut_id, True):
        try:
            import world
            out = world.render(cut_id, f)
            _WORLD_OK[cut_id] = True
            return out
        except NotImplementedError:
            _WORLD_OK[cut_id] = False
        except Exception as e:  # a world module mid-edit must not stop a render
            print(f"[render] world {cut_id} failed ({type(e).__name__}: {e}); placeholder", file=sys.stderr)
            _WORLD_OK[cut_id] = False
    import placeholder
    return placeholder.layers(cut_id, f)


def paper_layers(cut_id, f):
    try:
        import paper
        out = paper.render(cut_id, f)
        return out if out else {}
    except Exception as e:
        print(f"[render] paper {cut_id} failed ({type(e).__name__}: {e})", file=sys.stderr)
        return {}


# ---------------------------------------------------------------- actors

_FOX_MEMO = {}


def _draw_fox(sh, ss=3):
    key = (repr(sh.fox), sh.light, sh.lantern_screen, sh.lantern_reach, sh.hard_warm)
    if key in _FOX_MEMO:
        return _FOX_MEMO[key]
    L = T.LIGHTS[sh.light]
    tile, x0, y0, fd = fox.draw(sh.fox, sh.light, L, lantern_screen=sh.lantern_screen, lantern_reach=sh.lantern_reach,
                                ss=ss, hard_warm=sh.hard_warm)
    out = (tile, x0, y0, list(fd.glow))
    if len(_FOX_MEMO) > 6:
        _FOX_MEMO.pop(next(iter(_FOX_MEMO)))
    _FOX_MEMO[key] = out
    return out


def actor_layers(cut_id, f):
    """(premultiplied layer, glows, shot) for everything the lead draws in this frame."""
    layer = comp.empty()
    glows = []
    sh = anim.shot(cut_id, f)
    lantern_tile = None
    if sh.lantern_prop is not None:
        lp = sh.lantern_prop
        tile, x0, y0, g = LT.draw(lp["x"], lp["y"], lp["size"], lp.get("swing", 0.0), lp.get("flame_lean", 0.0),
                                  flicker=0.5 + 0.5 * math.sin((f // 2) * 1.7))
        lantern_tile = (tile, x0, y0)
        if g:
            glows.append(g)
    ex = sh.extras
    # ink sits on the paper, under the paw that makes it
    if ex.get("page9a") or ex.get("title"):
        from actors import title
        title.draw_page(layer, cut_id, f)
    if ex.get("title"):
        from actors import title
        title.draw_title(layer, f)
    if ex.get("inkbird"):
        from actors import inkbird
        inkbird.draw(layer, f)
    if sh.fox is not None:
        tile, x0, y0, g = _draw_fox(sh)
        comp.place(layer, tile, x0, y0)
        glows += g
    if lantern_tile is not None:
        # the set-down lantern stands in front of the fox (it put it down before itself)
        comp.place(layer, *lantern_tile)
    if ex.get("paw"):
        from actors import paw
        tile, x0, y0 = paw.draw(**ex["paw"])
        comp.place(layer, tile, x0, y0)
    return layer, glows, sh


def shake(cut_id, f):
    """Whole-frame shake offsets (px), on twos, decaying after each hit."""
    t = (f - f % 2) / T.FPS
    c = T.CUES
    amp = 0.0
    if cut_id in ("5a", "5b", "5c"):
        amp = 1.5 if cut_id == "5a" else 2.5
        if cut_id == "5a" and t < c["tremble_one"]:
            amp = 0.0
    if cut_id == "6a":
        amp = 14.0 * math.exp(-max(0.0, t - c["burst"]) * 2.2)
    if cut_id == "6c" and t >= c["shutters_break"]:
        amp = 10.0 * math.exp(-(t - c["shutters_break"]) * 3.0)
    if amp < 0.3:
        return 0, 0
    r = np.random.default_rng(int(f // 2) * 7919 + 17)
    return int(round(r.normal(0, amp))), int(round(r.normal(0, amp * 0.7)))


def _shift(rgb, dx, dy):
    if dx == 0 and dy == 0:
        return rgb
    pad = max(abs(dx), abs(dy))
    p = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[pad - dy:pad - dy + rgb.shape[0], pad - dx:pad - dx + rgb.shape[1]]


_DOF = {}


def _close_up_bg(bg, cut_id, f):
    """Shallow depth of field behind a close-up: the library goes soft, darker and
    cooler so the fox pops, with the lantern's spill warming the bottom right.
    Applied by the compositor until the world plate for the cut is painted that way."""
    from scipy import ndimage
    key = (cut_id, f // 2)
    if key in _DOF:
        return _DOF[key].copy()
    rgb = bg[..., :3]
    small = rgb[::2, ::2]
    small = ndimage.gaussian_filter(small, (5.0, 5.0, 0))
    soft = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[: rgb.shape[0], : rgb.shape[1]]
    lum = soft @ np.array([0.299, 0.587, 0.114], np.float32)
    soft = lum[..., None] + (soft - lum[..., None]) * 0.62           # less saturated
    soft = soft * np.array([0.60, 0.60, 0.72], np.float32)          # darker, cooler
    ys, xs = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]].astype(np.float32)
    lx, ly = T.ANCHORS[cut_id].get("lantern_glow_from_screen", (1450, 1200)) if cut_id in T.ANCHORS else (1450, 1200)
    d = np.sqrt(((xs - lx) / 900.0) ** 2 + ((ys - ly) / 700.0) ** 2)
    warm = np.clip(1.0 - d, 0, 1) ** 1.6
    soft = soft * (1 + warm[..., None] * np.array([0.9, 0.45, 0.1], np.float32)) + warm[..., None] * np.array([0.05, 0.025, 0.0])
    out = bg.copy()
    out[..., :3] = soft * bg[..., 3:4]
    if len(_DOF) > 4:
        _DOF.pop(next(iter(_DOF)))
    _DOF[key] = out
    return out.copy()


DOF_CUTS = ()   # cut 3 now has A2's dark defocused plate (c3_bg, 19:25)


def compose_rgb(f):
    """Float RGB of frame f before post."""
    cut_id = T.cut_at(f)
    w = world_layers(cut_id, f)
    if cut_id in DOF_CUTS:
        w = dict(w)
        w["bg"] = _close_up_bg(w["bg"], cut_id, f)
    pl = paper_layers(cut_id, f)
    frame = w["bg"].copy()
    if pl.get("back") is not None:
        comp.over(frame, pl["back"])
    actors, glows, sh = actor_layers(cut_id, f)
    comp.over(frame, actors)
    if w.get("fg") is not None:
        comp.over(frame, w["fg"])
    if pl.get("front") is not None:
        comp.over(frame, pl["front"])
    rgb = comp.flatten(frame)
    if w.get("glow") is not None:
        rgb = comp.add(rgb, w["glow"])
    if pl.get("glow") is not None:
        rgb = comp.add(rgb, pl["glow"])
    for (gx, gy, gr, gc) in glows:
        rgb = comp.add(rgb, post.radial_glow(T.H, T.W, gx, gy, gr * 3.0, gc, 0.55))
    return rgb, cut_id


def finish(rgb, f, cut_id):
    lname = T.CUT_LIGHT[cut_id]
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = post.bloom(rgb, threshold=0.88, strength=0.22)
    rgb = post.grade(rgb, **GRADES.get(lname, {}))
    dx, dy = shake(cut_id, f)
    rgb = _shift(rgb, dx, dy)
    rgb = post.vignette(rgb, amount=0.26 if lname in ("hall", "night_out") else 0.18)
    t = f / T.FPS
    c = T.CUES
    if t < c["fade_up_end"]:
        k = min(max((t - c["fade_up_start"]) / (c["fade_up_end"] - c["fade_up_start"]), 0.0), 1.0)
        rgb = rgb * (k * k * (3 - 2 * k))
    if t >= c["fade_paper_start"]:
        k = min(max((t - c["fade_paper_start"]) / (c["fade_black_start"] - c["fade_paper_start"]), 0.0), 1.0)
        rgb = rgb * (1 - k) + PAPER_RGB * k
        if t >= c["fade_black_start"]:
            k2 = min(max((t - c["fade_black_start"]) / (c["end"] - c["fade_black_start"] - 2 / T.FPS), 0.0), 1.0)
            rgb = rgb * (1 - k2)
    rgb = post.grain(rgb, f, amount=0.009)
    return post.to8(rgb, f)


def frame8(f):
    rgb, cut_id = compose_rgb(f)
    return finish(rgb, f, cut_id)


# ---------------------------------------------------------------- outputs

def save(rgb8, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb8).save(path)


def ffmpeg_writer(path, crf=14):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{T.W}x{T.H}",
           "-r", str(T.FPS), "-i", "-", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", str(crf), "-preset", "medium", "-r", str(T.FPS), path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_range(f0, f1, path, crf=14):
    proc = ffmpeg_writer(path, crf)
    t0 = time.time()
    for f in range(f0, f1):
        proc.stdin.write(frame8(f).tobytes())
        if (f - f0) % 24 == 0:
            print(f"[render] frame {f} ({T.cut_at(f)}) {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"[render] {path}: frames {f0}..{f1 - 1} in {time.time() - t0:.0f} s", file=sys.stderr)


def contact(cut_id, step=4, cols=6):
    from PIL import Image
    f0, f1 = T.cut_range(cut_id)
    frames = list(range(f0, f1, step))
    tw, th = 480, 270
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
    for i, f in enumerate(frames):
        im = Image.fromarray(frame8(f)).resize((tw, th), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    out = os.path.join(REVIEW, "contact", f"cut{cut_id}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    return out


def write_events():
    ev = anim.events()
    for modname in ("inkbird", "title"):
        try:
            mod = __import__("actors." + modname, fromlist=["events"])
            ev.update(mod.events())
        except Exception as e:
            print(f"[render] events from {modname} skipped: {e}", file=sys.stderr)
    out = os.path.join(HERE, "build", "events.json")
    with open(out, "w") as fh:
        json.dump(ev, fh, indent=1)
    return out


# seconds of cut 3 that open the X cut: it starts on frame 362, a new drawing with the
# right ear mid-swing, then the sniffs and the blink with its held closed frame
X_COLD_OPEN = (362 / 24.0, 16.9)


def deliver(master=None, wav=None):
    """Mux the master with the score, and cut the X version (cold open, then the film)."""
    master = master or os.path.join(OUT, "the-last-page-master.mp4")
    wav = wav or os.path.join(HERE, "build", "audio", "the-last-page.wav")
    full = os.path.join(OUT, "the-last-page-full.mp4")
    xcut = os.path.join(OUT, "the-last-page-x.mp4")
    enc = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
           "-r", str(T.FPS), "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-movflags", "+faststart"]
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", master, "-i", wav, "-map", "0:v", "-map", "1:a",
                           "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", full])
    a, b = X_COLD_OPEN
    cold = os.path.join(os.path.dirname(wav), "the-last-page-coldopen.wav")
    if os.path.exists(cold):
        # A3's cold open mix (close-up sounds and the motif head), exactly the cold open's length
        fc = (f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v0];[0:v]setpts=PTS-STARTPTS[v1];"
              f"[2:a]apad=whole_dur={b - a:.4f},atrim=end={b - a:.4f},asetpts=PTS-STARTPTS[a0];"
              f"[1:a]asetpts=PTS-STARTPTS[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        ins = ["-i", master, "-i", wav, "-i", cold]
    else:
        fc = (f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v0];[0:v]setpts=PTS-STARTPTS[v1];"
              f"[1:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,afade=t=out:st={b - a - 0.06:.3f}:d=0.06[a0];"
              f"[1:a]asetpts=PTS-STARTPTS[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        ins = ["-i", master, "-i", wav]
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y"] + ins + ["-filter_complex", fc,
                           "-map", "[v]", "-map", "[a]"] + enc + [xcut])
    return full, xcut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default=None, help="comma separated frame numbers")
    ap.add_argument("--contact", default=None, help="cut id(s), comma separated")
    ap.add_argument("--range", default=None, help="f0:f1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--concat", nargs="*", default=None)
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--cut3-composite", action="store_true")
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--deliver", action="store_true", help="mux audio and cut the X version from the master")
    args = ap.parse_args()
    if args.cut3_composite:
        out = args.out or os.path.join(REVIEW, "a1-stage1", "cut3-composite.png")
        save(frame8(T.sec(14.25)), out)
        print(out)
    if args.stills:
        for s in args.stills.split(","):
            f = int(s)
            out = os.path.join(REVIEW, "stills", f"f{f:04d}.png")
            save(frame8(f), out)
            print(out)
    if args.contact:
        for c in args.contact.split(","):
            print(contact(c))
    if args.range:
        a, b = (int(x) for x in args.range.split(":"))
        render_range(a, b, args.out or os.path.join(OUT, f"seg_{a:04d}_{b:04d}.mp4"), args.crf)
    if args.film:
        render_range(0, T.FRAMES, os.path.join(OUT, "the-last-page-master.mp4"), args.crf)
    if args.concat:
        lst = os.path.join(OUT, "concat.txt")
        with open(lst, "w") as fh:
            for p in args.concat:
                fh.write(f"file '{os.path.abspath(p)}'\n")
        out = args.out or os.path.join(OUT, "the-last-page-master.mp4")
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                               "-c", "copy", out])
        print(out)
    if args.events:
        print(write_events())
    if args.deliver:
        for p in deliver(args.out):
            print(p)


if __name__ == "__main__":
    main()
