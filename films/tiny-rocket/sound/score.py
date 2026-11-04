"""TINY ROCKET: render the whole soundtrack.

    python3 score.py            render both WAVs and measure them
    python3 score.py --quick    skip the loudness loop (one pass, for tests)

Outputs (48 kHz, stereo, 24 bit PCM):
    films/tiny-rocket/build/audio/tiny-rocket.wav           65.000 s
    films/tiny-rocket/build/audio/tiny-rocket-coldopen.wav   1.750 s

Every cue comes from films/tiny-rocket/cues.json. Nothing is sampled.
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dsp  # noqa: E402
import music  # noqa: E402
import sfx  # noqa: E402
from dsp import SR, Bus, db, env_points, n_of  # noqa: E402

FILM = os.path.dirname(HERE)
CUES_PATH = os.path.join(FILM, "cues.json")
OUT_DIR = os.path.join(FILM, "build", "audio")
OUT_FILM = os.path.join(OUT_DIR, "tiny-rocket.wav")
OUT_COLD = os.path.join(OUT_DIR, "tiny-rocket-coldopen.wav")

TARGET_LUFS = -14.0
CEILING_DBTP = -1.3


class Mixer:
    """Four buses: dry effects, dry music, a hall send and a field send."""

    def __init__(self, seconds):
        self.len = seconds
        self.buses = {name: Bus(seconds) for name in ("sfx", "music", "hall", "field")}

    def put(self, bus, sig, at, gain_db=0.0, p=0.0, send_db=None):
        sig = np.asarray(sig, dtype=np.float64)
        g = db(gain_db)
        self.buses[bus].add(sig, at, g, p)
        if send_db is not None:
            send_bus = "hall" if bus == "music" else "field"
            self.buses[send_bus].add(sig, at, g * db(send_db), p)

    def put_hall(self, sig, at, gain_db, p=0.0):
        """Effects that belong in the big space (ignition, the climax)."""
        self.buses["hall"].add(np.asarray(sig, dtype=np.float64), at, db(gain_db), p)


# --------------------------------------------------------------- the effects

def place_effects(m, c):
    """Every sound effect from cues.json, within a frame of its cue."""
    put = m.put

    # ---- the dusk bed: crickets and a breeze, then the night air
    put("sfx", sfx.crickets(13.0, density=7, seed=1, level=0.3), 0.0, -24.0, 0.0, -18.0)
    put("sfx", dsp.fade(sfx.crickets(9.0, density=4, rate_scale=0.7, seed=2, level=0.3), 2.0, 2.5), 12.0, -27.0, 0.0, -18.0)
    put("sfx", dsp.fade(sfx.breeze(13.0, seed=1, level=0.3, gustiness=1.0), 0.0, 1.5), 0.0, -24.0)
    put("sfx", dsp.fade(sfx.night_air(8.0, seed=1, level=0.3), 2.0, 1.5), 12.0, -24.0)
    put("sfx", dsp.fade(sfx.breeze(11.0, seed=3, level=0.3, gustiness=0.6, grass=0.4), 2.0, 0.5), 18.5, -31.0)
    put("sfx", dsp.fade(sfx.breeze(6.5, seed=4, level=0.3, gustiness=0.5, grass=0.3), 0.2, 1.0), 30.5, -33.0)
    # two far birds before dawn
    put("sfx", sfx.morning_birds(6.0, seed=9, level=0.3, count=1), 20.0, -36.0, 0.6, -14.0)

    # ---- shot 1: the knot, string, ears, tail, blinks
    put("sfx", sfx.knot_creak(), c["knot_tug"] - 0.2, -13.0, -0.25, -16.0)
    put("sfx", sfx.string_release(), c["string_release"], -22.0, 0.1, -18.0)
    for i, t in enumerate(c["ear_flick"]):
        put("sfx", sfx.ear_flick(seed=i), t, -27.0 if t < 12 else -32.0, 0.25, -18.0)
    for i, t in enumerate(c["tail_swish"]):
        put("sfx", sfx.tail_swish(seed=i), t - 0.05, -27.0 if t < 5 else -31.0, -0.3, -18.0)
    for i, t in enumerate(c["blink"]):
        g = -31.0 if t < 5 else -36.0
        put("sfx", sfx.blink(seed=i), t, g, 0.0)

    # ---- shot 2: paw steps on the hill, the village lights, the map
    for i, t in enumerate(c["paw_steps"]):
        if t < 12:
            g, w = -26.0, 0.6
        elif t < 25:
            g, w = -24.0, 0.8
        else:
            g, w = -22.0, 1.0
        put("sfx", sfx.paw_step(seed=i, weight=w), t - 0.01, g, -0.2 + 0.08 * (i % 3), -16.0)
    for i, t in enumerate(c["village_lights"]):
        put("sfx", sfx.village_light(seed=i), t, -40.0, -0.7 + 1.4 * (i / 10.0), -10.0)
    put("sfx", sfx.paper_rustle(seed=1, length=0.35), c["turn_to_map"] + 0.3, -32.0, 0.3, -18.0)

    # ---- shot 3: the sleeping breath, the shooting star, fireflies
    put("sfx", dsp.fade(sfx.breath_cycle(7.4, period=4.2, seed=1, level=0.3), 1.0, 0.8), 12.0, -34.0, -0.1)
    star_p = np.linspace(-0.8, 0.7, n_of(1.1))
    put("sfx", dsp.pan_moving(sfx.shooting_star(), star_p), c["shooting_star"] - 0.05, -27.0, 0.0, -10.0)
    rng = dsp.rng_for("fireflies")
    f0, f1 = c["fireflies"]
    for i in range(26):
        t = rng.uniform(f0 + 0.6, f1 - 0.3)
        put("sfx", sfx.firefly(seed=i), t, rng.uniform(-44.0, -38.0), rng.uniform(-0.9, 0.9), -14.0)

    # ---- shot 4: wake, yawn, stand, climb into the cockpit, goggles, swallow, nod
    put("sfx", sfx.cloth(seed=1, length=0.3), c["wake"], -30.0, 0.0, -18.0)
    put("sfx", sfx.yawn(), c["yawn"], -27.0, 0.0, -18.0)
    put("sfx", sfx.fur_rustle(seed=2, length=0.35), c["stand"], -30.0, 0.1, -18.0)
    put("sfx", sfx.cloth(seed=3, length=0.25), c["crouch"], -31.0, 0.0, -18.0)
    put("sfx", sfx.fur_rustle(seed=4, length=0.3), c["climb_jump"], -26.0, 0.0, -16.0)
    put("sfx", sfx.paw_step(seed=40, weight=0.5), c["climb_jump"], -30.0, 0.0, -16.0)
    put("sfx", sfx.teacup_land(), c["land_in_cockpit"], -20.0, 0.0, -14.0)
    put("sfx", sfx.goggles_snap(), c["goggles_down"] - 0.07, -24.0, 0.05, -18.0)
    put("sfx", sfx.swallow(), c["swallow"], -28.0, 0.0)
    put("sfx", sfx.cloth(seed=5, length=0.18), c["nod"], -34.0, 0.0)

    # ---- shot 5: the match, the fuse, the ticking, the stillness
    put("sfx", sfx.match_strike(), c["match_strike"] - 0.02, -16.0, -0.1, -16.0)
    fuse_len = c["fuse_end"] - c["fuse_start"]
    fz = sfx.fuse(fuse_len)
    fuse_p = np.linspace(-0.7, 0.6, fz.shape[0])
    put("sfx", dsp.pan_moving(fz, fuse_p), c["fuse_start"], -22.0, 0.0, -18.0)
    # ticking: accelerating from 25.0 to the squeeze, stopping dead at 29.35
    t = 25.0
    k = 0
    while t < 29.36:
        put("sfx", sfx.tick(seed=k), t, -25.0 + (k % 2) * -2.0, 0.15 if k % 2 else -0.15, -20.0)
        u = (t - 25.0) / 4.4
        t += 0.46 - 0.17 * u
        k += 1
    put("sfx", sfx.cloth(seed=6, length=0.2), c["eyes_squeeze"], -32.0, 0.0)
    put("sfx", sfx.blink(seed=99), c["one_eye_open"], -34.0, 0.0)

    # ---- shot 6: ignition, shudder, liftoff, the roar, the birds
    put("sfx", sfx.ignition(), c["ignition"], -6.0, 0.0)
    m.put_hall(sfx.ignition(), c["ignition"], -16.0)
    s0, s1 = c["shudder"]
    put("sfx", sfx.tin_rattle(s1 - s0, seed=1, density=32), s0, -16.0, -0.2, -16.0)
    put("sfx", sfx.paper_flutter(s1 - s0, seed=1, rate=24.0), s0, -18.0, 0.25, -16.0)
    roar_len = 15.0
    n = n_of(roar_len)
    lvl = env_points(n, [(0, 0), (0.12, 0.6), (1.2, 1.0), (2.2, 1.0), (3.5, 0.85), (5.0, 0.5), (6.0, 0.3),
                          (6.5, 0.4), (9.5, 0.4), (11.6, 0.35), (13.0, 0.12), (15.0, 0.0)])
    pitch = env_points(n, [(0, 1.0), (2.5, 1.0), (4.0, 0.6), (6.0, 0.3), (6.5, 0.55), (11.6, 0.5), (13.0, 0.2), (15.0, 0.1)])
    put("sfx", sfx.rocket_roar(roar_len, lvl, pitch), c["ignition"], -11.0, 0.0)
    m.put_hall(sfx.rocket_roar(roar_len, lvl, pitch, seed=2), c["ignition"], -24.0)
    put("sfx", sfx.bird_burst(), c["birds"] - 0.02, -16.0, 0.0, -14.0)

    # ---- shot 7: wind, cloud layers, fins rattling, open sky
    wl = 9.0
    n = n_of(wl)
    inten = env_points(n, [(0, 0.55), (3.5, 1.0), (5.6, 1.0), (6.4, 0.45), (7.0, 0.25), (9.0, 0.0)])
    close = env_points(n, [(0, 0.0), (3.4, 0.0), (3.6, 1.0), (5.6, 1.0), (6.2, 0.1), (9.0, 0.0)])
    put("sfx", sfx.wind_high(wl, seed=1, level=0.6, intensity=inten, closeness=close), c["shots"]["s7a_clouds"], -14.0)
    for i, t in enumerate(c["cloud_layers"]):
        put("sfx", sfx.cloud_punch(seed=i), t - 0.04, -14.0, 0.0)
    for i, t in enumerate(c["fins_rattle"]):
        put("sfx", sfx.paper_flutter(1.3, seed=10 + i, rate=19.0 + 4 * i), t, -20.0, 0.3, -18.0)
        put("sfx", sfx.tin_rattle(1.3, seed=10 + i, density=22, brightness=1.1), t, -22.0, -0.3, -18.0)
    put("sfx", sfx.open_sky_whoosh(), c["break_into_open_sky"] - 0.9, -14.0)
    m.put_hall(sfx.open_sky_whoosh(), c["break_into_open_sky"] - 0.9, -22.0)

    # ---- shot 8: the far speck, the star igniting, its sheen across the sky
    r0, r1 = c["rocket_speck"]
    put("sfx", dsp.fade(sfx.speck_rumble(r1 - r0 + 0.5, seed=1), 0.3, 0.4), r0, -30.0, 0.1)
    put("sfx", sfx.star_ignite(), c["star_ignites"] - 0.05, -14.0)
    m.put_hall(sfx.star_ignite(), c["star_ignites"] - 0.05, -20.0)
    sheen = sfx.star_sheen(5.5)
    sheen_p = np.linspace(-0.7, 0.8, sheen.shape[0])
    put("sfx", dsp.pan_moving(sheen, sheen_p), c["star_ignites"] + 0.4, -30.0, 0.0)
    m.put_hall(dsp.pan_moving(sheen, sheen_p), c["star_ignites"] + 0.4, -36.0)

    # ---- shot 9: parachute, descent, touchdown, climbing out, the map, the brush
    put("sfx", sfx.parachute_open(), c["parachute_open"] - 0.02, -14.0, 0.0, -14.0)
    m.put_hall(sfx.parachute_open(), c["parachute_open"] - 0.02, -26.0)
    put("sfx", dsp.fade(sfx.canopy_sway(3.0, seed=1), 0.5, 0.3), c["parachute_open"] + 0.6, -32.0, 0.0, -16.0)
    put("sfx", dsp.fade(sfx.breeze(13.5, seed=7, level=0.3, gustiness=0.5, grass=0.5), 1.5, 1.0), 52.0, -30.0)
    put("sfx", dsp.fade(sfx.morning_birds(12.5, seed=3, level=0.3, count=5), 1.0, 1.0), 52.3, -31.0, 0.0, -14.0)
    put("sfx", sfx.touchdown(), c["touchdown"] - 0.02, -16.0, 0.0, -16.0)
    p0, p1 = c["parachute_collapse"]
    put("sfx", dsp.fade(sfx.canopy_sway(p1 - p0, seed=2), 0.1, 0.4), p0, -27.0, 0.1, -16.0)
    put("sfx", sfx.cloth(seed=8, length=0.5), p0 + 0.3, -29.0, 0.1, -16.0)
    put("sfx", sfx.tin_rattle(0.25, seed=30, density=10), c["at_window"], -30.0, 0.0, -16.0)
    put("sfx", sfx.fur_rustle(seed=9, length=0.3), c["hop_out"], -28.0, 0.0, -16.0)
    put("sfx", sfx.paw_step(seed=60, weight=1.0), c["land"] - 0.01, -22.0, 0.0, -16.0)
    put("sfx", sfx.paw_step(seed=61, weight=0.7), c["land"] + 0.07, -27.0, 0.15, -16.0)
    put("sfx", sfx.cloth(seed=10, length=0.3), c["look_up_fading_star"], -33.0, 0.0)
    put("sfx", sfx.paper_rustle(seed=2), c["take_out_map"], -26.0, 0.2, -16.0)
    put("sfx", sfx.brush_stroke(length=0.22, seed=1, pressure=0.6), c["brush_stroke"], -32.0, 0.1, -16.0)
    put("sfx", sfx.cloth(seed=11, length=0.25), c["look_at_us"], -34.0, 0.0)
    for i, t in enumerate(c["tail_curl"]):
        put("sfx", sfx.fur_rustle(seed=20 + i, length=0.35), t, -31.0, -0.3, -18.0)

    # ---- the title: two brush strokes, the hanko, the small name strokes
    t0, t1 = c["title_brush"]
    put("sfx", sfx.brush_stroke(length=0.65, seed=2, pressure=1.0), t0, -26.0, -0.2, -14.0)
    put("sfx", sfx.brush_stroke(length=0.6, seed=3, pressure=0.9), t0 + 0.8, -27.0, 0.2, -14.0)
    put("sfx", sfx.hanko(), c["hanko"], -22.0, 0.1, -14.0)
    n0, n1 = c["model_name"]
    for i in range(4):
        put("sfx", sfx.brush_stroke(length=0.14, seed=10 + i, pressure=0.5), n0 + i * (n1 - n0) / 4.0, -34.0, -0.1 + 0.07 * i, -16.0)


# --------------------------------------------------------- stillness, master

def stillness_gate(seconds, c):
    """Per sample gain that cuts everything at stillness_start and brings it
    back at stillness_end (the beat when nothing happens)."""
    n = n_of(seconds)
    a, b = c["stillness_start"], c["stillness_end"]
    return env_points(n, [(0, 1.0), (a - 0.03, 1.0), (a, 0.0), (b, 0.0), (b + 0.4, 1.0), (seconds, 1.0)])


TRIM_SHOT1_DB = 7.0

# scene trim in dB: lifts the intimate scenes so they are audible on a phone
# and shaves the loud block so the limiter barely works. Measured from the
# first full render (per shot short term loudness).
TRIM_POINTS = [(0.0, TRIM_SHOT1_DB), (4.8, TRIM_SHOT1_DB), (5.3, 0.5), (11.6, 0.5), (12.2, 4.0), (18.8, 4.0),
               (19.4, 8.0), (24.8, 8.0), (25.1, 7.0), (29.4, 7.0), (30.9, 7.0), (31.0, -1.5), (43.8, -1.5),
               (44.3, 1.0), (45.6, 1.0), (46.0, -1.5), (52.0, -1.0), (52.6, 3.0), (59.6, 3.0), (60.0, 5.0), (65.0, 5.0)]


def scene_trim(seconds):
    n = n_of(seconds)
    return db(env_points(n, TRIM_POINTS))


def air_floor(seconds, c):
    """A whisper of air during the stillness so it is a held breath rather
    than a dropout: about 55 dB under the mix."""
    n = n_of(seconds)
    rng = dsp.rng_for("airfloor")
    a, b = c["stillness_start"], c["stillness_end"]
    env = env_points(n, [(0, 0), (a - 0.03, 0), (a + 0.02, 1.0), (b, 1.0), (b + 0.3, 0), (seconds, 0)])
    out = np.zeros((n, 2))
    for ch in range(2):
        out[:, ch] = dsp.lowpass(dsp.white(n, rng), 1500.0, 2) * env
    return out / (np.max(np.abs(out)) + 1e-9) * db(-52.0)


def master_chain(x):
    """Gentle top shelf (nothing harsh above 8 kHz), a sub rumble guard, and
    the lookahead limiter."""
    y = dsp.highpass(x, 24.0, order=2)
    y = dsp.highshelf(y, 8500.0, -4.0)
    y = dsp.lowpass(y, 15000.0, order=2)
    return y


def measure(path):
    """Integrated loudness, LRA and true peak from ffmpeg's ebur128."""
    cmd = ["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    txt = res.stderr
    tail = txt[txt.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    lra = re.search(r"LRA:\s+([\d.]+) LU", tail)
    pk = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    return (float(i.group(1)) if i else None, float(lra.group(1)) if lra else None,
            float(pk.group(1)) if pk else None, tail.strip())


def k_weight(x):
    """BS.1770 K weighting: the head shelf and the low cut, as RBJ biquads."""
    y = dsp.highshelf(x, 1681.0, 3.99, slope=0.7)
    y = dsp.highpass(y, 38.0, order=2)
    return y


def section_loudness(x, c):
    """Short term loudness (3 s windows) per shot, to check the quiet parts."""
    y = k_weight(x)
    p = y ** 2
    win = n_of(3.0)
    kernel = np.ones(win) / win
    ms = np.convolve(p[:, 0], kernel, "same") + np.convolve(p[:, 1], kernel, "same")
    lufs = -0.691 + 10 * np.log10(ms + 1e-12)
    rows = []
    shots = list(c["shots"].items())
    for (name, t0), (_, t1) in zip(shots, shots[1:]):
        a, b = n_of(t0), n_of(t1)
        seg = lufs[a:b]
        rows.append((name, t0, t1, float(np.max(seg)), float(np.median(seg))))
    return rows


# ------------------------------------------------------------------- render

def render_film(c, quick=False):
    seconds = c["duration"]
    m = Mixer(seconds)
    t0 = time.time()
    music.arrange(m.put)
    print(f"music placed in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    place_effects(m, c)
    print(f"effects placed in {time.time() - t0:.1f} s", flush=True)

    hall = dsp.Reverb(rt60=2.8, damping=0.45, predelay=0.02, size=1.0, early_gain=0.35, seed=3)
    field = dsp.Reverb(rt60=0.75, damping=0.6, predelay=0.008, size=0.55, early_gain=0.5, seed=5)
    t0 = time.time()
    wet_hall = hall.process(m.buses["hall"].buf)
    wet_field = field.process(m.buses["field"].buf)
    print(f"reverbs in {time.time() - t0:.1f} s", flush=True)

    gate = stillness_gate(seconds, c)
    mix = (m.buses["sfx"].buf + m.buses["music"].buf + wet_hall * db(-2.0) + wet_field * db(-3.0)) * gate[:, None]
    mix *= scene_trim(seconds)[:, None]
    mix += air_floor(seconds, c)
    # the final fade: paper at 63.5, black at 64.5, silence by 65.0
    mix *= env_points(mix.shape[0], [(0, 1), (c["shots"]["fade_paper"], 1.0), (c["shots"]["black"], 0.08), (64.9, 0.0), (65.0, 0)])[:, None]
    mix = master_chain(mix)
    return mix


def finish(mix, path, quick=False, label="film"):
    """Normalise to the target loudness in a measure and correct loop, then
    limit the true peak and write."""
    os.makedirs(OUT_DIR, exist_ok=True)
    gain = 1.0
    for it in range(3 if not quick else 1):
        y = dsp.limiter(mix * gain, ceiling_db=CEILING_DBTP)
        dsp.write_wav24(path, y)
        lufs, lra, peak, _ = measure(path)
        print(f"[{label}] pass {it + 1}: gain {20 * np.log10(gain):+.2f} dB  I {lufs} LUFS  LRA {lra} LU  TP {peak} dBTP", flush=True)
        if lufs is None or quick or abs(lufs - TARGET_LUFS) < 0.25:
            break
        gain *= db(TARGET_LUFS - lufs)
    return y


def render_cold_open(c):
    """1.75 s: the knot tug (picture 0.85 to 2.6 s), the paws, a breath of
    the dusk ambience and a hint of the motif, fading to near silence."""
    seconds = 1.75
    off = c["cold_open"][0]
    m = Mixer(seconds)
    put = m.put
    put("sfx", sfx.crickets(seconds + 0.5, density=7, seed=1, level=0.3), 0.0, -24.0, 0.0, -18.0)
    put("sfx", sfx.breeze(seconds + 0.5, seed=1, level=0.3, gustiness=1.0), 0.0, -24.0)
    put("sfx", sfx.knot_creak(), c["knot_tug"] - off - 0.2, -13.0, -0.25, -16.0)
    put("sfx", sfx.paw_step(seed=70, weight=0.4), 0.02, -30.0, -0.3, -16.0)
    put("sfx", sfx.fur_rustle(seed=71, length=0.2), c["knot_tug"] - off - 0.12, -30.0, -0.2, -16.0)
    for i, t in enumerate(c["ear_flick"]):
        if off <= t < off + seconds:
            put("sfx", sfx.ear_flick(seed=i), t - off, -30.0, 0.25, -18.0)
    for i, t in enumerate(c["tail_swish"]):
        if off <= t < off + seconds:
            put("sfx", sfx.tail_swish(seed=i), t - off - 0.05, -27.0, -0.3, -18.0)
    # a hint of the motif: its first two notes, soft, in the hall
    put("music", music.felt_piano("D4", length=2.0, velocity=0.35), 0.18, -20.0, 0.05, -6.0)
    put("music", music.felt_piano("A4", length=1.6, velocity=0.3), 0.18 + music.BEAT, -23.0, 0.1, -6.0)
    hall = dsp.Reverb(rt60=2.8, damping=0.45, predelay=0.02, size=1.0, early_gain=0.35, seed=3)
    field = dsp.Reverb(rt60=0.75, damping=0.6, predelay=0.008, size=0.55, early_gain=0.5, seed=5)
    mix = m.buses["sfx"].buf + m.buses["music"].buf + hall.process(m.buses["hall"].buf) * db(-2.0) + field.process(m.buses["field"].buf) * db(-3.0)
    mix *= db(TRIM_SHOT1_DB)
    mix *= env_points(mix.shape[0], [(0, 0.0), (0.03, 1.0), (1.25, 1.0), (1.7, 0.02), (1.75, 0.0)])[:, None]
    mix = master_chain(mix)
    return mix


def main():
    quick = "--quick" in sys.argv
    with open(CUES_PATH) as fh:
        c = json.load(fh)
    start = time.time()
    mix = render_film(c, quick=quick)
    assert mix.shape[0] == 3120000, mix.shape
    y = finish(mix, OUT_FILM, quick=quick, label="film")
    print("per shot short term loudness (max / median LUFS):")
    for name, t0, t1, mx, md in section_loudness(y, c):
        print(f"  {name:16s} {t0:5.1f} to {t1:5.1f}   {mx:7.1f} / {md:7.1f}")
    lufs, lra, peak, summary = measure(OUT_FILM)
    print(summary)

    cold = render_cold_open(c)
    assert cold.shape[0] == 84000, cold.shape
    # match the film's gain so the hand over is seamless: the film's final
    # gain is the ratio between its limited output and its mix
    film_gain = np.sqrt(np.mean(y[: n_of(4.0)] ** 2)) / (np.sqrt(np.mean(mix[: n_of(4.0)] ** 2)) + 1e-12)
    cold_y = dsp.limiter(cold * film_gain, ceiling_db=CEILING_DBTP)
    dsp.write_wav24(OUT_COLD, cold_y)
    lufs_c, lra_c, peak_c, summary_c = measure(OUT_COLD)
    print(f"[cold open] gain {20 * np.log10(film_gain):+.2f} dB  I {lufs_c} LUFS  TP {peak_c} dBTP")
    print(summary_c)
    print(f"total render {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
