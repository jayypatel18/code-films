#!/usr/bin/env python3
"""THE LAST PAGE: the original score and sound design, synthesized in code. Entry script.

Every sound comes from numpy and scipy here (no samples, no impulse response files, nothing
downloaded). Times come from films/the-last-page/timeline.py CUES.

Run (heavy job, through the slot wrapper, in the background):
    python3 tools/heavy.py -- \
        python3 films/the-last-page/sound/score.py

Options:
    --stems       also print per-bus loudness for every section
    --no-ffmpeg   skip the ffmpeg ebur128 / loudnorm measurement

Output: films/the-last-page/build/audio/the-last-page.wav
        48 kHz, stereo, 24 bit PCM, exactly 60.000 s (2,880,000 samples).
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from dsp import (FILM, FRAME, NS, SR, TIME, Bus, C, chain, compressor, convolve_stereo, curve,  # noqa: E402
                 db2lin, filt, kpower, limiter, log, loudness_curve, lufs_integrated, make_ir,
                 read_wav24, rng_for, sos_bp, sos_hp, sos_lp, sos_shelf_hi, true_peak, window_loudness,
                 write_wav24)
from music import compose, music_cue_points, render_music  # noqa: E402
from sfx import render_sfx  # noqa: E402
import coldopen  # noqa: E402

OUT_WAV = os.path.join(FILM, "build", "audio", "the-last-page.wav")
OUT_COLD = os.path.join(FILM, "build", "audio", "the-last-page-coldopen.wav")
TARGET_LUFS = -14.0
CEILING_DBTP = -1.3
RANGE_LIFT_DB = 4.0     # opening wind, first piano, ink (films session decision: X plays on phones)
RANGE_INK_DB = 6.0      # the silence after the paw: the ink drawing itself
RANGE_END_DB = 3.0      # quiet return
RANGE_LAST_DB = 6.0     # the last note, the title ink, the hanko

MUSIC = ["piano", "strings", "brass", "choir", "perc", "harp"]
SFX = ["foley", "paper", "fx", "amb"]
ROOM = ["room"]
DEFAULT_SEND = {"foley": 0.3, "paper": 0.35, "fx": 0.3, "amb": 0.12, "room": 0.2}


def space_at(t):
    if t < C("cut_hall"):
        return "out"
    if t < C("cut_sunrise"):
        return "hall"
    return "out"


class Mixer:
    """Buses plus per-space reverb sends. SFX call add(); the music renders straight to its buses."""

    def __init__(self):
        self.buses = {k: Bus(k) for k in MUSIC + SFX + ROOM}
        self.sends = {k: Bus(k) for k in ("hall", "out", "close")}
        self.events = []    # (bus, t0, audible onset offset s, peak)
        self.tagged = []    # (cue, bus, t0, audible onset offset s)

    def add(self, bus, sig, t0, gain=1.0, pan=0.0, space=None, send=None, cue=None):
        sig = np.asarray(sig, dtype=np.float64)
        self.buses[bus].add(sig, t0, gain, pan)
        s = DEFAULT_SEND[bus] if send is None else send
        sp = space or space_at(t0)
        if s > 0:
            self.sends[sp].add(sig, t0, gain * s, pan)
            if sp == "close":      # the close-ups still sit inside the big hall or the open air
                self.sends[space_at(t0)].add(sig, t0, gain * s * 0.35, pan)
        mono = np.abs(sig if sig.ndim == 1 else sig.max(axis=0))
        pk = float(mono.max()) if mono.size else 0.0
        on = int(np.argmax(mono > 0.1 * pk)) if pk > 0 else 0
        self.events.append((bus, t0, on / SR, pk * gain))
        if cue:
            self.tagged.append((cue, bus, t0, on / SR))


# ================================================================================ report helpers
SECTIONS = [
    ("1 before dawn, wind", "wind_in", "cut_hall"),
    ("2 hall, piano motif", "cut_hall", "cut_closeup"),
    ("3 close-up", "cut_closeup", "cut_lectern"),
    ("4a lectern, hop", "cut_lectern", "paw_down"),
    ("4b silence, ink, bird", "paw_down", "cut_shelf"),
    ("5 tremor", "cut_shelf", "stillness_start"),
    ("5d stillness", "stillness_start", "stillness_end"),
    ("6 burst, vortex", "burst", "cut_dome"),
    ("6c shutters, gold", "cut_dome", "cut_sunrise"),
    ("7 sunrise, wind ring", "cut_sunrise", "cut_great_bird"),
    ("8 flock forms", "cut_great_bird", "wingbeat"),
    ("8 climax", "wingbeat", "cut_last_page"),
    ("9a quiet returns", "cut_last_page", "final_paw"),
    ("9b last note, title", "final_paw", "end"),
]

# (label, cue, bus, band Hz) checked on the rendered stems: steepest rise of the band envelope
AUDIO_CUES = [  # music cues and the big hits on the rendered stems (tagged effects are checked per event)
    ("first paw step", "first_paw_step", "foley", (100, 1000)),
    ("piano enters", "piano_in", "piano", (100, 3000)),
    ("lantern tap", "lantern_down", "foley", (300, 3000)),
    ("eye glint celesta", "eye_glint", "harp", (1200, 1600)),
    ("hop takeoff", "hop_takeoff", "foley", (2000, 6500)),
    ("hop landing", "hop_land", "foley", (60, 400)),
    ("paw on paper", "paw_down", "foley", (1000, 5000)),
    ("bird motif (piano)", "bird_flap_start", "piano", (800, 3000)),
    ("one book trembles", "tremble_one", "paper", (150, 1200)),
    ("row trembles", "tremble_row", "paper", (150, 1200)),
    ("creak", "creak", "fx", (40, 400)),
    ("burst (paper)", "burst", "paper", (100, 3000)),
    ("burst (tutti)", "burst", "perc", (40, 400)),
    ("shutters crack", "shutters_break", "fx", (600, 7000)),
    ("gold light (harp)", "gold_light", "harp", (150, 1500)),
    ("wind ring", "wind_ring", "fx", (20, 120)),
    ("wingbeat (sub)", "wingbeat", "fx", (20, 120)),
    ("wingbeat (tutti)", "wingbeat", "perc", (40, 400)),
    ("page lands", "page_land", "foley", (250, 2500)),
    ("final piano note", "final_paw", "piano", (300, 1500)),
    ("paw on the last page", "final_paw", "foley", (1200, 5000)),
    ("hanko", "hanko", "foley", (60, 400)),
    ("tail flick", "tail_flick", "foley", (600, 4000)),
]


def detect_onset(x, tc, band, search=0.12):
    """Onset near cue tc: the first 2 ms hop where the band energy climbs 10 dB over the level just
    before the search window and within 15 dB of the window's peak; if the sound starts on top of a
    louder bed, the steepest rise instead."""
    i0, i1 = int((tc - 0.4) * SR), int((tc + 0.4) * SR)
    seg = x[:, max(0, i0):i1].astype(np.float64).mean(0)
    y = filt(sos_bp(band[0], band[1], 4), seg) ** 2
    hop = int(0.002 * SR)
    n = len(y) // hop
    e = 10 * np.log10(y[: n * hop].reshape(n, hop).mean(1) + 1e-16)
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    k0, k1 = int((0.4 - search) / 0.002), int((0.4 + search) / 0.002)
    bg = np.median(e[max(0, k0 - 40):k0 - 5])
    thr = max(bg + 10.0, e[k0:k1].max() - 15.0)
    above = np.nonzero(e[k0:k1] > thr)[0]
    if above.size and e[k0:k1].max() > bg + 10.0:
        return tc - 0.4 + (k0 + int(above[0])) * 0.002
    d = np.diff(e[k0 - 1:k1])
    k = int(np.argmax(d))
    return tc - 0.4 + (k0 + k) * 0.002


def ffmpeg_measure(path):
    out = {}
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    i = txt.rfind("Summary:")
    summ = txt[i:].strip() if i >= 0 else "(no summary)"
    out["ebur128"] = summ
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        "loudnorm=I=-14:TP=-1:LRA=20:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr
    j0, j1 = txt.rfind("{"), txt.rfind("}")
    try:
        out["loudnorm"] = json.loads(txt[j0:j1 + 1])
    except ValueError:
        out["loudnorm"] = None
    return out


# ========================================================================================= main
def main():
    t_all = time.time()
    stems = "--stems" in sys.argv
    end_t = C("end")
    log("composing")
    S, M = compose()
    for m in M.values():
        log("  tempo " + m.describe())
    mx = Mixer()

    log("sound design")
    t0 = time.time()
    info = render_sfx(mx)
    log(f"  {len(mx.events)} effect events, {len(info['paw_steps'])} paw steps ({time.time() - t0:.1f}s)")

    log("music")
    render_music(S, mx.buses, end_t)

    # -------------------------------------------------------------------------------- mix
    log("mixing")
    B = mx.buses
    level = {"piano": 0.0, "strings": 0.0, "brass": -0.5, "choir": -3.0, "perc": -1.0, "harp": -3.0,
             "foley": 0.0, "paper": 0.0, "fx": 0.0, "amb": 0.0, "room": 0.0}
    for k, v in level.items():
        B[k].x *= np.float32(db2lin(v))
    # the felt piano comes forward when the motif returns at full scale
    B["piano"].x *= curve(TIME, [(0, 0.8), (C("wingbeat") - 0.9, 0.8), (C("wingbeat") - 0.8, 1.2),
                                 (C("wingbeat") - 0.02, 1.2), (C("wingbeat"), 1.5),
                                 (C("cut_last_page"), 1.5), (C("cut_last_page") + 0.3, 1.15), (60, 1.15)]).astype(np.float32)
    # a held breath in the last half second of the build, so the wingbeat lands as the biggest moment
    tw = C("wingbeat")
    for k, dip in (("strings", 0.78), ("brass", 0.65), ("choir", 0.65), ("perc", 0.65)):
        # the air of the downstroke leads for a third of a second, then the orchestra blooms
        B[k].x *= curve(TIME, [(0, 1.0), (tw - 1.1, 1.0), (tw - 0.03, dip), (tw - 0.004, dip), (tw, 0.85),
                               (tw + 0.35, 1.2), (C("arc_start") + 1.0, 1.2), (C("arc_start") + 2.5, 1.05),
                               (60, 1.05)]).astype(np.float32)
    # room for the paper roar after the burst hit, and for the wind ring over the sunrise
    tb_, twr = C("burst"), C("wind_ring")
    for k in ("strings", "brass"):
        B[k].x *= curve(TIME, [(0, 1.0), (tb_ + 0.06, 1.0), (tb_ + 0.2, 0.7), (tb_ + 1.2, 0.7),
                               (C("cut_stairwell"), 1.0), (60, 1.0)]).astype(np.float32)
    for k in MUSIC:
        B[k].x *= curve(TIME, [(0, 1.0), (twr - 0.05, 1.0), (twr + 0.05, 0.6), (twr + 0.9, 0.62),
                               (twr + 1.8, 1.0), (60, 1.0)]).astype(np.float32)
    tremor_auto = curve(TIME, [(0, 1.0), (C("cut_shelf"), 1.0), (C("cut_shelf") + 0.05, 0.71),
                                (C("stillness_start"), 0.71), (C("stillness_start") + 0.05, 1.0), (60, 1.0)]).astype(np.float32)
    for k in MUSIC:
        B[k].x *= tremor_auto
    # transparent peak control: only the rare transient peaks of these buses are caught
    for k, below in (("strings", 5.0), ("brass", 3.0), ("paper", 4.0), ("perc", 2.0)):
        xk = B[k].x.astype(np.float64)
        pk = 20 * np.log10(np.abs(xk).max() + 1e-12)
        xk, gk = limiter(xk, pk - below, look=0.002, release=0.06)
        B[k].x[:] = xk.astype(np.float32)
        log(f"  {k} bus peak control: max {-20 * np.log10(gk.min()):.1f} dB, >1 dB {100 * np.mean(gk < db2lin(-1.0)):.2f}% of time")
    score_send = {"piano": 0.28, "strings": 0.5, "brass": 0.45, "choir": 0.6, "perc": 0.4, "harp": 0.55}
    score_in = np.zeros((2, NS))
    for k, s in score_send.items():
        score_in += B[k].x * s
    log("  reverbs")
    ir_score = make_ir(2.6, 1.5, 3.2, 0.022, rng_for("ir-score"), er=10, er_span=0.06)
    ir_hall = make_ir(4.6, 2.0, 5.5, 0.035, rng_for("ir-hall"), er=16, er_span=0.09, er_gain=7.0)
    ir_out = make_ir(1.3, 0.6, 1.8, 0.05, rng_for("ir-out"), er=4, er_span=0.25, er_gain=3.0, onset=0.08)
    ir_close = make_ir(0.45, 0.28, 0.6, 0.004, rng_for("ir-close"), er=12, er_span=0.02)
    rv_score = convolve_stereo(score_in, ir_score, cross=0.25)
    rv_hall = convolve_stereo(mx.sends["hall"].x, ir_hall, cross=0.25)
    rv_out = convolve_stereo(mx.sends["out"].x, ir_out, cross=0.3)
    rv_close = convolve_stereo(mx.sends["close"].x, ir_close, cross=0.15)
    # picture cuts between inside and outside cut the room sound too
    rv_hall *= curve(TIME, [(0, 1.0), (C("cut_sunrise"), 1.0), (C("cut_sunrise") + 0.25, 0.0), (60, 0.0)])
    rv_out *= curve(TIME, [(0, 1.0), (C("cut_hall"), 1.0), (C("cut_hall") + 0.2, 0.0),
                           (C("cut_sunrise") - 0.01, 0.0), (C("cut_sunrise"), 1.0), (60, 1.0)])
    wet = rv_score * 0.85 + rv_hall * 0.9 + rv_out * 0.7 + rv_close * 0.6
    dry = np.zeros((2, NS))
    for k in MUSIC + SFX:
        dry += B[k].x
    main_mix = dry + wet
    # the stillness: everything stops dead, only the room tone stays (a held breath)
    st0, st1 = C("stillness_start"), C("stillness_end")
    gate = curve(TIME, [(0, 1.0), (st0, 1.0), (st0 + 0.03, 0.0), (st1 - 0.002, 0.0), (st1, 1.0), (60, 1.0)])
    # the held breath: the room tone comes up about 12 dB inside the stillness so it reads as a
    # breath, not a dropout
    room = B["room"].x.astype(np.float64) * curve(TIME, [(0, 1.0), (st0, 1.0), (st0 + 0.08, 3.2), (st1 - 0.05, 3.6),
                                                         (st1, 1.0), (60, 1.0)])
    mix = main_mix * gate + room
    # loudness range for phones: the quiet first half and the ending come up (dB), the loud middle stays
    tp_, tb_ = C("paw_down"), C("cut_shelf")
    rng_db = curve(TIME, [(0, RANGE_LIFT_DB), (tp_ - 0.1, RANGE_LIFT_DB), (tp_ + 0.1, RANGE_INK_DB),
                          (tb_ - 0.1, RANGE_INK_DB), (tb_, RANGE_LIFT_DB), (C("cut_lantern"), 1.5),
                          (C("stillness_start"), 1.0), (C("stillness_end"), 0.0), (C("arc_end") - 0.5, 0.0),
                          (C("cut_last_page") + 0.5, RANGE_END_DB), (C("final_paw") - 0.3, RANGE_END_DB),
                          (C("final_paw"), RANGE_LAST_DB), (60, RANGE_LAST_DB)])
    mix *= db2lin(rng_db)
    # fades: in at the first sample, out to black with the picture
    f_in = np.clip(TIME / 0.005, 0, 1)
    xo = np.clip((TIME - C("fade_paper_start")) / (C("end") - 0.03 - C("fade_paper_start")), 0, 1)
    f_out = np.cos(0.5 * np.pi * xo) ** 1.5
    mix *= f_in * f_out

    if stems:
        log("\nper-bus loudness per section (LUFS, ungated power mean, pre-master)")
        cols = MUSIC + SFX + ROOM + ["wet", "MIX"]
        kp = {k: kpower(B[k].x.astype(np.float64) * (gate if k != "room" else 1.0)) for k in MUSIC + SFX + ROOM}
        kp["wet"] = kpower(wet * gate)
        kp["MIX"] = kpower(mix)
        log(f"{'section':24s}" + "".join(f"{c[:6]:>7s}" for c in cols))
        for label, a, b in SECTIONS:
            row = [window_loudness(kp[c], C(a), C(b), win=min(3.0, C(b) - C(a)))[0] for c in cols]
            log(f"{label:24s}" + "".join(f"{v:7.1f}" for v in row))
        log("\nper-bus max momentary loudness per section (LUFS, 0.4 s windows, pre-master)")
        log(f"{'section':24s}" + "".join(f"{c[:6]:>7s}" for c in cols))
        for label, a, b in SECTIONS:
            row = []
            for c in cols:
                _, mm_ = loudness_curve(kp[c][int(C(a) * SR):int(C(b) * SR)], min(0.4, C(b) - C(a)))
                row.append(mm_.max())
            log(f"{label:24s}" + "".join(f"{v:7.1f}" for v in row))
        log("\nkey moments: per-bus loudness over the 0.4 s after each cue (LUFS, pre-master)")
        log(f"{'cue':24s}" + "".join(f"{c[:6]:>7s}" for c in cols))
        for cue in ("fox_enter", "lantern_down", "ear_left", "sniff_1", "hop_land", "paw_down", "ink_start",
                    "tremble_one", "tremble_row", "creak", "flame_lean", "burst", "shutters_break", "gold_light",
                    "wind_ring", "wingbeat", "page_land", "final_paw", "hanko"):
            i0 = int(C(cue) * SR)
            row = [-0.691 + 10 * np.log10(kp[c][i0:i0 + int(0.4 * SR)].mean() + 1e-20) for c in cols]
            log(f"{cue:24s}" + "".join(f"{v:7.1f}" for v in row))
        log("\nper-bus sample peak (dBFS, pre-master) around the big hits")
        log(f"{'window':24s}" + "".join(f"{c[:6]:>7s}" for c in MUSIC + SFX + ["wet", "MIX"]))
        for cue in ("burst", "shutters_break", "gold_light", "wind_ring", "wingbeat", "arc_start"):
            i0, i1 = int((C(cue) - 0.05) * SR), int((C(cue) + 0.45) * SR)
            pk = [np.abs(B[k].x[:, i0:i1]).max() for k in MUSIC + SFX] + [np.abs(wet[:, i0:i1]).max(), np.abs(mix[:, i0:i1]).max()]
            log(f"{cue:24s}" + "".join(f"{20 * np.log10(v + 1e-12):7.1f}" for v in pk))

    # ----------------------------------------------------------------------------- master
    log("\nmastering")
    x = filt(sos_hp(25.0), mix)
    x = filt(chain(sos_shelf_hi(9000.0, -2.5), sos_lp(15500.0, 2)), x)
    x -= x.mean(axis=1, keepdims=True)
    g_lin = float(db2lin(TARGET_LUFS - 2.0 - lufs_integrated(x)))
    x *= g_lin
    # low band (below 110 Hz) peak control: drums, timpani and the wingbeat's sub barely count in
    # loudness but eat peak headroom; only the top 1% of low-band peaks are held back (3:1)
    lo = filt(sos_lp(110.0, 2), filt(sos_lp(110.0, 2), x))
    hi = x - lo
    env = np.abs(lo).max(axis=0)
    ka, kr = np.exp(-1.0 / (0.005 * SR)), np.exp(-1.0 / (0.12 * SR))
    env_s = np.maximum(signal.lfilter([1 - ka], [1, -ka], env), signal.lfilter([1 - kr], [1, -kr], env))
    ed = 20 * np.log10(env_s + 1e-9)
    thr_lo = float(np.percentile(ed[ed > -80], 99.0)) - 3.0
    gr_lo = np.maximum(ed - thr_lo, 0.0) * (1.0 - 1.0 / 3.0)
    gr_lo = signal.lfilter([1 - kr], [1, -kr], gr_lo)
    x = hi + lo * db2lin(-gr_lo)[None, :]
    log(f"  low band peak control: max {gr_lo.max():.1f} dB, >1 dB {100 * np.mean(gr_lo > 1):.2f}% of time")
    x, grc = compressor(x, thr_db=-21.0, ratio=2.0, knee=8.0, tau_det=0.03, tau_gain=0.12)
    y = x
    g = np.ones(NS)
    for _ in range(8):
        gk = float(db2lin(TARGET_LUFS - lufs_integrated(y)))
        x *= gk
        g_lin *= gk
        y, g = limiter(x, CEILING_DBTP)
        if abs(lufs_integrated(y) - TARGET_LUFS) < 0.03:
            break
    y[:, -int(0.03 * SR):] = 0.0
    grl = -20 * np.log10(g)
    clipped = write_wav24(OUT_WAV, y, rng_for("dither"))

    # cold open for the X cut: the film's close-up window, same master gain and compressor curve
    c = coldopen.premix(B, rv_close, rv_hall, rng_db, ir_score, piano_level=0.8, score_send=score_send["piano"])
    c = filt(sos_hp(25.0), c)
    c = filt(chain(sos_shelf_hi(9000.0, -2.5), sos_lp(15500.0, 2)), c)
    sl = slice(coldopen.I0, coldopen.I0 + coldopen.LENGTH)
    c = c * g_lin
    c_lo = filt(sos_lp(110.0, 2), filt(sos_lp(110.0, 2), c))
    c = (c - c_lo) + c_lo * db2lin(-gr_lo[sl])[None, :]
    c = c * db2lin(-grc[sl])[None, :]
    c, _ = limiter(c, CEILING_DBTP)
    c = coldopen.finish(c)
    clipped_c = write_wav24(OUT_COLD, c, rng_for("dither-cold"))

    # ----------------------------------------------------------------------------- report
    yf = read_wav24(OUT_WAV)
    assert yf.shape == (2, NS), yf.shape
    lu = lufs_integrated(yf)
    sp = float(np.abs(yf).max())
    tp = true_peak(yf, 4)
    log("\nRESULT")
    log(f"  file: {OUT_WAV}")
    log(f"  length: {yf.shape[1]} samples = {yf.shape[1] / SR:.4f} s, {SR} Hz, stereo, 24 bit PCM")
    log(f"  integrated loudness (own BS.1770 meter): {lu:.2f} LUFS (target {TARGET_LUFS})")
    log(f"  sample peak {20 * np.log10(sp):.2f} dBFS, true peak (4x) {20 * np.log10(tp):.2f} dBTP, clipped samples {clipped}")
    worst = np.argsort(grl)[::-1]
    spots = []
    for i in worst:
        if len(spots) >= 4 or grl[i] < 0.5:
            break
        if all(abs(i - j) > SR for j in spots):
            spots.append(i)
    log(f"  master compressor max {grc.max():.1f} dB; limiter max {grl.max():.1f} dB, >1 dB {100 * np.mean(grl > 1):.1f}%, "
        f">3 dB {100 * np.mean(grl > 3):.2f}% of time" + (", worst at " + ", ".join(f"{i / SR:.2f}s" for i in spots) if spots else ""))

    p = kpower(yf)
    tm, mom = loudness_curve(p, 0.4)
    ts_, stl = loudness_curve(p, 3.0)
    log("\nsections (LUFS: mean over the section, max momentary 0.4 s, short-term 3 s at its middle; peak dBFS)")
    for label, a, b in SECTIONS:
        ta, tb = C(a), C(b)
        mean = window_loudness(p, ta, tb, win=min(3.0, tb - ta))[0]
        mm = mom[(tm >= ta + 0.2) & (tm <= tb - 0.2)]
        mmax = mm.max() if mm.size else float("nan")
        stm = float(np.interp(0.5 * (ta + tb), ts_, stl))
        pk = 20 * np.log10(np.abs(yf[:, int(ta * SR):int(tb * SR)]).max() + 1e-12)
        log(f"  {label:24s} {ta:5.2f}-{tb:5.2f}  mean {mean:6.1f}  mom max {mmax:6.1f}  short {stm:6.1f}  peak {pk:6.1f}")
    i_m, i_s = int(np.argmax(mom)), int(np.argmax(stl))
    log(f"  loudest momentary at {tm[i_m]:.2f} s ({mom[i_m]:.1f} LUFS); loudest short-term window centred "
        f"{ts_[i_s]:.2f} s ({stl[i_s]:.1f} LUFS)")
    inside = (tm >= C("stillness_start") + 0.25) & (tm <= C("stillness_end") - 0.25)
    log(f"  stillness: momentary {mom[inside].max():.1f} LUFS max inside the beat "
        f"(the tremor just before peaks at {mom[(tm > C('cut_lantern')) & (tm < C('stillness_start'))].max():.1f})")
    f, psd = signal.welch(yf.mean(0), SR, nperseg=8192)
    hf = 10 * np.log10(psd[f > 8000].sum() / psd.sum())
    i0, i1 = int(C("wingbeat") * SR), int(C("cut_last_page") * SR)
    f, psd = signal.welch(yf[:, i0:i1].mean(0), SR, nperseg=8192)
    hfc = 10 * np.log10(psd[f > 8000].sum() / psd.sum())
    log(f"  energy above 8 kHz: {hf:.1f} dB of total (climax {hfc:.1f} dB)")
    corr = float(np.corrcoef(yf[0], yf[1])[0, 1])
    mono = lufs_integrated(np.vstack([yf.mean(0), yf.mean(0)]))
    log(f"  stereo: L/R correlation {corr:.2f}, mono fold-down loudness {mono:.2f} LUFS ({mono - lu:+.2f} LU vs stereo)")

    log(f"\ncue check (one frame = {1000 * FRAME:.0f} ms)")
    log("  rendered stems: steepest rise of the band envelope near the cue")
    bad = 0
    for label, cue, bus, band in AUDIO_CUES:
        tf = detect_onset(B[bus].x, C(cue), band, search=0.05 if bus in ("perc", "piano", "harp") else 0.12)
        dt = 1000 * (tf - C(cue))
        flag = "" if abs(dt) <= 1000 * FRAME else "  <-- OFF"
        bad += flag != ""
        log(f"    {label:24s} {cue:16s} {C(cue):7.3f}  found {tf:7.3f}  ({dt:+5.0f} ms){flag}")
    log("  placed effects: audible onset of the event tagged with each cue (exact, per event)")
    nbad = 0
    for cue, bus, t0, on in sorted(mx.tagged, key=lambda e: C(e[0])):
        dt = 1000 * (t0 + on - C(cue))
        design = cue in ("wind_in", "cut_sunrise", "cut_great_bird", "cut_stairwell", "arc_start")
        flag = "" if abs(dt) <= 1000 * FRAME else ("  (fade-in by design)" if design else "  <-- OFF")
        nbad += flag.endswith("OFF")
        log(f"    {cue:16s} {C(cue):7.3f}  {bus:6s} audible from {t0 + on:7.3f}  ({dt:+5.0f} ms){flag}")
    log(f"  placed effects more than one frame off: {nbad}")
    from dsp import EV, EVENT_OVERRIDES
    for name, label in (("paw_steps", "paw steps"), ("bird_wingbeats", "ink bird wingbeats")):
        want = [float(v) for v in (EV(name) or [])]
        got = [t0 + on for bus, t0, on, pk in mx.events if bus == "foley"]
        hits = sum(1 for w in want if min(abs(g - w) for g in got) <= FRAME)
        if want:
            worst = max(min(abs(g - w) for g in got) for w in want)
            log(f"  {label}: {hits} of {len(want)} from events.json sound within a frame (worst {1000 * worst:.0f} ms)")
    if EVENT_OVERRIDES:
        log("  events.json overrides of planned cues: " + ", ".join(
            f"{k} {('%.2f' % o) if o is not None else 'new'}->{v:.3f}" for k, o, v in EVENT_OVERRIDES))
    late = [(bus, t0, on) for bus, t0, on, pk in mx.events if on > FRAME and pk > 1e-4]
    log(f"  effect events whose sound starts more than a frame after their placement: {len(late)}")
    for bus, t0, on in late[:12]:
        log(f"    {bus} at {t0:.3f}: audible from +{1000 * on:.0f} ms (a swell or a fade-in by design)")
    log("  score points from the tempo maps:")
    for label, t, bus in music_cue_points(S, M):
        log(f"    {label:30s} {t:7.3f}")
    log(f"  cue detections more than one frame off: {bad}")

    cf = read_wav24(OUT_COLD)
    pc = kpower(cf)
    _, momc = loudness_curve(pc, 0.4)
    film_ref = window_loudness(kpower(yf), coldopen.I0 / SR, (coldopen.I0 + coldopen.LENGTH) / SR, win=1.8)[0]
    log("\ncold open (X cut)")
    log(f"  file: {OUT_COLD}")
    log(f"  length: {cf.shape[1]} samples = {cf.shape[1] / SR:.4f} s, from film {coldopen.I0 / SR:.4f} s "
        f"(frame {coldopen.START_FRAME}); clipped {clipped_c}")
    log(f"  loudness: mean {-0.691 + 10 * np.log10(pc.mean() + 1e-20):.1f} LUFS, momentary max {momc.max():.1f} "
        f"(the film's mix in the same window: {film_ref:.1f}); true peak {20 * np.log10(true_peak(cf, 4)):.2f} dBTP")
    tail = cf[:, -int(0.1 * SR):]
    log(f"  last 100 ms peak {20 * np.log10(np.abs(tail).max() + 1e-12):.1f} dBFS, last sample "
        f"{20 * np.log10(np.abs(cf[:, -1]).max() + 1e-12):.1f} dBFS (hands over to the film's wind from silence)")

    if "--no-ffmpeg" not in sys.argv:
        log("\nffmpeg measurement")
        res = ffmpeg_measure(OUT_WAV)
        log("  " + res["ebur128"].replace("\n", "\n  "))
        ln = res["loudnorm"]
        if ln:
            log(f"  loudnorm: input_i {ln['input_i']} LUFS, input_tp {ln['input_tp']} dBTP, "
                f"input_lra {ln['input_lra']} LU, input_thresh {ln['input_thresh']}")
    log(f"\nrender time {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
