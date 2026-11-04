"""The score of THE LAST PAGE: tempo maps pinned to timeline cues, the composition, rendering.

The motif (7 notes, two bars of 4/4: "da-da DAAA, da-da da DAAA"):

    6  1'  5'  |  4'  3'  1'  2'        in Db major:  Bb4 Db5 Ab5 | Gb5 F5 Db5 Eb5

It rises a minor third and a fifth, then falls by step and settles on the second degree: a
question that is left open every time it is played, until the very last note of the film.

Plan (keys and cues)
    0.0 to 7.6     wind and near silence, no music
    7.6 to 21.3    solo felt piano, Db major. Motif (fox enters), developed variant reaching a sixth
                   (close-up), a Bbm(add9) chord lands with the hop and is damped at paw_down
    24.0           the ink bird flies: the motif head, high and light
    25.0 to 31.0   Bb minor. Muted low string tremolo, motif fragments in low piano, violas (with a
                   raised seventh), dark horns at the creak, a semitone bend as the flame leans,
                   F7(b9) crescendo cut dead at stillness_start
    32.0 to 37.75  Bb minor, 120 bpm: tutti hit on the burst, string ostinato, rising vortex
                   arpeggios, the motif head in horns, F7 roll, caesura before the shutters crack
    38.0 to 46.5   gold light: deceptive resolution to Gb major. Horns and violins carry the motif
                   over the sunrise; its last note Eb becomes D# over B major (the pivot)
    46.5 to 52.0   climax in E major (a chromatic third above the opening key): the motif at full
                   scale, piano in octaves with strings, horns, choir, timpani; ends on the open
                   F# over B and fades as quiet returns
    52.0 to 55.0   piano alone echoes the second half; the question F#5 rings, the pedal lifts
    55.4           one soft E5: the answer, with the pedal down so the strings ring along

Each metric section has its own tempo map whose pins are cue names, so moving a cue in
timeline.py stretches the music between pins instead of breaking it. Free-time passages are
written as offsets from cues.
"""

import time

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from dsp import (NS, SR, C, Bus, log, nn, rng_for, taxis, limiter)
from instruments import (STR_V2, bass_drum, bowed_note, celesta_tone, choir_note, cymbal,
                         felt_piano_note, harp_note, horn_note, key_noise, pedal_noise,
                         sympathetic_ir, timp_roll, timpani_hit)


# ================================================================================ tempo maps
class TempoMap:
    """Beats to seconds through pins (beat, cue name or seconds), monotone cubic in between so the
    tempo changes smoothly, linear outside the pins."""

    def __init__(self, name, pins):
        self.name = name
        self.beats = np.array([b for b, _ in pins], dtype=float)
        self.times = np.array([C(c) if isinstance(c, str) else float(c) for _, c in pins])
        self.pins = pins
        self.f = PchipInterpolator(self.beats, self.times)
        self.s0 = (self.times[1] - self.times[0]) / (self.beats[1] - self.beats[0])
        self.s1 = (self.times[-1] - self.times[-2]) / (self.beats[-1] - self.beats[-2])

    def __call__(self, b):
        if b < self.beats[0]:
            return float(self.times[0] + (b - self.beats[0]) * self.s0)
        if b > self.beats[-1]:
            return float(self.times[-1] + (b - self.beats[-1]) * self.s1)
        return float(self.f(b))

    def describe(self):
        segs = []
        for i in range(len(self.beats) - 1):
            spb = (self.times[i + 1] - self.times[i]) / (self.beats[i + 1] - self.beats[i])
            segs.append(f"{self.pins[i][1]}->{self.pins[i + 1][1]} {60.0 / spb:.0f} bpm")
        return f"{self.name}: " + ", ".join(segs)


def maps():
    return {
        # the melody's Db lands as the lantern is set down and its open Eb as the fox looks up
        "P": TempoMap("piano solo", [(0, "piano_in"), (5, "lantern_down"), (6, "look_up"), (8, "cut_closeup"),
                                     (14, "hop_land"), (16, "paw_down")]),
        "B": TempoMap("burst", [(0, "burst"), (5, "cut_stairwell"), (10, "vortex_peak"), (12, "gold_light")]),
        "S": TempoMap("sunrise", [(0, "gold_light"), (1, "cut_sunrise"), (6, "cut_great_bird"), (8.5, "wingbeat")]),
        "X": TempoMap("climax", [(0, "wingbeat"), (2, "arc_start"), (7.3333, "arc_end")]),
    }


# ===================================================================================== score
class Score:
    def __init__(self):
        self.piano = []      # (t_on, t_key_up, midi, vel, felt)
        self.pedal = []      # (t_down, t_up)
        self.strings = []    # (sec, t0, dur, midi, vel, vel_end, kind, att, rel, glide, mute)
        self.horns = []      # (t0, dur, midi, vel, vel_end, att, rel, dark)
        self.choir = []      # (part, t0, dur, midi, vel, vel_end, va, vb, morph_t, att, rel)
        self.harp = []       # (t0, midi, vel, pan)
        self.celesta = []    # (t0, midi, vel, pan)
        self.perc = []       # (kind, t0, args...)
        self.rng = rng_for("score-humanize")

    def pa(self, t0, dur, note, vel, felt=1.0, exact=False):
        if not exact:
            t0 += self.rng.normal(0.0, 0.004)
            vel += self.rng.normal(0.0, 0.015)
        self.piano.append((t0, t0 + dur, nn(note), float(np.clip(vel, 0.05, 1.0)), felt))

    def roll(self, t0, dur, notes, vel, spread=0.05, felt=1.0):
        for i, nm in enumerate(notes):
            self.pa(t0 + i * spread, dur - i * spread, nm, vel - 0.01 * i, felt, exact=(i == 0))

    def st(self, sec, t0, dur, note, vel, vel_end=None, kind="legato", att=0.35, rel=0.6, glide=None, mute=False):
        self.strings.append((sec, t0, dur, nn(note), vel, vel_end, kind, att, rel, glide, mute))

    def hn(self, t0, dur, note, vel, vel_end=None, att=0.14, rel=0.6, dark=0.0):
        self.horns.append((t0, dur, nn(note), vel, vel_end, att, rel, dark))

    def ch(self, part, t0, dur, note, vel, vel_end=None, va="a", vb=None, morph_t=0.0, att=0.9, rel=1.2):
        self.choir.append((part, t0, dur, nn(note), vel, vel_end, va, vb or va, morph_t, att, rel))


MOTIF = [(0.0, 0.5, 0), (0.5, 0.5, 3), (1.0, 3.0, 10), (4.0, 0.5, 8), (4.5, 0.5, 7), (5.0, 1.0, 3), (6.0, 2.0, 5)]
# (beat, length, semitones above the motif's first note); first note is the 6th degree.


def motif_in(first, variant=None):
    """The motif as (beat, length, midi) starting on note `first` (the 6th degree of the key)."""
    m0 = nn(first)
    return [(b, d, m0 + s) for b, d, s in (variant or MOTIF)]


def compose():
    S = Score()
    M = maps()

    # ------------------------------------------------------------ 7.6 to 21.3: solo felt piano, Db
    P = M["P"]

    def pb(b, d, note, vel, felt=1.0):
        t0 = P(b)
        S.pa(t0, P(b + d) - t0, note, vel, felt)

    # statement 1 (fox enters, stops, sets the lantern down, looks up on the open Eb)
    for b, d, m in motif_in("Bb4"):
        pb(b, d, m, 0.40 + (0.04 if d >= 2 else 0.0) - (0.03 if d < 1 else 0.0))
    for b, nm, v in ((0, "Gb2", 0.31), (1, "Db3", 0.24), (2, "Ab3", 0.22), (3, "Bb3", 0.22),
                     (4, "Ab2", 0.30), (5, "Eb3", 0.24), (6, "Gb3", 0.22), (7, "Db4", 0.20)):
        pb(b, 4 - (b % 4), nm, v)
    S.pedal += [(P(0) + 0.03, P(4) + 0.06), (P(4) + 0.13, P(8) + 0.06)]

    # statement 2, developed: reaches a sixth (Bb5) over the close-up, settles on F5 at hop_land
    var2 = [(0.0, 0.5, 0), (0.5, 0.5, 3), (1.0, 3.0, 12), (4.0, 0.5, 10), (4.5, 0.5, 8), (5.0, 1.0, 5)]
    for b, d, s in var2:
        pb(8 + b, d, nn("Bb4") + s, 0.42 + (0.05 if d >= 2 else 0.0) - (0.03 if d < 1 else 0.0))
    for b, nm, v in ((8, "Eb2", 0.30), (9, "Bb2", 0.24), (10, "Gb3", 0.22), (11, "Db4", 0.20),
                     (12, "Ab2", 0.28), (13, "Eb3", 0.22)):
        pb(b, (12 - b) if b < 12 else (14 - b), nm, v)
    t_land, t_paw = C("hop_land"), C("paw_down")
    S.pa(t_land, t_paw - t_land, "F5", 0.40, exact=True)
    S.roll(t_land, t_paw - t_land, ["Bb1", "F2", "C3", "Db3", "Ab3"], 0.26, spread=0.055)
    S.pedal += [(P(8) + 0.03, P(12) + 0.06), (P(12) + 0.13, t_land + 0.03), (t_land + 0.09, t_paw)]

    # eye glint: a tiny celesta dyad
    tg = C("eye_glint")
    S.celesta += [(tg, nn("F6"), 0.24, 0.25), (tg + 0.035, nn("Bb6"), 0.17, 0.35)]

    # ------------------------------------------------------------ 24.0: the ink bird flies
    tb = C("bird_flap_start")
    for i, (nm, v) in enumerate((("Bb5", 0.30), ("Db6", 0.30), ("Ab6", 0.35))):
        S.pa(tb + 0.16 * i, 1.0, nm, v, exact=(i == 0))
    S.celesta.append((tb + 0.32, nn("Ab6"), 0.16, 0.5))
    S.pedal.append((tb - 0.02, C("tremble_one") - 0.03))

    # ------------------------------------------------------------ 25.0 to 31.0: the tremor, Bb minor
    t25, t1, tr, tcr = C("cut_shelf"), C("tremble_one"), C("tremble_row"), C("creak")
    tl, tf, te, tst = C("cut_lantern"), C("flame_lean"), C("ears_flat"), C("stillness_start")
    end = tst - 0.01
    for sec, a, b in (("cb", "Bb1", "F1"), ("vc", "Bb2", "F2")):
        S.st(sec, t25, te - t25 + 0.05, a, 0.10, 0.40, kind="trem", att=0.9, rel=0.15, mute=True)
        S.st(sec, te, end - te, b, 0.42, 0.62, kind="trem", att=0.08, rel=0.04, mute=True)
    S.st("vc", te, end - te, "C3", 0.38, 0.6, kind="trem", att=0.08, rel=0.04)
    # piano: the motif head, low and dark
    for i, (nm, v) in enumerate((("Bb3", 0.33), ("Db4", 0.31), ("Ab4", 0.35))):
        S.pa(t1 + 0.27 * i, 1.5 - 0.27 * i, nm, v, exact=(i == 0))
    S.pedal.append((t1 + 0.04, tr - 0.06))
    # the row: violas and celli take the head with a raised seventh that bends down as the flame leans
    bend = (tf - (tr + 0.5), tf - (tr + 0.5) + 0.55, -100.0)
    for sec, notes, v0 in (("vla", ("Bb3", "Db4", "A4"), 0.30), ("vc", ("Bb2", "Db3", "A3"), 0.26)):
        S.st(sec, tr, 0.27, notes[0], v0, att=0.06, rel=0.1)
        S.st(sec, tr + 0.25, 0.27, notes[1], v0, att=0.06, rel=0.1)
        S.st(sec, tr + 0.5, te - (tr + 0.5), notes[2], v0, v0 + 0.14, att=0.08, rel=0.15, glide=bend)
    S.pa(tr, 1.6, "Bb1", 0.36, exact=True)
    S.pa(tr, 1.6, "Bb2", 0.30, exact=True)
    S.pedal.append((tr + 0.03, tcr + 0.6))
    # the creak: dark horns, a semitone cluster
    S.hn(tcr, te - tcr + 0.05, "Bb2", 0.18, 0.42, att=0.8, rel=0.2, dark=0.6)
    S.hn(tcr, te - tcr + 0.05, "B2", 0.15, 0.36, att=0.9, rel=0.2, dark=0.6)
    # the lantern: high violins tremolo, a minor second apart
    S.st("vln1", tl, end - tl, "F5", 0.12, 0.62, kind="trem", att=0.25, rel=0.04)
    S.st("vln2", tl, end - tl, "Gb5", 0.10, 0.56, kind="trem", att=0.3, rel=0.04)
    # ears flat: F7(b9) crescendo, cut dead at the stillness
    for nm in ("F3", "A3", "Eb4"):
        S.hn(te, end - te, nm, 0.28, 0.85, att=0.12, rel=0.04, dark=0.15)
    S.st("vla", te, end - te, "A3", 0.35, 0.75, kind="trem", att=0.06, rel=0.04)
    S.st("vla", te, end - te, "Eb4", 0.33, 0.72, kind="trem", att=0.06, rel=0.04)
    S.perc += [("timp", t1, "Bb2", 0.32), ("timp", tr, "Bb2", 0.42), ("timp", tl, "F2", 0.46),
               ("bd", tr, 0.28), ("bd", tl, 0.34),
               ("roll", te, end, "F2", 0.15, 0.85), ("swell", te - 0.4, end, 0.34)]

    # ------------------------------------------------------------ 32.0 to 37.75: the burst, Bb minor
    B = M["B"]
    t_cut = min(B(11.5), C("shutters_break") - 0.04)
    roots = [(0, "Bb"), (4, "Gb"), (6, "Ab"), (8, "F")]

    def root_at(b):
        r = roots[0][1]
        for bb, nm in roots:
            if b >= bb:
                r = nm
        return r

    tb0 = B(0)
    S.perc += [("timp", tb0, "Bb2", 0.9), ("timp", tb0, "F2", 0.65), ("bd", tb0, 0.8), ("crash", tb0, 0.75)]
    for sec, nm, v in (("cb", "Bb1", 0.85), ("vc", "Bb2", 0.85), ("vla", "F3", 0.75), ("vla", "Bb3", 0.72),
                       ("vln2", "Db5", 0.72), ("vln1", "F5", 0.70)):
        S.st(sec, tb0, 0.42, nm, v, kind="marc", att=0.012, rel=0.25)
    fifth = {"Bb": "F3", "Gb": "Db4", "Ab": "Eb4", "F": "C4"}
    upper = {"Bb": "Bb3", "Gb": "Gb3", "Ab": "Ab3", "F": "F3"}
    b = 0.5
    while B(b) < t_cut - 0.05:
        r = root_at(b)
        x = b / 11.5
        acc = 0.72 if b % 1 == 0 else 0.55
        v = acc + 0.18 * x
        for sec, octv in (("cb", 1), ("vc", 2)):
            S.st(sec, B(b), B(b + 0.45) - B(b), f"{r}{octv}", v, kind="spicc")
        S.st("vla", B(b), B(b + 0.45) - B(b), fifth[r] if b % 1 == 0 else upper[r], v * 0.85, kind="spicc")
        b += 0.5
    trem = [(0, 4, "F5", "Db5"), (4, 6, "Gb5", "Bb4"), (6, 8, "Ab5", "C5"), (8, 10, "A5", "Eb5"), (10, 11.5, "C6", "Gb5")]
    for b0, b1, w1, w2 in trem:
        t0 = B(b0) + (0.2 if b0 == 0 else 0.0)
        t1_ = min(B(b1), t_cut)
        v0, v1 = 0.40 + 0.04 * b0, 0.44 + 0.04 * b1
        S.st("vln1", t0, t1_ - t0 + (0.04 if b1 < 11.5 else 0.0), w1, v0, v1, kind="trem", att=0.1, rel=0.06)
        S.st("vln2", t0, t1_ - t0 + (0.04 if b1 < 11.5 else 0.0), w2, v0 * 0.95, v1 * 0.95, kind="trem", att=0.1, rel=0.06)
    # the vortex: a rising spiral of 16ths, alternating violas and second violins
    chord_pcs = {"Bb": [10, 1, 5], "Gb": [6, 10, 1], "Ab": [8, 0, 3], "F": [5, 9, 0, 3]}
    i = 0
    b = 5.0
    while B(b) < t_cut - 0.06:
        r = root_at(b)
        ladder = [m for m in range(nn("Ab3"), nn("C7")) if m % 12 in chord_pcs[r]]
        k = int((b - 5.0) * 1.0)
        start = min(len(ladder) - 5, 2 + k)
        m = ladder[start + (i % 4)]
        x = (b - 5.0) / 6.5
        sec = "vln2" if i % 2 == 0 else "vla"
        S.st(sec, B(b), B(b + 0.22) - B(b), m, 0.42 + 0.3 * x + (0.06 if i % 4 == 0 else 0.0), kind="spicc")
        b += 0.25
        i += 1
    # horns: sfz chord, the motif head at the stairwell cut, F7(b9) to the caesura
    for nm in ("Bb3", "Db4", "F4"):
        S.hn(tb0, B(4) - tb0, nm, 0.88, 0.62, att=0.02, rel=0.15)
    for nm, b0, d in (("Bb3", 5.0, 0.5), ("Db4", 5.5, 0.5), ("Ab4", 6.0, 2.0)):
        for octv in (0, -12):
            S.hn(B(b0), B(b0 + d) - B(b0) + 0.03, nn(nm) + octv, 0.80 if octv == 0 else 0.62, att=0.03, rel=0.15)
    for nm in ("F3", "A3", "C4", "Eb4"):
        S.hn(B(8), t_cut - B(8), nm, 0.55, 0.95, att=0.05, rel=0.05)
    S.hn(B(10), t_cut - B(10), "Gb4", 0.6, 0.95, att=0.05, rel=0.05)
    S.perc += [("timp", B(4), "Gb2", 0.70), ("timp", B(6), "Ab2", 0.75), ("timp", B(8), "F2", 0.80),
               ("roll", B(10), t_cut, "F2", 0.40, 1.0), ("bd", B(4), 0.55), ("bd", B(8), 0.65),
               ("swell", B(8), t_cut, 0.45)]
    gl = [m for m in range(nn("Bb3"), nn("Bb6") + 1) if m % 12 in (10, 0, 1, 3, 5, 6, 8)]
    for j, m in enumerate(gl):
        S.harp.append((B(5) + 0.6 * j / len(gl), m, 0.42 + 0.2 * j / len(gl), -0.4 + 0.8 * j / len(gl)))

    # ------------------------------------------------------------ 38.0 to 46.5: gold light, sunrise
    Sx = M["S"]
    s = Sx
    t38 = s(0)
    # melody: violins 1 and horns an octave below
    mel = motif_in("Bb4")[:-1] + [(6.0, 1.75, nn("Eb5"))]
    for b0, d, m in mel:
        t0, t1_ = s(b0), s(b0 + d)
        x = b0 / 7.75
        S.st("vln1", t0, t1_ - t0 + 0.05, m, 0.60 + 0.2 * x, att=0.10 if d < 1 else 0.18, rel=0.4)
        S.hn(t0, t1_ - t0 + 0.04, m - 12, 0.72 + 0.22 * x, att=0.06 if d < 1 else 0.12, rel=0.4)
        S.hn(t0, t1_ - t0 + 0.04, m - 24, 0.55 + 0.2 * x, att=0.08 if d < 1 else 0.14, rel=0.4, dark=0.3)
    harm = [  # (beat0, beat1, cb, vc, vc, vla, vla, vln2, vln2, choir alto, choir sop, choir sop)
        (0.0, 2.5, "Gb1", "Gb2", "Db3", "Bb3", "Db4", "F4", "Ab4", "Bb4", "Db5", "F5"),
        (2.5, 3.5, "Eb1", "Eb2", "Bb2", "Gb3", "Db4", "F4", "Bb4", "Bb4", "Db5", "Gb5"),
        (3.5, 4.0, "Bb1", "F2", "Db3", "Ab3", "Db4", "F4", "Ab4", "Ab4", "Db5", "F5"),
        (4.0, 6.0, "Ab1", "Ab2", "Eb3", "Db4", "Eb4", "Gb4", "Bb4", "Ab4", "Db5", "Eb5"),
        (6.0, 8.5, "B1", "B2", "F#3", "D#4", "F#4", "F#4", "B4", "D#5", "F#5", "B5"),
    ]
    for j, (b0, b1, cbn, vc1, vc2, va1, va2, w1, w2, ca, cs1, cs2) in enumerate(harm):
        t0, t1_ = s(b0), s(b1)
        p0, p1 = b0 / 8.5, b1 / 8.5
        v0, v1 = 0.50 + 0.33 * p0, 0.50 + 0.33 * p1
        att = 0.25 if j == 0 else 0.15
        over = 0.06
        kind = "trem" if j == 4 else "legato"
        if j == 4:
            v1 = 0.82
        for sec, nm, g in (("cb", cbn, 1.0), ("vc", vc1, 1.0), ("vc", vc2, 0.9), ("vla", va1, 0.9),
                           ("vla", va2, 0.85), ("vln2", w1, 0.85), ("vln2", w2, 0.8)):
            if j == 4 and sec == "vla" and nm == va2:
                tm = s(7.5)
                S.st(sec, t0, tm - t0 + over, nm, v0 * g, v1 * g * 0.9, kind=kind, att=att, rel=0.2)
                S.st(sec, tm, t1_ - tm, "A4", v1 * g * 0.9, v1 * g, kind=kind, att=0.08, rel=0.1)
                continue
            S.st(sec, t0, t1_ - t0 + (over if j < 4 else 0.0), nm, v0 * g, v1 * g, kind=kind, att=att,
                 rel=0.25 if j < 4 else 0.1)
        for part, nm in (("a", ca), ("s", cs1), ("s", cs2)):
            S.ch(part, t0, t1_ - t0 + (0.1 if j < 4 else 0.0), nm, 0.34 + 0.38 * p0, 0.34 + 0.38 * p1 + (0.12 if j == 4 else 0),
                 va="u" if j == 0 else "a", vb="a", morph_t=0.8, att=0.9 if j == 0 else 0.3,
                 rel=0.6 if j < 4 else 0.08)
    # pivot horns: the held Eb becomes D#, joined by F#3 and B3
    for nm in ("F#3", "B3"):
        S.hn(s(6), s(8.5) - s(6), nm, 0.5, 0.92, att=0.15, rel=0.08)
    # violas run up into the pivot's last bar
    run = ["B3", "C#4", "D#4", "E4", "F#4", "G#4", "A4"]
    for j, nm in enumerate(run):
        t0 = s(6.5) + 0.14 * j
        S.st("vla", t0, 0.16, nm, 0.55 + 0.04 * j, kind="legato", att=0.02, rel=0.08)
    # harp glissando on the gold light (Gb major pentatonic)
    gl = [m for m in range(nn("Gb3"), nn("Gb6") + 1) if m % 12 in (6, 8, 10, 1, 3)]
    for j, m in enumerate(gl):
        S.harp.append((t38 + 0.7 * j / len(gl), m, 0.45 + 0.2 * j / len(gl), 0.4 - 0.8 * j / len(gl)))
    S.perc += [("timp", t38, "Gb2", 0.70), ("crash", t38, 0.55), ("swell", s(2.5), s(3.5), 0.32),
               ("timp", s(3.5), "Bb2", 0.72), ("bd", s(3.5), 0.45), ("timp", s(6), "B2", 0.62),
               ("roll", s(6.5), s(8.5) - 0.04, "B2", 0.25, 0.95), ("swell", s(7.0), s(8.5), 0.55)]

    # ------------------------------------------------------------ 46.5 to 52.0: the climax, E major
    X = M["X"]
    melx = [(-1.0, 0.5, "C#5"), (-0.5, 0.5, "E5"), (0.0, 3.0, "B5"), (3.0, 0.5, "A5"),
            (3.5, 0.5, "G#5"), (4.0, 1.0, "E5"), (5.0, 2.3333, "F#5")]
    for b0, d, nm in melx:
        t0, t1_ = X(b0), X(b0 + d)
        m = nn(nm)
        last = b0 == 5.0
        pv = 0.95 if b0 == 0.0 else (0.70 if b0 < 0 else (0.72 if last else 0.86))
        S.pa(t0, t1_ - t0, m, pv, felt=0.3, exact=(b0 == 0.0))
        S.pa(t0, t1_ - t0, m + 12, pv * 0.88, felt=0.3, exact=(b0 == 0.0))
        if last:
            S.st("vln1", t0, t1_ - t0, m, 0.85, 0.22, att=0.08, rel=1.0)
            S.st("vln2", t0, t1_ - t0, m - 12, 0.75, 0.2, att=0.08, rel=1.0)
            S.hn(t0, t1_ - t0 - 0.2, m - 12, 0.85, 0.2, att=0.08, rel=0.8)
        else:
            S.st("vln1", t0, t1_ - t0 + 0.04, m, 0.88 if b0 >= 0 else 0.8, att=0.05 if b0 == 0 else 0.07, rel=0.35)
            S.st("vln2", t0, t1_ - t0 + 0.04, m - 12, 0.78, att=0.05 if b0 == 0 else 0.07, rel=0.35)
            if b0 >= 0:
                S.hn(t0, t1_ - t0 + 0.04, m - 12, 0.92, att=0.04 if b0 == 0 else 0.06, rel=0.35)
    harmx = [  # (beat0, beat1, cb, vc, vc, vla, vla, horn, horn, low horn, low horn, choir a, choir s, choir s, piano LH)
        (0.0, 2.0, "E1", "E2", "B2", "G#3", "E4", "E4", "G#4", "E3", "B2", "G#4", "E5", "B5", ("E1", "E2")),
        (2.0, 3.0, "C#2", "C#3", "G#3", "E4", "B3", "E4", "G#4", "C#3", "G#2", "G#4", "E5", "B5", ("C#2", "C#3")),
        (3.0, 5.0, "A1", "A2", "E3", "C#4", "E4", "C#4", "E4", "A2", "E3", "A4", "C#5", "E5", ("A1", "A2")),
        (5.0, 6.0, "B1", "B2", "F#3", "E4", "F#4", "E4", "B3", "B2", "F#3", "B4", "E5", "F#5", ("B1", "B2")),
        (6.0, 7.3333, "B1", "B2", "F#3", "D#4", "F#4", "D#4", "B3", "B2", "F#3", "B4", "D#5", "F#5", None),
    ]
    for j, (b0, b1, cbn, vc1, vc2, va1, va2, h1, h2, lh1, lh2, ca, cs1, cs2, plh) in enumerate(harmx):
        t0, t1_ = X(b0), X(b1)
        last = j == 4
        v0 = [0.95, 0.85, 0.85, 0.82, 0.78][j]
        v1 = [0.88, 0.85, 0.82, 0.78, 0.16][j]
        att = 0.03 if j == 0 else 0.12
        rel = 1.1 if last else 0.25
        over = 0.0 if last else 0.06
        for sec, nm, g in (("cb", cbn, 1.0), ("vc", vc1, 1.0), ("vc", vc2, 0.9), ("vla", va1, 0.9), ("vla", va2, 0.85)):
            S.st(sec, t0, t1_ - t0 + over, nm, v0 * g, v1 * g, att=att, rel=rel)
        for nm in (h1, h2):
            S.hn(t0, t1_ - t0 + over, nm, v0 * 0.88, v1 * 0.85, att=0.04 if j == 0 else 0.12, rel=rel)
        for nm in (lh1, lh2):
            S.hn(t0, t1_ - t0 + over, nm, v0 * 0.8, v1 * 0.8, att=0.05 if j == 0 else 0.12, rel=rel, dark=0.3)
        for part, nm in (("a", ca), ("s", cs1), ("s", cs2)):
            S.ch(part, t0, t1_ - t0 + over, nm, v0 * 0.72, v1 * 0.72, va="a", att=0.06 if j == 0 else 0.25, rel=rel + 0.3)
        if plh:
            S.pa(t0, t1_ - t0, plh[0], 0.9 if j == 0 else 0.7, felt=0.3, exact=(j == 0))
            S.pa(t0, t1_ - t0, plh[1], 0.84 if j == 0 else 0.65, felt=0.3, exact=(j == 0))
        S.pedal.append((t0 + 0.02, t1_ - 0.01 if not last else C("cut_last_page") + 0.1))
    S.pedal.append((X(-1.0) - 0.02, X(0.0) - 0.04))
    xs = X(0.0)
    S.perc += [("timp", xs, "E2", 0.88), ("timp", xs, "B2", 0.75), ("bd", xs, 0.72), ("crash", xs, 0.85),
               ("timp", X(2.0), "C#3", 0.6), ("timp", X(3.0), "A2", 0.66), ("timp", X(5.0), "B2", 0.7),
               ("crash", X(5.0), 0.38), ("roll", X(6.0), X(7.3333), "B2", 0.5, 0.08)]
    gl = [m for m in range(nn("E3"), nn("E6") + 1) if m % 12 in (4, 6, 8, 9, 11, 1, 3)]
    for j, m in enumerate(gl):
        S.harp.append((X(2.0) + 0.6 * j / len(gl), m, 0.5 + 0.2 * j / len(gl), -0.5 + 1.0 * j / len(gl)))
    gl = [m for m in range(nn("B6"), nn("B3") - 1, -1) if m % 12 in (4, 6, 8, 9, 11, 1, 3)]
    for j, m in enumerate(gl):
        S.harp.append((X(5.0) + 0.7 * j / len(gl), m, 0.5 - 0.15 * j / len(gl), 0.5 - 1.0 * j / len(gl)))

    # ------------------------------------------------------------ 52.0 to 60.0: quiet returns
    tq = C("cut_last_page")
    for sec, nm, v in (("vc", "A2", 0.14), ("vc", "E3", 0.12), ("vla", "C#4", 0.10), ("vln2", "B4", 0.09),
                       ("vln1", "E5", 0.08)):
        S.st(sec, tq, 2.3, nm, v, v * 0.5, att=0.5, rel=1.8)
    S.roll(tq + 0.35, C("fox_looks_page") - tq - 0.37, ["A2", "E3", "B3", "C#4"], 0.22, spread=0.07)
    for dt, d, nm, v in ((0.40, 0.4, "A5", 0.30), (0.75, 0.45, "G#5", 0.28), (1.15, 1.0, "E5", 0.30)):
        S.pa(tq + dt, d, nm, v)
    tp, tt = C("fox_looks_page"), C("cut_title")
    S.pa(tp, tt - tp, "F#5", 0.32, exact=True)
    S.roll(tp, tt - tp, ["B2", "F#3", "E4"], 0.20, spread=0.06)
    S.pedal += [(tq + 0.33, tp - 0.02), (tp + 0.05, tt)]
    tfin = C("final_paw")
    S.pa(tfin, C("end") - tfin + 1.0, "E5", 0.40, exact=True)
    S.pedal.append((tfin + 0.006, C("end") + 2.0))
    return S, M


# ================================================================================ rendering
def damper_time(t_up, pedal):
    for d, u in pedal:
        if d <= t_up < u:
            return u
    return t_up


def pedal_state(pedal):
    """Per-sample 0..1 sustain pedal position with short mechanical ramps."""
    ps = np.zeros(NS)
    for d, u in pedal:
        a, b = int(max(d, 0) * SR), int(min(u, NS / SR) * SR)
        if b > a:
            ps[a:b] = 1.0
    k = np.ones(int(0.02 * SR)) / int(0.02 * SR)
    return np.convolve(ps, k, mode="same")


def render_music(S, buses, end_t):
    t_start = time.time()
    piano = buses["piano"]
    for i, (t0, t_up, m, v, felt) in enumerate(sorted(S.piano)):
        td = max(0.05, damper_time(t_up, S.pedal) - t0)
        r = rng_for(f"pno{i}:{m}:{t0:.4f}")
        y = felt_piano_note(m, v, td, r, felt=felt, cap=end_t - t0 + 0.05)
        pan = float(np.clip((m - 64) / 40.0, -0.38, 0.38))
        piano.add(y, t0, pan=pan)
        piano.add(key_noise(v, r, "down"), t0 + 0.004, pan=pan)
        if t0 + td < end_t - 0.1 and m < 89:
            piano.add(key_noise(v, r, "up"), t0 + td, pan=pan)
    for j, (d, u) in enumerate(S.pedal):
        r = rng_for(f"pedal{j}")
        piano.add(pedal_noise(r, True), d)
        if u < end_t - 0.2:
            piano.add(pedal_noise(r, False), u)
    # sympathetic resonance: the undamped strings answer while the pedal is down
    ps = pedal_state(S.pedal)
    ir = sympathetic_ir(rng_for("sympathetic"))
    src = piano.x.astype(np.float64) * ps[None, :]
    res = np.vstack([signal.oaconvolve(src[0], ir[0])[:NS], signal.oaconvolve(src[1], ir[1])[:NS]])
    # the dampers fall when the pedal lifts: the halo dies in about a tenth of a second
    k = np.exp(-1.0 / (0.04 * SR))
    damp = np.maximum(ps, signal.lfilter([1.0 - k], [1.0, -k], ps))
    res *= damp[None, :]
    piano.x += (res * 0.20).astype(np.float32)
    px = piano.x.astype(np.float64)
    pk = 20 * np.log10(np.abs(px).max() + 1e-12)
    px, g = limiter(px, pk - 4.0, release=0.05)
    piano.x[:] = px.astype(np.float32)
    log(f"  piano: {len(S.piano)} notes, {len(S.pedal)} pedals ({time.time() - t_start:.1f}s), "
        f"bus peak control max {-20 * np.log10(g.min()):.1f} dB")

    t_start = time.time()
    for i, (sec, t0, dur, m, v, ve, kind, att, rel, glide, mute) in enumerate(S.strings):
        y = bowed_note(sec, m, dur, v, ve, kind, att, rel, rng_for(f"str{i}:{sec}:{m}:{t0:.3f}"),
                       glide=glide, mute=mute)
        buses["strings"].add(y, t0, pan=STR_V2[sec]["pan"])
    log(f"  strings: {len(S.strings)} notes ({time.time() - t_start:.1f}s)")

    t_start = time.time()
    for i, (t0, dur, m, v, ve, att, rel, dark) in enumerate(S.horns):
        y = horn_note(m, dur, v, ve, att, rel, rng_for(f"hn{i}:{m}:{t0:.3f}"), dark=dark)
        buses["brass"].add(y, t0, pan=-0.25 if m >= nn("C3") else 0.1)
    for i, (part, t0, dur, m, v, ve, va, vb, mt, att, rel) in enumerate(S.choir):
        y = choir_note(part, m, dur, v, ve, va, vb, mt, att, rel, rng_for(f"ch{i}:{m}:{t0:.3f}"))
        buses["choir"].add(y, t0)
    log(f"  horns {len(S.horns)}, choir {len(S.choir)} ({time.time() - t_start:.1f}s)")

    for i, (t0, m, v, p) in enumerate(S.harp):
        buses["harp"].add(harp_note(m, v, rng_for(f"harp{i}")), t0, pan=p)
    for i, (t0, m, v, p) in enumerate(S.celesta):
        buses["harp"].add(celesta_tone(m, v, rng_for(f"cel{i}")), t0, pan=p)

    for i, ev in enumerate(S.perc):
        r = rng_for(f"perc{i}")
        kind = ev[0]
        if kind == "timp":
            buses["perc"].add(timpani_hit(nn(ev[2]), ev[3], r), ev[1], pan=-0.15)
        elif kind == "bd":
            buses["perc"].add(bass_drum(ev[2], r), ev[1], pan=0.05)
        elif kind == "crash":
            buses["perc"].add(cymbal("crash", 4.5, ev[2], r), ev[1], pan=0.25)
        elif kind == "swell":
            n = int((ev[2] - ev[1]) * SR)
            buses["perc"].add(cymbal("swell", ev[2] - ev[1], ev[3], r)[:, :n], ev[1], pan=0.25)
        elif kind == "roll":
            t0, t1, note, v0, v1 = ev[1], ev[2], nn(ev[3]), ev[4], ev[5]
            y = timp_roll(note, t1 - t0, v0, v1, r)
            n = int((t1 - t0) * SR)
            if v1 >= v0:          # crescendo rolls stop dead on their target
                y[n:] = 0.0
                y[max(0, n - int(0.006 * SR)):n] *= np.linspace(1, 0, min(n, int(0.006 * SR)))
            buses["perc"].add(y, t0, pan=-0.15)


def music_cue_points(S, M):
    """Where the score is meant to land, for the cue report: (label, time, bus)."""
    return [
        ("piano_in (first note)", M["P"](0), "piano"),
        ("lantern_down (Db5)", M["P"](5), "piano"),
        ("look_up (open Eb5)", M["P"](6), "piano"),
        ("cut_closeup (statement 2)", M["P"](8), "piano"),
        ("hop_land (chord)", C("hop_land"), "piano"),
        ("bird_flap_start (motif head)", C("bird_flap_start"), "piano"),
        ("tremble_one (low motif)", C("tremble_one"), "piano"),
        ("burst (tutti)", M["B"](0), "perc"),
        ("cut_stairwell (horn motif)", M["B"](5), "brass"),
        ("gold_light (bloom)", M["S"](0), "harp"),
        ("cut_sunrise (melody Ab)", M["S"](1), "strings"),
        ("cut_great_bird (pivot)", M["S"](6), "perc"),
        ("wingbeat (climax)", M["X"](0), "perc"),
        ("final_paw (single note)", C("final_paw"), "piano"),
    ]
