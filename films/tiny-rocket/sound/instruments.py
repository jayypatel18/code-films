"""Instruments for the TINY ROCKET score. Every one is a small physical
caricature built from sines, band limited saws and filtered noise.

felt_piano   additive string with inharmonic partials, a felt dampened
             spectrum, two detuned strings per note and a soft hammer knock
music_box    a steel tine: a few inharmonic partials, a pluck click
celesta      softer bar with a resonator body
strings      six detuned band limited saws with slow vibrato, lowpassed
brass        saw through a soft tanh with a filter that opens on the attack
timpani      sine with a pitch drop, membrane modes and a mallet thud
bass_drum    low sine thump
cymbal       noise through a bank of narrow resonators, swelled
glock        bright bar for the climax sparkle
pad          slow detuned sines
"""

import math

import numpy as np

from dsp import (SR, additive, bandpass, env_adsr, env_exp, env_perc,
                 env_points, highpass, hz, lowpass, midi, n_of, resonator,
                 rng_for, saw_bl, sine, svf, t_axis, white)


def _dur(seconds):
    return n_of(seconds)


# ------------------------------------------------------------------ felt piano

def felt_piano(note, length=3.0, velocity=0.5, seed=0):
    """A felt piano note. note: midi number or name."""
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("piano", note, seed)
    t = t_axis(n)
    # the lower the note, the longer it rings
    base_tau = 1.6 * (110.0 / f) ** 0.45 + 0.25
    partials = []
    nk = 14 if f < 400 else 9
    for k in range(1, nk + 1):
        # felt: the hammer is soft, so upper partials start weak and die fast
        amp = (1.0 / k ** 1.9) * (1.0 if k < 4 else 0.55) * (0.5 + velocity)
        if k > 6:
            amp *= 0.45
        tau = base_tau / (1.0 + 0.55 * (k - 1))
        partials.append((k, amp, tau))
    inh = 0.00025 if f < 300 else 0.0006
    phases = rng.uniform(0, 2 * math.pi, nk)
    s1 = additive(f * (1 + rng.uniform(-0.0006, 0.0006)), n, partials, inharmonic=inh, phases=phases)
    s2 = additive(f * (1 + rng.uniform(0.0008, 0.0016)), n, partials, inharmonic=inh,
                  phases=rng.uniform(0, 2 * math.pi, nk))
    s = s1 + 0.7 * s2
    # the attack of a felt hammer is round, not instant
    att = 1.0 - np.exp(-t / 0.006)
    s *= att
    # knock: a soft lowpassed noise thud that depends on velocity
    knock = white(n, rng) * env_perc(n, 0.001, 0.012)
    knock = lowpass(knock, 700.0 + 900.0 * velocity, order=2)
    s += knock * 0.35 * velocity
    # the felt layer rolls the top off
    s = lowpass(s, 2200.0 + 2600.0 * velocity, order=1)
    # release: the damper comes down near the end
    s *= env_points(n, [(0, 1), (length - 0.25, 1), (length - 0.02, 0.0), (length, 0)])
    peak = np.max(np.abs(s)) + 1e-9
    return s / peak * (0.25 + 0.75 * velocity)


# ------------------------------------------------------------------- music box

def music_box(note, length=2.0, velocity=0.6, seed=0):
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("box", note, seed)
    partials = [(1.0, 1.0, 0.9), (2.77, 0.28, 0.25), (5.4, 0.10, 0.09), (8.9, 0.03, 0.05)]
    s = additive(f, n, partials, phases=rng.uniform(0, 2 * math.pi, 4), max_hz=8500.0)
    # the comb plucking the tine: tiny click
    click = white(n, rng) * env_perc(n, 0.0005, 0.004)
    click = bandpass(click, 1800.0, 6000.0, order=2)
    s = s * (1.0 - np.exp(-t_axis(n) / 0.0015)) + click * 0.3
    s = lowpass(s, 8000.0, order=2)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


def celesta(note, length=2.5, velocity=0.5, seed=0):
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("cel", note, seed)
    partials = [(1.0, 1.0, 1.1), (4.0, 0.14, 0.3), (9.2, 0.03, 0.08)]
    s = additive(f, n, partials, phases=rng.uniform(0, 2 * math.pi, 3), max_hz=8500.0)
    mallet = lowpass(white(n, rng) * env_perc(n, 0.001, 0.006), 2500.0, order=2)
    s = s * (1.0 - np.exp(-t_axis(n) / 0.003)) + mallet * 0.2
    # the wooden resonator box under the bar
    s += 0.25 * resonator(s, f, 12.0)
    s = lowpass(s, 7500.0, order=2)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


def glock(note, length=1.8, velocity=0.5, seed=0):
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("glk", note, seed)
    partials = [(1.0, 1.0, 1.3), (2.71, 0.35, 0.5), (5.15, 0.12, 0.2)]
    s = additive(f, n, partials, phases=rng.uniform(0, 2 * math.pi, 3), max_hz=8500.0)
    s *= (1.0 - np.exp(-t_axis(n) / 0.001))
    s = lowpass(s, 8000.0, order=2)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


# --------------------------------------------------------------------- strings

def strings(note, length=4.0, velocity=0.5, attack=0.6, release=0.8,
            voices=6, brightness=1.0, seed=0, tremolo=0.0):
    """A string section: detuned band limited saws with slow independent
    vibrato, a bow noise layer, and a lowpass that opens with velocity."""
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("str", note, seed, voices)
    t = t_axis(n)
    out = np.zeros(n)
    for v in range(voices):
        det = rng.uniform(-0.006, 0.006)
        vib_rate = rng.uniform(4.2, 5.6)
        vib_depth = rng.uniform(0.003, 0.006)
        vib_on = 1.0 - np.exp(-t / rng.uniform(0.4, 0.9))
        ph0 = rng.uniform(0, 2 * math.pi)
        freq = f * (1 + det) * (1 + vib_depth * vib_on * np.sin(2 * math.pi * vib_rate * t + ph0))
        out += saw_bl(freq, n, max_hz=min(7000.0, f * 40), rolloff=1.15) * rng.uniform(0.7, 1.0)
    out /= voices
    # bow noise, mostly in the mids, keyed to the envelope
    bow = bandpass(white(n, rng), 800.0, 3500.0, order=2) * 0.015
    out += bow
    cutoff = 1200.0 + 4200.0 * velocity * brightness
    out = lowpass(out, min(cutoff, 7000.0), order=2)
    if tremolo > 0:
        trem = 1.0 - 0.5 * tremolo * (1 + np.sin(2 * math.pi * 11.0 * t + rng.uniform(0, 6)))
        out *= trem
    env = env_adsr(n, attack, 0.3, 0.85, release, hold=length - release)
    out *= env
    return out / (np.max(np.abs(out)) + 1e-9) * velocity


def string_run(notes, start_len, end_len, velocity=0.5, seed=0):
    """An ascending run: each note slightly shorter than the last so it
    accelerates. Returns a mono signal of the whole run."""
    lens = np.linspace(start_len, end_len, len(notes))
    total = float(np.sum(lens)) + 1.5
    out = np.zeros(_dur(total))
    pos = 0.0
    for i, (note, ln) in enumerate(zip(notes, lens)):
        s = strings(note, length=ln + 0.6, velocity=velocity, attack=min(0.08, ln * 0.3),
                    release=0.4, voices=4, seed=seed + i)
        a = n_of(pos)
        out[a:a + s.shape[0]] += s[: out.shape[0] - a]
        pos += ln
    return out


# ----------------------------------------------------------------------- brass

def brass(note, length=3.0, velocity=0.6, attack=0.12, release=0.5, seed=0, voices=3):
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("brs", note, seed)
    t = t_axis(n)
    out = np.zeros(n)
    for v in range(voices):
        det = rng.uniform(-0.004, 0.004)
        vib = 1 + 0.003 * np.sin(2 * math.pi * rng.uniform(4.5, 5.5) * t + rng.uniform(0, 6)) * (1 - np.exp(-t / 0.5))
        # a horn's lip buzz starts flat and settles up
        scoop = 1 - 0.02 * np.exp(-t / 0.05)
        out += saw_bl(f * (1 + det) * vib * scoop, n, max_hz=min(6500.0, f * 30), rolloff=1.0)
    out /= voices
    out = np.tanh(out * (1.5 + 2.5 * velocity)) / (1.5 + 2.5 * velocity)
    # the filter opens over the attack: that is what makes the swell "brassy"
    cutoff = 500.0 + (900.0 + 3600.0 * velocity) * (1 - np.exp(-t / max(attack, 0.03)))
    out = svf(out, cutoff, 0.9, "low")
    env = env_adsr(n, attack, 0.1, 0.9, release, hold=length - release)
    out *= env
    return out / (np.max(np.abs(out)) + 1e-9) * velocity


# ----------------------------------------------------------------- percussion

def timpani(note="D2", length=2.5, velocity=0.7, seed=0):
    f = hz(note) if isinstance(note, str) else midi(note)
    n = _dur(length)
    rng = rng_for("timp", note, seed)
    t = t_axis(n)
    drop = f * (1 + 0.35 * np.exp(-t / 0.05))
    body = sine(drop, n) * np.exp(-t / 0.9)
    # membrane modes of a kettle drum: 1.5, 1.98, 2.44 ratios
    modes = additive(f, n, [(1.5, 0.5, 0.5), (1.98, 0.25, 0.35), (2.44, 0.15, 0.25), (2.9, 0.08, 0.2)],
                     phases=rng.uniform(0, 6, 4))
    thud = lowpass(white(n, rng) * env_perc(n, 0.0005, 0.02), 900.0, order=2)
    s = body + 0.6 * modes + 0.6 * thud * velocity
    s = lowpass(s, 2500.0, order=1)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


def timpani_roll(note, length, vel_env, seed=0, rate=14.0):
    """A roll: hits at the given rate, loudness following vel_env (0..1 array
    sampled per hit)."""
    n = _dur(length + 2.5)
    out = np.zeros(n)
    rng = rng_for("roll", note, seed)
    k = 0
    pos = 0.0
    while pos < length:
        v = float(np.interp(pos, np.linspace(0, length, len(vel_env)), vel_env))
        h = timpani(note, length=1.6, velocity=0.3 + 0.7 * v, seed=k)
        a = n_of(pos)
        out[a:a + h.shape[0]] += h[: n - a] * (0.6 + 0.4 * v) * rng.uniform(0.85, 1.0)
        pos += (1.0 / rate) * rng.uniform(0.9, 1.1)
        k += 1
    return out


def bass_drum(length=1.5, velocity=0.8, seed=0):
    n = _dur(length)
    t = t_axis(n)
    rng = rng_for("bd", seed)
    f = 42.0 * (1 + 2.5 * np.exp(-t / 0.04))
    s = sine(f, n) * np.exp(-t / 0.45)
    s += 0.3 * lowpass(white(n, rng) * env_perc(n, 0.0005, 0.015), 600.0, order=2)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


def cymbal(length=3.0, swell=1.5, velocity=0.5, seed=0):
    """Suspended cymbal: white noise into many narrow resonators, swelled with
    a slow attack and a long shimmering decay. Capped under 7.5 kHz."""
    n = _dur(length)
    rng = rng_for("cym", seed)
    w = white(n, rng)
    s = np.zeros(n)
    freqs = rng.uniform(1800.0, 7200.0, 22)
    for i, fq in enumerate(freqs):
        s += resonator(w, fq, rng.uniform(25.0, 60.0)) * rng.uniform(0.5, 1.0)
    s += 0.15 * bandpass(w, 3000.0, 7000.0, order=2)
    s = lowpass(s, 7500.0, order=3)
    t = t_axis(n)
    env = np.minimum(1.0, (t / swell) ** 2.2) * np.exp(-np.maximum(0, t - swell) / 1.1)
    s *= env
    return s / (np.max(np.abs(s)) + 1e-9) * velocity


def pad(note, length=6.0, velocity=0.3, attack=1.5, release=1.5, seed=0):
    f = midi(note) if isinstance(note, (int, float)) else hz(note)
    n = _dur(length)
    rng = rng_for("pad", note, seed)
    t = t_axis(n)
    s = np.zeros(n)
    for k, (ratio, amp) in enumerate([(1.0, 1.0), (2.0, 0.3), (3.0, 0.1), (0.5, 0.25)]):
        det = rng.uniform(-0.003, 0.003)
        s += amp * np.sin(2 * math.pi * f * ratio * (1 + det) * t + rng.uniform(0, 6))
        s += amp * 0.8 * np.sin(2 * math.pi * f * ratio * (1 - det) * t + rng.uniform(0, 6))
    s *= 1 + 0.08 * np.sin(2 * math.pi * 0.15 * t)
    s *= env_adsr(n, attack, 0.5, 0.9, release, hold=length - release)
    return s / (np.max(np.abs(s)) + 1e-9) * velocity
