"""Sound effects for TINY ROCKET. Every effect is synthesized: filtered
noise, pulse trains, resonators and sines shaped by envelopes. Each function
returns a mono or stereo float array at 48 kHz; the placement is in score.py.
"""

import math

import numpy as np

from dsp import (SR, additive, bandpass, brown, env_exp, env_perc, env_points,
                 highpass, lowpass, n_of, pan, pan_moving, pink, resonator,
                 rng_for, sine, svf, t_axis, white)


def _norm(s, level):
    return s / (np.max(np.abs(s)) + 1e-9) * level


# ------------------------------------------------------------------ ambiences

def crickets(length, density=6, rate_scale=1.0, seed=0, level=0.2):
    """A field of crickets. Each cricket is a chirp train: a chirp is three
    or four short pulses of a tone near 4 kHz (a real cricket's wing scrape),
    repeated at its own rate, with its own pitch and position. Returns (n, 2)."""
    n = n_of(length)
    out = np.zeros((n, 2))
    rng = rng_for("crickets", seed)
    t = t_axis(n)
    for c in range(density):
        f = rng.uniform(3600.0, 4900.0)
        chirp_rate = rng.uniform(1.4, 2.6) * rate_scale
        pulses = int(rng.integers(3, 5))
        pulse_len = rng.uniform(0.018, 0.028)
        gap = rng.uniform(0.012, 0.02)
        amp_env = np.zeros(n)
        pos = rng.uniform(0, 1.0 / chirp_rate)
        while pos < length:
            # crickets pause now and then
            if rng.random() < 0.08:
                pos += rng.uniform(0.8, 2.5)
                continue
            for p in range(pulses):
                a = n_of(pos + p * (pulse_len + gap))
                b = min(n, a + n_of(pulse_len))
                if a >= n:
                    break
                k = b - a
                w = np.sin(np.linspace(0, math.pi, k)) ** 0.7
                amp_env[a:b] = np.maximum(amp_env[a:b], w)
            pos += (1.0 / chirp_rate) * rng.uniform(0.9, 1.1)
        tone = np.sin(2 * math.pi * f * t + rng.uniform(0, 6)) * (1 + 0.3 * np.sin(2 * math.pi * f * 0.5 * t))
        tone = tone * amp_env * rng.uniform(0.5, 1.0)
        # distance: far crickets are softer and duller
        dist = rng.uniform(0.3, 1.0)
        tone = lowpass(tone, 5500.0 + 1500.0 * dist, order=2) * (0.25 + 0.75 * dist)
        out += pan(tone, rng.uniform(-0.9, 0.9))
    return _norm(out, level)


def breeze(length, seed=0, level=0.15, gustiness=1.0, grass=0.6):
    """Wind in grass at dusk: pink noise with slow random gusts, plus a higher
    grass rustle band whose level follows the gusts. Independent per side."""
    n = n_of(length)
    rng = rng_for("breeze", seed)
    out = np.zeros((n, 2))
    gust_base = _slow_random(n, 0.18, rng)
    for ch in range(2):
        gust = 0.5 + 0.5 * (0.7 * gust_base + 0.3 * _slow_random(n, 0.3, rng))
        gust = gust ** 1.8 * gustiness + 0.15
        w = pink(n, rng)
        body = svf(highpass(w, 60.0, 1), 500.0 + 500.0 * gust, 0.7, "low")
        rustle = bandpass(white(n, rng), 1200.0, 6000.0, order=2)
        flutter = 0.5 + 0.5 * _slow_random(n, 6.0, rng)
        rustle *= flutter * gust ** 2 * grass * 0.08
        out[:, ch] = body * gust * 0.5 + rustle
    return _norm(out, level)


def _slow_random(n, rate_hz, rng):
    """Smooth random modulation in [-1, 1] at about rate_hz."""
    k = max(4, int(n * rate_hz / SR) + 2)
    pts = rng.uniform(-1, 1, k)
    xs = np.linspace(0, n, k)
    return np.interp(np.arange(n), xs, pts)


def night_air(length, seed=0, level=0.05):
    """Deep night: a very soft low air bed."""
    n = n_of(length)
    rng = rng_for("night", seed)
    out = np.zeros((n, 2))
    for ch in range(2):
        out[:, ch] = lowpass(brown(n, rng), 250.0, order=2) * (0.8 + 0.2 * _slow_random(n, 0.1, rng))
    return _norm(out, level)


def wind_high(length, seed=0, level=0.5, intensity=None, closeness=None):
    """Rushing wind for the climb: noise through a resonant lowpass whose
    cutoff wanders, and a hiss layer. intensity and closeness are per sample
    arrays in 0..1 (or None)."""
    n = n_of(length)
    rng = rng_for("wind", seed)
    if intensity is None:
        intensity = np.ones(n)
    if closeness is None:
        closeness = np.zeros(n)
    out = np.zeros((n, 2))
    for ch in range(2):
        w = white(n, rng)
        wander = 0.5 + 0.5 * _slow_random(n, 0.7, rng)
        wander2 = 0.5 + 0.5 * _slow_random(n, 2.5, rng)
        fc = 250.0 + 900.0 * wander + 500.0 * wander2 * intensity
        howl = svf(w, fc, 2.2 + 1.5 * intensity, "band")
        body = svf(w, 300.0 + 1500.0 * intensity, 0.8, "low")
        hiss = bandpass(w, 2500.0, 7000.0, order=2) * 0.08 * (0.3 + 0.7 * wander2)
        close = lowpass(w, 180.0, 2) * closeness * 2.0
        out[:, ch] = (howl * 0.9 + body * 0.5 + hiss + close) * (0.2 + 0.8 * intensity)
    return _norm(out, level)


def morning_birds(length, seed=0, level=0.12, count=5):
    """Distant songbirds: short FM chirp phrases between 2.5 and 6 kHz."""
    n = n_of(length)
    rng = rng_for("birds", seed)
    out = np.zeros((n, 2))
    for b in range(count):
        base = rng.uniform(2600.0, 4800.0)
        p = rng.uniform(-0.8, 0.8)
        dist = rng.uniform(0.2, 0.8)
        pos = rng.uniform(0.3, 3.0)
        while pos < length - 0.5:
            syll = int(rng.integers(2, 6))
            for s in range(syll):
                ln = rng.uniform(0.05, 0.14)
                k = n_of(ln)
                tt = t_axis(k)
                shape = rng.choice([1, 2, 3])
                if shape == 1:
                    f = base * (1 + 0.25 * np.sin(2 * math.pi * tt / ln * 0.5))
                elif shape == 2:
                    f = base * (1.2 - 0.3 * tt / ln)
                else:
                    f = base * (1 + 0.15 * np.sin(2 * math.pi * 60 * tt))
                env = np.sin(np.pi * tt / ln) ** 1.5
                sig = np.sin(2 * math.pi * np.cumsum(f) / SR) * env
                a = n_of(pos)
                if a + k < n:
                    out[a:a + k] += pan(sig, p) * (0.3 + 0.7 * dist)
                pos += ln + rng.uniform(0.03, 0.12)
            pos += rng.uniform(1.2, 4.0)
    out = lowpass(out, 7000.0, order=2)
    return _norm(out, level)


# ------------------------------------------------------------ small intimate

def knot_creak(seed=0):
    """A rope knot pulled tight on bamboo: stick slip creak (a pulse train
    whose rate climbs as the string tightens), fibre tension noise, and the
    final tug thunk. About 0.3 s."""
    n = n_of(0.34)
    rng = rng_for("knot", seed)
    t = t_axis(n)
    rate = 70.0 + 520.0 * (t / 0.34) ** 2.5
    ph = np.cumsum(rate) / SR
    pulses = (np.mod(ph, 1.0) < 0.1).astype(float)
    pulses = np.diff(np.concatenate([[0.0], pulses])) > 0
    imp = pulses.astype(float) * rng.uniform(0.6, 1.0, n)
    creak = resonator(imp, 1400.0, 9.0) + 0.6 * resonator(imp, 2300.0, 12.0) + 0.4 * resonator(imp, 760.0, 7.0)
    creak *= env_points(n, [(0, 0.25), (0.08, 0.7), (0.22, 1.0), (0.27, 0.0), (0.34, 0)])
    fibre = bandpass(white(n, rng), 900.0, 4500.0, order=2) * env_points(n, [(0, 0.1), (0.2, 0.5), (0.26, 0.0), (0.34, 0)])
    thunk = lowpass(white(n, rng), 380.0, 2)
    thunk_env = np.zeros(n)
    a = n_of(0.24)
    thunk_env[a:] = env_perc(n - a, 0.002, 0.03)
    thunk *= thunk_env
    bamboo = resonator(np.diff(np.concatenate([[0.0], thunk_env])) * 20, 620.0, 25.0) * 0.6
    s = creak * 0.6 + fibre * 0.5 + thunk * 1.2 + bamboo
    s = lowpass(s, 6500.0, order=2)
    return _norm(s, 1.0)


def string_release(seed=0):
    """The string end slipping from the teeth: a short dry fwip."""
    n = n_of(0.12)
    rng = rng_for("srel", seed)
    w = white(n, rng)
    fc = np.linspace(3200.0, 900.0, n)
    s = svf(w, fc, 2.5, "band") * env_perc(n, 0.003, 0.025)
    return _norm(lowpass(s, 5500.0, 2), 1.0)


def ear_flick(seed=0):
    """A tiny soft puff of fur, 40 ms."""
    n = n_of(0.05)
    rng = rng_for("ear", seed)
    s = bandpass(white(n, rng), 1500.0, 4500.0, order=2) * env_perc(n, 0.004, 0.012)
    return _norm(lowpass(s, 5000.0, 1), 1.0)


def blink(seed=0):
    """An almost subliminal soft tick: 25 ms, lowpassed. Intimate close up."""
    n = n_of(0.03)
    rng = rng_for("blink", seed)
    s = lowpass(white(n, rng), 1800.0, order=2) * env_perc(n, 0.002, 0.007)
    return _norm(s, 1.0)


def tail_swish(seed=0, length=0.3):
    n = n_of(length)
    rng = rng_for("tail", seed)
    w = white(n, rng)
    t = t_axis(n)
    fc = 900.0 + 1600.0 * np.sin(np.pi * t / length)
    s = svf(w, fc, 1.6, "band") * np.sin(np.pi * t / length) ** 1.4
    return _norm(lowpass(s, 5000.0, 1), 1.0)


def fur_rustle(seed=0, length=0.25):
    n = n_of(length)
    rng = rng_for("fur", seed)
    s = bandpass(white(n, rng), 1200.0, 5000.0, order=2) * np.sin(np.pi * t_axis(n) / length) ** 2
    s *= 0.6 + 0.4 * _slow_random(n, 40.0, rng)
    return _norm(s, 1.0)


def paw_step(seed=0, weight=1.0):
    """A paw landing in grass: a crinkly grass crunch (noise with sparse micro
    transients) over a soft low thud."""
    n = n_of(0.11)
    rng = rng_for("paw", seed)
    w = white(n, rng)
    crunch = bandpass(w, 1100.0, 5200.0, order=2)
    micro = (rng.random(n) < 0.012).astype(float) * rng.uniform(0.3, 1.0, n)
    micro = lowpass(micro, 4000.0, 1) * 6
    crunch = (crunch + micro) * env_perc(n, 0.003, 0.03)
    thud = lowpass(w, 160.0, 2) * env_perc(n, 0.002, 0.04) * weight
    s = crunch * 0.35 + thud * 1.4
    return _norm(lowpass(s, 6000.0, 1), 1.0)


def breath_cycle(length, period=4.2, seed=0, level=0.2):
    """Sleeping breath: in and out, filtered noise with a slow sine shape."""
    n = n_of(length)
    rng = rng_for("breath", seed)
    t = t_axis(n)
    phase = 2 * math.pi * t / period
    inhale = np.maximum(0, np.sin(phase)) ** 1.6
    exhale = np.maximum(0, -np.sin(phase)) ** 1.3
    w = white(n, rng)
    s = lowpass(w, 900.0, 2) * inhale * 0.6 + lowpass(w, 500.0, 2) * exhale
    s = highpass(s, 120.0, 1)
    return _norm(s, level)


def yawn(seed=0):
    """A small animal yawn: a breathy noise swell with a tiny squeak at the
    top, 0.7 s. Mostly breath, the squeak is quiet."""
    n = n_of(0.7)
    rng = rng_for("yawn", seed)
    t = t_axis(n)
    env = np.sin(np.pi * t / 0.7) ** 1.3
    w = white(n, rng)
    breath = svf(w, 400.0 + 900.0 * env, 1.0, "band") * env
    sq_env = env_points(n, [(0, 0), (0.2, 0), (0.3, 1), (0.45, 0.7), (0.55, 0), (0.7, 0)])
    f = 1150.0 + 250.0 * np.sin(np.pi * t / 0.7) + 20 * np.sin(2 * math.pi * 7 * t)
    squeak = (sine(f, n) + 0.3 * sine(f * 2, n)) * sq_env
    s = breath * 1.0 + squeak * 0.18
    return _norm(lowpass(s, 5000.0, 2), 1.0)


def swallow(seed=0):
    n = n_of(0.22)
    rng = rng_for("swallow", seed)
    t = t_axis(n)
    f = 200.0 * np.exp(-t / 0.08) + 90.0
    s = sine(f, n) * env_perc(n, 0.01, 0.05)
    s += lowpass(white(n, rng), 500.0, 2) * env_points(n, [(0, 0), (0.03, 1), (0.1, 0.2), (0.22, 0)]) * 0.7
    return _norm(s, 1.0)


def cloth(seed=0, length=0.2):
    """Soft cloth or fur movement."""
    n = n_of(length)
    rng = rng_for("cloth", seed)
    s = bandpass(white(n, rng), 600.0, 3500.0, order=2) * np.sin(np.pi * t_axis(n) / length) ** 2
    return _norm(s, 1.0)


# --------------------------------------------------------------- night magic

def village_light(seed=0):
    """A far window lighting up: a tiny soft glassy ting, barely there."""
    n = n_of(0.6)
    rng = rng_for("vl", seed)
    f = rng.uniform(2200.0, 3600.0)
    s = additive(f, n, [(1.0, 1.0, 0.18), (2.4, 0.2, 0.06)], phases=rng.uniform(0, 6, 2))
    s *= (1 - np.exp(-t_axis(n) / 0.004))
    return _norm(s, 1.0)


def shooting_star(seed=0):
    """A rising whistle with shimmer, a sparkle of bell partials, and a soft
    air sweep. 1.1 s, meant to be panned across the sky."""
    n = n_of(1.1)
    rng = rng_for("star", seed)
    t = t_axis(n)
    f = 1500.0 * 2 ** (1.6 * (t / 1.1) ** 0.8)
    env = np.sin(np.pi * t / 1.1) ** 1.2
    whistle = sine(f * (1 + 0.004 * np.sin(2 * math.pi * 9 * t)), n) * env
    sparkle = np.zeros(n)
    for k in range(14):
        a = n_of(rng.uniform(0.05, 0.9))
        ln = n_of(0.25)
        fq = rng.uniform(3000.0, 7000.0)
        b = min(n, a + ln)
        sparkle[a:b] += np.sin(2 * math.pi * fq * t[: b - a]) * np.exp(-t[: b - a] / 0.05) * rng.uniform(0.3, 1.0)
    air = svf(white(n, rng), 2500.0 + 3000.0 * t / 1.1, 1.5, "band") * env * 0.4
    s = whistle * 0.5 + sparkle * 0.35 + air
    return _norm(lowpass(s, 7500.0, 2), 1.0)


def firefly(seed=0):
    """One firefly blink: a soft, very short sine blip."""
    n = n_of(0.09)
    rng = rng_for("ff", seed)
    f = rng.uniform(2300.0, 3800.0)
    s = sine(f, n) * np.sin(np.pi * t_axis(n) / 0.09) ** 2.5
    return _norm(s, 1.0)


# ------------------------------------------------------------------- cockpit

def teacup_land(seed=0):
    """Landing in the teacup cockpit: a ceramic clonk (a few ringing modes),
    a thud, and tin rattling around it."""
    n = n_of(0.8)
    rng = rng_for("cup", seed)
    t = t_axis(n)
    imp = np.zeros(n)
    imp[0] = 1.0
    modes = [(1180.0, 40.0, 1.0), (2360.0, 60.0, 0.5), (3420.0, 70.0, 0.3), (640.0, 25.0, 0.4)]
    ring = np.zeros(n)
    for fq, q, g in modes:
        ring += resonator(imp, fq, q) * g
    ring *= np.exp(-t / 0.25)
    thud = lowpass(white(n, rng), 220.0, 2) * env_perc(n, 0.002, 0.06)
    s = ring * 25 + thud * 1.2
    tin = tin_rattle(0.5, seed=seed, density=12)
    s[: tin.shape[0]] += tin * 0.5 * np.exp(-t[: tin.shape[0]] / 0.2)
    return _norm(lowpass(s, 6500.0, 1), 1.0)


def tin_rattle(length, seed=0, density=25, brightness=1.0):
    """Loose tin: random impacts, each a few metallic modes, at the given
    density per second. Mono."""
    n = n_of(length)
    rng = rng_for("tin", seed)
    out = np.zeros(n)
    count = max(1, int(length * density))
    for i in range(count):
        a = n_of(rng.uniform(0, length))
        k = n_of(0.08)
        b = min(n, a + k)
        if b <= a:
            continue
        imp = np.zeros(b - a)
        imp[0] = rng.uniform(0.3, 1.0)
        hit = np.zeros(b - a)
        for _ in range(3):
            fq = rng.uniform(700.0, 3200.0) * brightness
            hit += resonator(imp, fq, rng.uniform(15.0, 40.0))
        hit *= np.exp(-t_axis(b - a) / rng.uniform(0.01, 0.04))
        out[a:b] += hit
    return _norm(out, 1.0)


def paper_flutter(length, seed=0, rate=22.0):
    """Paper flapping in wind: a noise band whose amplitude flaps at rate Hz
    with irregularity."""
    n = n_of(length)
    rng = rng_for("paper", seed)
    t = t_axis(n)
    flap = np.maximum(0, np.sin(2 * math.pi * rate * t + 1.5 * _slow_random(n, 3.0, rng))) ** 3
    w = white(n, rng)
    s = bandpass(w, 700.0, 3500.0, order=2) * flap + lowpass(w, 300.0, 2) * flap * 0.5
    return _norm(s, 1.0)


def goggles_snap(seed=0):
    n = n_of(0.25)
    rng = rng_for("gog", seed)
    w = white(n, rng)
    stretch = svf(w, np.linspace(2500.0, 4500.0, n), 2.0, "band") * env_points(n, [(0, 0), (0.02, 0.3), (0.07, 0.0), (0.25, 0)])
    imp = np.zeros(n)
    imp[n_of(0.075)] = 1.0
    thwack = resonator(imp, 320.0, 6.0) * 10 + resonator(imp, 1500.0, 8.0) * 3
    thwack += lowpass(w, 600.0, 2) * np.concatenate([np.zeros(n_of(0.075)), env_perc(n - n_of(0.075), 0.001, 0.02)])
    s = stretch * 0.5 + thwack
    return _norm(s, 1.0)


# ---------------------------------------------------------------- fire, fuse

def match_strike(seed=0):
    """A match on the box: a 0.11 s granular rasp, then the flare, a soft
    low whump with a sizzle that settles into a tiny flame flicker."""
    n = n_of(0.9)
    rng = rng_for("match", seed)
    t = t_axis(n)
    w = white(n, rng)
    grain = (rng.random(n) < 0.08).astype(float)
    rasp = bandpass(w * (0.4 + grain * 2.0), 1500.0, 6000.0, order=2)
    rasp *= env_points(n, [(0, 0), (0.01, 1.0), (0.1, 0.8), (0.13, 0.0), (0.9, 0)])
    a = n_of(0.12)
    flare_env = np.zeros(n)
    flare_env[a:] = env_perc(n - a, 0.012, 0.09)
    flare = lowpass(w, 900.0, 2) * flare_env * 1.2
    sizzle_env = np.zeros(n)
    sizzle_env[a:] = env_points(n - a, [(0, 0.6), (0.15, 1.0), (0.4, 0.3), (0.78, 0.15)])
    sizzle = bandpass(w, 2500.0, 7000.0, order=2) * sizzle_env * (0.5 + 0.5 * _slow_random(n, 35.0, rng)) * 0.35
    s = rasp * 0.7 + flare + sizzle
    return _norm(lowpass(s, 7500.0, 1), 1.0)


def fuse(length, seed=0):
    """A burning fuse: a fizzing noise band with sparse crackles and a slow
    rise in pitch as it races along the grass. Mono; score pans it."""
    n = n_of(length)
    rng = rng_for("fuse", seed)
    t = t_axis(n)
    w = white(n, rng)
    fc = 2200.0 + 1800.0 * (t / length)
    fizz = svf(w, fc, 1.2, "band") * (0.6 + 0.4 * _slow_random(n, 25.0, rng))
    crack = (rng.random(n) < 0.0025).astype(float) * rng.uniform(0.3, 1.0, n)
    crack = lowpass(crack, 5000.0, 1) * 8
    crack = bandpass(crack, 1200.0, 6500.0, order=1)
    hiss = bandpass(w, 4000.0, 7500.0, order=2) * 0.25
    s = (fizz + hiss + crack) * env_points(n, [(0, 0), (0.05, 1), (length - 0.1, 1), (length, 0.0)])
    return _norm(lowpass(s, 7800.0, 2), 1.0)


def tick(seed=0):
    """A dry wooden clock tick, 30 ms."""
    n = n_of(0.05)
    rng = rng_for("tick", seed)
    imp = np.zeros(n)
    imp[0] = 1.0
    s = resonator(imp, 2100.0, 14.0) * 10 + resonator(imp, 3900.0, 20.0) * 4 + lowpass(white(n, rng), 1200.0, 2) * env_perc(n, 0.0005, 0.004) * 0.6
    s *= np.exp(-t_axis(n) / 0.012)
    return _norm(s, 1.0)


# ------------------------------------------------------------------- ignition

def ignition(seed=0):
    """Ignition: a sub boom, a huge noise burst whose lowpass sweeps from
    bright to dull over 1.5 s, sparks crackling, and a widening whoosh for the
    ring of flattened grass. Returns stereo, 3 s."""
    n = n_of(3.0)
    rng = rng_for("ign", seed)
    t = t_axis(n)
    sub = sine(55.0 * (1 + 1.2 * np.exp(-t / 0.12)) * np.exp(-t / 6.0) + 0.0, n) * np.exp(-t / 0.7)
    sub += 0.5 * sine(38.0, n) * env_points(n, [(0, 0), (0.05, 1), (1.0, 0.6), (3.0, 0)])
    burst_fc = 7000.0 * np.exp(-t / 0.5) + 350.0
    out = np.zeros((n, 2))
    for ch in range(2):
        w = white(n, rng)
        burst = svf(w, burst_fc, 0.9, "low") * env_points(n, [(0, 0), (0.008, 1), (0.25, 0.8), (1.5, 0.35), (3.0, 0.15)])
        crack = (rng.random(n) < 0.004 * np.exp(-t / 0.8)).astype(float) * rng.uniform(0.2, 1.0, n)
        crack = bandpass(lowpass(crack, 6000.0, 1) * 10, 900.0, 6500.0, order=1)
        ring_delay = n_of(0.06 + 0.03 * ch)
        whoosh = np.zeros(n)
        k = n - ring_delay
        whoosh[ring_delay:] = svf(white(k, rng), np.linspace(2500.0, 400.0, k), 1.8, "band") * env_perc(k, 0.03, 0.22)
        out[:, ch] = burst * 1.0 + crack * 0.8 + whoosh * 0.9 + sub * 1.3
    out = lowpass(out, 7500.0, 2)
    return _norm(out, 1.0)


def rocket_roar(length, level_env, pitch_env, seed=0):
    """The sustained rocket exhaust: brown and white noise with a resonant
    body, a crackling layer, and a slow filter that follows pitch_env (0..1)
    so the roar drops in brightness and pitch as the rocket moves away.
    level_env and pitch_env: per sample arrays."""
    n = n_of(length)
    rng = rng_for("roar", seed)
    out = np.zeros((n, 2))
    flutter = 0.5 + 0.5 * _slow_random(n, 18.0, rng)
    for ch in range(2):
        w = white(n, rng)
        b = brown(n, rng)
        body = svf(w, 120.0 + 900.0 * pitch_env, 1.3, "low")
        growl = np.tanh(b * 3.0 * (0.6 + 0.6 * flutter))
        roarmid = svf(w, 350.0 + 1800.0 * pitch_env, 1.6, "band") * (0.6 + 0.4 * flutter)
        crack = (rng.random(n) < 0.0035).astype(float) * rng.uniform(0.2, 1.0, n)
        crack = bandpass(lowpass(crack, 5000.0, 1) * 8, 700.0, 5500.0, order=1) * (0.3 + 0.7 * pitch_env)
        out[:, ch] = (body * 1.2 + growl * 0.9 + roarmid * 0.7 + crack * 0.5) * level_env
    out = lowpass(out, 7000.0, 2)
    return _norm(out, 1.0)


def bird_burst(seed=0, count=9):
    """A flock bursting from the tree: each bird a flutter of wing beats
    (noise bursts at 9 to 13 Hz for about a second) and an alarm chirp,
    scattering across the stereo field. 2.2 s stereo."""
    n = n_of(2.2)
    rng = rng_for("flock", seed)
    out = np.zeros((n, 2))
    t = t_axis(n)
    for b in range(count):
        start = rng.uniform(0.0, 0.35)
        rate = rng.uniform(9.0, 13.5)
        dur = rng.uniform(0.7, 1.3)
        p0 = rng.uniform(-0.5, 0.5)
        p1 = p0 + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
        p_env = np.clip(np.interp(t, [start, start + dur], [p0, p1]), -1, 1)
        beats = np.maximum(0, np.sin(2 * math.pi * rate * (t - start) + rng.uniform(0, 6))) ** 4
        beats *= ((t >= start) & (t < start + dur)).astype(float)
        beats *= np.exp(-(t - start) / (dur * 0.8))
        w = white(n, rng)
        wing = (bandpass(w, 400.0, 2500.0, order=2) + 0.6 * lowpass(w, 250.0, 2)) * beats
        chirp = np.zeros(n)
        if rng.random() < 0.8:
            ca = n_of(start + rng.uniform(0.0, 0.3))
            k = n_of(0.09)
            tt = t_axis(k)
            f = rng.uniform(3200.0, 5200.0) * (1 + 0.2 * np.sin(2 * math.pi * 30 * tt))
            chirp[ca:ca + k] = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.sin(np.pi * tt / 0.09) * 0.5
        out += pan_moving(wing * rng.uniform(0.6, 1.0) + chirp, p_env)
    return _norm(lowpass(out, 7000.0, 2), 1.0)


def cloud_punch(seed=0):
    """Punching through a cloud layer: a soft wide whoomp of low noise, 0.6 s."""
    n = n_of(0.7)
    rng = rng_for("cloud", seed)
    out = np.zeros((n, 2))
    for ch in range(2):
        w = white(n, rng)
        out[:, ch] = svf(w, np.linspace(1800.0, 250.0, n), 1.0, "low") * env_points(n, [(0, 0), (0.06, 1), (0.3, 0.5), (0.7, 0)])
    return _norm(out, 1.0)


def open_sky_whoosh(seed=0):
    """Breaking into open sky: a rising sweep that releases into air. 1.6 s."""
    n = n_of(1.6)
    rng = rng_for("open", seed)
    t = t_axis(n)
    out = np.zeros((n, 2))
    for ch in range(2):
        w = white(n, rng)
        fc = 300.0 * 2 ** (4.2 * np.minimum(1.0, t / 0.9))
        out[:, ch] = svf(w, fc, 2.0, "band") * env_points(n, [(0, 0), (0.6, 0.7), (0.9, 1.0), (1.6, 0)])
    return _norm(lowpass(out, 7500.0, 2), 1.0)


def star_ignite(seed=0):
    """The trail igniting into a shooting star: a shimmering upward sweep
    of many sines plus a bright sparkle cloud. 2.5 s stereo."""
    n = n_of(2.5)
    rng = rng_for("starign", seed)
    t = t_axis(n)
    out = np.zeros((n, 2))
    for k in range(24):
        f0 = rng.uniform(600.0, 2400.0)
        f = f0 * 2 ** (1.5 * np.minimum(1.0, t / 0.45) ** 0.7)
        env = np.sin(np.pi * np.minimum(1.0, t / 1.6)) ** 1.5 * (t < 1.6)
        s = np.sin(2 * math.pi * np.cumsum(f) / SR + rng.uniform(0, 6)) * env * rng.uniform(0.3, 1.0)
        out += pan(s, rng.uniform(-0.8, 0.8))
    for k in range(40):
        a = n_of(rng.uniform(0.2, 2.0))
        ln = n_of(0.3)
        b = min(n, a + ln)
        fq = rng.uniform(2500.0, 7000.0)
        s = np.sin(2 * math.pi * fq * t[: b - a]) * np.exp(-t[: b - a] / 0.06) * rng.uniform(0.2, 1.0)
        out[a:b] += pan(s, rng.uniform(-0.9, 0.9)) * 0.5
    air = svf(white(n, rng), 3000.0 + 2500.0 * np.minimum(1, t / 0.5), 1.5, "band") * env_points(n, [(0, 0), (0.3, 1), (1.2, 0.3), (2.5, 0)])
    out += pan(air, 0.0) * 0.5
    return _norm(lowpass(out, 7800.0, 2), 1.0)


def star_sheen(length, seed=0):
    """The arc's faint high sheen while it crosses the sky. Mono."""
    n = n_of(length)
    rng = rng_for("sheen", seed)
    t = t_axis(n)
    s = svf(white(n, rng), 3500.0 + 800.0 * _slow_random(n, 0.4, rng), 2.5, "band")
    s *= np.sin(np.pi * t / length) ** 0.8
    return _norm(lowpass(s, 7500.0, 2), 1.0)


# ------------------------------------------------------------------- landing

def parachute_open(seed=0):
    """A cloth canopy snapping open: a crack of noise, a low whump as it
    fills, and a flutter that settles. 1.6 s stereo."""
    n = n_of(1.6)
    rng = rng_for("chute", seed)
    t = t_axis(n)
    out = np.zeros((n, 2))
    for ch in range(2):
        w = white(n, rng)
        crack = bandpass(w, 900.0, 5000.0, order=2) * env_perc(n, 0.002, 0.03)
        whump = lowpass(w, 180.0, 2) * env_points(n, [(0, 0), (0.05, 1), (0.3, 0.3), (1.6, 0)])
        flutter = bandpass(w, 500.0, 3000.0, order=2) * np.maximum(0, np.sin(2 * math.pi * 7.0 * t + ch)) ** 2 * np.exp(-t / 0.6)
        out[:, ch] = crack * 1.2 + whump * 1.6 + flutter * 0.45
    return _norm(out, 1.0)


def canopy_sway(length, seed=0):
    """Gentle cloth and cord creaks while the canopy sways down. Stereo."""
    n = n_of(length)
    rng = rng_for("sway", seed)
    t = t_axis(n)
    out = np.zeros((n, 2))
    sway = 0.5 + 0.5 * np.sin(2 * math.pi * 0.45 * t)
    for ch in range(2):
        w = white(n, rng)
        cloth = bandpass(w, 400.0, 2500.0, order=2) * (sway ** 2) * (0.5 + 0.5 * _slow_random(n, 8.0, rng))
        out[:, ch] = cloth
    return _norm(out, 1.0)


def touchdown(seed=0):
    """Bamboo legs meeting grass: a soft double thud, a tin settle, a bamboo
    knock and a grass crunch. 1.2 s."""
    n = n_of(1.2)
    rng = rng_for("touch", seed)
    w = white(n, rng)
    thud = lowpass(w, 140.0, 2) * (env_perc(n, 0.003, 0.08) + np.concatenate([np.zeros(n_of(0.09)), env_perc(n - n_of(0.09), 0.003, 0.06)]) * 0.7)
    imp = np.zeros(n)
    imp[n_of(0.01)] = 1.0
    imp[n_of(0.1)] = 0.6
    bamboo = (resonator(imp, 540.0, 22.0) + 0.5 * resonator(imp, 1100.0, 30.0)) * 12 * np.exp(-t_axis(n) / 0.15)
    crunch = bandpass(w, 1200.0, 5000.0, order=2) * env_points(n, [(0, 0), (0.01, 1), (0.12, 0.3), (0.3, 0)]) * 0.5
    tin = tin_rattle(0.8, seed=seed + 3, density=18)[: n_of(0.8)] * env_exp(n_of(0.8), 0.25)
    s = thud * 1.5 + bamboo + crunch
    s[: n_of(0.8)] += tin * 0.6
    return _norm(lowpass(s, 6000.0, 1), 1.0)


def paper_rustle(seed=0, length=0.45):
    """A paper map taken out and unfolded: crinkly noise with crisp folds."""
    n = n_of(length)
    rng = rng_for("map", seed)
    w = white(n, rng)
    folds = (rng.random(n) < 0.004).astype(float) * rng.uniform(0.3, 1.0, n)
    folds = lowpass(folds, 5000.0, 1) * 6
    s = (bandpass(w, 900.0, 4500.0, order=2) * (0.5 + 0.5 * _slow_random(n, 30.0, rng)) + folds) * np.sin(np.pi * t_axis(n) / length) ** 0.8
    return _norm(lowpass(s, 6500.0, 2), 1.0)


def brush_stroke(length=0.5, seed=0, pressure=1.0):
    """A brush dragged on paper: a bristle noise band whose level has a
    loaded start and a dry, grainier end."""
    n = n_of(length)
    rng = rng_for("brush", seed)
    t = t_axis(n)
    u = t / length
    w = white(n, rng)
    grain = 0.6 + 0.4 * _slow_random(n, 120.0, rng) * u
    s = bandpass(w, 1500.0, 5500.0, order=2) * grain
    s += lowpass(w, 400.0, 2) * 0.5 * (1 - u)
    env = env_points(n, [(0, 0), (0.03 * length, 1.0), (0.7 * length, 0.75 * pressure), (length, 0)])
    s *= env
    return _norm(lowpass(s, 6500.0, 2), 1.0)


def hanko(seed=0):
    """A seal pressed into paper: a dull soft thump and a short paper contact."""
    n = n_of(0.35)
    rng = rng_for("hanko", seed)
    w = white(n, rng)
    t = t_axis(n)
    thump = sine(110.0 * (1 + 0.5 * np.exp(-t / 0.02)), n) * env_perc(n, 0.004, 0.05)
    contact = lowpass(w, 1200.0, 2) * env_perc(n, 0.002, 0.012)
    s = thump * 1.0 + contact * 0.6
    return _norm(s, 1.0)


def speck_rumble(length, seed=0):
    """The rocket as a far speck: a faint, distant rumble. Mono."""
    n = n_of(length)
    rng = rng_for("speck", seed)
    s = lowpass(brown(n, rng), 160.0, 2) * (0.6 + 0.4 * _slow_random(n, 2.0, rng))
    return _norm(s, 1.0)
