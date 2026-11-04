"""Synthesized instruments for THE LAST PAGE.

Felt piano (additive, inharmonic partials, felt hammer, key and damper noise, sympathetic
resonance), string sections (detuned band-limited saw ensembles with vibrato, tremolo, mutes,
glides), horns, choir (formant filtered), timpani, bass drum, suspended cymbal, harp, celesta.
Every tone is computed here from oscillators and noise.
"""

import numpy as np

from scipy import signal as _sig

from dsp import (SR, TAU, chain, edge_fade, env_note, filt, modal, mtof, onepole_lp, pan_gains, rng_for,
                 saw_blep, smooth_rand, smoothstep, sos_bp, sos_hp, sos_lp, sos_peak, sos_reson, svf,
                 taxis, db2lin)


# =============================================================================== felt piano
def felt_piano_note(m, vel, t_damp, rng, felt=1.0, cap=None):
    """One felt piano note, stereo.

    Inharmonic partials (string stiffness rising with pitch), two-stage decays (prompt sound and
    aftersound), 1 to 3 detuned strings per key, strike-point comb, and a felt strip between hammer
    and string: felt=1 is the soft, dark, close felt piano; felt=0 is an open grand. Louder notes
    are brighter. The damper falls at t_damp seconds (key up with no pedal)."""
    f0 = mtof(m)
    B = 10.0 ** (-3.45 + 0.031 * (m - 60))
    T1 = float(np.interp(m, [21, 36, 48, 60, 72, 84, 96, 108], [22, 16, 12, 8.5, 5.5, 3.4, 1.9, 1.0]))
    d1 = 6.91 / T1
    has_damper = m < 89
    rel = float(np.interp(m, [21, 60, 88], [0.40, 0.24, 0.14]))
    length = (t_damp + rel * 1.4) if has_damper else T1 * 0.8
    length = min(length, T1 * 1.2, 14.0)
    if cap is not None:
        length = min(length, cap)
    length = max(length, 0.08)
    n = int(length * SR)
    t = taxis(n)

    kmax = int(min(90, 12000.0 / f0))
    k = np.arange(1, kmax + 1, dtype=float)
    fk = k * f0 * np.sqrt(1.0 + B * k * k)
    keep = fk < 12500.0
    k, fk = k[keep], fk[keep]
    x0 = 0.12 + 0.006 * rng.standard_normal()
    comb = 0.25 + 0.75 * np.abs(np.sin(np.pi * k * x0))
    fc_grand = 900.0 + 6200.0 * vel ** 1.5
    fc_felt = 300.0 + 1700.0 * vel ** 1.4 + 1.6 * f0
    fc = np.exp(np.log(fc_felt) * felt + np.log(fc_grand) * (1.0 - felt))
    amp = comb / k ** 0.45 / (1.0 + (fk / fc) ** 2) ** (1.0 + 0.4 * felt)
    if f0 < 200.0:
        amp[0] *= (f0 / 200.0) ** 0.9
        if len(amp) > 1:
            amp[1] *= (f0 / 200.0) ** 0.4
    amp /= np.sqrt(np.sum(amp ** 2))
    dk = d1 + (0.85 + 1.3 * felt) * (fk / 1000.0) ** 1.6

    nstr = 3 if m >= 52 else (2 if m >= 40 else 1)
    base_det = {1: [0.0], 2: [-0.45, 0.45], 3: [-0.8, 0.05, 0.75]}[nstr]
    det = np.array(base_det) * rng.uniform(0.5, 1.4) + rng.normal(0.0, 0.1, nstr)
    wl = {1: [0.707], 2: [0.85, 0.53], 3: [0.82, 0.62, 0.42]}[nstr]
    wr = wl[::-1]

    relenv = None
    i_d = n
    if has_damper:
        i_d = int(t_damp * SR)
        if i_d < n:
            relenv = np.ones(n)
            relenv[i_d:] = np.exp(-6.91 * taxis(n - i_d) / rel)
    L = np.zeros(n)
    R = np.zeros(n)
    for i in range(len(k)):
        a = amp[i]
        if a < 3e-4:
            continue
        nk = min(n, int(SR * 6.91 / (0.62 * dk[i])) + 2)
        if relenv is not None:
            nk = min(nk, i_d + int(rel * 1.3 * SR))
        tt = t[:nk]
        e = a * (0.68 * np.exp(-1.9 * dk[i] * tt) + 0.32 * np.exp(-0.62 * dk[i] * tt))
        if relenv is not None:
            e = e * relenv[:nk]
        for j in range(nstr):
            fj = fk[i] * 2.0 ** (det[j] / 1200.0)
            s = np.cos(TAU * ((fj * tt + rng.random()) % 1.0)) * e
            L[:nk] += s * wl[j]
            R[:nk] += s * wr[j]

    # felt hammer: a soft woolly thump rather than a hard click; brighter for a grand
    nh = min(n, int(0.05 * SR))
    th = taxis(nh)
    hn = rng.standard_normal(nh) * np.exp(-th / (0.004 + 0.004 * felt))
    hc = min(max(f0 * 3.0, 350.0), 2500.0 + 1500.0 * (1 - felt))
    ham = filt(sos_bp(hc / 2.2, hc * 1.6), hn) * (0.045 + 0.03 * felt) * vel ** 1.1
    ham += np.sin(TAU * 70.0 * th) * np.exp(-th / 0.012) * (0.035 + 0.03 * felt) * vel
    L[:nh] += ham
    R[:nh] += ham * 0.92
    out = np.vstack([L, R])
    edge_fade(out, 0.0012, 0.004)
    g = vel ** 1.45 * float(np.interp(m, [21, 36, 60, 84, 108], [0.75, 0.85, 1.0, 0.9, 0.7]))
    return out * g


def key_noise(vel, rng, kind="down"):
    """Close-miked mechanism: key bottoming (woody tock) or the damper felt landing."""
    n = int(0.06 * SR)
    t = taxis(n)
    if kind == "down":
        y = modal([310.0 * rng.uniform(0.9, 1.1), 760.0, 1400.0], [1.0, 0.45, 0.2],
                  [0.03, 0.018, 0.01], 0.06, rng, 0.0008)
        y += filt(sos_bp(200.0, 1800.0), rng.standard_normal(n)) * np.exp(-t / 0.004) * 0.6
        return edge_fade(y, 0.0005, 0.01) * 0.012 * (0.3 + vel)
    y = filt(sos_bp(250.0, 1400.0), rng.standard_normal(n)) * np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.018)
    return edge_fade(y, 0.001, 0.01) * 0.008


def pedal_noise(rng, down=True):
    """Sustain pedal: the dampers lifting off all strings (a soft swell of string air) or landing."""
    n = int(0.35 * SR)
    t = taxis(n)
    thump = modal([95.0, 210.0, 480.0], [1.0, 0.5, 0.2], [0.06, 0.04, 0.02], 0.35, rng, 0.002)
    air = filt(sos_bp(300.0, 3000.0), rng.standard_normal((2, n)))
    if down:
        air *= np.clip(t / 0.06, 0, 1) * np.exp(-t / 0.12) * 0.25
    else:
        air *= np.clip(t / 0.01, 0, 1) * np.exp(-t / 0.05) * 0.18
    y = air + thump[None, :] * 0.6
    return edge_fade(y, 0.002, 0.02) * 0.006


def sympathetic_ir(rng, length=6.0):
    """Stereo bank of free string resonances (A0 to C8, first partials, piano-like decay times).
    Convolving the piano with it while the pedal is down gives the halo of undamped strings
    ringing along with what is played (sympathetic resonance)."""
    n = int(length * SR)
    t = taxis(n)
    ir = np.zeros((2, n))
    for m in range(21, 109):
        f0 = mtof(m)
        B = 10.0 ** (-3.45 + 0.031 * (m - 60))
        T1 = float(np.interp(m, [21, 36, 48, 60, 72, 84, 96, 108], [16, 12, 9, 6.5, 4.0, 2.5, 1.4, 0.8]))
        pan = float(np.clip((m - 64) / 40.0, -0.6, 0.6))
        gl, gr = pan_gains(pan)
        for kk in range(1, 9):
            fk = kk * f0 * np.sqrt(1.0 + B * kk * kk)
            if fk > 5000.0:
                break
            t60 = T1 / (1.0 + 0.35 * (kk - 1))
            nk = min(n, int(t60 * 1.1 * SR))
            a = rng.uniform(0.6, 1.0) / kk ** 0.8
            ph = rng.random()
            s = a * np.cos(TAU * ((fk * t[:nk] + ph) % 1.0)) * np.exp(-6.91 * t[:nk] / t60)
            ir[0, :nk] += s * gl
            ir[1, :nk] += s * gr
    ir *= smoothstep(t / 0.01)[None, :]
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return ir


# ===================================================================================== choir
VOWELS = {
    ("a", "s"): ([800, 1150, 2900, 3900, 4950], [0, -6, -32, -20, -50], [80, 90, 120, 130, 140]),
    ("u", "s"): ([350, 750, 2700, 3800, 4950], [0, -14, -30, -36, -55], [55, 65, 150, 170, 190]),
    ("a", "a"): ([800, 1150, 2800, 3500, 4950], [0, -4, -20, -36, -60], [80, 90, 120, 130, 140]),
    ("u", "a"): ([350, 600, 2700, 2900, 3300], [0, -20, -32, -28, -36], [45, 60, 100, 120, 120]),
    ("a", "t"): ([650, 1080, 2650, 2900, 3250], [0, -6, -7, -8, -22], [80, 90, 120, 130, 140]),
    ("u", "t"): ([290, 750, 2300, 3000, 3350], [0, -20, -17, -14, -26], [40, 60, 100, 120, 120]),
}


# ================================================================================ percussion
def timpani_hit(m, vel, rng):
    f = mtof(m)
    y = modal([0.86 * f, f, 1.504 * f, 1.742 * f, 2.0 * f, 2.44 * f, 2.9 * f],
              [0.5, 1.0, 0.55, 0.25, 0.32, 0.18, 0.08],
              [0.25, 2.4, 1.6, 0.7, 1.0, 0.6, 0.4], 2.8, rng, 0.002)
    nb = int(0.03 * SR)
    y[:nb] += filt(sos_lp(900.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.006) * 0.6
    y = filt(sos_lp(500.0 + 2400.0 * vel), y)
    return edge_fade(y, 0.001, 0.02) * vel ** 1.4


def timp_roll(m, t_len, v0, v1, rng, shape=1.5):
    """Single-drum roll, strokes 50 to 60 ms apart, dynamic v0 to v1. Ends dead at t_len."""
    n = int(t_len * SR)
    acc = np.zeros(n + int(2.9 * SR))
    t = 0.0
    while t < t_len - 0.01:
        x = t / t_len
        h = timpani_hit(m, (v0 + (v1 - v0) * x ** shape) * rng.uniform(0.85, 1.0), rng)
        i0 = int(t * SR)
        acc[i0:i0 + len(h)] += h[: len(acc) - i0]
        t += rng.uniform(0.048, 0.060)
    return acc


def bass_drum(vel, rng, f=47.0):
    n = int(2.6 * SR)
    t = taxis(n)
    fr = f * (1.0 + 0.3 * np.exp(-t / 0.04))
    y = np.sin(TAU * np.cumsum(fr) / SR) * np.exp(-6.91 * t / 2.2)
    y += 0.4 * np.sin(TAU * np.cumsum(fr * 1.58) / SR) * np.exp(-6.91 * t / 0.9)
    nb = int(0.05 * SR)
    y[:nb] += filt(sos_lp(350.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.01) * 0.8
    return edge_fade(y, 0.002, 0.05) * vel ** 1.3


def cymbal(kind, length, vel, rng):
    """Suspended cymbal: 'crash' (soft mallet hit) or 'swell' (mallet roll crescendo).
    Kept below 9 kHz so it never turns harsh."""
    n = int(length * SR)
    t = taxis(n)
    nz = rng.standard_normal((2, n))
    bright = filt(chain(sos_bp(2400.0, 8500.0), sos_lp(8000.0, 2)), nz)
    body = filt(sos_bp(450.0, 2400.0), nz[::-1]) * 0.55
    if kind == "crash":
        y = bright * np.exp(-6.91 * t / 1.6) * 0.75 + body * np.exp(-6.91 * t / 3.8)
        a = int(0.004 * SR)
        y[:, :a] *= np.linspace(0, 1, a)
    else:
        x = t / length
        e = x ** 2.6 * (1.0 + 0.25 * np.sin(TAU * 17.0 * t))
        y = (bright * 0.75 + body * 0.7) * e
    return edge_fade(y, 0.002, 0.012) * vel


# ===================================================================================== harp
def harp_note(m, vel, rng):
    """Plucked harp string: harmonic partials with slight stretch, pluck-point comb, faster decay
    for higher partials, and a short finger noise."""
    f0 = mtof(m)
    T60 = float(np.interp(m, [36, 48, 60, 72, 84, 96, 103], [8.0, 6.0, 4.5, 3.2, 2.0, 1.2, 0.8]))
    K = int(min(24, 9000.0 / f0))
    freqs, amps, t60s = [], [], []
    x0 = 0.42 + 0.03 * rng.standard_normal()
    for k in range(1, K + 1):
        freqs.append(k * f0 * (1.0 + 0.00015 * k * k))
        amps.append(abs(np.sin(np.pi * k * x0)) / k ** (1.1 + 0.4 * (1 - vel)))
        t60s.append(T60 / (1.0 + 0.45 * (k - 1)))
    y = modal(freqs, amps, t60s, T60 * 1.05, rng, 0.0015)
    nb = int(0.006 * SR)
    y[:nb] += filt(sos_bp(800.0, 4000.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0012) * 0.05
    return edge_fade(y, 0.0015, 0.02) * vel ** 1.3 * 0.6


def celesta_tone(m, vel, rng):
    f = mtof(m)
    T60 = float(np.interp(m, [72, 84, 96, 108], [2.4, 1.8, 1.1, 0.6]))
    y = modal([f, 2.0 * f * 1.0006, 3.01 * f, 4.17 * f],
              [1.0, 0.10, 0.05, 0.035], [T60, T60 * 0.35, T60 * 0.2, T60 * 0.08], T60 * 1.15, rng, 0.001)
    nb = int(0.004 * SR)
    y[:nb] += filt(sos_bp(2500.0, 6500.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0008) * 0.05
    return edge_fade(y, 0.0008, 0.01) * vel ** 1.3


# ===================================================================== realistic ensembles (v2)
def saw_phase(ph, dt):
    """Band-limited sawtooth from a phase array (0..1) and its per-sample increment (polyBLEP)."""
    y = 2.0 * ph - 1.0
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = ph > 1.0 - dt
    x = (ph[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


BODY_SCALE = {"vln1": 1.0, "vln2": 1.0, "vla": 0.82, "vc": 0.42, "cb": 0.27}
_BODY = {}


def body_ir(sec, desk):
    """Synthesized body response of one instrument of a section (two desks per section, each its own
    instrument). Violin family: the air mode A0, the main wood modes, dense higher modes with a
    bridge hill, scaled down in frequency for viola, cello and bass. Modal synthesis only."""
    key = (sec, desk)
    if key in _BODY:
        return _BODY[key]
    s = BODY_SCALE[sec]
    rng = rng_for(f"body:{sec}:{desk}")
    n = int(0.09 * SR)
    t = taxis(n)
    ir = np.zeros(n)
    ir[0] = 1.0
    for f, a, tau in ((275.0, 1.0, 0.014), (405.0, 0.45, 0.010), (460.0, 0.9, 0.012), (540.0, 0.85, 0.011),
                      (710.0, 0.45, 0.008), (1050.0, 0.3, 0.006)):
        fm = f * s * (1.0 + 0.04 * rng.standard_normal())
        ir += a * 0.11 * np.sin(TAU * fm * t + rng.uniform(0, TAU)) * np.exp(-t / (tau / s ** 0.5))
    hill = 2500.0 * s ** 0.55
    for _ in range(80):
        f = float(np.exp(rng.uniform(np.log(650.0 * s), np.log(min(7000.0, 6500.0 * s ** 0.4)))))
        g = 1.0 + 1.6 * np.exp(-0.5 * (np.log2(f / hill) / 0.45) ** 2)
        a = rng.normal() * g * (f / 1000.0) ** -0.35 * 0.05
        ir += a * np.sin(TAU * f * t + rng.uniform(0, TAU)) * np.exp(-t / rng.uniform(0.0015, 0.007))
    ir = filt(sos_lp(6500.0, 2), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    _BODY[key] = ir
    return ir


STR_V2 = {  # voices per section, vibrato depth (cents), stereo width, detune spread (cents)
    "vln1": dict(voices=10, vib=16.0, width=0.5, det=5.0, pan=-0.50),
    "vln2": dict(voices=9, vib=15.0, width=0.5, det=5.0, pan=-0.20),
    "vla": dict(voices=7, vib=14.0, width=0.45, det=4.5, pan=0.15),
    "vc": dict(voices=6, vib=13.0, width=0.4, det=4.0, pan=0.38),
    "cb": dict(voices=5, vib=8.0, width=0.3, det=4.0, pan=0.55),
}
STR_GAIN = {k: float(db2lin(v)) for k, v in  # loudness matched to the v1 sections so the mix balance holds
            {"vln1": -11.9, "vln2": -11.7, "vla": -14.4, "vc": -16.2, "cb": -19.7}.items()}


def bowed_note(sec, m, dur, vel, vel_end, kind, att, rel, rng, glide=None, mute=False):
    """String section note, modelled player by player.

    Each player: a bow-position pulse (a sawtooth minus its copy shifted by the bow position beta,
    which carves the harmonic notches a real bow makes; beta drifts as the bow moves), intonation
    offset, slow pitch drift, attack pitch settling, vibrato with its own rate wobble, depth and
    delayed onset, bow pressure flutter, rosin noise pulsed at each slip of the string, a bow-bite
    scratch on the attack (and on every stroke in tremolo). The two desks of a section each go
    through their own synthesized instrument body; brightness opens with the attack and closes on
    the release. kind: 'legato', 'marc', 'spicc', 'trem'. glide: (t0, t1, cents). mute: con sordino."""
    P = STR_V2[sec]
    f0 = mtof(m)
    short = kind == "spicc"
    if short:
        att, rel = 0.008, 0.08
        voices = max(3, P["voices"] // 2)
    else:
        voices = P["voices"]
    n = int((dur + rel + 0.1) * SR)
    t = taxis(n)
    gl_c = 0.0
    if glide is not None:
        g0, g1, gc = glide
        gl_c = gc * smoothstep((t - g0) / max(g1 - g0, 1e-3))
    dyn = vel if vel_end is None else np.interp(t, [0.0, max(dur, 1e-3)], [vel, vel_end])
    dyn = np.asarray(dyn) * np.ones(n)
    vmax = float(dyn.max())
    desks = [np.zeros((2, n)), np.zeros((2, n))]
    for v in range(voices):
        d0 = int(rng.uniform(0.0, 0.006 if short or kind == "marc" else 0.035) * SR)
        cents = (rng.normal(0.0, P["det"]) + 4.0 * smooth_rand(n, 0.35, rng) + 1.2 * smooth_rand(n, 3.5, rng)
                 + gl_c + rng.normal(0.0, 9.0) * np.exp(-t / 0.06))
        if not short:
            vr = rng.uniform(4.8, 6.2) * (1.0 + 0.08 * smooth_rand(n, 0.5, rng))
            vph = np.cumsum(vr) / SR + rng.random()
            onset = np.clip((t - rng.uniform(0.12, 0.4)) / rng.uniform(0.3, 0.6), 0.0, 1.0)
            vd = P["vib"] * rng.uniform(0.5, 1.3) * (0.8 + 0.25 * smooth_rand(n, 0.6, rng))
            if kind == "trem":
                vd *= 0.4
            cents = cents + vd * np.sin(TAU * vph) * onset
        f = f0 * 2.0 ** (cents / 1200.0)
        dt = f / SR
        ph = (rng.random() + np.cumsum(dt)) % 1.0
        beta = np.clip(rng.uniform(0.07, 0.14) * (1.0 + 0.12 * smooth_rand(n, 0.4, rng)), 0.05, 0.2)
        ph2 = (ph + beta) % 1.0
        pulse = (saw_phase(ph, dt) - saw_phase(ph2, dt + np.gradient(beta))) * (0.5 / np.sin(np.pi * beta))
        e = env_note(n, att * rng.uniform(0.8, 1.3), dur, rel * rng.uniform(0.9, 1.2))
        e *= 1.0 + 0.06 * smooth_rand(n, 4.5, rng)
        if short:
            e *= np.exp(-t / 0.11)
        if kind == "marc":
            e *= 1.0 + 0.7 * np.exp(-t / 0.08)
        nz = rng.standard_normal(n)
        scratch_env = np.exp(-t / 0.03) * (0.28 if kind in ("marc", "spicc") else 0.08 * min(1.0, 0.06 / max(att, 0.01)))
        if kind == "trem":
            rate = rng.uniform(11.0, 15.0) * (1.0 + 0.05 * smooth_rand(n, 2.0, rng))
            sph = np.cumsum(rate) / SR + rng.random()
            stroke = (0.5 + 0.5 * np.cos(TAU * sph)) ** 1.2
            e *= 0.3 + 0.7 * stroke
            fr = sph % 1.0
            scratch_env = scratch_env + 0.05 * np.exp(-np.minimum(fr, 1.0 - fr) / 0.05) * stroke
        slip = np.exp(-0.5 * ((((ph + 0.5) % 1.0) - 0.5) / 0.06) ** 2)
        rosin = filt(sos_hp(1800.0), nz) * slip * (0.04 + (0.02 if kind == "trem" else 0.0) + 0.03 * vmax)
        scratch = filt(sos_bp(900.0, 5000.0), rng.standard_normal(n)) * scratch_env
        sig = (pulse + rosin) * e + scratch * env_note(n, 0.002, dur, rel)
        if d0:
            sig = np.concatenate([np.zeros(d0), sig[:-d0]])
        pan = rng.uniform(-1.0, 1.0) * P["width"]
        gl, gr = pan_gains(pan)
        dk = 0 if pan < 0 else 1
        desks[dk][0] += sig * gl
        desks[dk][1] += sig * gr
    fc_t = float(np.clip(1000.0 + f0 * (2.4 + 7.0 * vmax), 900.0, 7000.0))
    if mute:
        fc_t = min(fc_t, 2200.0)
    open_ = 0.35 + 0.65 * smoothstep(t / max(1.2 * att, 0.015))
    close = np.where(t > dur, np.exp(-(t - dur) / max(rel, 0.05)), 1.0)
    fc = fc_t * open_ * (0.5 + 0.5 * close)
    out = np.zeros((2, n))
    for dk in range(2):
        y = svf(desks[dk], fc, 0.707, "lp")
        ir = body_ir(sec, dk)
        out += _sig.oaconvolve(y, ir[None, :], axes=1)[:, :n]
    if mute:
        out = filt(sos_peak(1300.0, 3.0, 1.5), out)
    out = filt(sos_lp(7500.0, 2), out)
    return out * (dyn ** 1.4) / np.sqrt(voices) * STR_GAIN[sec] * (0.7 if mute else 1.0)


HORN_GAIN = float(db2lin(-1.5))
CHOIR_GAIN = float(db2lin(1.3))


def horn_note(m, dur, vel, vel_end, att, rel, rng, voices=3, dark=0.0):
    """French horn section, player by player: a lip scoop up into the pitch, brightness that follows
    each player's level (brass gets brighter as it gets louder, with a blip on a tongued attack),
    almost no vibrato, slow drift, tongue and breath noise, the mellow formants and bell of a horn,
    a brassy edge only at fortissimo. dark > 0: stopped or distant colour."""
    f0 = mtof(m)
    n = int((dur + rel + 0.1) * SR)
    t = taxis(n)
    dyn = vel if vel_end is None else np.interp(t, [0.0, max(dur, 1e-3)], [vel, vel_end])
    dyn = np.asarray(dyn) * np.ones(n)
    out = np.zeros((2, n))
    for v in range(voices):
        d0 = int(rng.uniform(0.0, 0.025) * SR)
        cents = (rng.normal(0.0, 4.0) + 3.0 * smooth_rand(n, 0.4, rng) + 1.2 * smooth_rand(n, 4.0, rng)
                 - rng.uniform(25.0, 50.0) * np.exp(-t / rng.uniform(0.025, 0.05))
                 + 4.0 * np.sin(TAU * rng.uniform(4.5, 5.5) * t) * np.clip((t - 0.6) / 0.8, 0.0, 1.0))
        f = f0 * 2.0 ** (cents / 1200.0)
        dt = f / SR
        ph = (rng.random() + np.cumsum(dt)) % 1.0
        s = saw_phase(ph, dt)
        e = env_note(n, att * rng.uniform(0.85, 1.2), dur, rel)
        lev = e * dyn
        blip = 1.0 + 0.7 * np.exp(-t / 0.04) * float(np.clip((dyn[0] - 0.4) / 0.5, 0.0, 1.0)) * (att < 0.1)
        fc = np.clip(f0 * (1.2 + 6.0 * lev ** 1.8) * blip * (1.0 - 0.45 * dark), 250.0, 3500.0)
        s = svf(s, fc, 0.8, "lp") * e
        nz = rng.standard_normal(n)
        tongue = filt(sos_bp(1000.0, 4000.0), nz) * np.exp(-t / 0.018) * 0.12 * (att < 0.15)
        breath = filt(sos_bp(1200.0, 3500.0), nz[::-1]) * e * 0.012
        sig = s + tongue + breath
        if d0:
            sig = np.concatenate([np.zeros(d0), sig[:-d0]])
        gl, gr = pan_gains(rng.uniform(-0.4, 0.4))
        out[0] += sig * gl
        out[1] += sig * gr
    out /= np.sqrt(voices)
    edge = np.clip((dyn - 0.7) / 0.3, 0.0, 1.0)
    if edge.max() > 0:           # brassy edge at fortissimo, before the bell so it stays round
        drive = 1.0 + 1.5 * edge
        out = out * (1 - edge) + np.tanh(out * drive * 1.2) / 1.2 * edge
    out = filt(chain(sos_peak(450.0, 4.0, 1.0), sos_peak(1100.0, 1.5, 1.2), sos_lp(3200.0 * (1 - 0.4 * dark), 2),
                     sos_lp(6000.0, 2)), out)
    return out * dyn ** 1.4 * HORN_GAIN


def choir_note(part, m, dur, vel, vel_end, va, vb, morph_t, att, rel, rng, voices=5):
    """Choir section, singer by singer: each voice has its own vocal tract (formants scaled a few
    percent up or down and drifting), vibrato with its own rate and onset, jitter and shimmer,
    breath noise shaped by the same formants, a soft onset and a slight scoop; vowels morph
    from one to the other."""
    f0 = mtof(m)
    n = int((dur + rel + 0.1) * SR)
    t = taxis(n)
    dyn = vel if vel_end is None else np.interp(t, [0.0, max(dur, 1e-3)], [vel, vel_end])
    out = np.zeros((2, n))
    w = smoothstep((t - morph_t) / 0.9) if vb != va else None
    for v in range(voices):
        d0 = int(rng.uniform(0.0, 0.05) * SR)
        cents = (rng.normal(0.0, 7.0)
                 + rng.uniform(14.0, 26.0) * np.sin(TAU * np.cumsum(rng.uniform(4.7, 5.7) * (1 + 0.06 * smooth_rand(n, 0.7, rng))) / SR
                                                   + rng.uniform(0, TAU)) * np.clip((t - rng.uniform(0.15, 0.4)) / 0.7, 0.0, 1.0)
                 + 4.0 * smooth_rand(n, 0.5, rng) + 2.0 * smooth_rand(n, 25.0, rng)
                 - 20.0 * np.exp(-t / 0.08))
        f = f0 * 2.0 ** (cents / 1200.0)
        dt = f / SR
        ph = (rng.random() + np.cumsum(dt)) % 1.0
        src = saw_phase(ph, dt)
        src = onepole_lp(onepole_lp(src, 700.0), 2500.0)
        e = env_note(n, att * rng.uniform(0.8, 1.25), dur, rel) * (1.0 + 0.06 * smooth_rand(n, 18.0, rng))
        asp = filt(sos_hp(800.0), rng.standard_normal(n)) * 0.03
        x = (src + asp) * e
        k = rng.uniform(0.94, 1.06) * (1.0 + 0.01 * smooth_rand(n, 0.3, rng).mean())
        y = _formants(x, va, part, k)
        if w is not None:
            y = y * (1.0 - w) + _formants(x, vb, part, k) * w
        if d0:
            y = np.concatenate([np.zeros(d0), y[:-d0]])
        gl, gr = pan_gains(rng.uniform(-0.7, 0.7))
        out[0] += y * gl
        out[1] += y * gr
    return out * (np.asarray(dyn) ** 1.3) * 2.2 / np.sqrt(voices) * CHOIR_GAIN


def _formants(x, vowel, part, k):
    fs, ls, bws = VOWELS[(vowel, part)]
    y = np.zeros_like(x)
    for fr, lv, bw in zip(fs, ls, bws):
        y += filt(sos_reson(fr * k, fr * k / (bw * 1.4)), x) * db2lin(lv)
    return y
