"""Core DSP for the TINY ROCKET score: envelopes, filters, noise, panning,
a feedback delay network reverb, a lookahead limiter and 24 bit WAV output.

Everything here is written from scratch with numpy and scipy. The few sample
by sample loops (reverb, time varying filters, limiter) are compiled with
numba when it is available and fall back to plain Python otherwise.
"""

import math
import struct

import numpy as np
from scipy import signal

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

SR = 48000


# ---------------------------------------------------------------- time helpers

def n_of(seconds):
    return int(round(seconds * SR))


def t_axis(n):
    return np.arange(n, dtype=np.float64) / SR


def rng_for(*keys):
    """Deterministic generator seeded from a tuple of keys (cue, element)."""
    seed = 0
    for k in keys:
        for ch in str(k):
            seed = (seed * 131 + ord(ch)) % (2 ** 31 - 1)
    return np.random.default_rng(seed)


# ------------------------------------------------------------------- envelopes

def env_points(n, points):
    """Piecewise linear envelope. points: list of (time_s, value)."""
    ts = np.array([p[0] for p in points], dtype=np.float64)
    vs = np.array([p[1] for p in points], dtype=np.float64)
    t = t_axis(n)
    return np.interp(t, ts, vs)


def env_adsr(n, a, d, s, r, hold=None):
    """Attack, decay, sustain level, release. hold is the gate length in s
    (defaults to the whole buffer minus the release)."""
    total = n / SR
    if hold is None:
        hold = max(total - r, a + d)
    pts = [(0.0, 0.0), (a, 1.0), (a + d, s), (hold, s), (hold + r, 0.0)]
    # keep times monotonic
    fixed = []
    last = -1.0
    for t, v in pts:
        if t <= last:
            t = last + 1e-4
        fixed.append((t, v))
        last = t
    return env_points(n, fixed)


def env_exp(n, tau, start=1.0):
    """Exponential decay with time constant tau seconds."""
    return start * np.exp(-t_axis(n) / tau)


def env_perc(n, attack, tau):
    """Percussive: fast linear attack then exponential decay."""
    e = env_exp(n, tau)
    na = max(1, n_of(attack))
    e[:na] *= np.linspace(0.0, 1.0, na)
    return e


def fade(x, fin=0.0, fout=0.0):
    x = np.array(x, dtype=np.float64, copy=True)
    n = x.shape[0]
    if fin > 0:
        k = min(n, n_of(fin))
        ramp = np.linspace(0.0, 1.0, k)
        if x.ndim == 2:
            x[:k] *= ramp[:, None]
        else:
            x[:k] *= ramp
    if fout > 0:
        k = min(n, n_of(fout))
        ramp = np.linspace(1.0, 0.0, k)
        if x.ndim == 2:
            x[n - k:] *= ramp[:, None]
        else:
            x[n - k:] *= ramp
    return x


# ----------------------------------------------------------------------- noise

def white(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    """Pink noise by summing octave spaced one pole lowpassed white noises."""
    out = np.zeros(n)
    w = white(n, rng)
    for k in range(7):
        fc = 40.0 * (2 ** k)
        out += lowpass(w, fc, order=1) * (0.7 ** k)
    out /= np.max(np.abs(out)) + 1e-9
    return out


def brown(n, rng):
    x = np.cumsum(white(n, rng))
    x = highpass(x, 20.0, order=1)
    x /= np.max(np.abs(x)) + 1e-9
    return x


# --------------------------------------------------------------------- filters

def _sos(kind, fc, order, fs=SR):
    if isinstance(fc, (tuple, list)):
        lo = max(10.0, min(fc[0], fs * 0.49))
        hi = max(lo * 1.01, min(fc[1], fs * 0.49))
        return signal.butter(order, [lo, hi], btype=kind, fs=fs, output="sos")
    fc = max(10.0, min(fc, fs * 0.49))
    return signal.butter(order, fc, btype=kind, fs=fs, output="sos")


def lowpass(x, fc, order=2):
    return signal.sosfilt(_sos("low", fc, order), x, axis=0)


def highpass(x, fc, order=2):
    return signal.sosfilt(_sos("high", fc, order), x, axis=0)


def bandpass(x, lo, hi, order=2):
    return signal.sosfilt(_sos("band", (lo, hi), order), x, axis=0)


def peaking(x, fc, q, gain_db):
    """Peaking EQ (RBJ cookbook) as a biquad."""
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * q)
    b0 = 1 + alpha * a
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / a
    b = np.array([b0, b1, b2]) / a0
    aa = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, aa, x, axis=0)


def highshelf(x, fc, gain_db, slope=0.7):
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / 2 * math.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    sq = 2 * math.sqrt(a) * alpha
    b0 = a * ((a + 1) + (a - 1) * cw + sq)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - sq)
    a0 = (a + 1) - (a - 1) * cw + sq
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - sq
    b = np.array([b0, b1, b2]) / a0
    aa = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, aa, x, axis=0)


def resonator(x, fc, q, gain=1.0):
    """Two pole resonant bandpass (constant peak gain)."""
    w0 = 2 * math.pi * max(20.0, min(fc, SR * 0.45)) / SR
    alpha = math.sin(w0) / (2 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * math.cos(w0), 1 - alpha])
    return gain * signal.lfilter(b / a[0], a / a[0], x)


@njit(cache=True)
def _svf_tv(x, fc, res, mode):
    """Chamberlin state variable filter with per sample cutoff.
    mode 0 lowpass, 1 bandpass, 2 highpass."""
    n = x.shape[0]
    out = np.zeros(n)
    low = 0.0
    band = 0.0
    for i in range(n):
        f = 2.0 * math.sin(math.pi * min(fc[i], 20000.0) / 48000.0)
        if f > 1.2:
            f = 1.2
        q = 1.0 / max(res[i], 0.5)
        # two passes per sample for stability at high cutoff
        for _ in range(2):
            high = x[i] - low - q * band
            band += 0.5 * f * high
            low += 0.5 * f * band
        if mode == 0:
            out[i] = low
        elif mode == 1:
            out[i] = band
        else:
            out[i] = x[i] - low - q * band
    return out


def svf(x, fc, res=0.7, mode="low"):
    """Time varying state variable filter. fc may be a scalar or an array."""
    n = x.shape[0]
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), (n,)).copy()
    res = np.broadcast_to(np.asarray(res, dtype=np.float64), (n,)).copy()
    m = {"low": 0, "band": 1, "high": 2}[mode]
    return _svf_tv(np.ascontiguousarray(x, dtype=np.float64), fc, res, m)


# --------------------------------------------------------------- oscillators

def phase_of(freq, n=None):
    """Integrated phase in radians for a scalar or per sample frequency."""
    if np.isscalar(freq):
        return 2 * math.pi * freq * t_axis(n)
    return 2 * math.pi * np.cumsum(np.asarray(freq, dtype=np.float64)) / SR


def sine(freq, n=None, phase0=0.0):
    return np.sin(phase_of(freq, n) + phase0)


def additive(freq, n, partials, inharmonic=0.0, max_hz=9000.0, phases=None):
    """Sum of sines. partials: list of (ratio, amplitude, decay_tau or None).
    A decay tau makes that partial die with its own time constant."""
    t = t_axis(n)
    out = np.zeros(n)
    base = freq if np.isscalar(freq) else np.asarray(freq)
    for idx, (ratio, amp, tau) in enumerate(partials):
        k = ratio * math.sqrt(1 + inharmonic * ratio * ratio)
        f0 = (base if np.isscalar(base) else base.max())
        if f0 * k >= max_hz:
            continue
        ph = phase_of(base * k, n)
        if phases is not None:
            ph = ph + phases[idx % len(phases)]
        comp = np.sin(ph) * amp
        if tau is not None:
            comp *= np.exp(-t / tau)
        out += comp
    return out


def saw_bl(freq, n, max_hz=7000.0, rolloff=1.0):
    """Band limited sawtooth by additive synthesis with 1/k^rolloff."""
    f0 = freq if np.isscalar(freq) else float(np.max(freq))
    kmax = max(1, int(max_hz / f0))
    partials = [(k, ((-1) ** (k + 1)) / (k ** rolloff), None) for k in range(1, kmax + 1)]
    return additive(freq, n, partials, max_hz=max_hz) * (2 / math.pi)


def square_bl(freq, n, max_hz=7000.0):
    f0 = freq if np.isscalar(freq) else float(np.max(freq))
    kmax = max(1, int(max_hz / f0))
    partials = [(k, 1.0 / k, None) for k in range(1, kmax + 1, 2)]
    return additive(freq, n, partials, max_hz=max_hz) * (4 / math.pi)


def midi(note):
    return 440.0 * 2 ** ((note - 69) / 12.0)


NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nn(name):
    """Note name like 'F#4' to midi number."""
    base = NOTE_NAMES[name[0]]
    rest = name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + base + acc


def hz(name):
    return midi(nn(name))


# ----------------------------------------------------------------- stereo mix

def pan(x, p):
    """Constant power pan: p in [-1, 1]. Returns (n, 2)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x
    a = (p + 1) * 0.25 * math.pi
    return np.stack([x * math.cos(a), x * math.sin(a)], axis=1)


def pan_moving(x, p_env):
    """Pan with a per sample position array."""
    a = (np.asarray(p_env) + 1) * 0.25 * math.pi
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


class Bus:
    """A stereo buffer that sounds are placed on at a time in seconds."""

    def __init__(self, seconds):
        self.n = n_of(seconds)
        self.buf = np.zeros((self.n, 2), dtype=np.float64)

    def add(self, x, at, gain=1.0, p=0.0):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = pan(x, p)
        start = n_of(at)
        if start >= self.n:
            return
        if start < 0:
            x = x[-start:]
            start = 0
        end = min(self.n, start + x.shape[0])
        self.buf[start:end] += x[: end - start] * gain

    def mix_from(self, other, gain=1.0):
        self.buf += other.buf * gain


def db(x):
    return 10 ** (x / 20.0)


# ------------------------------------------------------------------- reverb

@njit(cache=True)
def _fdn_run(x, lengths, gains, damp, out):
    """8 line feedback delay network with a Householder feedback matrix and a
    one pole damping lowpass in every line. x is mono input, out is (n, 2)."""
    n = x.shape[0]
    nl = lengths.shape[0]
    maxlen = 0
    for i in range(nl):
        if lengths[i] > maxlen:
            maxlen = lengths[i]
    bufs = np.zeros((nl, maxlen))
    heads = np.zeros(nl, dtype=np.int64)
    lp = np.zeros(nl)
    reads = np.zeros(nl)
    for i in range(n):
        s = 0.0
        for k in range(nl):
            idx = heads[k]
            r = bufs[k, idx]
            lp[k] = lp[k] + damp[k] * (r - lp[k])
            reads[k] = lp[k] * gains[k]
            s += reads[k]
        # Householder: y = x - (2/N) * sum(x)
        s2 = s * (2.0 / nl)
        xin = x[i]
        left = 0.0
        right = 0.0
        for k in range(nl):
            v = reads[k] - s2 + xin
            bufs[k, heads[k]] = v
            heads[k] += 1
            if heads[k] >= lengths[k]:
                heads[k] = 0
            if k % 2 == 0:
                left += reads[k]
            else:
                right += reads[k]
        out[i, 0] = left
        out[i, 1] = right


class Reverb:
    """A space built in code: early reflections (a handful of taps) feeding an
    8 line FDN with per line damping, so the tail darkens as it decays."""

    def __init__(self, rt60=2.5, damping=0.35, predelay=0.012, size=1.0,
                 early_gain=0.4, seed=7):
        rng = np.random.default_rng(seed)
        base = np.array([1123, 1327, 1559, 1823, 2131, 2399, 2777, 3203], dtype=np.float64)
        lengths = (base * size).astype(np.int64)
        # per line gain so that the loop decays by 60 dB in rt60 seconds
        gains = np.array([10 ** (-3.0 * (L / SR) / rt60) for L in lengths])
        # damping coefficient: higher means brighter tail
        damp = np.full(8, 1.0 - damping) * (0.85 + 0.3 * rng.random(8))
        damp = np.clip(damp, 0.05, 0.98)
        self.lengths = lengths
        self.gains = gains
        self.damp = damp
        self.predelay = n_of(predelay)
        # early reflections: 10 taps over 8 to 80 ms with decaying gains
        taps = np.sort(rng.uniform(0.008, 0.08, 10))
        self.taps = [(n_of(t), early_gain * (0.9 ** i) * (1 if i % 2 == 0 else -1) * rng.uniform(0.6, 1.0),
                      rng.uniform(-0.8, 0.8)) for i, t in enumerate(taps)]

    def process(self, x):
        """x: (n, 2) or (n,). Returns the wet signal (n, 2)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            mono = x.mean(axis=1)
        else:
            mono = x
        n = mono.shape[0]
        wet = np.zeros((n, 2))
        for d, g, p in self.taps:
            if d < n:
                a = (p + 1) * 0.25 * math.pi
                wet[d:, 0] += mono[: n - d] * g * math.cos(a)
                wet[d:, 1] += mono[: n - d] * g * math.sin(a)
        fed = np.zeros(n)
        if self.predelay < n:
            fed[self.predelay:] = mono[: n - self.predelay]
        fed = highpass(fed, 90.0, order=1)
        tail = np.zeros((n, 2))
        _fdn_run(np.ascontiguousarray(fed), self.lengths, self.gains, self.damp, tail)
        tail *= 0.35
        return wet + tail


# ------------------------------------------------------------------- limiter

@njit(cache=True)
def _limiter_gain(peak, threshold, attack_n, release_coef):
    n = peak.shape[0]
    gain = np.ones(n)
    g = 1.0
    for i in range(n):
        # lookahead: the loudest peak in the next attack_n samples
        j_end = i + attack_n
        if j_end > n:
            j_end = n
        m = 0.0
        for j in range(i, j_end):
            if peak[j] > m:
                m = peak[j]
        target = 1.0
        if m > threshold:
            target = threshold / m
        if target < g:
            g = target
        else:
            g = g + (target - g) * release_coef
        gain[i] = g
    return gain


def limiter(x, ceiling_db=-1.5, lookahead=0.004, release=0.15):
    """Lookahead peak limiter. x: (n, 2). Measured on 4x oversampled peaks so
    the true peak stays under the ceiling."""
    thr = db(ceiling_db)
    n = x.shape[0]
    up = signal.resample_poly(x, 4, 1, axis=0)
    peak_up = np.max(np.abs(up), axis=1)
    peak = peak_up.reshape(n, 4).max(axis=1)
    # smooth the peak track a little so the gain does not chatter
    la = n_of(lookahead)
    peak_s = np.maximum(peak, signal.sosfilt(_sos("low", 3000.0, 1), peak))
    rel = 1.0 - math.exp(-1.0 / (release * SR))
    g = _limiter_gain(np.ascontiguousarray(peak_s), thr, la, rel)
    # slide the gain earlier by the lookahead so it is in place before the peak
    g2 = np.ones(n)
    g2[: n - la] = g[la:]
    g2 = np.minimum(g2, g)
    return x * g2[:, None]


# ------------------------------------------------------------------- WAV out

def write_wav24(path, x):
    """Write (n, 2) float in [-1, 1] as 48 kHz 24 bit PCM with TPDF dither."""
    x = np.asarray(x, dtype=np.float64)
    n, ch = x.shape
    rng = np.random.default_rng(12345)
    lsb = 1.0 / (2 ** 23)
    dith = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    y = np.clip(x + dith, -1.0, 1.0 - lsb)
    ints = np.round(y * (2 ** 23)).astype(np.int32)
    ints = np.clip(ints, -(2 ** 23), 2 ** 23 - 1)
    flat = ints.reshape(-1)
    b = flat.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].reshape(-1)
    data = b.tobytes()
    byte_rate = SR * ch * 3
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, SR, byte_rate, ch * 3, 24)
    header += b"data" + struct.pack("<I", len(data))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data)
