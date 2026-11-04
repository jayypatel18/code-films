"""Shared DSP for the sound of THE LAST PAGE.

Constants, filters, envelopes, noise textures, mix buses, synthetic reverb, loudness
measurement (ITU-R BS.1770), a true peak limiter and a 24 bit WAV writer.
numpy, scipy and numba only. Nothing here reads an audio file.
"""

import os
import struct
import sys
import zlib

import numpy as np
from numba import njit
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
if FILM not in sys.path:
    sys.path.insert(0, FILM)
import timeline as TL  # noqa: E402

SR = 48000
DUR = TL.FRAMES / TL.FPS            # 60.0 s
NS = int(round(SR * DUR))           # 2,880,000 samples
FRAME = 1.0 / TL.FPS                # one picture frame, 41.7 ms
TAU = 2.0 * np.pi
SEED = 20261104
TAX = np.arange(int(SR * 20)) / SR  # time axis for single events (20 s max)
TIME = np.arange(NS) / SR

# Exact event times from the animation (written by the lead: render.py --events) override the
# planned cues of the same name, so a re-run of score.py re-times every sound to the picture.
EVENTS_PATH = os.path.join(FILM, "build", "events.json")
CUES = dict(TL.CUES)
EVENTS = {}
EVENT_OVERRIDES = []      # (name, planned cue or None, event time)


def _load_events():
    import json
    if not os.path.exists(EVENTS_PATH):
        return
    with open(EVENTS_PATH) as fh:
        ev = json.load(fh)
    EVENTS.update(ev)
    scal = {k: float(v) for k, v in ev.items() if isinstance(v, (int, float))}
    if isinstance(ev.get("sniffs"), list) and len(ev["sniffs"]) >= 2:
        scal["sniff_1"], scal["sniff_2"] = float(ev["sniffs"][0]), float(ev["sniffs"][1])
    if ev.get("paw_steps"):
        scal["first_paw_step"] = float(ev["paw_steps"][0])
    if ev.get("title_strokes"):
        scal["title_ink_start"] = float(ev["title_strokes"][0][0])
        scal["title_ink_end"] = float(ev["title_strokes"][-1][1])
    if "line_strokes_start" in scal:
        scal["line_ink_start"] = scal.pop("line_strokes_start")
    for k, v in scal.items():
        old = CUES.get(k)
        if old is None or abs(old - v) > 1e-6:
            EVENT_OVERRIDES.append((k, old, v))
        CUES[k] = v


_load_events()


def C(name):
    """Time in seconds: the animation's exact event if events.json has it, else timeline.CUES."""
    return float(CUES[name])


def EV(name, default=None):
    """A raw entry of events.json (lists: paw_steps, bird_wingbeats, title_strokes, ...)."""
    return EVENTS.get(name, default)


def log(msg=""):
    print(msg, flush=True)


def rng_for(tag):
    """Deterministic generator per element: editing one sound never reshuffles the others."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode("utf-8"))])


NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nn(x):
    """Note name ('F#5', 'Db3') or MIDI number to MIDI number (C4 = 60)."""
    if not isinstance(x, str):
        return int(x)
    base = NOTE_BASE[x[0]]
    i = 1
    while i < len(x) and x[i] in "#b":
        base += 1 if x[i] == "#" else -1
        i += 1
    return 12 * (int(x[i:]) + 1) + base


def mtof(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def db2lin(d):
    return 10.0 ** (np.asarray(d, dtype=float) / 20.0)


def lin2db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))


def pan_gains(p):
    th = (np.clip(p, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def taxis(n):
    return TAX[:n] if n <= len(TAX) else np.arange(n) / SR


# ------------------------------------------------------------------------------------- filters
def sos_lp(fc, order=2):
    return signal.butter(order, min(fc, 0.45 * SR), "lowpass", fs=SR, output="sos")


def sos_hp(fc, order=2):
    return signal.butter(order, fc, "highpass", fs=SR, output="sos")


def sos_bp(lo, hi, order=2):
    return signal.butter(order, [lo, min(hi, 0.45 * SR)], "bandpass", fs=SR, output="sos")


def sos_peak(f0, gain_db, q):
    """RBJ peaking EQ."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TAU * f0 / SR
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = np.array([1 + al * A, -2 * c, 1 - al * A])
    a = np.array([1 + al / A, -2 * c, 1 - al / A])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def sos_shelf_hi(f0, gain_db, s=0.7):
    """RBJ high shelf."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TAU * f0 / SR
    c, sn = np.cos(w0), np.sin(w0)
    al = sn / 2.0 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al)
    b1 = -2 * A * ((A - 1) + (A + 1) * c)
    b2 = A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)
    a0 = (A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al
    a1 = 2 * ((A - 1) - (A + 1) * c)
    a2 = (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def sos_reson(f0, q):
    """RBJ band-pass with 0 dB peak gain (formants, resonances)."""
    w0 = TAU * f0 / SR
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = np.array([al, 0.0, -al])
    a = np.array([1 + al, -2 * c, 1 - al])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def chain(*soss):
    return np.vstack(soss)


def filt(sos, x):
    return signal.sosfilt(sos, x, axis=-1)


def onepole_lp(x, fc):
    k = np.exp(-TAU * fc / SR)
    return signal.lfilter([1.0 - k], [1.0, -k], x, axis=-1)


@njit(cache=False)
def _svf(x, fc, q, mode, sr):
    """Topology preserving state variable filter with per-sample cutoff and Q.
    mode 0 lowpass, 1 bandpass (0 dB peak), 2 highpass."""
    n = x.shape[0]
    y = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    for i in range(n):
        f = fc[i]
        if f > 0.45 * sr:
            f = 0.45 * sr
        if f < 5.0:
            f = 5.0
        g = np.tan(np.pi * f / sr)
        k = 1.0 / q[i]
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1 * k
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def svf(x, fc, q, mode="bp"):
    """Time varying filter. x 1-D or 2-D (channels first); fc and q scalars or per-sample arrays."""
    md = {"lp": 0, "bp": 1, "hp": 2}[mode]
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    fc = np.ascontiguousarray(np.broadcast_to(np.asarray(fc, dtype=np.float64), (n,)))
    q = np.ascontiguousarray(np.broadcast_to(np.asarray(q, dtype=np.float64), (n,)))
    if x.ndim == 1:
        return _svf(np.ascontiguousarray(x), fc, q, md, float(SR))
    return np.stack([_svf(np.ascontiguousarray(c), fc, q, md, float(SR)) for c in x])


# ----------------------------------------------------------------------------------- envelopes
def env_note(n, att, t_off, rel):
    """Raised-cosine attack, sustain, exponential release (60 dB in rel s) from t_off."""
    e = np.ones(n)
    na = min(n, max(2, int(att * SR)))
    e[:na] = 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    i_off = int(t_off * SR)
    if i_off < n:
        e[i_off:] *= np.exp(-6.91 * taxis(n - i_off) / max(rel, 1e-3))
    nf = min(n, int(0.006 * SR))
    e[n - nf:] *= np.linspace(1.0, 0.0, nf)
    return e


def env_ar(n, att, dec, shape=1.0):
    """Attack then exponential decay (time constant dec)."""
    t = taxis(n)
    a = np.clip(t / max(att, 1e-4), 0.0, 1.0) ** shape
    return a * np.exp(-np.maximum(t - att, 0.0) / max(dec, 1e-4))


def env_bell(n, peak=0.5, width=0.22):
    x = np.linspace(0.0, 1.0, n)
    return np.exp(-0.5 * ((x - peak) / width) ** 2)


def smooth_rand(n, rate, rng):
    """Smooth random curve, unit-ish variance, about `rate` wiggles per second."""
    m = int(n / SR * rate) + 3
    pts = rng.standard_normal(m)
    pos = np.arange(n) * (rate / SR)
    i = pos.astype(np.int64)
    w = 0.5 - 0.5 * np.cos(np.pi * (pos - i))
    return pts[i] * (1.0 - w) + pts[i + 1] * w


def edge_fade(y, fin=0.002, fout=0.004):
    """Short raised-cosine fades at both ends (no clicks)."""
    n = y.shape[-1]
    a = min(n // 2, max(1, int(fin * SR)))
    b = min(n // 2, max(1, int(fout * SR)))
    y[..., :a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    y[..., n - b:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return y


def curve(t, knots):
    """Piecewise linear curve over time array t from [(time, value), ...]."""
    ts, vs = zip(*knots)
    return np.interp(t, ts, vs)


# ------------------------------------------------------------------------------- oscillators
def saw_blep(freq, phase0=0.0):
    """Band-limited sawtooth (polyBLEP) for a per-sample frequency array."""
    dt = freq / SR
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = ph > 1.0 - dt
    x = (ph[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def modal(freqs, amps, t60s, length, rng, att=0.0015):
    """Sum of exponentially decaying sines (wood, metal, drums, clicks)."""
    n = max(4, int(length * SR))
    y = np.zeros(n)
    for f, a, t60 in zip(freqs, amps, t60s):
        if f >= 0.45 * SR or a == 0.0:
            continue
        nk = min(n, int(t60 * 1.15 * SR) + 2)
        tt = taxis(nk)
        y[:nk] += a * np.sin(TAU * ((f * tt + rng.random()) % 1.0)) * np.exp(-6.91 * tt / t60)
    na = max(2, int(att * SR))
    y[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    return y


# ------------------------------------------------------------------------------------ noise
def crackle_env(n, rate, rng, tau=0.003, sigma=0.8):
    """Envelope of random grains: Poisson events (rate per second, scalar or per sample), each a
    lognormal spike with exponential decay tau. Multiply noise by it for rustle, crinkle, dust."""
    r = np.broadcast_to(np.asarray(rate, dtype=float), (n,))
    ev = rng.random(n) < r / SR
    imp = np.zeros(n)
    k = int(ev.sum())
    if k:
        imp[ev] = rng.lognormal(0.0, sigma, k)
    a = np.exp(-1.0 / (max(tau, 1e-5) * SR))
    return signal.lfilter([1.0], [1.0, -a], imp)


def band_noise(n, lo, hi, rng, channels=1, order=2):
    nz = rng.standard_normal((channels, n)) if channels > 1 else rng.standard_normal(n)
    return filt(sos_bp(lo, hi, order), nz)


def brown(n, rng, fc=60.0, channels=1):
    nz = rng.standard_normal((channels, n)) if channels > 1 else rng.standard_normal(n)
    return onepole_lp(nz, fc) * np.sqrt(SR / (TAU * fc))


FB_CENTERS = np.geomspace(110.0, 9000.0, 19)


def swept_noise(n, fc, rng, width_oct=0.8, channels=2):
    """Noise whose spectral peak follows the per-sample curve fc (Hz), via a crossfaded filterbank."""
    noise = rng.standard_normal((channels, n))
    out = np.zeros((channels, n))
    lfc = np.log2(np.maximum(fc, 20.0))
    for c in FB_CENTERS:
        w = np.exp(-0.5 * ((lfc - np.log2(c)) / width_oct) ** 2)
        if w.max() < 1e-3:
            continue
        out += filt(sos_bp(c / 2 ** 0.26, c * 2 ** 0.26), noise) * w
    return out


def place_stereo(mono, pan):
    """Mono signal to stereo with a per-sample (or scalar) equal power pan."""
    gl, gr = pan_gains(pan)
    return np.vstack([mono * gl, mono * gr])


# ---------------------------------------------------------------------------------- mix buses
class Bus:
    def __init__(self, name):
        self.name = name
        self.x = np.zeros((2, NS), np.float32)

    def add(self, sig, t0, gain=1.0, pan=0.0):
        """Mix a mono (panned, equal power) or stereo (balanced) signal in at time t0."""
        sig = np.asarray(sig)
        i0 = int(round(t0 * SR))
        n = sig.shape[-1]
        a, b, s0 = i0, i0 + n, 0
        if a < 0:
            s0, a = -a, 0
        b = min(b, NS)
        if b <= a:
            return
        seg = slice(s0, s0 + (b - a))
        if sig.ndim == 1:
            gl, gr = pan_gains(pan)
            self.x[0, a:b] += sig[seg] * (gain * gl)
            self.x[1, a:b] += sig[seg] * (gain * gr)
        else:
            gl = gain * min(1.0, 1.0 - pan)
            gr = gain * min(1.0, 1.0 + pan)
            self.x[0, a:b] += sig[0, seg] * gl
            self.x[1, a:b] += sig[1, seg] * gr


# ------------------------------------------------------------------------------------ reverb
def make_ir(rt_lo, rt_hi, length, predelay, rng, er=10, er_span=0.07, er_gain=6.0, onset=0.035):
    """Stereo synthetic reverb: band-split decaying noise (highs die faster), soft onset,
    a few early reflections, unit energy per channel. Pure synthesized noise, no recordings."""
    n = int(length * SR)
    t = np.arange(n) / SR
    edges = [20.0, 250.0, 800.0, 2500.0, 6000.0, 12000.0, 21000.0]
    out = np.zeros((2, n))
    for ch in range(2):
        w = rng.standard_normal(n)
        acc = np.zeros(n)
        for lo, hi in zip(edges[:-1], edges[1:]):
            fcen = np.sqrt(lo * hi)
            rt60 = float(np.interp(np.log(fcen), np.log([250.0, 9000.0]), [rt_lo, rt_hi]))
            acc += filt(sos_bp(lo, hi), w) * np.exp(-6.91 * t / rt60)
        acc *= smoothstep(t / onset)
        for _ in range(er):
            d = rng.uniform(0.004, er_span)
            acc[int(d * SR)] += rng.choice([-1, 1]) * rng.uniform(0.3, 0.8) * np.exp(-d / 0.05) * er_gain
        out[ch] = acc
    pd = int(predelay * SR)
    out = np.concatenate([np.zeros((2, pd)), out], axis=1)
    out /= np.sqrt(np.sum(out ** 2, axis=1, keepdims=True))
    return out


def convolve_stereo(x, ir, cross=0.2):
    wl = signal.oaconvolve(x[0].astype(np.float64), ir[0])[:NS]
    wr = signal.oaconvolve(x[1].astype(np.float64), ir[1])[:NS]
    return np.vstack([wl + cross * wr, wr + cross * wl]) / (1.0 + cross)


# -------------------------------------------------------------------------- loudness, peaks
def kweight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def kpower(x):
    y = kweight(np.asarray(x, dtype=np.float64))
    return y[0] ** 2 + y[1] ** 2


def lufs_integrated(x):
    """ITU-R BS.1770-4 integrated loudness (K-weighting, 400 ms blocks, 75% overlap, gates)."""
    p = kpower(x)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    st = np.arange(0, len(p) - blk + 1, hop)
    z = (cs[st + blk] - cs[st]) / blk
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    g1 = lk > -70.0
    if not g1.any():
        return -np.inf
    rel = -0.691 + 10 * np.log10(z[g1].mean()) - 10.0
    g2 = g1 & (lk > rel)
    return -0.691 + 10 * np.log10(z[g2].mean())


def window_loudness(p, t0, t1, win=3.0):
    """(mean LUFS over [t0, t1], max short-term LUFS of windows `win` s inside it) from K power p."""
    i0, i1 = int(t0 * SR), int(t1 * SR)
    mean = -0.691 + 10 * np.log10(p[i0:i1].mean() + 1e-20)
    cs = np.concatenate([[0.0], np.cumsum(p[i0:i1])])
    w = min(int(win * SR), i1 - i0)
    hop = int(0.1 * SR)
    st = np.arange(0, (i1 - i0) - w + 1, hop)
    z = (cs[st + w] - cs[st]) / w
    return mean, -0.691 + 10 * np.log10(z.max() + 1e-20)


def loudness_curve(p, win, hop=0.1):
    """(times at window centre, LUFS) for sliding windows of `win` seconds."""
    cs = np.concatenate([[0.0], np.cumsum(p)])
    w, h = int(win * SR), int(hop * SR)
    st = np.arange(0, len(p) - w + 1, h)
    z = (cs[st + w] - cs[st]) / w
    return (st + w / 2) / SR, -0.691 + 10 * np.log10(z + 1e-20)


def true_peak(x, os_=4):
    up = signal.resample_poly(np.asarray(x, dtype=np.float64), os_, 1, axis=-1)
    return float(np.abs(up).max())


def compressor(x, thr_db, ratio, knee=6.0, tau_det=0.06, tau_gain=0.15):
    p = 0.5 * (x[0] ** 2 + x[1] ** 2)
    k = np.exp(-1.0 / (tau_det * SR))
    ms = signal.lfilter([1 - k], [1, -k], p)
    over = 10 * np.log10(ms + 1e-12) - thr_db
    s = 1.0 - 1.0 / ratio
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * s, s * (over + knee / 2) ** 2 / (2 * knee)))
    kg = np.exp(-1.0 / (tau_gain * SR))
    gr = signal.lfilter([1 - kg], [1, -kg], gr)
    return x * db2lin(-gr), gr


def limiter(x, ceiling_db, look=0.003, release=0.03):
    """Look-ahead true peak limiter: needed gain from a 4x oversampled peak envelope,
    min-filtered over the look-ahead window and box-smoothed (never above the needed gain)."""
    c = db2lin(ceiling_db)
    up = signal.resample_poly(x, 4, 1, axis=-1)
    pk = np.abs(up).max(axis=0)[: 4 * x.shape[1]].reshape(-1, 4).max(axis=1)
    pk = np.maximum(pk, np.abs(x).max(axis=0))
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = int(look * SR)
    g = uniform_filter1d(minimum_filter1d(g, size=2 * L + 1), size=L + 1)
    R = int(release * SR)
    g = uniform_filter1d(minimum_filter1d(g, size=2 * R + 1), size=R + 1)
    return x * g, g


# ----------------------------------------------------------------------------- 24 bit output
def write_wav24(path, y, rng):
    """Write float stereo (2, n) as 24 bit PCM WAV with TPDF dither. Returns clipped count."""
    lsb = 1.0 / 8388608.0
    d = (rng.random(y.shape) - rng.random(y.shape)) * lsb
    q = np.round((y + d) * 8388607.0)
    clipped = int(np.sum(np.abs(q) > 8388607))
    q = np.clip(q, -8388607, 8388607).astype("<i4")
    inter = np.ascontiguousarray(q.T).reshape(-1)
    b = inter.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    nch, bits = 2, 24
    ba = nch * bits // 8
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(b)) + b"WAVE"
           + b"fmt " + struct.pack("<IHHIIHH", 16, 1, nch, SR, SR * ba, ba, bits)
           + b"data" + struct.pack("<I", len(b)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(b)
    return clipped


def read_wav24(path):
    """Read back a 24 bit stereo WAV written by write_wav24 (for verification)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    i = raw.find(b"data")
    n = struct.unpack("<I", raw[i + 4:i + 8])[0]
    b = np.frombuffer(raw[i + 8:i + 8 + n], dtype=np.uint8).reshape(-1, 3)
    v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return (v.reshape(-1, 2).T.astype(np.float64)) / 8388608.0
