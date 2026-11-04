"""Cold open for the X cut of THE LAST PAGE.

The X cut opens on cut 3 (the close-up) from frame 362 (15.083 s) for 87,216 samples (1.817 s):
the right ear mid swing, the second ear, two sniffs, the blink. Then the film plays from 0.

The cold open uses the film's own close-up sounds in that window (ears, sniffs, blink, the lantern
flame, room tone, and their reverbs), plus a hint of the motif on the felt piano: its head as the
film first plays it (Bb4, Db5, then Ab5 left ringing). It goes through the same master gain and
compressor curve as the film, so it sits at the film's level, and fades to near silence at the
end so it hands over cleanly to the opening wind.
"""

import numpy as np
from scipy import signal

from dsp import SR, TL, db2lin, edge_fade, nn, rng_for, taxis
from instruments import felt_piano_note, key_noise, pedal_noise, sympathetic_ir

START_FRAME = 362                 # first frame of the cold open (cut 3)
LENGTH = 87216                    # samples (1.817 s at 48 kHz)
I0 = int(round(START_FRAME / TL.FPS * SR))

# the motif head, soft, on the felt piano: (offset s, note, velocity)
GESTURE = [(0.12, "Bb4", 0.30), (0.44, "Db5", 0.31), (0.76, "Ab5", 0.36)]
PEDAL_DOWN = 0.10


def piano_gesture(ir_score, piano_level, score_send):
    """Render the cold open's piano hint, dry plus its share of the scoring stage reverb."""
    n = LENGTH + int(0.5 * SR)
    acc = np.zeros((2, n))
    for k, (dt, nm, v) in enumerate(GESTURE):
        m = nn(nm)
        r = rng_for(f"coldopen-pno{k}")
        y = felt_piano_note(m, v, LENGTH / SR + 1.0 - dt, r, felt=1.0, cap=LENGTH / SR - dt + 0.2)
        pan = float(np.clip((m - 64) / 40.0, -0.38, 0.38))
        y = y[:, : n - int(dt * SR)]
        i = int(dt * SR)          # stereo balance, as the film's buses do it
        acc[0, i:i + y.shape[1]] += y[0] * min(1.0, 1.0 - pan)
        acc[1, i:i + y.shape[1]] += y[1] * min(1.0, 1.0 + pan)
        kn = key_noise(v, r, "down")
        acc[:, i + int(0.004 * SR):i + int(0.004 * SR) + len(kn)] += kn[None, :] * 0.7
    pn = pedal_noise(rng_for("coldopen-pedal"), True)
    i = int(PEDAL_DOWN * SR)
    acc[:, i:i + pn.shape[1]] += pn
    # sympathetic resonance: the pedal is down from PEDAL_DOWN on
    ps = np.clip((np.arange(n) / SR - PEDAL_DOWN) / 0.02, 0.0, 1.0)
    ir = sympathetic_ir(rng_for("sympathetic"))
    src = acc * ps[None, :]
    res = np.vstack([signal.oaconvolve(src[0], ir[0])[:n], signal.oaconvolve(src[1], ir[1])[:n]])
    acc += res * 0.20 * ps[None, :]
    acc *= piano_level
    wet = np.vstack([signal.oaconvolve(acc[0] * score_send, ir_score[0])[:n],
                     signal.oaconvolve(acc[1] * score_send, ir_score[1])[:n]])
    return (acc + wet * 0.85)[:, :LENGTH]


def premix(buses, rv_close, rv_hall, range_db, ir_score, piano_level, score_send):
    """Cold open mix before the master: the film's close-up sound plus the piano hint."""
    sl = slice(I0, I0 + LENGTH)
    dry = sum(buses[k].x[:, sl].astype(np.float64) for k in ("foley", "room", "amb", "paper"))
    wet = rv_close[:, sl] * 0.6 + rv_hall[:, sl] * 0.9
    sfx = dry + wet
    pno = piano_gesture(ir_score, piano_level, score_send)
    return (sfx + pno) * db2lin(range_db[sl])[None, :]


def finish(c):
    """Fades: in over 12 ms (the scene is already alive), out over the last 0.55 s to silence."""
    t = taxis(LENGTH)
    c = c.copy()
    c *= np.clip(t / 0.012, 0.0, 1.0)[None, :]
    x = np.clip((t - (LENGTH / SR - 0.55)) / 0.55, 0.0, 1.0)
    c *= (np.cos(0.5 * np.pi * x) ** 2)[None, :]
    return edge_fade(c, 0.0005, 0.003)
