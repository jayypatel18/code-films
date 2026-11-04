# TINY ROCKET: the sound

Everything you hear is synthesized here with numpy and scipy (numba speeds
up three small per sample loops). No samples, no impulse response files, no
borrowed melodies. Run `python3 score.py` to render both WAVs and print the
ffmpeg ebur128 measurements.

## Files
- `dsp.py` envelopes, Butterworth and RBJ filters, a time varying state
  variable filter, noise colours, band limited additive oscillators, pan,
  the reverb, a lookahead true peak limiter and the 24 bit WAV writer.
- `instruments.py` the orchestra (below).
- `sfx.py` every sound effect (below).
- `music.py` the motif and the arrangement on the timeline.
- `score.py` the effects timeline from `cues.json`, buses, reverbs, the
  stillness gate, mastering, loudness loop, the cold open.

## The motif
D major, seven notes, 72 bpm: `D4 A4 B4 A4 F#4 E4 D4` with the rhythm
`1 1 .5 .5 1 1 2` beats. A leap up a fifth (the reach), one step higher,
then it falls back home. Heard:
- 5.6 s alone on the felt piano as the panda admires the rocket;
- 13.6 to 18.8 s in augmentation on the celesta over a music box arpeggio;
- 46.0 s at full scale (horns, strings in two octaves, piano octaves,
  glockenspiel, timpani) as the trail ignites, the F# landing on the arc's
  peak at 48.5 s;
- 53.2 s its last three notes, quiet, on the piano, at home;
- 58.9 s a single felt piano D5 on the brush stroke; 60.0 s a rolled D major
  chord under the title.

## Instruments (how each is made)
- Felt piano: two detuned additive strings per note with inharmonic
  partials, each partial with its own decay (higher ones die faster), a soft
  lowpassed noise hammer knock, a rounded 6 ms attack, a felt lowpass tied
  to velocity, and a damper release.
- Music box: a steel tine as four inharmonic partials (1, 2.77, 5.4, 8.9)
  with a tiny comb click. Celesta: the same idea with a wooden resonator
  under the bar. Glockenspiel: brighter bar partials.
- Strings: six band limited saws (additive, 1/k^1.15) each with its own
  detune and slow vibrato, a bow noise layer, a lowpass opened by velocity,
  an optional 11 Hz tremolo. Runs are chains of short notes that accelerate.
- Horns: three detuned saws through a soft tanh, a lowpass that opens over
  the attack (the "brassy" swell), a lip scoop at the onset.
- Timpani: a sine with a pitch drop plus membrane modes (1.5, 1.98, 2.44)
  and a mallet thud; rolls are rapid hits following a loudness curve.
- Bass drum: a pitched sine thump. Cymbal: white noise through 22 narrow
  resonators between 1.8 and 7.2 kHz with a slow swell envelope.
- Pad: detuned sine pairs with a slow amplitude wobble.

## Effects (all in `sfx.py`)
Crickets (chirp trains of 4 kHz pulses, each cricket its own pitch, rate,
distance and position), breeze (pink noise under slow gusts plus a grass
rustle band), night air, rushing wind (resonant band filters wandering over
white noise plus a close low layer for the face shot), morning birds (FM
chirp phrases), the knot creak (stick slip pulse train through three
resonators, fibre noise, a tug thunk with a bamboo ring), string release,
ear flick, blink, tail swish, fur and cloth rustles, paw steps in grass
(crinkle band with micro transients over a thud), village light tings,
sleeping breath, the shooting star (rising whistle with a sparkle cloud and
an air sweep, panned across the sky), fireflies (soft sine blips), the
teacup landing (ceramic modes, a thud, tin), goggles snap, swallow, match
strike (granular rasp then a flare whump and sizzle), the fuse (fizz band
rising in pitch with sparse crackles, panned along the grass), the clock
tick, ignition (sub boom, a bright to dull noise burst, sparks, a widening
ring whoosh), the rocket roar (brown and white noise, a growl, a resonant
body following a pitch curve so it drops as the rocket passes and leaves),
tin rattle and paper flutter for the shudder and fins, the bird burst (nine
birds, each a flutter of wing beats and an alarm chirp scattering in
stereo), cloud punches, the open sky whoosh, the star ignition (24 sweeping
sines and a sparkle cloud), the arc's high sheen, the parachute (crack,
whump, settling flutter), canopy sway, touchdown (double thud, bamboo
knocks, tin settling, grass), paper map, brush strokes (bristle band with a
loaded start and a dry, grainy end), the hanko thump.

## Space, mix, master
Two reverbs built in code: each is ten early reflection taps feeding an 8
line feedback delay network (Householder mixing, a one pole damping lowpass
in every line so the tail darkens as it decays). The hall (RT60 about 2.8 s)
takes the music and the big sky events; the field (about 0.75 s) takes the
close outdoor effects at a low send.

The stillness beat (29.4 to 30.5 s) is a hard gate on everything with a
whisper of air at about -52 dBFS so it reads as a held breath, not a
dropout. A scene trim curve (`TRIM_POINTS` in `score.py`, set from the per
shot short term loudness of the first render) lifts the intimate scenes so
they stay audible on a phone and shaves the loud block. Master: 24 Hz
highpass, a -4 dB shelf above 8.5 kHz, 15 kHz lowpass, then a loudness loop
(render, measure with ffmpeg ebur128, correct the gain, up to three passes)
to -14 LUFS integrated, and a 4x oversampled lookahead limiter at
-1.3 dBTP. TPDF dither to 24 bit. `score.py` prints the integrated
loudness, LRA, true peak and the short term loudness of every shot.

The cold open WAV (1.75 s) covers picture 0.85 to 2.6 s: dusk bed, the knot
creak, paws and fur, the ear flick and tail swish, the first two motif
notes in the hall, fading to near silence at the end.
