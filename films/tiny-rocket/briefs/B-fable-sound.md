# B sound: the score and sound design of Film B, "TINY ROCKET", made entirely by Fable

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

The picture of this film is finished and was made entirely by another Fable agent. You make every sound, in code you write yourself.

Read first:
1. `films/tiny-rocket/PROMPT.md`, especially the SOUND section.
2. `films/tiny-rocket/briefs/COMMON.md`: production rules (binding). For sound that means no samples, no impulse response files, no borrowed melodies, nothing downloaded.
3. `films/tiny-rocket/cues.json`: the start of every shot and every story beat and sound event, in seconds, measured from the finished animation. `cold_open` is the X cut's opening segment.
4. `films/tiny-rocket/NOTES.md`: what the picture agent made, shot by shot.
5. Watch the picture through stills: `films/tiny-rocket/build/sheets/sheet-*.png` (contact sheets per shot) and the master `films/tiny-rocket/out/tiny-rocket-master.mp4` (65.0 s, no audio; you can pull frames with ffmpeg to look at them).

## Why you
This film tests what Fable can do on its own, so the work must be yours alone: write every line of the synthesis code yourself. Do not read or import code from `engine/`, from `films/the-last-page/` (another model's film, including its `sound/` folder), or from `our previous film's repo (github.com/jayypatel18/last-train)`. Start from an empty folder. Python with numpy and scipy is available.

## What you own
- `films/tiny-rocket/sound/` (your code; one entry script `score.py`, plus modules if you want, and a short `README.md` saying what every layer is and how it is made).
- Outputs:
  - `films/tiny-rocket/build/audio/tiny-rocket.wav`: 48 kHz, stereo, 24 bit PCM, exactly 65.000 s (3,120,000 samples).
  - `films/tiny-rocket/build/audio/tiny-rocket-coldopen.wav`: the X cut opens cold on the picture from 0.85 s to 2.6 s (the knot tug), then plays the film from 0. This WAV is exactly 1.75 s (84,000 samples): the knot, the paws, a breath of the dusk ambience and a hint of the piano motif, fading to near silence so it hands over cleanly to the film's opening.
- Log: append to `films/tiny-rocket/NOTES.md` with entries starting `SOUND HH:MM IST:`. Write your start time first (`TZ=Asia/Kolkata date`) and your end time when you stop; captions will state the hours, so keep them accurate.
- Do not touch the picture files, `cues.json` or `PROMPT.md`.

## Craft
- An original motif of 5 to 9 notes, simple enough to hum, first on a solo felt piano when the panda admires its rocket, a music box or celesta colour under the night sky, ticking held-breath tension for the fuse, true near silence for the stillness beat (`stillness_start` to `stillness_end`), a roar at `ignition`, rattling tin and paper, a full orchestral swell as the rocket breaks through the clouds (`break_into_open_sky`), the motif at full scale at the climax (`star_ignites`), quiet at home, and a single soft piano note on the paw's brush stroke (`brush_stroke`).
- Every effect in `cues.json` placed within one frame (42 ms): the knot tug, paws in grass, blinks and ear flicks as tiny intimate sounds, crickets and breeze at dusk, the night shooting star, fireflies, the match strike, the fuse fizz, ignition, the birds bursting from the tree, wind, fins rattling, the parachute, the touchdown, the brush stroke, the hanko, the tail curl.
- Synthesize the space with a reverb you build in code.
- Mix and master: integrated loudness about -14 LUFS, true peak at most -1 dBTP, loudness range about 15 LU (quiet parts must still be audible on a phone), nothing harsh above 8 kHz, no clipping. Measure with ffmpeg's `ebur128` or `loudnorm` filter and print the numbers.
- Long renders go through the slot wrapper with one slot only: `python3 tools/heavy.py --slots 1 -- python3 <script>`.
- No em dashes or en dashes anywhere (code, comments, README, notes). No git commands.

## Report
When both WAVs are rendered and measured, report back: the paths, the loudness numbers, the motif, how the main instruments are made, what you are unsure about (you cannot listen, so say what you checked by measurement), and your hours.
