# B: Film B, "TINY ROCKET" (picture), made entirely by Fable

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

You are making a whole animated short film, every frame of it, in code you write yourself. Read first:
1. `films/tiny-rocket/PROMPT.md`: the creative prompt. This is your film.
2. `films/tiny-rocket/briefs/COMMON.md`: the production rules (binding) and a house style you may use or improve on.
3. Reference frames from the films that inspired this project: `reference/chetas-film1-sheet.png` and `chetas-film2-last-page-sheet.png`, plus the `*-first3s.png` strips. That is the bar to beat. Do not copy any shot or character.

## Why you
The person who inspired this project says Fable gave him better art than other models. This film tests that. So the work must be yours alone: write every line of the picture code yourself. Do not import or read the code in `engine/` or in any other film folder, and do not read `our previous film's repo (github.com/jayypatel18/last-train)`. Start from an empty folder. The only shared file you may use is `tools/heavy.py` (the render slot wrapper).

The end card must name you correctly: write the exact model name you are, as given in your own system prompt (for example "Fable 5.1"). Never write a model name that is not yours.

## What you own
- Everything in `films/tiny-rocket/` except `PROMPT.md` (do not edit the prompt; if you change the film, as the prompt allows, write what you changed and why in `NOTES.md`). Sound will be made later by a second Fable agent in `films/tiny-rocket/sound/`; leave that folder to it.
- Your log: `films/tiny-rocket/NOTES.md`. Write your start time first (`TZ=Asia/Kolkata date`), then log progress, decisions and what you fixed after each inspection. If your session is interrupted, this file is how the work resumes, so keep it current.

## Format
- 1920x1080 (16:9) full frame, no letterbox bars inside the video. 24 fps. About 60 s.
- Pre-render offline. Stream frames to ffmpeg; never write a full PNG sequence of the whole film.
- Master: `films/tiny-rocket/out/tiny-rocket-master.mp4`, H.264 high, yuv420p, 24 fps, CRF 14 or lower, no audio.
- A cue sheet for the sound agent: `films/tiny-rocket/cues.json` with the start of every shot and every story beat and sound event (each paw step, the knot tug, blinks, the match strike, the fuse start and end, the stillness beat, ignition, the birds, breaking through the clouds, the shooting star, the parachute, the brush stroke, the hanko, the tail curl) as `{"name": seconds}` or lists of seconds. Write a first version as soon as your shot timing is set, and update it when the animation is final.
- Later, for X: an X cut that opens cold on the panda in motion (the most striking 1.5 to 2 s of it moving: a blink, an ear flick, the knot tug), then the film from the start. Choose the cold-open segment and record it in `cues.json` as `"cold_open": [start, end]`.
- Three stills in `out/`: the best close-up of the panda, the climax, the title card.

## Checkpoints (return to the director session at each one)
1. **Look test:** within about 2 hours, the red panda in at least five poses (a model sheet) and two finished frames: shot 1 (the close-up at the knot) and shot 2 (the hill at dusk, wide) at full 1920x1080. Save to `films/tiny-rocket/build/review/`. Inspect them hard against the prompt's quality standard and fix what fails. Then stop and report the paths, what you think is working, what is not yet, and hours so far.
2. **Full rough render:** every shot animated, the whole film rendered, contact sheets per shot, `cues.json` written. Report.
3. **Final:** after your own inspection loop over every shot, the final master and the three stills. Report.

When you report, be honest about anything you could not do or that still falls short.
