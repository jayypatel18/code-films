# Code films: rules and house style every film agent reads

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo, the opening paragraph was shortened because it quoted view counts, and a remark comparing our aims with the reference films was trimmed. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

Repo: this one, `jayypatel18/code-films`.

## Why this exists
@chetaslua made two short anime films where a model wrote code that painted every frame and synthesized every sound. Our previous code film (The Last Train, real-time three.js) had no living character, no motion in the first seconds, and looked like 3D. These films fix that: offline pre-rendered, hand-painted 2D anime look, an animal the audience falls for.

Reference frames from his films: `reference/*-sheet.png` (one frame every 4 s) and `*-first3s.png`. His "THE LAST PAGE" prompt: `films/the-last-page/PROMPT.md`. That is the bar. We want backgrounds at least as painterly as his first film's gouache and a character as appealing as a theatrical anime lead.

## Hard rules
1. Every pixel and every sound comes from code you write in this repo. No image, video or music models, no downloaded or stock assets, no samples, no impulse responses from files, no font files. Do not use `PIL.ImageFont` or any system font: all lettering is drawn as brush strokes in code.
2. Tools: Python 3.12 with numpy, scipy, numba and PIL (no OpenCV, no cairo, no skia). C or C++ through `clang` for hot loops is fine. ffmpeg has libx264 and aac but no drawtext. Do not pip install anything without asking the director session.
3. Never write an em dash or an en dash anywhere: code, comments, docs, notes, reports. Do not use a spaced hyphen as a stand-in either. Restructure the sentence.
4. Do not run `git commit`, `git push` or any GitHub command. The director session commits everything with the right author.
5. Never read `.env` files. Never post anything to X or anywhere else.
6. Stay in your own folders. You may read anything in the repo; write only where your brief says you own files.
7. Heavy jobs (frame renders, background painting, long audio renders) always run through the slot wrapper so at most two run on this 8 core, 16 GB machine:
   `python3 tools/heavy.py -- python3 <script> <args>`
   Anything longer than about 5 minutes runs with `run_in_background` and you get notified when it ends; do not sit in a foreground command for 10 minutes.
8. Disk has about 44 GB free. Never write a full PNG sequence of a whole film. Stream frames to ffmpeg over a pipe. Keep your caches under 5 GB in `build/` and delete your own scratch files.
9. Rendering must be deterministic and random access: frame f renders the same every time and can be rendered alone, so stills and partial re-renders work. Seed every random generator from (cut, element, frame) style keys.
10. Absolute paths in every Bash command. No `cd X && ...`.

## House style: how a frame is made to look like theatrical 2D anime
The look comes from contrast: crisp, flat, inked characters over soft, richly painted backgrounds, tied together by one light and one atmosphere.

Backgrounds (painted, gouache)
- Compose a reference image in code first (layout, values, light, shapes), then paint over it with a stroke-based painter: several passes from big brushes to small ones, each stroke a textured, slightly opaque dab or drag oriented along the form (cloud contours, stone courses, foliage clumps, book spines), colour sampled from the reference with jitter in hue and value. Then hand-place detail strokes and crisp edges where a background artist would sharpen (architecture edges, highlights on stone, leaf clusters against sky).
- Every book, stone, pine and cloud is individually different. No repeated stamps that a viewer can spot.
- Colourful shadows (violet, teal, rose), never grey. Glowing mist and atmospheric perspective: far things lighter, bluer and lower in contrast.
- Paper or canvas grain modulates the paint subtly. Slight edge darkening where paint pools.
- Never: hard computer gradients, perfect perspective grids, uniform boxes, smooth plastic shading.

Characters (cel animation)
- Shapes are filled with flat colour, one hard-edged shadow tone (hue shifted toward the ambient light) and occasionally one highlight tone. No soft airbrushed shading on characters.
- Ink lines have real pressure: thicker on the shadow side and at overlaps, tapered at stroke ends, thinner on lit edges, slightly broken where light hits. Line colour is a dark warm umber, not pure black; inner lines (fur tufts, shadow borders) may use coloured lines.
- Draw at 3x and downsample for clean anti-aliasing. Lines may wobble a fraction of a pixel and "boil" a tiny amount from drawing to drawing, never enough to flicker.
- Animate on twos (12 drawings a second) for most acting, threes for slow movement, ones only for the fastest action and camera moves. Strong key poses, holds, anticipation, overshoot and settle, smears on the fastest frames, overlapping action in ears, tail and fur. No constant bobbing, no smooth software easing, no rigid parts rotating on pins: in-betweens are redrawn shapes, not rotated sprites.
- The character takes the scene light: shadow colour, rim light on the light side, lantern warmth on faces, and atmosphere over distant figures.

Effects and finishing
- Light shafts, dust motes (only visible inside light), glow and bloom done like a compositing department: soft, warm, restrained.
- One shared grade per cut, gentle vignette, fine animated film grain, and dithering before 8 bit so gradients never band.
- Text appears only where the story says, brushed in ink in code.

## The inspection loop (not optional)
- Export stills from every cut you touch and look at them with the Read tool. Make contact sheets (a grid of small frames) to check a whole cut, and full resolution crops to check line work and brushwork.
- For every still ask: would this pass as a frame from a finished theatrical anime film? Does anything look like 3D, vectors, web animation, AI art or amateur work? Fix it before moving on.
- Check motion by looking at consecutive frames side by side, not only single stills.
- Keep a log in your notes file: what you saw, what you changed.

## Record keeping
- Write the start time (IST, `TZ=Asia/Kolkata date`) when you begin and the end time when you stop, in your notes file. Captions will state how many hours the work took, so this must be accurate.
- When you finish or hit a checkpoint, report back: the paths of your best stills, what is done, what still looks weak, anything you could not do, and hours spent.
