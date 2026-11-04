# Code Films

Short anime films where every frame and every sound is code a model wrote.

No image models, no video models, no stock footage, no samples, no font files. Every background is painted stroke by stroke by a Python painter, every character is inked and cel shaded by Python, the lettering is brushed by code, and every note of the score and every sound effect is synthesized with numpy. The finished frames are rendered offline and encoded to video.

**Watch them:** https://jayypatel18.github.io/code-films/

| Film | Made by | Length | Story |
|---|---|---|---|
| [THE LAST PAGE](films/the-last-page/) | Opus 5.5 | 60 s | A small fox wakes a forgotten library that holds every story ever written. |

## Where this comes from

The idea and the original prompt come from [@chetaslua](https://x.com/chetaslua), who made the first films like this and published his prompt: [film one](https://x.com/chetaslua/status/2117169796290916732) and [THE LAST PAGE](https://x.com/chetaslua/status/2117224839010459987). THE LAST PAGE here runs his prompt again, word for word except for the model name, with a different model. The agents that made it also worked from team briefs (a house style, the fox design, the split into departments) and from a directing session's notes between passes; all of it is in [films/the-last-page/briefs/](films/the-last-page/briefs/), so you can see exactly what was added on top of his prompt. More films made the same way, with new stories in the same prompt format, are on the way.

## How a frame is made

1. **Backgrounds** are painted by a stroke painter (`engine/paint*.py`): a layout and light study is composed in code, then painted over in passes from big brushes to small ones, each stroke a textured gouache dab or drag that follows the form, with paper grain and pooled edges. Plates are painted once and cached.
2. **Characters** are built from simple posed 3D masses for layout (so they stay on model from every angle), then drawn as flat 2D cel shapes with hard shadow shapes and pressure ink lines (`engine/cel.py`, `engine/ink.py`), animated on twos and threes like hand-drawn anime.
3. **Effects** (paper birds, light shafts, dust, the great bird) are drawn by batch rasterizers in numba.
4. **Lettering** is brush strokes authored glyph by glyph and written on with a pressure brush (`engine/brush.py`). No font is loaded anywhere.
5. **Sound** is synthesized: a felt piano, strings, horns and choir modelled voice by voice, reverbs built from synthesized noise, and every effect placed on the exact frame of the animation.
6. **Compositing** stacks the layers, grades each shot, adds bloom, film grain and dither, and streams the frames to ffmpeg.

## Repo layout

- `engine/`: the shared drawing engine and stroke painter.
- `films/<film>/`: one folder per film, with its `PROMPT.md`, its generator code, its sound code and a production log (`NOTES.md`).
- `tools/`: the render slot wrapper and the page builder.
- `docs/`: the GitHub Pages gallery.

## Rebuild a film

Python 3.12 with numpy, scipy, numba and Pillow, and ffmpeg with libx264.

```
python3 films/the-last-page/sound/score.py              # the soundtrack
python3 films/the-last-page/render.py --film            # every frame, streamed to ffmpeg
python3 films/the-last-page/render.py --deliver         # mux the sound, cut the X version
```

Each film's folder has its own notes on rendering.

## License

MIT. See [LICENSE](LICENSE).
