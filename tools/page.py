#!/usr/bin/env python3
"""Build the web pages for the films.

Each film gets a page that is only a black cinema frame with one quiet play
button (browsers block autoplay with sound). The gallery lists every film.

    python3 tools/page.py film the-last-page      # docs/the-last-page/index.html
    python3 tools/page.py single the-last-page    # out/the-last-page.html, the film embedded in one file
    python3 tools/page.py gallery                 # docs/index.html

Film facts live in FILMS below. The page expects the web encode at
docs/<slug>/film.mp4 and a poster still at docs/<slug>/poster.jpg.
"""

import base64
import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/jayypatel18/code-films"

FILMS = {
    "the-last-page": {
        "title": "THE LAST PAGE",
        "line": "A small fox wakes a library that holds every story ever written.",
        "model": "Opus 5.5",
        "seconds": 60,
        "prompt_by": "@chetaslua",
    },
    "tiny-rocket": {
        "title": "TINY ROCKET",
        "line": "A red panda builds a rocket out of paper and scrap.",
        "model": "Fable 5.1",
        "seconds": 65,
        "prompt_by": None,
    },
}

PLAYER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
{og_image}
<link rel="icon" href="data:,">
<style>
  html, body {{ margin: 0; height: 100%; background: #000; overflow: hidden; }}
  body {{ display: grid; place-items: center; }}
  .frame {{ position: relative; width: min(100vw, calc(100vh * 16 / 9)); aspect-ratio: 16 / 9; background: #000; }}
  video {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; background: #000; cursor: pointer; }}
  button {{
    position: absolute; left: 50%; top: 50%; width: 68px; height: 68px; margin: -34px 0 0 -34px;
    border-radius: 50%; border: 1px solid rgba(255, 255, 255, 0.34); background: rgba(0, 0, 0, 0.28);
    padding: 0; cursor: pointer; transition: opacity 0.6s ease, border-color 0.3s ease, background 0.3s ease;
    -webkit-tap-highlight-color: transparent;
  }}
  button:hover, button:focus-visible {{ border-color: rgba(255, 255, 255, 0.7); background: rgba(0, 0, 0, 0.42); outline: none; }}
  button svg {{ display: block; margin: 0 auto; }}
  .playing button {{ opacity: 0; pointer-events: none; }}
</style>
</head>
<body>
<div class="frame" id="frame">
  <video id="film" src="{src}" {poster} preload="metadata" playsinline></video>
  <button id="play" aria-label="Play with sound">
    <svg width="22" height="24" viewBox="0 0 22 24" aria-hidden="true"><path d="M3 2.5v19l16-9.5z" fill="rgba(255,255,255,0.82)"/></svg>
  </button>
</div>
<script>
  const film = document.getElementById('film');
  const frame = document.getElementById('frame');
  const play = document.getElementById('play');
  function toggle() {{
    if (film.paused || film.ended) {{ film.muted = false; film.play(); }} else {{ film.pause(); }}
  }}
  play.addEventListener('click', toggle);
  film.addEventListener('click', toggle);
  film.addEventListener('play', () => frame.classList.add('playing'));
  film.addEventListener('pause', () => frame.classList.remove('playing'));
  film.addEventListener('ended', () => frame.classList.remove('playing'));
  document.addEventListener('keydown', (e) => {{
    if (e.code === 'Space' || e.key === 'k') {{ e.preventDefault(); toggle(); }}
  }});
</script>
</body>
</html>
"""

GALLERY = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Code Films</title>
<meta name="description" content="Short anime films where every frame and every sound is code a model wrote.">
<link rel="icon" href="data:,">
<style>
  :root {{ color-scheme: dark; }}
  html, body {{ margin: 0; background: #000; color: #e9e4da; }}
  body {{ font: 16px/1.55 ui-serif, Georgia, "Times New Roman", serif; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 56px 16px 72px; }}
  h1 {{ font-weight: 400; font-size: 30px; letter-spacing: 0.06em; margin: 0 0 8px; }}
  .lede {{ color: #a39d92; max-width: 640px; margin: 0 0 40px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 28px; }}
  a.card {{ color: inherit; text-decoration: none; display: block; }}
  .still {{ aspect-ratio: 16 / 9; background: #111 center / cover no-repeat; border-radius: 2px; transition: filter 0.3s ease; }}
  a.card:hover .still, a.card:focus-visible .still {{ filter: brightness(1.12); }}
  a.card:focus-visible {{ outline: 1px solid #e9e4da; outline-offset: 6px; }}
  .name {{ margin: 14px 0 2px; letter-spacing: 0.08em; font-size: 15px; }}
  .meta {{ color: #8f897e; font-size: 14px; }}
  footer {{ color: #8f897e; font-size: 14px; margin-top: 56px; max-width: 720px; }}
  footer a {{ color: #c9c2b5; }}
</style>
</head>
<body>
<main>
  <h1>CODE FILMS</h1>
  <p class="lede">Short anime films where every frame and every sound is code a model wrote. No image models, no stock, no samples, no font files. Each film's generator code and prompt are in the repo.</p>
  <div class="grid">
{cards}
  </div>
  <footer>
    <p>The idea and the original prompt come from <a href="https://x.com/chetaslua">@chetaslua</a>, who made the first films like this (<a href="https://x.com/chetaslua/status/2117169796290916732">film one</a>, <a href="https://x.com/chetaslua/status/2117224839010459987">THE LAST PAGE</a>).</p>
    <p>Code and films: <a href="{repo}">{repo_short}</a>, MIT license. Made for <a href="https://x.com/jayypatel18">@jayypatel18</a>.</p>
  </footer>
</main>
</body>
</html>
"""

CARD = """    <a class="card" href="{slug}/">
      <div class="still" style="background-image: url('{slug}/poster.jpg')"></div>
      <div class="name">{title}</div>
      <div class="meta">{line}<br>{model} · {seconds} s</div>
    </a>"""


def player(slug, src, poster_src):
    f = FILMS[slug]
    desc = f"{f['line']} Every frame and every sound is code {f['model']} wrote."
    return PLAYER.format(
        title=html.escape(f["title"].title()),
        desc=html.escape(desc),
        src=src,
        poster=f'poster="{poster_src}"' if poster_src else "",
        og_image=f'<meta property="og:image" content="{poster_src}">' if poster_src and not poster_src.startswith("data:") else "",
    )


def build_film(slug):
    out_dir = os.path.join(ROOT, "docs", slug)
    os.makedirs(out_dir, exist_ok=True)
    poster = "poster.jpg" if os.path.exists(os.path.join(out_dir, "poster.jpg")) else None
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(player(slug, "film.mp4", poster))
    print("wrote", path)


def build_single(slug, mp4=None):
    mp4 = mp4 or os.path.join(ROOT, "docs", slug, "film.mp4")
    with open(mp4, "rb") as fh:
        data = "data:video/mp4;base64," + base64.b64encode(fh.read()).decode("ascii")
    poster_path = os.path.join(ROOT, "docs", slug, "poster.jpg")
    poster = None
    if os.path.exists(poster_path):
        with open(poster_path, "rb") as fh:
            poster = "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode("ascii")
    out_dir = os.path.join(ROOT, "films", slug, "out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(player(slug, data, poster))
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def build_gallery():
    cards = []
    for slug, f in FILMS.items():
        if not os.path.exists(os.path.join(ROOT, "docs", slug, "index.html")):
            continue
        cards.append(CARD.format(slug=slug, title=html.escape(f["title"]), line=html.escape(f["line"]),
                                 model=html.escape(f["model"]), seconds=f["seconds"]))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(GALLERY.format(cards="\n".join(cards), repo=REPO_URL,
                                repo_short=REPO_URL.replace("https://", "")))
    open(os.path.join(ROOT, "docs", ".nojekyll"), "w").close()
    print("wrote", path, f"({len(cards)} films)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "film":
        build_film(sys.argv[2])
    elif cmd == "single":
        build_single(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "gallery":
        build_gallery()
    else:
        sys.exit(__doc__)
