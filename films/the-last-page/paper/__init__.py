"""THE LAST PAGE: the paper department (everything made of paper that flies).

    import paper
    layers = paper.render("8", f)

render(cut_id, f) returns
    'back':  float32 (1080, 1920, 4) premultiplied RGBA or None, behind the fox
    'front': same or None, in front of the fox
    'glow':  float32 (1080, 1920, 3) or None, additive light

Modules: raster (numba batch rasterizer), pages (the page atlas), bird (the
small paper bird: fold states and flap keys), great (the great bird of pages).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_FILM = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _FILM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PAPER_CUTS = ("6a", "6b", "6c", "7", "8", "9a", "9b")


def render(cut_id, f):
    empty = {"back": None, "front": None, "glow": None}
    if cut_id not in PAPER_CUTS:
        return empty
    if cut_id == "8":
        from . import scene8b
        return scene8b.render(f)
    from . import cuts
    return cuts.RENDER[cut_id](f)
