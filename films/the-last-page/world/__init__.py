"""THE LAST PAGE: the painted world (background department).

    from world import render
    layers = render("1", f)

render(cut_id, f) returns
    'bg':   float32 (1080, 1920, 4) premultiplied RGBA, display space
    'fg':   same shape or None, drawn over the fox
    'glow': float32 (1080, 1920, 3) or None, additive light

Plates are painted once (see each cut module's `paint` command, run through
tools/heavy.py) and cached in build/plates/. render() only samples the cached
plates through timeline.camera() and draws the animated layers.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402,F401

_MODULES = {}


def _module(cut_id):
    name = {
        "1": "cut1_dawn",
        "2": "cut2_hall",
        "5d": "cut2_hall",
        "3": "hall_family", "4a": "hall_family", "4b": "hall_family", "5a": "hall_family",
        "5b": "hall_family", "5c": "hall_family", "6a": "hall_family",
        "6b": "dome", "6c": "dome",
        "7": "sky_family", "8": "sky_family", "9a": "sky_family", "9b": "sky_family",
    }.get(cut_id)
    if cut_id == "6c" and common.have_plate("c6c_shaft"):
        name = "dome6c"     # the ray cast dome, once its plates are painted
    if name is None:
        raise NotImplementedError(f"world: cut {cut_id} is not painted yet (stage 2)")
    if name not in _MODULES:
        _MODULES[name] = __import__(name)
    return _MODULES[name]


def render(cut_id, f):
    mod = _module(cut_id)
    if hasattr(mod, "render_cut"):
        return mod.render_cut(cut_id, f)
    return mod.render(f)
