"""Checkpoint 1: model sheet and two finished frames (shot 1, shot 2).

python3 look_test.py sheet|shot1|shot2
"""

import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tr import raster as R
from tr import bg as B
from tr import post
from tr.cel import Cel
from tr.panda import Pose, Transform, draw_head, pose_sit_front, pose_stand, pose_look_back, pose_sleep
from tr import scenes

OUT = f"{ROOT}/build/review"
W2, H2 = scenes.W2, scenes.H2


def model_sheet():
    # warm paper
    paper = np.full((H2, W2, 3), 0.0, np.float32)
    paper[..., 0] = 0.93
    paper[..., 1] = 0.89
    paper[..., 2] = 0.80
    paper *= B.paper(H2, W2, 5, 0.05)[..., None]
    cel = Cel(W2, H2)
    light = (-0.6, -0.8)
    # row 1: five poses
    pose_sit_front(cel, Transform(420, 520, 150), Pose(seed=1, mouth="smile", look=(0.1, 0.1)), light)
    pose_stand(cel, Transform(1480, 560, 150), Pose(seed=2, yaw=0.6, mouth="smile"), light, stride=0.6)
    pose_stand(cel, Transform(2060, 560, 150, flip=True), Pose(seed=3, yaw=0.35, pitch=-0.3, mouth="o", mouth_open=0.4, goggles=True, brow=0.6), light)
    pose_look_back(cel, Transform(2800, 520, 150), Pose(seed=4, yaw=0.95, pitch=-0.7, mouth="line"), light)
    pose_sleep(cel, Transform(3180, 1380, 150), Pose(seed=5, yaw=0.5, pitch=0.3, eye_open=0.0, mouth="line", ear_r=0.2), light)
    # row 2: expressions, head only, larger
    heads = [
        Pose(seed=11, yaw=-0.3, pitch=0.3, mouth="teeth", look=(-0.3, 0.4), brow=0.4),          # the knot
        Pose(seed=12, yaw=0.0, squeeze=1.0, mouth="line", cheek=0.3),                             # eyes squeezed shut
        Pose(seed=13, yaw=0.0, eye_open_l=0.0, eye_open_r=1.0, mouth="line", look=(0.2, -0.2)),  # one eye open
        Pose(seed=14, yaw=0.2, pitch=-0.4, mouth="o", mouth_open=1.0, eye_open=1.0, look=(0.0, -0.3), goggles=True, goggles_up=False),  # wonder
        Pose(seed=15, yaw=0.0, mouth="smile", brow=0.9, look=(0, 0), goggles=True, goggles_up=False),  # determined nod
    ]
    for i, p in enumerate(heads):
        draw_head(cel, Transform(360 + i * 520, 1600, 215), 0, 0, p, light)
    frame = paper.copy()
    cel.over(frame)
    out = post.finish(frame, 5, grain=0.02, vig=0.0, do_bloom=False)
    post.save(out, f"{OUT}/model-sheet.png")


def shot2():
    t0 = time.time()
    bgimg, masks = scenes.paint_hill(2)
    print("bg painted", time.time() - t0, flush=True)
    np.save(f"{ROOT}/build/cache/shot2_bg.npy", bgimg)
    frame = bgimg.copy()
    cel = scenes.shot2_cels()
    # cels take the dusk: slightly warmer and darker as the light fades, violet in the shadows
    cel.over(frame, tint=(1.0, 0.92, 0.88))
    scenes.rim_light(frame, cel, (0.9, 0.55, 0.2), -0.8, -0.5, strength=0.6, width=5, ink_w=2)
    # soft dusk haze over the far valley
    frame = post.grade(frame, lift=(0.01, 0.0, 0.03), gain=(1.0, 0.97, 0.98), gamma=1.0, sat=1.08)
    out = post.finish(frame, 2, grain=0.03, vig=0.3, do_bloom=True, bloom_kw=dict(threshold=0.7, sigma=30, strength=0.3))
    post.save(out, f"{OUT}/shot2-hill.png")
    print("shot2 done", time.time() - t0, flush=True)


def shot1():
    t0 = time.time()
    bgimg = scenes.paint_closeup_bg(1)
    np.save(f"{ROOT}/build/cache/shot1_bg.npy", bgimg)
    print("bg painted", time.time() - t0, flush=True)
    frame = bgimg.copy()
    cel = scenes.shot1_cels()
    cel.over(frame, tint=(1.0, 0.94, 0.90))
    scenes.rim_light(frame, cel, (1.0, 0.70, 0.30), 0.8, -0.5, strength=0.9, width=16, ink_w=7)
    frame = post.grade(frame, lift=(0.02, 0.0, 0.0), gain=(1.02, 0.97, 0.92), sat=1.05)
    out = post.finish(frame, 1, grain=0.03, vig=0.32, do_bloom=True, bloom_kw=dict(threshold=0.72, sigma=40, strength=0.35))
    post.save(out, f"{OUT}/shot1-knot.png")
    print("shot1 done", time.time() - t0, flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    {"sheet": model_sheet, "shot1": shot1, "shot2": shot2}[what]()
