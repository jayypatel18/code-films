#!/usr/bin/env python3
"""Run a heavy command while holding one of the machine's render slots.

The build machine is an 8 core laptop with 16 GB of RAM, so at most two heavy
jobs (frame renders, background painting, long audio renders) run at once.
Every heavy job goes through this wrapper:

    python3 tools/heavy.py -- python3 films/the-last-page/render.py --all

It waits for a free slot, sets thread counts so two jobs share the cores
fairly, runs the command, and frees the slot when the command exits.

Options:
    --slots N     number of slots on the machine (default 2)
    --threads N   threads per job for numba, numpy and BLAS (default 4)
    --name TEXT   label shown in the slot file (default: the command)
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time

LOCK_DIR = "/tmp/code-films-slots"


def acquire(slots, label):
    os.makedirs(LOCK_DIR, exist_ok=True)
    waited = 0.0
    announced = False
    while True:
        for i in range(slots):
            path = os.path.join(LOCK_DIR, f"slot{i}.lock")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {time.strftime('%H:%M:%S')} {label}\n".encode())
            if announced:
                print(f"[heavy] got slot {i} after {waited:.0f} s", file=sys.stderr, flush=True)
            return fd
        if not announced:
            print("[heavy] all slots busy, waiting. Holders:", file=sys.stderr, flush=True)
            for i in range(slots):
                try:
                    with open(os.path.join(LOCK_DIR, f"slot{i}.lock")) as fh:
                        print("   ", fh.read().strip(), file=sys.stderr, flush=True)
                except OSError:
                    pass
            announced = True
        time.sleep(2.0)
        waited += 2.0


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("give a command after --")
    label = args.name or " ".join(cmd)[:160]
    fd = acquire(args.slots, label)
    env = dict(os.environ)
    t = str(args.threads)
    for key in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = t
    start = time.time()
    try:
        code = subprocess.call(cmd, env=env)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    print(f"[heavy] done in {time.time() - start:.0f} s, exit {code}", file=sys.stderr, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
