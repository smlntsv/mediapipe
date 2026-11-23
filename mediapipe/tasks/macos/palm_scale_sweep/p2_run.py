#!/usr/bin/env python3
"""Runs palm search configurations on the phase 2 scenes (p2_scenes.py).

    python3 p2_run.py --set border|nearfar|crowd|timing [--jobs N] [--delegate cpu|gpu|coreml]
                      [--scenes a,b] [--only substring] [--list]

Every configuration of a scene reads the same lossless file. Output:
$P2_OUT/raw/<scene>__<config>[__<delegate>].jsonl, one line per frame
(palm_search_dump). The `timing` set is meant to be run with --jobs 1.

Configuration names: off (feature off), full (full frame through the search
path), g<rows>x<columns>_o<overlap %>_<all|rrK> plus _tr (truncated
detections rejected), _best (best-score merge), _nofull (no full-frame
region), _clipb (clip to numHands before the association).
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SHARED = os.path.normpath(os.path.join(HERE, "..", "test_projects", "shared"))
MODEL = os.path.join(SHARED, "models", "hand_landmarker.task")
COREML_CACHE = os.path.join(SHARED, "models", "coreml_models")
TOOL = os.path.join(ROOT, "bazel-bin", "mediapipe", "tasks", "macos", "palm_scale_sweep", "palm_search_dump")
OUT = os.environ.get("P2_OUT", "/tmp/palmp2")
W, H, FPS = 1280, 720, 100000 / 3333

GRIDS = ["2x2", "2x3", "3x3", "3x4", "4x4"]   # rows x columns


def grid(g, overlap, schedule="all", tr=False, best=False, nofull=False, clipb=False):
    name = f"g{g}_o{round(overlap * 100):02d}_{schedule}"
    args = ["--search", "on", "--grid", g, "--overlap", str(overlap)]
    if schedule != "all":
        args += ["--rr", schedule[2:]]
    if tr:
        name += "_tr"
        args += ["--reject-truncated", "1"]
    if best:
        name += "_best"
        args += ["--merge", "best"]
    if nofull:
        name += "_nofull"
        args += ["--full-frame", "0"]
    if clipb:
        name += "_clipb"
        args += ["--clip-before", "1"]
    return name, args


OFF = ("off", [])
FULL = ("full", ["--search", "on"])
FULL_CLIPB = ("full_clipb", ["--search", "on", "--clip-before", "1"])


def configs(which):
    out = [OFF, FULL]
    if which == "border":
        for g in GRIDS:
            for overlap in (0, 0.1, 0.2, 0.3):
                out += [grid(g, overlap), grid(g, overlap, tr=True)]
            out += [grid(g, 0.2, best=True), grid(g, 0.2, nofull=True)]
    elif which == "nearfar":
        for g in GRIDS:
            for overlap in (0, 0.2):
                out += [grid(g, overlap), grid(g, overlap, "rr1")]
            out += [grid(g, 0.2, tr=True)]
        for g in ("3x4", "4x4"):
            out += [grid(g, 0.2, "rr4"), grid(g, 0.2, clipb=True), grid(g, 0.2, nofull=True)]
    elif which == "crowd":
        out += [FULL_CLIPB]
        for g in GRIDS:
            out += [grid(g, 0.2), grid(g, 0.2, clipb=True)]
        for g in ("3x4", "4x4"):
            out += [grid(g, 0.2, "rr1"), grid(g, 0.2, tr=True)]
    elif which == "timing":
        for g in GRIDS:
            out += [grid(g, 0.2), grid(g, 0.2, "rr1")]
    return out


SETS = {
    "border": (["border_s018", "border_s035"], "image"),
    "nearfar": (["nearfar_s018", "nearfar_s025", "faronly_s018"], "video"),
    "crowd": (["crowd4", "crowd10"], "video"),
    "timing": (["nearfar_s018"], "video"),
}


def run(scene, mode, name, args, delegate, suffix):
    out = os.path.join(OUT, "raw", f"{scene}__{name}{suffix}.jsonl")
    if os.path.exists(out) and os.path.getsize(out) > 0 and not os.path.exists(out + ".partial"):
        return out, "cached", 0.0
    open(out + ".partial", "w").close()
    src = (f"ffmpeg -v error -i {shlex.quote(os.path.join(OUT, 'scenes', scene + '.mkv'))} "
           "-f rawvideo -pix_fmt bgra -")
    extra = ["--delegate", delegate]
    if delegate == "coreml":
        extra += ["--coreml-cache", COREML_CACHE]
    tool = " ".join(shlex.quote(a) for a in
                    [TOOL, "--model", MODEL, "--width", str(W), "--height", str(H), "--mode", mode,
                     "--fps", str(FPS), "--num-hands", "2", "--out", out] + extra + args)
    log = os.path.join(OUT, "logs", os.path.basename(out) + ".log")
    start = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(f"set -o pipefail; {src} | {tool}", shell=True, stderr=lf,
                            executable="/bin/bash").returncode
    if rc != 0:
        return out, f"FAILED rc={rc} (see {log})", time.time() - start
    os.remove(out + ".partial")
    return out, "ok", time.time() - start


for _scenes, _ in SETS.values():
    _scenes += [s + "_wall" for s in _scenes]      # the same layers on the plain wall
SETS["timing"] = (["faronly_s018_wall"], "video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--delegate", default="cpu")
    ap.add_argument("--scenes")
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    scenes, mode = SETS[args.set]
    if args.scenes:
        scenes = args.scenes.split(",")
    suffix = "" if args.set != "timing" else f"__t_{args.delegate}"
    if args.set != "timing" and args.delegate != "cpu":
        suffix = f"__{args.delegate}"
    work = [(s, mode, n, a, args.delegate, suffix) for s in scenes for n, a in configs(args.set)
            if not args.only or args.only in n]
    if args.list:
        print("\n".join(f"{w[0]}__{w[2]}{suffix}" for w in work))
        return
    for d in ("raw", "logs"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for out, status, seconds in pool.map(lambda w: run(*w), work):
            failed += status.startswith("FAILED")
            print(f"{os.path.basename(out)}: {status} {seconds:.0f}s", flush=True)
    print(f"done, {len(work)} runs, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
