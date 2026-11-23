#!/usr/bin/env python3
"""Runs the palm-size sweep: ffmpeg builds each synthetic clip (area-filtered
shrink onto a full-size canvas, optional degradation) and pipes raw BGRA into
palm-scale-sweep, which writes one JSON line per frame.

    python3 sweep.py run --set gt|image|video|degraded|gpu [--scales ...] [--jobs N]

Raw output: $SWEEP_OUT/raw/<config>.jsonl (default /tmp/palmsweep).
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "test_projects", "shared"))
MODEL = os.path.join(SHARED, "models", "hand_landmarker.task")
TOOL = os.path.join(HERE, ".build", "release", "palm-scale-sweep")
OUT = os.environ.get("SWEEP_OUT", "/tmp/palmsweep")
RAW = os.path.join(OUT, "raw")

W, H = 1280, 720
FPS = 100000 / 3333
CLIPS = {"two-hands": os.path.join(SHARED, "two-hands-only.mov"),
         "rotation": os.path.join(SHARED, "media-pipe-rotation-test.mov")}
BACKGROUNDS = {"black": None, "room": os.path.join(OUT, "bg_room.png")}
SCALES = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09]
POSITIONS = [f"{r}{c}" for r in "tmb" for c in "lcr"]  # t/m/b row, l/c/r column

# Degradation, applied to the composed canvas in this order.
BLUR_SIGMA = 1.0          # Gaussian blur sigma, canvas pixels
NOISE_STRENGTH = 10       # ffmpeg noise=alls (temporal, Gaussian)
H264_BITRATE = "2M"       # libx264, preset veryfast, yuv420p, 720p30


def geometry(scale, pos):
    w = round(W * scale)
    h = round(H * scale)
    xs = {"l": 0, "c": (W - w) // 2, "r": W - w}
    ys = {"t": 0, "m": (H - h) // 2, "b": H - h}
    return w, h, xs[pos[1]], ys[pos[0]]


def config_name(clip, scale, pos, bg, mode, delegate, degraded):
    # degraded: "" (clean), "all" (blur + noise + H.264) or one of "blur", "noise", "h264".
    suffix = "" if not degraded else ("_deg" if degraded == "all" else f"_deg-{degraded}")
    return f"{clip}_s{scale:g}_{pos}_{bg}_{mode}_{delegate}" + suffix


def compose_filter(scale, pos, bg, degraded):
    w, h, x, y = geometry(scale, pos)
    fg = (f"[0:v]scale=in_color_matrix=bt709:in_range=tv,format=rgb24,"
          f"scale={w}:{h}:flags=area,pad={W}:{H}:{x}:{y}:color=black")
    if bg == "black":
        chain = fg + "[c]"
    else:
        # The still goes on top with a transparent hole where the clip sits;
        # overlay repeats the single still frame for every clip frame.
        chain = (fg + "[m];[1:v]format=rgba,"
                 f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black@0:t=fill:replace=1[o];"
                 "[m][o]overlay=0:0:format=rgb[c]")
    tail = "[c]"
    filters = []
    if degraded in ("all", "blur"):
        filters.append(f"gblur=sigma={BLUR_SIGMA}")
    if degraded in ("all", "noise"):
        filters.append(f"noise=alls={NOISE_STRENGTH}:allf=t")
    if filters:
        chain += ";[c]" + ",".join(filters) + "[d]"
        tail = "[d]"
    return chain, tail


def run_config(clip, scale, pos, bg, mode, delegate, degraded, frame_step=1):
    name = config_name(clip, scale, pos, bg, mode, delegate, degraded)
    out = os.path.join(RAW, name + ".jsonl")
    if os.path.exists(out) and os.path.getsize(out) > 0 and not os.path.exists(out + ".partial"):
        return name, "cached", 0.0
    open(out + ".partial", "w").close()
    chain, tail = compose_filter(scale, pos, bg, degraded)
    inputs = ["-i", CLIPS[clip]]
    if bg != "black":
        inputs += ["-i", BACKGROUNDS[bg]]
    select = f",select='not(mod(n\\,{frame_step}))',setpts=N/FRAME_RATE/TB" if frame_step > 1 else ""
    start = time.time()
    encode = degraded in ("all", "h264")
    if encode:
        mp4 = os.path.join(OUT, "tmp", name + ".mp4")
        os.makedirs(os.path.dirname(mp4), exist_ok=True)
        enc = (["ffmpeg", "-v", "error", "-y"] + inputs +
               ["-filter_complex", chain + f";{tail}scale=out_color_matrix=bt709:out_range=tv,format=yuv420p{select}[e]",
                "-map", "[e]", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-c:v", "libx264", "-preset", "veryfast", "-b:v", H264_BITRATE,
                "-maxrate", H264_BITRATE, "-bufsize", "4M", "-an", mp4])
        subprocess.run(enc, check=True)
        src = (f"ffmpeg -v error -i {shlex.quote(mp4)} -vf "
               "scale=in_color_matrix=bt709:in_range=tv,format=bgra -f rawvideo -pix_fmt bgra -")
    else:
        fc = chain + f";{tail}format=bgra{select}[e]"
        src = ("ffmpeg -v error " + " ".join(shlex.quote(a) for a in inputs) +
               f" -filter_complex {shlex.quote(fc)} -map '[e]' -f rawvideo -pix_fmt bgra -")
    tool = (f"{shlex.quote(TOOL)} --model {shlex.quote(MODEL)} --width {W} --height {H} "
            f"--mode {mode} --delegate {delegate} --fps {FPS / frame_step} --candidates 50 "
            f"--out {shlex.quote(out)}")
    log = os.path.join(OUT, "logs", name + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as lf:
        rc = subprocess.run(f"set -o pipefail; {src} | {tool}", shell=True, stderr=lf,
                            executable="/bin/bash").returncode
    if rc != 0:
        return name, f"FAILED rc={rc} (see {log})", time.time() - start
    if encode:
        os.remove(mp4)
    os.remove(out + ".partial")
    return name, "ok", time.time() - start


def configs_for(args):
    scales = [float(s) for s in args.scales.split(",")] if args.scales else SCALES
    positions = args.positions.split(",") if args.positions else POSITIONS
    bgs = args.backgrounds.split(",") if args.backgrounds else list(BACKGROUNDS)
    out = []
    if args.set == "gt":
        return [(args.clip, 1.0, "mc", "black", "image", "cpu", "")]
    mode = "video" if args.set == "video" else "image"
    delegate = "gpu" if args.set == "gpu" else "cpu"
    degraded = (args.degrade or "all") if args.set == "degraded" else ""
    for scale, pos, bg in itertools.product(scales, positions, bgs):
        if scale == 1.0 and (pos != "mc" or bg != "black"):
            continue  # at full size the canvas is the frame itself
        out.append((args.clip, scale, pos, bg, mode, delegate, degraded))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "list"])
    ap.add_argument("--set", required=True, choices=["gt", "image", "video", "degraded", "gpu"])
    ap.add_argument("--clip", default="two-hands", choices=list(CLIPS))
    ap.add_argument("--scales")
    ap.add_argument("--positions")
    ap.add_argument("--backgrounds")
    ap.add_argument("--degrade", choices=["all", "blur", "noise", "h264"])
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    configs = configs_for(args)
    if args.command == "list":
        for c in configs:
            print(config_name(*c))
        return
    print(f"{len(configs)} configs, {args.jobs} jobs", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_config, *c, frame_step=args.frame_step) for c in configs]
        for n, f in enumerate(futures, 1):
            name, status, secs = f.result()
            print(f"[{n}/{len(configs)}] {name} {status} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
