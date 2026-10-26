#!/usr/bin/env python3
# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compares two mp-arch-validate result directories with tolerances.

  python3 -I compare_reports.py BASELINE_DIR CANDIDATE_DIR [--tol-xy 0.02]

Typical: BASELINE = results/native-arm64, CANDIDATE = results/native-x86_64
(or results/rosetta-x86_64, or a released-binary baseline). Standard library
only.

Gates (exit 1 if any fails), per delegate present in BOTH directories:
  image  identical model + image SHA-256; same hand count and handedness
         labels per image; max |dx|,|dy| over all 21 landmarks <= --tol-xy
         (normalized; 0.02 = the hand tolerance compare.py uses for Web
         parity); handedness score delta <= --tol-score; both plausible.
  video  identical model + video SHA-256 and frame count; per-loop
         hand-detection rate within --tol-rate percentage points; first-loop
         frame-level agreement (same hand count) >= --min-agreement and p90
         of per-frame max |dxy| <= --tol-xy. Video inputs are decoded on each
         machine, so decoder differences legitimately add noise here.
Latency, throughput and memory are printed side by side but never gated
(different hardware). A delegate the candidate rejects at creation time
(status init_failed, e.g. the GPU guard on Intel Macs) is reported as
UNSUPPORTED; it fails the comparison unless --allow-unsupported lists it.
"""

import argparse
import json
import math
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def env_line(env):
    return (f"{env.get('execution')} | {env.get('hw_model')} | {env.get('cpu_brand')} | "
            f"macOS {env.get('os_version')} | GPU {env.get('metal_default_device')}")


def match_hands(a, b):
    """Pairs hands by handedness label when labels are unique, else by index."""
    la = [h.get("handedness") for h in a]
    lb = [h.get("handedness") for h in b]
    if len(set(la)) == len(la) and sorted(la) == sorted(lb):
        return [(h, next(x for x in b if x.get("handedness") == h.get("handedness"))) for h in a]
    return list(zip(a, b))


def max_xy(la, lb):
    worst = 0.0
    for p, q in zip(la, lb):
        if None in p[:2] or None in q[:2]:
            return math.inf
        worst = max(worst, abs(p[0] - q[0]), abs(p[1] - q[1]))
    return worst


def unsupported(base, cand, args, out):
    """Returns True/False if the candidate rejected the delegate, else None."""
    if cand.get("status") != "init_failed":
        return None
    allowed = cand.get("delegate") in args.allow_unsupported
    out.append(f"  UNSUPPORTED on candidate ({'allowed' if allowed else 'FAIL'}): {cand.get('error')}")
    return allowed


def compare_image(base, cand, args, out):
    ok = True
    u = unsupported(base, cand, args, out)
    if u is not None:
        return u
    if base.get("status") != "ok" or cand.get("status") != "ok":
        out.append(f"  FAIL status baseline={base.get('status')} candidate={cand.get('status')}")
        return False
    if base["model"]["sha256"] != cand["model"]["sha256"]:
        out.append("  FAIL model sha256 differs")
        ok = False
    cand_images = {i["image"]: i for i in cand["images"]}
    for bi in base["images"]:
        ci = cand_images.get(bi["image"])
        name = bi["image"]
        if ci is None:
            out.append(f"  FAIL {name}: missing in candidate")
            ok = False
            continue
        if bi["sha256"] != ci["sha256"]:
            out.append(f"  FAIL {name}: input sha256 differs")
            ok = False
        bh, ch = bi["hands"], ci["hands"]
        labels_b = [h["handedness"] for h in bh]
        labels_c = [h["handedness"] for h in ch]
        line = f"  {name:30s} hands {len(bh)}/{len(ch)} [{','.join(labels_b)}]/[{','.join(labels_c)}]"
        if len(bh) != len(ch) or sorted(labels_b) != sorted(labels_c):
            out.append(line + "  FAIL hand count/handedness")
            ok = False
            continue
        worst_xy = worst_z = worst_world = worst_score = 0.0
        for hb, hc in match_hands(bh, ch):
            worst_xy = max(worst_xy, max_xy(hb["landmarks"], hc["landmarks"]))
            worst_z = max(worst_z, max(abs(p[2] - q[2]) for p, q in zip(hb["landmarks"], hc["landmarks"])))
            worst_world = max(worst_world, max(
                max(abs(p[k] - q[k]) for k in range(3))
                for p, q in zip(hb["world_landmarks"], hc["world_landmarks"])))
            worst_score = max(worst_score, abs(hb["handedness_score"] - hc["handedness_score"]))
        verdict = []
        if worst_xy > args.tol_xy:
            verdict.append(f"FAIL xy>{args.tol_xy}")
        if worst_score > args.tol_score:
            verdict.append(f"FAIL score>{args.tol_score}")
        for tag, img in (("baseline", bi), ("candidate", ci)):
            if not img["plausibility"]["pass"]:
                verdict.append(f"FAIL implausible({tag}: {img['plausibility']['failures']})")
        ok = ok and not verdict
        out.append(f"{line} max|dxy|={worst_xy:.5f} max|dz|={worst_z:.5f} "
                   f"max|dworld|={worst_world:.5f}m d_score={worst_score:.4f} "
                   f"{' '.join(verdict) or 'PASS'}")
        bl, cl = bi["latency_ms"], ci["latency_ms"]
        out.append(f"  {'':30s} latency mean {bl['mean']:.2f} / {cl['mean']:.2f} ms "
                   f"(p90 {bl['p90']:.2f} / {cl['p90']:.2f})")
    return ok


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * (len(s) - 1) + 0.5))]


def compare_video(base, cand, args, out):
    u = unsupported(base, cand, args, out)
    if u is not None:
        return u
    if base.get("status") != "ok" or cand.get("status") != "ok":
        out.append(f"  FAIL status baseline={base.get('status')} candidate={cand.get('status')}")
        return False
    ok = True
    if base["model"]["sha256"] != cand["model"]["sha256"] or base["video"]["sha256"] != cand["video"]["sha256"]:
        out.append("  FAIL model/video sha256 differs")
        ok = False
    for bl, cl in zip(base["per_loop"], cand["per_loop"]):
        if bl["frames"] != cl["frames"]:
            out.append(f"  FAIL loop {bl['loop']}: frames {bl['frames']} vs {cl['frames']}")
            ok = False
            continue
        rb = 100 * bl["frames_with_hand"] / max(bl["frames"], 1)
        rc = 100 * cl["frames_with_hand"] / max(cl["frames"], 1)
        tb = 100 * bl["frames_with_two_hands"] / max(bl["frames"], 1)
        tc = 100 * cl["frames_with_two_hands"] / max(cl["frames"], 1)
        bad = abs(rb - rc) > args.tol_rate
        ok = ok and not bad
        out.append(f"  loop {bl['loop']}: hand-rate {rb:.1f}% / {rc:.1f}%  two-hands {tb:.1f}% / {tc:.1f}%"
                   f"{'  FAIL rate' if bad else ''}")
    agree = 0
    deltas = []
    fb, fc = base["first_loop_frames"], cand["first_loop_frames"]
    for a, b in zip(fb, fc):
        if len(a["hands"]) != len(b["hands"]):
            continue
        agree += 1
        if not a["hands"]:
            continue
        pairs = match_hands([{"handedness": h["h"], **h} for h in a["hands"]],
                            [{"handedness": h["h"], **h} for h in b["hands"]])
        worst = 0.0
        for ha, hb in pairs:
            xa, xb = ha["xy"], hb["xy"]
            worst = max(worst, max(abs(p - q) for p, q in zip(xa, xb)))
        deltas.append(worst)
    agreement = agree / max(len(fb), 1)
    p50, p90, mx = pct(deltas, 50), pct(deltas, 90), (max(deltas) if deltas else float("nan"))
    bad_agree = agreement < args.min_agreement
    bad_dxy = not (p90 <= args.tol_xy)
    ok = ok and not bad_agree and not bad_dxy
    out.append(f"  first loop: hand-count agreement {100 * agreement:.1f}%"
               f"{'  FAIL' if bad_agree else ''}; per-frame max|dxy| p50 {p50:.4f} p90 {p90:.4f} "
               f"max {mx:.4f}{'  FAIL p90' if bad_dxy else ''}")
    for tag, r in (("baseline ", base), ("candidate", cand)):
        m = r["memory"]
        slope = m.get("slope_mb_per_1000_frames_after_loop0")
        out.append(f"  {tag}: {r['frames_total']} frames, latency mean {r['latency_ms']['mean']:.2f} "
                   f"p50 {r['latency_ms']['p50']:.2f} p90 {r['latency_ms']['p90']:.2f} "
                   f"p99 {r['latency_ms']['p99']:.2f} max {r['latency_ms']['max']:.1f} ms; "
                   f"{r['throughput_fps_inference']:.1f} fps; first frame {r['first_frame_ms']:.1f} ms; "
                   f"footprint {m['footprint_mb_after_loop0']:.0f}->{m['footprint_mb_end']:.0f} MB "
                   f"(peak {m['footprint_mb_peak']:.0f}, slope "
                   f"{'n/a' if slope is None else f'{slope:+.2f}'} MB/1k frames)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--tol-xy", type=float, default=0.02)
    ap.add_argument("--tol-score", type=float, default=0.05)
    ap.add_argument("--tol-rate", type=float, default=3.0)
    ap.add_argument("--min-agreement", type=float, default=0.95)
    ap.add_argument("--allow-unsupported", default="", type=lambda v: [d for d in v.split(",") if d],
                    help="comma-separated delegates the candidate may reject (e.g. gpu)")
    args = ap.parse_args()

    out = []
    all_ok = True
    eb = load(os.path.join(args.baseline, "env.json"))
    ec = load(os.path.join(args.candidate, "env.json"))
    out.append(f"baseline : {env_line(eb)}")
    out.append(f"candidate: {env_line(ec)}")
    compared = 0
    for name in sorted(os.listdir(args.baseline)):
        if not name.endswith(".json") or name == "env.json":
            continue
        bp, cp = os.path.join(args.baseline, name), os.path.join(args.candidate, name)
        if not os.path.exists(cp):
            out.append(f"\n## {name}: not in candidate (skipped)")
            continue
        base, cand = load(bp), load(cp)
        out.append(f"\n## {name}")
        fn = compare_image if base.get("command") == "image" else compare_video
        ok = fn(base, cand, args, out)
        out.append(f"  => {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
        compared += 1
    out.append(f"\nOVERALL: {'PASS' if all_ok and compared else 'FAIL'} ({compared} report pair(s))")
    print("\n".join(out))
    return 0 if all_ok and compared else 1


if __name__ == "__main__":
    sys.exit(main())
