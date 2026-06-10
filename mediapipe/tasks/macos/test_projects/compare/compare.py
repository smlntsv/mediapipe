#!/usr/bin/env python3
# Copyright 2025 The MediaPipe Authors.
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
"""Compare MediaPipe parity outputs from the web and native macOS projects.

Reads two JSON envelopes (default: ../shared/output/{web,macos}.json) and:
  * verifies detection counts and per-instance landmark counts
    (hand 21, pose 33, face 478);
  * compares normalized x/y with a tolerance and reports max/mean abs diff;
  * compares hand handedness labels;
  * reports z and world-landmark differences (report-only by default).

Exit code is non-zero if any *enforced* check (counts, x/y tolerance, handedness
labels) fails. z / world-landmark diffs are report-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

EXPECTED = {"hand": 21, "pose": 33, "face": 478}
# Key holding the per-instance landmark arrays for each task.
LANDMARK_KEY = {"hand": "landmarks", "pose": "landmarks", "face": "faceLandmarks"}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEB = os.path.join(HERE, "..", "shared", "output", "web.json")
DEFAULT_MACOS = os.path.join(HERE, "..", "shared", "output", "macos.json")


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, msg: str) -> None:
        print(f"  \033[32mPASS\033[0m {msg}")

    def fail(self, msg: str) -> None:
        self.failed = True
        print(f"  \033[31mFAIL\033[0m {msg}")

    def info(self, msg: str) -> None:
        print(f"       {msg}")


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def abs_diffs(web_pts: list, mac_pts: list, axis: str) -> list[float]:
    n = min(len(web_pts), len(mac_pts))
    return [abs(float(web_pts[i][axis]) - float(mac_pts[i][axis])) for i in range(n)]


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (max(values), sum(values) / len(values))


def report_models_and_options(rep: Report, task: str, web: dict, mac: dict) -> None:
    """Model SHA256 (enforced if both present) and options (report-only)."""
    wh = (web.get("models") or {}).get(task)
    mh = (mac.get("models") or {}).get(task)
    if wh and mh:
        if wh == mh:
            rep.ok(f"model sha256 match ({wh[:12]}…)")
        else:
            rep.fail(f"model sha256 differ: web={wh[:12]}… macos={mh[:12]}… (parity invalid)")
    else:
        rep.info("model sha256: not present in one/both envelopes (skipped)")

    wo = (web.get("options") or {}).get(task)
    mo = (mac.get("options") or {}).get(task)
    if wo and mo:
        diffs = [k for k in set(wo) | set(mo) if wo.get(k) != mo.get(k)]
        if diffs:
            rep.info(f"options differ (report-only): {', '.join(sorted(diffs))}")
            for k in sorted(diffs):
                rep.info(f"    {k}: web={wo.get(k)!r} macos={mo.get(k)!r}")
        else:
            rep.info("options match")


def compare_task(rep: Report, task: str, web: dict, mac: dict, xy_tol: float) -> None:
    print(f"\n=== {task.upper()} (xy-tol={xy_tol}) ===")
    report_models_and_options(rep, task, web, mac)
    key = LANDMARK_KEY[task]
    web_inst = web.get(task, {}).get(key) or []
    mac_inst = mac.get(task, {}).get(key) or []

    # Detection count parity.
    if len(web_inst) == 0 or len(mac_inst) == 0:
        rep.fail(f"detection count: web={len(web_inst)} macos={len(mac_inst)} (need >=1 each)")
        return
    if len(web_inst) != len(mac_inst):
        rep.fail(f"detection count mismatch: web={len(web_inst)} macos={len(mac_inst)}")
    else:
        rep.ok(f"detected {len(web_inst)} instance(s) on both")

    expected = EXPECTED[task]
    n_inst = min(len(web_inst), len(mac_inst))

    # Aggregate diffs across all aligned instances/landmarks.
    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []
    for i in range(n_inst):
        w, m = web_inst[i], mac_inst[i]
        if len(w) != expected or len(m) != expected:
            rep.fail(f"instance {i} landmark count: web={len(w)} macos={len(m)} (expected {expected})")
        else:
            rep.ok(f"instance {i} landmark count == {expected}")
        all_x += abs_diffs(w, m, "x")
        all_y += abs_diffs(w, m, "y")
        all_z += abs_diffs(w, m, "z")

    x_max, x_mean = stats(all_x)
    y_max, y_mean = stats(all_y)
    z_max, z_mean = stats(all_z)

    # Enforced: normalized x/y within tolerance.
    if x_max <= xy_tol and y_max <= xy_tol:
        rep.ok(f"normalized x/y within tol={xy_tol}: x_max={x_max:.5f} y_max={y_max:.5f}")
    else:
        rep.fail(f"normalized x/y exceed tol={xy_tol}: x_max={x_max:.5f} y_max={y_max:.5f}")
    rep.info(f"x: max={x_max:.5f} mean={x_mean:.5f} | y: max={y_max:.5f} mean={y_mean:.5f}")

    # Report-only: z.
    rep.info(f"z (report-only): max={z_max:.5f} mean={z_mean:.5f}")

    # Report-only: world landmarks.
    w_world = web.get(task, {}).get("worldLandmarks")
    m_world = mac.get(task, {}).get("worldLandmarks")
    if w_world and m_world:
        wd: list[float] = []
        for i in range(min(len(w_world), len(m_world))):
            for ax in ("x", "y", "z"):
                wd += abs_diffs(w_world[i], m_world[i], ax)
        wmax, wmean = stats(wd)
        rep.info(f"worldLandmarks (report-only): max={wmax:.5f} mean={wmean:.5f}")

    # Hand handedness labels (enforced).
    if task == "hand":
        wh = web.get("hand", {}).get("handedness") or []
        mh = mac.get("hand", {}).get("handedness") or []
        for i in range(min(len(wh), len(mh))):
            wl = wh[i][0].get("categoryName") if wh[i] else None
            ml = mh[i][0].get("categoryName") if mh[i] else None
            if wl == ml:
                rep.ok(f"handedness[{i}] label == {wl!r}")
            else:
                rep.fail(f"handedness[{i}] label: web={wl!r} macos={ml!r}")

    # Face blendshapes / matrices (report-only).
    if task == "face":
        wb = len(web.get("face", {}).get("faceBlendshapes") or [])
        mb = len(mac.get("face", {}).get("faceBlendshapes") or [])
        wm = len(web.get("face", {}).get("facialTransformationMatrixes") or [])
        mm = len(mac.get("face", {}).get("facialTransformationMatrixes") or [])
        rep.info(f"faceBlendshapes (report-only): web={wb} macos={mb}")
        rep.info(f"facialTransformationMatrixes (report-only): web={wm} macos={mm}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--web", default=DEFAULT_WEB, help="web.json path")
    ap.add_argument("--macos", default=DEFAULT_MACOS, help="macos.json path")
    # Per-task tolerances. Face is tightest; pose is loose while full-body /
    # options parity is still being validated.
    ap.add_argument("--hand-tol", type=float, default=0.02,
                    help="max allowed abs diff for hand normalized x/y (default 0.02)")
    ap.add_argument("--face-tol", type=float, default=0.01,
                    help="max allowed abs diff for face normalized x/y (default 0.01)")
    ap.add_argument("--pose-tol", type=float, default=0.04,
                    help="max allowed abs diff for pose normalized x/y (default 0.04)")
    ap.add_argument("--tasks", nargs="+", default=["hand", "pose", "face"],
                    choices=["hand", "pose", "face"])
    args = ap.parse_args()
    tol = {"hand": args.hand_tol, "face": args.face_tol, "pose": args.pose_tol}

    for label, path in (("web", args.web), ("macos", args.macos)):
        if not os.path.isfile(path):
            print(f"error: {label} JSON not found at {path}", file=sys.stderr)
            print("Run the web and macOS projects first (see ../README.md).", file=sys.stderr)
            return 2

    web = load(args.web)
    mac = load(args.macos)
    print(f"web   : {args.web}  (source={web.get('source')}, image={web.get('image')})")
    print(f"macos : {args.macos}  (source={mac.get('source')}, image={mac.get('image')})")

    rep = Report()
    for task in args.tasks:
        compare_task(rep, task, web, mac, tol[task])

    print("\n" + ("RESULT: FAILED" if rep.failed else "RESULT: PASSED"))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
