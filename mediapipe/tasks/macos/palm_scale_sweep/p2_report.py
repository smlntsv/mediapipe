#!/usr/bin/env python3
"""Markdown tables from p2_analyze.py output.

    python3 p2_report.py results.json [crowd-frames.jsonl.gz] > tables.md
"""
import gzip
import json
import re
import statistics
import sys

W, H = 1280, 720
GRIDS = ["2x2", "2x3", "3x3", "3x4", "4x4"]


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def num(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def grid_label(g):
    rows, columns = g.split("x")
    return f"{columns} × {rows}"          # columns × rows


def magnification(g, overlap):
    rows, columns = (int(v) for v in g.split("x"))
    tw = W / (1 + (columns - 1) * (1 - overlap))
    th = H / (1 + (rows - 1) * (1 - overlap))
    return max(W, H) / max(tw, th)


def config_label(name):
    if name in ("off", "full", "full_clipb"):
        return {"off": "off", "full": "full frame only", "full_clipb": "full frame only, clip before"}[name]
    m = re.match(r"g(\dx\d)_o(\d+)_(all|rr\d+)(.*)", name)
    g, overlap, schedule, rest = m.groups()
    parts = [grid_label(g), f"overlap {int(overlap) / 100:g}",
             "all tiles" if schedule == "all" else f"round-robin {schedule[2:]}"]
    extra = {"_tr": "truncated rejected", "_best": "best-score merge", "_nofull": "no full frame",
             "_clipb": "clip before"}
    parts += [extra[r] for r in re.findall(r"_[a-z]+", rest)]
    return ", ".join(parts)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def border(scene, cs):
    print(f"### {scene}\n")
    any_cfg = cs["full"]
    print(f"{any_cfg['frames']} frames, {any_cfg['hands_with_palm_box']} reference hands, "
          f"median palm {num(any_cfg['palm_px_median'])} px. IMAGE mode, CPU, `numHands` 2.\n")
    rows = []
    for name in ("off", "full"):
        e = cs[name]
        rows.append([config_label(name), "1.0", num(e["palm_runs_per_frame"], 0), pct(e["palm_detected"]),
                     pct(e["hand_found"]), "–", "–", num(e["duplicate_detections_per_1000"]),
                     num(e["unmatched_detections_per_1000"]), num(e["unmatched_sets_per_1000"])])
    for g in GRIDS:
        for overlap in (0, 10, 20, 30):
            e = cs.get(f"g{g}_o{overlap:02d}_all")
            t = cs.get(f"g{g}_o{overlap:02d}_all_tr")
            if not e or not t:
                continue
            rows.append([f"{grid_label(g)}, overlap {overlap / 100:g}", num(magnification(g, overlap / 100), 2),
                         num(e["palm_runs_per_frame"], 0), pct(e["palm_detected"]), pct(e["hand_found"]),
                         pct(t["palm_detected"]), pct(t["hand_found"]), num(e["duplicate_detections_per_1000"]),
                         num(e["unmatched_detections_per_1000"]), num(e["unmatched_sets_per_1000"])])
    print(table(["Configuration (columns × rows)", "Magnification", "Palm runs / frame", "Palm detected",
                 "Hand found", "Palm detected, truncated rejected", "Hand found, truncated rejected",
                 "Duplicate detections / 1000 frames", "Unmatched detections / 1000 frames",
                 "Unmatched landmark sets / 1000 frames"], rows))

    print("Palm detected by best margin (palm box sizes), all tiles, truncated detections kept:\n")
    names = [b["margin"] for b in cs["g4x4_o00_all"]["by_margin"]]
    rows = []
    for g in GRIDS:
        for overlap in (0, 20):
            e = cs.get(f"g{g}_o{overlap:02d}_all")
            if not e:
                continue
            rows.append([f"{grid_label(g)}, overlap {overlap / 100:g}"] +
                        [f"{pct(b['palm_detected'])} (n {b['hands']})" for b in e["by_margin"]])
    print(table(["Configuration"] + names, rows))

    print("Overlap 0.2, all tiles: merge and full-frame variants:\n")
    rows = []
    for g in GRIDS:
        for suffix, label in (("", "weighted merge"), ("_best", "best-score merge"), ("_nofull", "no full frame")):
            e = cs.get(f"g{g}_o20_all{suffix}")
            if not e:
                continue
            src = e["matched_detection_sources"]
            rows.append([grid_label(g), label, pct(e["palm_detected"]), pct(e["hand_found"]),
                         num(e["landmark_error_median"], 4), num(e["duplicate_detections_per_1000"]),
                         num(e["unmatched_sets_per_1000"]),
                         f"{src['full_only']} / {src['tiles_only']} / {src['both']}"])
    print(table(["Grid", "Variant", "Palm detected", "Hand found", "Landmark error, median",
                 "Duplicate detections / 1000 frames", "Unmatched landmark sets / 1000 frames",
                 "Matched detections from full frame only / tiles only / both"], rows))


def far_stats(e):
    apps = [a for hid, h in e["hands"].items() if hid.startswith("far") for a in h["appearances"]]
    acq = [a["acquisition_ms"] for a in apps if a["acquisition_ms"] is not None]
    ref = sum(a["frames_with_reference"] for a in apps)
    found = sum(a["frames_found"] for a in apps)
    reacq = [v for a in apps for v in a["reacquisition_ms"]]
    return len(apps), acq, (found / ref if ref else None), reacq


def nearfar(scene, cs):
    print(f"### {scene}\n")
    e = cs["full"]
    far_px = statistics.median(h["palm_px_median"] for hid, h in e["hands"].items() if hid.startswith("far"))
    line = f"{e['frames']} frames. Far hand: median palm {far_px:.1f} px, 8 appearances of 60 frames."
    if "near" in e["hands"]:
        line += f" Near hand: median palm {e['hands']['near']['palm_px_median']:.1f} px, in view throughout."
    print(line + " VIDEO mode, CPU, `numHands` 2.\n")
    order = ["off", "full"]
    for g in GRIDS:
        order += [f"g{g}_o00_all", f"g{g}_o20_all", f"g{g}_o20_all_tr", f"g{g}_o00_rr1", f"g{g}_o20_rr1"]
        if g in ("3x4", "4x4"):
            order += [f"g{g}_o20_rr4", f"g{g}_o20_all_clipb", f"g{g}_o20_all_nofull"]
    rows = []
    for name in order:
        e = cs.get(name)
        if not e:
            continue
        n, acq, found, reacq = far_stats(e)
        near = e["hands"].get("near")
        rows.append([config_label(name), num(e["palm_runs_per_frame"], 2), f"{len(acq)} of {n}",
                     num(statistics.median(acq), 0) if acq else "–", num(max(acq), 0) if acq else "–",
                     pct(found), f"{len(reacq)}" + (f" ({num(statistics.median(reacq), 0)})" if reacq else ""),
                     pct(near["found"]) if near else "–", num(e["unmatched_sets_per_1000"]),
                     num(e["duplicate_sets_per_1000"])])
    print(table(["Configuration", "Palm runs / frame", "Far appearances acquired", "Acquisition, median ms",
                 "Acquisition, max ms", "Far hand found", "Re-acquisitions (median ms)", "Near hand found",
                 "Unmatched landmark sets / 1000 frames", "Duplicate landmark sets / 1000 frames"], rows))


def crowd(scene, cs, frames):
    print(f"### {scene}\n")
    e = cs["full"]
    hands = sorted(e["hands"], key=lambda h: -e["hands"][h]["palm_px_median"])
    print(f"{e['frames']} frames, VIDEO mode, CPU, `numHands` 2. Hands, largest first: " +
          ", ".join(f"{h} {e['hands'][h]['palm_px_median']:.1f} px ({e['hands'][h]['frames_with_reference']} frames)"
                    for h in hands) + ".\n")
    order = ["off", "full", "full_clipb"]
    for g in GRIDS:
        order += [f"g{g}_o20_all", f"g{g}_o20_all_clipb"]
        if g in ("3x4", "4x4"):
            order += [f"g{g}_o20_all_tr", f"g{g}_o20_rr1"]
    rows = []
    for name in order:
        c = cs.get(name)
        if not c:
            continue
        rows.append([config_label(name), c["detector_frames"], num(c["palm_runs_per_frame"], 2),
                     num(c["landmark_sets_per_frame"], 2), len(c["handovers"]),
                     num(c["unmatched_sets_per_1000"]), num(c["duplicate_sets_per_1000"])] +
                    [pct(c["hands"][h]["found"]) for h in hands])
    print(table(["Configuration", "Detector frames", "Palm runs / frame", "Landmark sets / frame", "Handovers",
                 "Unmatched sets / 1000 frames", "Duplicate sets / 1000 frames"] + hands, rows))

    print("What became of the merged detections, summed over the detector frames:\n")
    rows = []
    for name in order:
        c = cs.get(name)
        if not c or name == "off":
            continue
        o = c["merged_outcomes"]
        rows.append([config_label(name), o["to_landmarks"], o["overlaps_tracked"], o["overlaps_earlier"],
                     o["clipped_num_hands"], o["clipped_before_association"]])
    print(table(["Configuration", "To the landmark model", "Overlaps a tracked rect", "Overlaps an earlier rect",
                 "Clipped by numHands after association", "Clipped before association"], rows))

    print("Longest unbroken stretch with landmarks per hand, frames (number of stretches):\n")
    rows = []
    for name in order:
        c = cs.get(name)
        if not c:
            continue
        cells = []
        for h in hands:
            apps = c["hands"][h]["appearances"]
            cells.append(f"{max(a['longest_run_frames'] for a in apps)} ({sum(a['tracked_runs'] for a in apps)})")
        rows.append([config_label(name)] + cells)
    print(table(["Configuration"] + hands, rows))

    if frames:
        print("Clip before against clip after the association, frame by frame:\n")
        rows = []
        for name in order:
            if not name.endswith("_clipb"):
                continue
            a = frames.get(f"{scene}__{name[:-6]}")
            b = frames.get(f"{scene}__{name}")
            if not a or not b:
                continue
            diff = [n for n, (x, y) in enumerate(zip(a, b)) if x["landmarks"] != y["landmarks"]]
            rows.append([config_label(name[:-6]), len(diff), diff[0] if diff else "–"])
        print(table(["Configuration", "Frames whose hands with landmarks differ", "First such frame"], rows))


def timing(scene, cs):
    delegates = [d for d in ("cpu", "gpu", "coreml") if any(k.endswith(f"__t_{d}") for k in cs)]
    if not delegates:
        return
    print(f"### Call time, {scene}\n")
    print("One run at a time, nothing else running. Call time is `call_end_ns - call_start_ns` of the "
          "search record, so the feature-off path has no row.\n")
    order = ["full"]
    for g in GRIDS:
        order += [f"g{g}_o20_all", f"g{g}_o20_rr1"]
    for d in delegates:
        base = cs.get(f"full__t_{d}")
        rows = []
        for name in order:
            e = cs.get(f"{name}__t_{d}")
            if not e:
                continue
            runs = e["palm_runs_per_detector_frame"]
            no_hands = e["call_ms_detector_frame_no_hands_median"]
            extra = "–"
            if base and name != "full" and runs and runs > 1 and no_hands is not None:
                extra = num((no_hands - base["call_ms_detector_frame_no_hands_median"]) / (runs - 1), 2)
            mean = e["call_ms_mean"]
            rows.append([config_label(name), num(runs, 0), e["detector_frames"], num(no_hands, 2),
                         num(e["call_ms_detector_frame_with_hands_median"], 2),
                         num(e["call_ms_detector_frame_p95"], 2), num(mean, 2),
                         num(1000 / mean, 1) if mean else "–", extra])
        print(f"{d}:\n")
        print(table(["Configuration", "Palm runs / detector frame", "Detector frames",
                     "Call ms, no hand with landmarks, median", "Call ms, a hand with landmarks, median",
                     "Call ms, 95th percentile", "Call ms, mean", "Calls / second at that mean",
                     "ms per region beyond the full frame"], rows))


def main():
    results = json.load(open(sys.argv[1]))
    frames = {}
    if len(sys.argv) > 2:
        with gzip.open(sys.argv[2], "rt") as f:
            for line in f:
                row = json.loads(line)
                frames.setdefault(row["run"], []).append(row)
    for scene in sorted(results):
        cs = results[scene]
        if scene.startswith("border"):
            border(scene, cs)
        elif scene.startswith("crowd"):
            crowd(scene, cs, frames)
        else:
            nearfar(scene, {k: v for k, v in cs.items() if "__t_" not in k})
            timing(scene, cs)


if __name__ == "__main__":
    main()
