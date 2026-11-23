#!/usr/bin/env python3
"""Analyses the phase 2 runs (p2_run.py) against the scene references.

    python3 p2_analyze.py --out results.json [--frames crowd-frames.jsonl.gz]

Definitions (scene pixels, 1280x720):
- Reference hand: a hand of the scene reference (p2_scenes.py) with landmarks.
  Hands shown without a reference (the full-size run had none in that source
  frame) are left out of every count.
- Palm px: sqrt(width x height) of the reference palm box x 192/1280.
- Palm detected: a palm detection handed on by the detector graph (after the
  cross-region merge when the search is on) has IoU >= 0.3 with the reference
  palm box; greedy one-to-one, highest IoU first. A further detection with
  IoU >= 0.3 to an already assigned reference hand is a duplicate; any other
  is unmatched.
- Hand found: a landmark set whose palm centroid (landmarks 0, 5, 9, 13, 17)
  is within 0.25 x reference hand size of the reference palm centroid; greedy
  one-to-one. A further set within that distance of an assigned hand is a
  duplicate set; any other is an unmatched set.
- Margin (grid runs): for a reference palm box and a tile's sampled window
  (the square the model input is taken from, not clipped to the frame), the
  smallest of the four distances from the box to the window's sides, negative
  when the box sticks out. Best margin: the largest over the tiles run on the
  frame, in palm box sizes (sqrt(width x height)).
- Latency: frames x 33.33 ms, i.e. clip time with every frame processed.
"""
import argparse
import glob
import gzip
import json
import math
import os
import statistics

OUT = os.environ.get("P2_OUT", "/tmp/palmp2")
W, H = 1280, 720
FRAME_MS = 3333 / 100
IOU_MIN = 0.3
HAND_MATCH = 0.25
PALM_IDX = [0, 5, 9, 13, 17]
MARGIN_BINS = [(-9, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5), (1.5, 99)]
MARGIN_NAMES = ["< -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1.5", ">= 1.5"]
OUTCOMES = ["to_landmarks", "overlaps_tracked", "overlaps_earlier", "clipped_num_hands",
            "clipped_before_association"]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def det_box(d):
    return (d[1] * W, d[2] * H, (d[1] + d[3]) * W, (d[2] + d[4]) * H)


def centroid(pts):
    return (sum(pts[k][0] for k in PALM_IDX) / 5, sum(pts[k][1] for k in PALM_IDX) / 5)


def hand_pts(hand):
    lm = hand["lm"]
    return [(lm[3 * k] * W, lm[3 * k + 1] * H) for k in range(21)]


def med(values):
    return round(statistics.median(values), 3) if values else None


def rate(a, b):
    return round(a / b, 4) if b else None


def match_dets(gts, boxes):
    """-> list per detection: ("hit"|"dup"|"none", gt index or None)."""
    pairs = sorted(((iou(g["box"], b), i, j) for i, g in enumerate(gts) if g["box"]
                    for j, b in enumerate(boxes)), reverse=True)
    out = [("none", None)] * len(boxes)
    used = set()
    for v, i, j in pairs:
        if v < IOU_MIN or out[j][0] != "none":
            continue
        out[j] = ("dup", i) if i in used else ("hit", i)
        used.add(i)
    return out


def match_hands(gts, hands):
    """-> list per landmark set: ("hit"|"dup"|"none", gt index, error)."""
    sets = [hand_pts(h) for h in hands]
    pairs = []
    for i, g in enumerate(gts):
        gc = centroid(g["pts"])
        for j, pts in enumerate(sets):
            d = math.dist(gc, centroid(pts))
            if d <= HAND_MATCH * g["size"]:
                pairs.append((d, i, j))
    out = [("none", None, None)] * len(hands)
    used = set()
    for d, i, j in sorted(pairs):
        if out[j][0] != "none":
            continue
        if i in used:
            out[j] = ("dup", i, None)
            continue
        used.add(i)
        g = gts[i]
        err = sum(math.dist(p, q) for p, q in zip(g["pts"], sets[j])) / 21 / g["size"]
        out[j] = ("hit", i, err)
    return out


def regions_of(fr):
    return fr.get("search", {}).get("regions", []) if fr.get("ran") else []


def palm_runs(fr):
    if "search" in fr:
        return len(regions_of(fr))
    return 1 if fr.get("ran") else 0


def best_margin(box, fr):
    best = None
    for region in regions_of(fr):
        if region["kind"] != 1:
            continue
        cx, cy, w, h = region["sampled"]
        x0, x1 = (cx - w / 2) * W, (cx + w / 2) * W
        y0, y1 = (cy - h / 2) * H, (cy + h / 2) * H
        m = min(box[0] - x0, x1 - box[2], box[1] - y0, y1 - box[3])
        best = m if best is None else max(best, m)
    if best is None:
        return None
    return best / math.sqrt((box[2] - box[0]) * (box[3] - box[1]))


def frame_matches(fr, gts):
    refs = [g for g in gts if g["pts"]]
    dets = match_dets(refs, [det_box(d) for d in fr.get("det", [])])
    hands = match_hands(refs, fr["hands"])
    return refs, dets, hands


# ----------------------------------------------------------------- border

def analyse_border(frames, gt):
    n = boxed = det_hit = hand_hit = dup = unmatched = dup_sets = false_sets = runs = 0
    errs, palm_px = [], []
    bins = [[0, 0, 0] for _ in MARGIN_BINS]          # hands, detected, found
    source = {"full_only": 0, "tiles_only": 0, "both": 0}
    rejected = 0
    for fr, gts in zip(frames, gt):
        refs, dets, hands = frame_matches(fr, gts)
        runs += palm_runs(fr)
        det_gt = {i for kind, i in dets if kind == "hit"}
        hand_gt = {i: e for kind, i, e in hands if kind == "hit"}
        dup += sum(kind == "dup" for kind, _ in dets)
        unmatched += sum(kind == "none" for kind, _ in dets)
        dup_sets += sum(kind == "dup" for kind, _, _ in hands)
        false_sets += sum(kind == "none" for kind, _, _ in hands)
        for region in regions_of(fr):
            rejected += sum(d["outcome"] != 0 for d in region["det"])
        merged = fr.get("search", {}).get("merged", [])
        for j, (kind, i) in enumerate(dets):
            if kind == "hit" and j < len(merged):
                ids = {s[0] for s in merged[j]["src"]}
                key = "both" if 0 in ids and len(ids) > 1 else ("full_only" if ids == {0} else "tiles_only")
                source[key] += 1
        for i, g in enumerate(refs):
            n += 1
            if not g["box"]:
                continue
            boxed += 1
            palm_px.append(g["palm_px"])
            det_hit += i in det_gt
            hand_hit += i in hand_gt
            if i in hand_gt:
                errs.append(hand_gt[i])
            m = best_margin(g["box"], fr)
            if m is not None:
                for b, (lo, hi) in enumerate(MARGIN_BINS):
                    if lo <= m < hi:
                        bins[b][0] += 1
                        bins[b][1] += i in det_gt
                        bins[b][2] += i in hand_gt
    k = 1000 / len(frames)
    return {"frames": len(frames), "hands": n, "hands_with_palm_box": boxed, "palm_px_median": med(palm_px),
            "palm_detected": rate(det_hit, boxed), "hand_found": rate(hand_hit, boxed),
            "duplicate_detections_per_1000": round(dup * k, 1),
            "unmatched_detections_per_1000": round(unmatched * k, 1),
            "duplicate_sets_per_1000": round(dup_sets * k, 1),
            "unmatched_sets_per_1000": round(false_sets * k, 1),
            "rejected_region_detections_per_1000": round(rejected * k, 1),
            "landmark_error_median": med(errs), "palm_runs_per_frame": round(runs / len(frames), 2),
            "matched_detection_sources": source,
            "by_margin": [{"margin": name, "hands": b[0], "palm_detected": rate(b[1], b[0]),
                           "hand_found": rate(b[2], b[0])} for name, b in zip(MARGIN_NAMES, bins)]}


# ------------------------------------------------------- video (per hand id)

def tracked_ids(fr, gts):
    refs, dets, hands = frame_matches(fr, gts)
    found = {refs[i]["id"] for kind, i, _ in hands if kind == "hit"}
    return refs, dets, hands, found


def runs_of(flags):
    """[(start, length)] of True runs."""
    out, start = [], None
    for i, v in enumerate(flags + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    return out


def analyse_video(frames, gt, frames_out=None, name=None):
    ids = []
    for gts in gt:
        for g in gts:
            if g["id"] not in ids:
                ids.append(g["id"])
    shown = {i: [False] * len(gt) for i in ids}       # in the scene, with or without reference
    present = {i: [False] * len(gt) for i in ids}     # with reference
    found = {i: [False] * len(gt) for i in ids}
    px = {i: [] for i in ids}
    errs = {i: [] for i in ids}
    palm = landmark = detector_frames = false_sets = dup_sets = unmatched = dup = sets = 0
    outcomes = [0] * len(OUTCOMES)
    region_rejected = 0
    two_sets = 0
    call_ms = {"detector": [], "skipped": [], "detector_no_hands": [], "detector_with_hands": []}
    detector_palm_runs = []
    handovers = []
    previous = set()
    for n, (fr, gts) in enumerate(zip(frames, gt)):
        refs, dets, hands, tracked = tracked_ids(fr, gts)
        for g in gts:
            shown[g["id"]][n] = True
        for g in refs:
            present[g["id"]][n] = True
            if g.get("palm_px"):
                px[g["id"]].append(g["palm_px"])
        for kind, i, e in hands:
            if kind == "hit":
                found[refs[i]["id"]][n] = True
                errs[refs[i]["id"]].append(e)
        palm += palm_runs(fr)
        landmark += len(fr.get("rois", []))
        detector_frames += bool(fr.get("ran"))
        sets += len(fr["hands"])
        two_sets += len(fr["hands"]) >= 2
        dup_sets += sum(k == "dup" for k, _, _ in hands)
        false_sets += sum(k == "none" for k, _, _ in hands)
        dup += sum(k == "dup" for k, _ in dets)
        unmatched += sum(k == "none" for k, _ in dets)
        search = fr.get("search")
        merged_rows = []
        if search:
            ms = (search["call_end_ns"] - search["call_start_ns"]) / 1e6
            call_ms["detector" if search["ran"] else "skipped"].append(ms)
            if search["ran"]:
                call_ms["detector_with_hands" if fr["hands"] else "detector_no_hands"].append(ms)
                detector_palm_runs.append(palm_runs(fr))
            for region in regions_of(fr):
                region_rejected += sum(d["outcome"] != 0 for d in region["det"])
            if fr.get("ran"):
                for j, m in enumerate(search["merged"]):
                    outcomes[m["outcome"]] += 1
                    kind, i = dets[j] if j < len(dets) else ("none", None)
                    merged_rows.append({"hand": refs[i]["id"] if i is not None else None, "match": kind,
                                        "score": round(m["d"][0], 3), "outcome": OUTCOMES[m["outcome"]],
                                        "regions": sorted({s[0] for s in m["src"]})})
        # A handover: a hand that had landmarks, is still in the scene and has
        # none now, on a frame where another hand gained landmarks.
        visible = {g["id"] for g in refs}
        lost = sorted((previous - tracked) & visible)
        gained = sorted(tracked - previous)
        if lost and gained:
            handovers.append({"frame": n, "lost": lost, "gained": gained})
        previous = tracked
        if frames_out is not None:
            row = {"run": name, "frame": n, "visible": sorted(visible), "landmarks": sorted(tracked),
                   "sets": len(fr["hands"]), "detector_ran": int(bool(fr.get("ran")))}
            if search and fr.get("ran"):
                row["regions"] = [r["id"] for r in regions_of(fr)]
                row["merged"] = merged_rows
            frames_out.write(json.dumps(row, separators=(",", ":")) + "\n")
    per_hand = {}
    for i in ids:
        appearances = []
        for start, length in runs_of(shown[i]):
            span = range(start, start + length)
            f = [found[i][k] for k in span]
            p = [present[i][k] for k in span]
            first = f.index(True) if True in f else None
            tracked_runs = runs_of(f)
            reacq = []
            for (s0, l0), (s1, _) in zip(tracked_runs, tracked_runs[1:]):
                if all(p[s0 + l0:s1]):
                    reacq.append(round((s1 - (s0 + l0)) * FRAME_MS, 1))
            appearances.append({
                "first_frame": start, "frames": length, "frames_with_reference": sum(p),
                "frames_found": sum(f),
                "acquisition_ms": None if first is None else round(first * FRAME_MS, 1),
                "tracked_runs": len(tracked_runs),
                "longest_run_frames": max((l for _, l in tracked_runs), default=0),
                "reacquisition_ms": reacq})
        per_hand[i] = {"palm_px_median": med(px[i]), "frames_with_reference": sum(present[i]),
                       "frames_found": sum(found[i]), "found": rate(sum(found[i]), sum(present[i])),
                       "landmark_error_median": med(errs[i]), "appearances": appearances}
    k = 1000 / len(frames)
    return {"frames": len(frames), "detector_frames": detector_frames,
            "palm_runs_per_frame": round(palm / len(frames), 3),
            "landmark_runs_per_frame": round(landmark / len(frames), 3),
            "landmark_sets_per_frame": round(sets / len(frames), 3), "frames_with_two_sets": two_sets,
            "duplicate_sets_per_1000": round(dup_sets * k, 1), "unmatched_sets_per_1000": round(false_sets * k, 1),
            "duplicate_detections_per_1000": round(dup * k, 1),
            "unmatched_detections_per_1000": round(unmatched * k, 1),
            "rejected_region_detections_per_1000": round(region_rejected * k, 1),
            "merged_outcomes": dict(zip(OUTCOMES, outcomes)),
            "call_ms_detector_frame_median": med(call_ms["detector"]),
            "call_ms_skipped_frame_median": med(call_ms["skipped"]),
            "call_ms_detector_frame_no_hands_median": med(call_ms["detector_no_hands"]),
            "call_ms_detector_frame_with_hands_median": med(call_ms["detector_with_hands"]),
            "call_ms_detector_frame_p95": (round(sorted(call_ms["detector"])[int(0.95 * (len(call_ms["detector"]) - 1))], 3)
                                           if call_ms["detector"] else None),
            "palm_runs_per_detector_frame": med(detector_palm_runs),
            "call_ms_mean": (round(statistics.mean(call_ms["detector"] + call_ms["skipped"]), 3)
                             if call_ms["detector"] or call_ms["skipped"] else None),
            "handovers": handovers, "hands": per_hand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames")
    args = ap.parse_args()
    gts = {}
    results = {}
    frames_out = gzip.open(args.frames, "wt") if args.frames else None
    for path in sorted(glob.glob(os.path.join(OUT, "raw", "*.jsonl"))):
        if os.path.exists(path + ".partial"):
            continue
        name = os.path.basename(path)[:-6]
        scene, config = name.split("__", 1)
        if scene not in gts:
            with open(os.path.join(OUT, "scenes", scene + ".gt.json")) as f:
                gts[scene] = json.load(f)["frames"]
        frames = load(path)
        if len(frames) != len(gts[scene]):
            print(f"{name}: {len(frames)} frames, expected {len(gts[scene])}; skipped")
            continue
        if scene.startswith("border"):
            entry = analyse_border(frames, gts[scene])
        else:
            entry = analyse_video(frames, gts[scene], frames_out if scene.startswith("crowd") else None, name)
        results.setdefault(scene, {})[config] = entry
    if frames_out:
        frames_out.close()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print({s: len(c) for s, c in results.items()})


if __name__ == "__main__":
    main()
