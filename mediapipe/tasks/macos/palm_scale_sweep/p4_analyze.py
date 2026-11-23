#!/usr/bin/env python3
"""Phase 4: compares the Inference Lab's runs of one recorded clip.

    python3 p4_analyze.py --runs OUT/<clip> --clip <clip.mp4> [--labels labels.json]
                          [--out DIR]

Input: one <configuration>.jsonl per configuration, written by ScreenBar's
InferenceLabPalmSearchTests.recordedClips (lines of type "pf" and
"palm-search-run"). Every configuration processed the same decoded frames.

There is no reference for a recorded clip. The script reports what each
configuration returned and how the configurations differ:

- Landmark set: one hand of a frame, 21 points. Its centre is the mean of
  landmarks 0, 5, 9, 13, 17; its size is the distance from landmark 0 (wrist)
  to landmark 9 (middle finger base), in frame pixels. "Detector px" is that
  size x 192 / max(frame width, frame height): the same length in the palm
  detector's full-frame input.
- Pooled hand (per frame): the landmark sets of all configurations on that
  frame, grouped when their centres are within 0.5 x the larger size. Each
  pooled hand lists the configurations that have it.
- Pooled track: pooled hands of successive frames linked when their centres
  are within 1.0 x size, with gaps of up to GAP frames. Tracks shorter than
  MIN_TRACK frames are counted apart as short tracks.
- For a configuration and a pooled track: frames on which it has the hand,
  and the delay from the track's first frame to the configuration's first
  frame with it (frames x frame time).
- Labels (optional, labels.json: {"<frame>": [[x, y], ...]}, one normalized
  point on each visible hand, frames numbered from 1): a labelled hand is
  found when the point lies in a landmark set's bounding box grown by 25 %;
  one set per point. Sets on a labelled frame that hold no point are listed
  as sets without a label.

Nothing here says which configuration is right.
"""
import argparse
import glob
import json
import math
import os
import statistics
import subprocess

PALM_IDX = [0, 5, 9, 13, 17]
GAP = 5
MIN_TRACK = 10
SIZE_BINS = [(0, 4), (4, 6), (6, 9), (9, 14), (14, 22), (22, 1e9)]
SIZE_NAMES = ["< 4", "4 to 6", "6 to 9", "9 to 14", "14 to 22", ">= 22"]
OUTCOMES = ["toLandmarks", "overlapsTrackedRect", "overlapsEarlierRect", "clippedByNumHands",
            "clippedBeforeAssociation"]


def probe(clip):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate", "-of", "json", clip])
    stream = json.loads(out)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return stream["width"], stream["height"], float(num) / float(den)


def med(values, digits=2):
    return round(statistics.median(values), digits) if values else None


def load_run(path, width, height):
    """frames: {frame: {"sets": [(cx, cy, size, box)], "ms", "ran", "runs"}}"""
    frames, counts = {}, None
    with open(path) as f:
        for text in f:
            line = json.loads(text)
            if line.get("type") == "palm-search-run":
                counts = line
            if line.get("type") != "pf":
                continue
            sets = []
            for flat in line["hands"]:
                pts = [(flat[2 * k] * width, flat[2 * k + 1] * height) for k in range(21)]
                cx = sum(pts[k][0] for k in PALM_IDX) / 5
                cy = sum(pts[k][1] for k in PALM_IDX) / 5
                size = math.dist(pts[0], pts[9])
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                sets.append((cx, cy, size, (min(xs), min(ys), max(xs), max(ys))))
            search = line.get("search")
            entry = {"sets": sets, "ms": line["ms"], "ran": None, "runs": None, "merged": []}
            if search is not None:
                entry["ran"] = bool(search["detectorRan"])
                entry["runs"] = len(search["regions"]) if entry["ran"] else 0
                entry["merged"] = [m.get("outcome") for m in search["merged"]]
            frames[line["frame"]] = entry
    return frames, counts


def pool_frame(per_config):
    """per_config: {config: [sets]} -> [{"c": centre, "size", "configs": set}]"""
    pooled = []
    for config, sets in per_config.items():
        for cx, cy, size, _ in sets:
            best = None
            for hand in pooled:
                limit = 0.5 * max(size, hand["size"])
                distance = math.dist((cx, cy), hand["c"])
                if distance <= limit and (best is None or distance < best[0]):
                    best = (distance, hand)
            if best is None:
                pooled.append({"c": (cx, cy), "size": size, "configs": {config}, "n": 1})
            else:
                hand = best[1]
                n = hand["n"]
                hand["c"] = ((hand["c"][0] * n + cx) / (n + 1), (hand["c"][1] * n + cy) / (n + 1))
                hand["size"] = (hand["size"] * n + size) / (n + 1)
                hand["n"] = n + 1
                hand["configs"].add(config)
    return pooled


def link_tracks(pooled_by_frame):
    tracks, open_tracks = [], []
    for frame in sorted(pooled_by_frame):
        open_tracks = [t for t in open_tracks if frame - t["last"] <= GAP + 1]
        used = set()
        for hand in pooled_by_frame[frame]:
            best = None
            for index, track in enumerate(open_tracks):
                if index in used or track["last"] == frame:
                    continue
                distance = math.dist(hand["c"], track["c"])
                if distance <= max(hand["size"], track["size"]) and (best is None or distance < best[0]):
                    best = (distance, index)
            if best is None:
                track = {"first": frame, "frames": {}, "sizes": []}
                tracks.append(track)
                open_tracks.append(track)
                used.add(len(open_tracks) - 1)
            else:
                track = open_tracks[best[1]]
                used.add(best[1])
            track["last"] = frame
            track["c"] = hand["c"]
            track["size"] = hand["size"]
            track["sizes"].append(hand["size"])
            track["frames"][frame] = hand["configs"]
    return tracks


def size_bin(px):
    for index, (low, high) in enumerate(SIZE_BINS):
        if low <= px < high:
            return index
    return len(SIZE_BINS) - 1


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("–" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--out")
    args = parser.parse_args()
    out_dir = args.out or args.runs
    width, height, fps = probe(args.clip)
    frame_ms = 1000 / fps
    to_detector = 192 / max(width, height)

    runs, counts = {}, {}
    for path in sorted(glob.glob(os.path.join(args.runs, "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        runs[name], counts[name] = load_run(path, width, height)
    order = [n for n in ("off", "full") if n in runs] + sorted(n for n in runs if n not in ("off", "full"))
    frame_ids = sorted(set().union(*[set(r) for r in runs.values()]))

    pooled_by_frame = {}
    for frame in frame_ids:
        pooled = pool_frame({n: runs[n][frame]["sets"] for n in order if frame in runs[n]})
        if pooled:
            pooled_by_frame[frame] = pooled
    tracks = link_tracks(pooled_by_frame)
    long_tracks = [t for t in tracks if len(t["frames"]) >= MIN_TRACK]
    short_tracks = [t for t in tracks if len(t["frames"]) < MIN_TRACK]
    for track in tracks:
        track["px"] = statistics.median(track["sizes"]) * to_detector

    result = {"clip": os.path.basename(args.clip), "width": width, "height": height, "fps": fps,
              "frames": len(frame_ids), "configs": {}, "tracks": []}
    md = [f"# {os.path.basename(args.clip)}\n",
          f"{width} x {height}, {fps:.2f} fps, {len(frame_ids)} frames, {len(order)} configurations. "
          f"Pooled tracks of {MIN_TRACK} frames or more: {len(long_tracks)}; shorter: {len(short_tracks)}.\n"]

    # 1. What each configuration returned.
    rows = []
    for name in order:
        run = runs[name]
        per_frame = [len(run[f]["sets"]) for f in sorted(run)]
        sizes = [s[2] * to_detector for f in run for s in run[f]["sets"]]
        detector = [run[f] for f in run if run[f]["ran"]]
        other = [run[f] for f in run if run[f]["ran"] is False]
        merged = [o for f in run for o in run[f]["merged"]]
        entry = {
            "frames": len(run),
            "frames_0_1_2plus": [sum(1 for n in per_frame if n == 0), sum(1 for n in per_frame if n == 1),
                                 sum(1 for n in per_frame if n >= 2)],
            "landmark_sets": sum(per_frame),
            "set_size_detector_px_min_median": [round(min(sizes), 1), med(sizes, 1)] if sizes else None,
            "detector_frames": len(detector) if detector or other else None,
            "palm_runs": sum(e["runs"] for e in detector) if detector or other else None,
            "merged_by_outcome": {o: merged.count(o) for o in OUTCOMES} if detector or other else None,
            "call_ms_median_detector": med([e["ms"] for e in detector], 1),
            "call_ms_median_other": med([e["ms"] for e in other], 1),
            "call_ms_median_all": med([run[f]["ms"] for f in run], 1),
            "short_track_frames": sum(1 for t in short_tracks for cs in t["frames"].values() if name in cs),
        }
        result["configs"][name] = entry
        rows.append([
            name, entry["detector_frames"], entry["palm_runs"],
            " / ".join(map(str, entry["frames_0_1_2plus"])), entry["landmark_sets"],
            " / ".join(map(str, entry["set_size_detector_px_min_median"])) if sizes else None,
            " / ".join(str(entry["merged_by_outcome"][o]) for o in OUTCOMES) if entry["merged_by_outcome"] else None,
            entry["call_ms_median_detector"], entry["call_ms_median_other"], entry["short_track_frames"],
        ])
    md.append("## What each configuration returned\n")
    md.append(table(["Configuration", "Detector frames", "Palm runs", "Frames with 0 / 1 / 2+ hands",
                     "Landmark sets", "Set size, detector px: min / median",
                     "Merged: " + " / ".join(OUTCOMES), "Call p50 ms, detector frames",
                     "Call p50 ms, other frames", f"Sets in pooled tracks under {MIN_TRACK} frames"], rows))

    # 2. Pooled tracks.
    rows = []
    for index, track in enumerate(sorted(long_tracks, key=lambda t: t["first"])):
        frames = sorted(track["frames"])
        per_config = {}
        for name in order:
            have = [f for f in frames if name in track["frames"][f]]
            per_config[name] = {"frames": len(have),
                                "delay_ms": round((have[0] - frames[0]) * frame_ms) if have else None}
        result["tracks"].append({"id": index + 1, "first": frames[0], "last": frames[-1],
                                 "pooled_frames": len(frames), "detector_px": round(track["px"], 1),
                                 "configs": per_config})
        track["id"], track["per_config"] = index + 1, per_config
        rows.append([index + 1, frames[0], frames[-1], len(frames), round(track["px"], 1),
                     sum(1 for n in order if per_config[n]["frames"]), len(order)])
    md.append("## Pooled tracks\n")
    md.append(table(["Track", "First frame", "Last frame", "Frames with the hand in any configuration",
                     "Median size, detector px", "Configurations that have it", "of"], rows))

    # 3. Per configuration, by track size.
    md.append("## Frames with the hand, of the pooled tracks' frames, by size (detector px)\n")
    rows = []
    for name in order:
        cells, delays, missing = [], [], 0
        for index in range(len(SIZE_BINS)):
            in_bin = [t for t in long_tracks if size_bin(t["px"]) == index]
            cells.append(pct(sum(t["per_config"][name]["frames"] for t in in_bin),
                             sum(len(t["frames"]) for t in in_bin)))
        for track in long_tracks:
            delay = track["per_config"][name]["delay_ms"]
            if delay is None:
                missing += 1
            else:
                delays.append(delay)
        result["configs"][name]["tracks_never"] = missing
        result["configs"][name]["delay_ms_median_max"] = [med(delays, 0), max(delays)] if delays else None
        rows.append([name] + cells + [missing, med(delays, 0), max(delays) if delays else None])
    md.append(table(["Configuration"] + SIZE_NAMES + ["Tracks never had", "Delay to first frame, ms: median",
                                                       "max"], rows))

    # 4. Labels.
    if args.labels:
        with open(args.labels) as f:
            labels = {int(k): v for k, v in json.load(f).items()}
        rows = []
        for name in order:
            found = total = extra = 0
            for frame, points in labels.items():
                sets = list(runs[name].get(frame, {"sets": []})["sets"])
                for x, y in points:
                    total += 1
                    px, py = x * width, y * height
                    for candidate in sets:
                        x0, y0, x1, y1 = candidate[3]
                        gx, gy = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
                        if x0 - gx <= px <= x1 + gx and y0 - gy <= py <= y1 + gy:
                            sets.remove(candidate)
                            found += 1
                            break
                extra += len(sets)
            result["configs"][name]["labels"] = {"hands": total, "found": found, "sets_without_label": extra}
            rows.append([name, pct(found, total), extra])
        md.append(f"## Labelled frames ({len(labels)} frames, "
                  f"{sum(len(v) for v in labels.values())} hands)\n")
        md.append(table(["Configuration", "Labelled hands found", "Sets without a label"], rows))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(out_dir, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print(os.path.join(out_dir, "tables.md"))


if __name__ == "__main__":
    main()
