#!/usr/bin/env python3
"""Analyses the raw per-frame JSON lines written by sweep.py.

    python3 analyze.py --out results.json [--instances instances.jsonl.gz]

Definitions (all in canvas pixels of the 1280x720 synthetic frame unless noted):
- Ground truth (GT): the full-size IMAGE-mode CPU run of the same clip. Each GT
  hand is a result hand (21 landmarks); its GT palm box is the palm detection
  of that frame whose centre is nearest the hand's palm centroid (landmarks
  0, 5, 9, 13, 17). GT is mapped through the shrink (w/1280, h/720) and offset.
- Palm px: sqrt(box width px * box height px) of the mapped GT palm box, times
  192/1280 (the letterbox factor of a 1280x720 frame into the 192x192 palm
  detector input).
- Detected: a palm detection after suppression (stage record) has IoU >= 0.3
  with the GT palm box (greedy one-to-one, highest IoU first).
- Hand found: a result hand whose palm centroid is within 0.25 x GT hand size
  of the GT palm centroid (greedy one-to-one). GT hand size: longer side of the
  GT landmark bounding box.
- Best matching candidate: the highest-scored palm candidate (top 50, no
  threshold, before suppression) with IoU >= 0.3 with the GT palm box.
- False detection: a palm detection after suppression not assigned to any GT
  palm.
- Landmark error: mean over the 21 landmarks of the distance to GT, divided by
  GT hand size, on GT hands that were found.
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import statistics

OUT = os.environ.get("SWEEP_OUT", "/tmp/palmsweep")
RAW = os.path.join(OUT, "raw")
W, H = 1280, 720
LETTERBOX = 192 / max(W, H)
IOU_MIN = 0.3
HAND_MATCH = 0.25
THRESHOLD = 0.5
PALM_IDX = [0, 5, 9, 13, 17]
NAME_RE = re.compile(r"^(?P<clip>[a-z-]+)_s(?P<scale>[0-9.]+)_(?P<pos>[tmb][lcr])_(?P<bg>[a-z]+)_"
                     r"(?P<mode>image|video)_(?P<delegate>cpu|gpu)(?P<deg>_deg(-[a-z0-9]+)?)?$")
BIN_EDGES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def det_box_px(d):
    _, x, y, w, h = d
    return (x * W, y * H, (x + w) * W, (y + h) * H)


def pts_px(lm):
    return [(lm[2 * k] * W, lm[2 * k + 1] * H) for k in range(len(lm) // 2)]


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def r(v, n=4):
    return None if v is None else round(v, n)


def build_gt(clip):
    frames = load(os.path.join(RAW, f"{clip}_s1_mc_black_image_cpu.jsonl"))
    gt = []
    unboxed = 0
    for fr in frames:
        hands = []
        dets = [det_box_px(d) for d in fr["det"]]
        used = set()
        labels = {}
        for hand in fr["hands"]:
            pts = pts_px(hand["lm"])
            c = centroid([pts[k] for k in PALM_IDX])
            best, best_d = None, None
            for j, b in enumerate(dets):
                if j in used:
                    continue
                d = math.dist(c, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                if best_d is None or d < best_d:
                    best, best_d = j, d
            box = None
            if best is not None:
                used.add(best)
                box = dets[best]
            else:
                unboxed += 1
            label = hand["h"]
            labels[label] = labels.get(label, 0) + 1
            ident = label if labels[label] == 1 else f"{label}#{labels[label]}"
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            hands.append({"id": ident, "pts": pts, "box": box,
                          "size": max(max(xs) - min(xs), max(ys) - min(ys))})
        gt.append(hands)
    return gt, unboxed


def map_gt(hands, scale, pos):
    w = round(W * scale)
    h = round(H * scale)
    sx, sy = w / W, h / H
    x0 = {"l": 0, "c": (W - w) // 2, "r": W - w}[pos[1]]
    y0 = {"t": 0, "m": (H - h) // 2, "b": H - h}[pos[0]]
    out = []
    for hand in hands:
        pts = [(x0 + x * sx, y0 + y * sy) for x, y in hand["pts"]]
        box = None
        if hand["box"] is not None:
            bx0, by0, bx1, by1 = hand["box"]
            box = (x0 + bx0 * sx, y0 + by0 * sy, x0 + bx1 * sx, y0 + by1 * sy)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append({"id": hand["id"], "pts": pts, "box": box,
                    "size": max(max(xs) - min(xs), max(ys) - min(ys)),
                    "palm_px": (math.sqrt((box[2] - box[0]) * (box[3] - box[1])) * LETTERBOX) if box else None})
    return out


def match_dets(gts, dets):
    """Greedy one-to-one by IoU. Returns {gt_index: det_index}, unassigned det count."""
    pairs = []
    for i, g in enumerate(gts):
        if g["box"] is None:
            continue
        for j, d in enumerate(dets):
            v = iou(g["box"], d)
            if v >= IOU_MIN:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    gi, dj, m = set(), set(), {}
    for _, i, j in pairs:
        if i in gi or j in dj:
            continue
        gi.add(i)
        dj.add(j)
        m[i] = j
    return m, len(dets) - len(dj)


def match_hands(gts, hands):
    pairs = []
    for i, g in enumerate(gts):
        gc = centroid([g["pts"][k] for k in PALM_IDX])
        for j, hand in enumerate(hands):
            pts = pts_px(hand["lm"])
            d = math.dist(gc, centroid([pts[k] for k in PALM_IDX]))
            if d <= HAND_MATCH * g["size"]:
                pairs.append((d, i, j))
    pairs.sort()
    gi, hj, m = set(), set(), {}
    for _, i, j in pairs:
        if i in gi or j in hj:
            continue
        gi.add(i)
        hj.add(j)
        m[i] = j
    return m


def analyse_image(meta, frames, gt, instances_out):
    scale, pos = meta["scale"], meta["pos"]
    n_gt = n_boxed = det_hits = hand_hits = 0
    best_scores, no_cand, match_above, above_counts, lm_errs, palm_pxs = [], 0, [], [], [], []
    false_dets = frames_with_false = 0
    for fr, hands in zip(frames, gt):
        gts = map_gt(hands, scale, pos)
        dets = [det_box_px(d) for d in fr.get("det", [])]
        cands = [(d[0], det_box_px(d)) for d in fr.get("cand", [])]
        above_counts.append(fr.get("above", 0))
        dm, unassigned = match_dets(gts, dets)
        false_dets += unassigned
        frames_with_false += unassigned > 0
        hm = match_hands(gts, fr["hands"])
        for i, g in enumerate(gts):
            n_gt += 1
            found = i in hm
            hand_hits += found
            err = None
            if found:
                pts = pts_px(fr["hands"][hm[i]]["lm"])
                err = sum(math.dist(a, b) for a, b in zip(pts, g["pts"])) / len(pts) / g["size"]
                lm_errs.append(err)
            rec = {"cfg": meta["name"], "f": fr["i"], "id": g["id"], "hand": found,
                   "lm_err": r(err)}
            if g["box"] is not None:
                n_boxed += 1
                palm_pxs.append(g["palm_px"])
                detected = i in dm
                det_hits += detected
                matching = [s for s, b in cands if iou(g["box"], b) >= IOU_MIN]
                best = max(matching) if matching else None
                if best is None:
                    no_cand += 1
                else:
                    best_scores.append(best)
                match_above.append(sum(s >= THRESHOLD for s in matching))
                rec.update({"palm_px": r(g["palm_px"], 3), "det": detected, "best": r(best),
                            "n_above": match_above[-1]})
            if instances_out is not None:
                instances_out.append(rec)
    return {
        "frames": len(frames),
        "gt_hands": n_gt,
        "gt_palms": n_boxed,
        "palm_px_median": r(pct(palm_pxs, 0.5), 2),
        "palm_px_p10": r(pct(palm_pxs, 0.1), 2),
        "palm_px_p90": r(pct(palm_pxs, 0.9), 2),
        "detected": det_hits,
        "detection_rate": r(det_hits / n_boxed if n_boxed else None),
        "hands_found": hand_hits,
        "hand_rate": r(hand_hits / n_gt if n_gt else None),
        "best_candidate_score_median": r(pct(best_scores, 0.5)),
        "best_candidate_score_p10": r(pct(best_scores, 0.1)),
        "palms_without_matching_candidate": no_cand,
        "matching_candidates_above_threshold_mean": r(statistics.mean(match_above) if match_above else None, 3),
        "above_threshold_per_frame_mean": r(statistics.mean(above_counts) if above_counts else None, 3),
        "false_detections": false_dets,
        "frames_with_false_detection": frames_with_false,
        "landmark_error_median": r(pct(lm_errs, 0.5)),
        "landmark_error_p90": r(pct(lm_errs, 0.9)),
    }


def analyse_video(meta, frames, gt):
    scale, pos = meta["scale"], meta["pos"]
    # Per GT identity: list of (frame, present, matched).
    timeline = {}
    n_gt = hits = 0
    first_gt = first_hit = None
    for fr, hands in zip(frames, gt):
        gts = map_gt(hands, scale, pos)
        hm = match_hands(gts, fr["hands"])
        present = {g["id"]: (i in hm) for i, g in enumerate(gts)}
        n_gt += len(gts)
        hits += len(hm)
        if gts and first_gt is None:
            first_gt = fr["i"]
        if hm and first_hit is None:
            first_hit = fr["i"]
        for ident in set(timeline) | set(present):
            timeline.setdefault(ident, []).append((ident in present, present.get(ident, False)))
    acq, never, loss_len, censored, losses = [], 0, [], 0, 0
    segments = 0
    for ident, tl in timeline.items():
        k = 0
        while k < len(tl):
            if not tl[k][0]:
                k += 1
                continue
            start = k
            while k < len(tl) and tl[k][0]:
                k += 1
            seg = [m for _, m in tl[start:k]]
            segments += 1
            if True in seg:
                acq.append(seg.index(True))
            else:
                never += 1
                continue
            j = seg.index(True)
            while j < len(seg):
                if seg[j]:
                    j += 1
                    continue
                losses += 1
                lost_at = j
                while j < len(seg) and not seg[j]:
                    j += 1
                if j < len(seg):
                    loss_len.append(j - lost_at)
                else:
                    censored += 1
    return {
        "frames": len(frames),
        "gt_hands": n_gt,
        "hands_found": hits,
        "hand_rate": r(hits / n_gt if n_gt else None),
        "clip_first_gt_frame": first_gt,
        "clip_first_found_frame": first_hit,
        "appearances": segments,
        "appearances_acquired": len(acq),
        "appearances_never_acquired": never,
        "frames_to_first_detection_median": pct(acq, 0.5),
        "frames_to_first_detection_p90": pct(acq, 0.9),
        "frames_to_first_detection_max": max(acq) if acq else None,
        "losses": losses,
        "losses_reacquired": len(loss_len),
        "losses_not_reacquired_before_hand_left": censored,
        "frames_to_reacquire_median": pct(loss_len, 0.5),
        "frames_to_reacquire_p90": pct(loss_len, 0.9),
        "frames_to_reacquire_max": max(loss_len) if loss_len else None,
        "frames_to_first_detection_list": acq,
        "frames_to_reacquire_list": loss_len,
    }


def bin_curve(instances, key):
    bins = []
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:]):
        sel = [x for x in instances if x.get("palm_px") is not None and lo <= x["palm_px"] < hi]
        if not sel:
            continue
        bins.append({"lo": lo, "hi": hi, "n": len(sel),
                     "palm_px_mean": r(statistics.mean(x["palm_px"] for x in sel), 2),
                     "rate": r(sum(bool(x[key]) for x in sel) / len(sel))})
    return bins


def crossing(bins, level, min_n=30):
    """Palm px where the binned rate first falls below `level`, walking from
    large palms to small ones; linear between neighbouring bin means. Bins
    with fewer than min_n instances are skipped."""
    pts = [(b["palm_px_mean"], b["rate"]) for b in bins if b["n"] >= min_n]
    pts.sort(reverse=True)
    for (x1, y1), (x0, y0) in zip(pts, pts[1:]):
        if y1 >= level > y0:
            return r(x0 + (x1 - x0) * (level - y0) / (y1 - y0), 2)
    if pts and pts[0][1] < level:
        return f">{pts[0][0]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="two-hands")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances")
    args = ap.parse_args()
    gt, unboxed = build_gt(args.clip)
    configs, videos, instances = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW, f"{args.clip}_*.jsonl"))):
        if os.path.exists(path + ".partial"):
            continue
        name = os.path.basename(path)[:-6]
        m = NAME_RE.match(name)
        if not m:
            continue
        meta = {"name": name, "clip": m["clip"], "scale": float(m["scale"]), "pos": m["pos"],
                "bg": m["bg"], "mode": m["mode"], "delegate": m["delegate"], "degraded": (m["deg"][5:] or "all") if m["deg"] else ""}
        frames = load(path)
        if len(frames) != len(gt):
            print(f"skip {name}: {len(frames)} frames, GT has {len(gt)}")
            continue
        if meta["mode"] == "image":
            inst = []
            meta.update(analyse_image(meta, frames, gt, inst))
            for x in inst:
                x.update({"scale": meta["scale"], "bg": meta["bg"], "delegate": meta["delegate"],
                          "degraded": meta["degraded"], "pos": meta["pos"]})
            instances += inst
            configs.append(meta)
        else:
            meta.update(analyse_video(meta, frames, gt))
            videos.append(meta)
    curves = {}
    groups = {}
    for x in instances:
        groups.setdefault((x["delegate"], x["degraded"], x["bg"]), []).append(x)
        groups.setdefault((x["delegate"], x["degraded"], "all"), []).append(x)
    for (delegate, degraded, bg), sel in sorted(groups.items()):
        key = f"{delegate}_{('degraded-' + degraded) if degraded else 'clean'}_{bg}"
        det = bin_curve(sel, "det")
        hand = bin_curve(sel, "hand")
        curves[key] = {"detection": det, "hand": hand,
                       "detection_below_90": crossing(det, 0.9), "detection_below_50": crossing(det, 0.5),
                       "hand_below_90": crossing(hand, 0.9), "hand_below_50": crossing(hand, 0.5)}
    result = {
        "definitions": __doc__,
        "gt": {"clip": args.clip, "frames": len(gt), "hands": sum(len(h) for h in gt),
               "hands_without_palm_box": unboxed},
        "image": configs,
        "video": videos,
        "curves": curves,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if args.instances:
        with gzip.open(args.instances, "wt") as f:
            for x in instances:
                f.write(json.dumps(x, separators=(",", ":")) + "\n")
    print(f"{len(configs)} image configs, {len(videos)} video configs, {len(instances)} instances")


if __name__ == "__main__":
    main()
