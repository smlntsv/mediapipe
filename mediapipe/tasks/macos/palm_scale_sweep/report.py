#!/usr/bin/env python3
"""Turns analyze.py output into Markdown tables and Mermaid charts.

    python3 report.py results.json instances.jsonl.gz > tables.md
"""
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict

import analyze

FALLOFF = (0.5, 0.35, 0.25)
W_ = 1280
POS_ORDER = ["tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br"]


def pct(v, p):
    return analyze.pct(v, p)


def f(v, n=2):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    return f"{v:.{n}f}"


def rate(a, b):
    return f"{100 * a / b:.1f}%" if b else "–"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def pooled_image(configs, keyfn):
    groups = defaultdict(list)
    for c in configs:
        groups[keyfn(c)].append(c)
    return groups


def image_rows(groups, instances_by_cfg):
    rows = []
    for key in sorted(groups, key=lambda k: (-k[0],) + tuple(str(x) for x in k[1:])):
        cs = groups[key]
        inst = [x for c in cs for x in instances_by_cfg[c["name"]]]
        px = [x["palm_px"] for x in inst if x.get("palm_px") is not None]
        best = [x["best"] for x in inst if x.get("best") is not None]
        lm = [x["lm_err"] for x in inst if x.get("lm_err") is not None]
        palms = sum(c["gt_palms"] for c in cs)
        frames = sum(c["frames"] for c in cs)
        rows.append(list(key) + [
            f"{f(pct(px, .5))} ({f(pct(px, .1))}–{f(pct(px, .9))})",
            palms,
            rate(sum(c["detected"] for c in cs), palms),
            rate(sum(c["hands_found"] for c in cs), sum(c["gt_hands"] for c in cs)),
            f(pct(best, .5), 3), f(pct(best, .1), 3),
            sum(c["palms_without_matching_candidate"] for c in cs),
            f(statistics.mean(x["n_above"] for x in inst if "n_above" in x), 2),
            f(sum(c["above_threshold_per_frame_mean"] * c["frames"] for c in cs) / frames, 2),
            sum(c["false_detections"] for c in cs),
            f(1000 * sum(c["false_detections"] for c in cs) / frames, 1),
            f(pct(lm, .5), 4), f(pct(lm, .9), 4),
        ])
    return rows


IMAGE_HEADER = ["palm px median (p10–p90)", "GT palms", "detection rate", "hand found rate",
                "best cand. score median", "best cand. score p10", "palms with no matching cand. in top 50",
                "matching cand. ≥ 0.5 per palm (mean)", "cand. ≥ 0.5 per frame (mean)",
                "false detections", "false det. per 1000 frames", "landmark error median", "landmark error p90"]


def curve(instances, key="det"):
    return analyze.bin_curve(instances, key)


def mermaid(title, series, bins_lo_hi):
    labels = [f"{lo:g}-{hi:g}" for lo, hi in bins_lo_hi]
    lines = ["```mermaid", "xychart-beta", f'    title "{title}"',
             "    x-axis \"palm px in the 192x192 detector input\" [" + ", ".join(f'"{l}"' for l in labels) + "]",
             '    y-axis "detection rate %" 0 --> 100']
    for _, values in series:
        lines.append("    line [" + ", ".join(f"{v:.1f}" for v in values) + "]")
    lines.append("```")
    return "\n".join(lines)


def curve_section(name, groups, min_n=30):
    """groups: [(label, instances)]. Prints a table and a Mermaid chart over the bins every series has."""
    curves = [(label, {(b["lo"], b["hi"]): b for b in curve(inst)}) for label, inst in groups]
    common = sorted(set.intersection(*[{k for k, b in c.items() if b["n"] >= min_n} for _, c in curves]))
    rows = []
    for k in common:
        rows.append([f"{k[0]:g}–{k[1]:g}"] + [f"{100 * c[k]['rate']:.1f}% (n={c[k]['n']}, mean {c[k]['palm_px_mean']})"
                                             for _, c in curves])
    out = [table(["palm px bin"] + [label for label, _ in groups], rows), ""]
    out.append(mermaid(name, [(label, [100 * c[k]["rate"] for k in common]) for label, c in curves], common))
    out.append("")
    out.append("Lines in order: " + "; ".join(f"{i + 1}. {label}" for i, (label, _) in enumerate(groups)) + ".")
    xs = []
    for label, c in curves:
        bins = [c[k] for k in sorted(c)]
        xs.append([label, f(analyze.crossing(bins, 0.9, min_n)), f(analyze.crossing(bins, 0.5, min_n))])
    return "\n".join(out), xs


def main():
    res = json.load(open(sys.argv[1]))
    with gzip.open(sys.argv[2], "rt") as fh:
        instances = [json.loads(line) for line in fh]
    by_cfg = defaultdict(list)
    for x in instances:
        by_cfg[x["cfg"]].append(x)
    image = res["image"]
    cpu_clean = [c for c in image if c["delegate"] == "cpu" and not c["degraded"]]
    out = []

    out.append("## T1. CPU, clean, per scale and background (all 9 positions pooled)\n")
    g = pooled_image(cpu_clean, lambda c: (c["scale"], c["bg"]))
    out.append(table(["scale", "background"] + IMAGE_HEADER, image_rows(g, by_cfg)))

    out.append("\n## T2. Detection rate per position, CPU clean (rows: scale, background; columns: position)\n")
    out.append("Positions: t/m/b = top/middle/bottom row, l/c/r = left/centre/right column of the 3×3 grid.\n")
    rows = []
    for scale in sorted({c["scale"] for c in cpu_clean if c["scale"] < 1}, reverse=True):
        for bg in ("black", "room"):
            cells = []
            for pos in POS_ORDER:
                cs = [c for c in cpu_clean if c["scale"] == scale and c["bg"] == bg and c["pos"] == pos]
                cells.append(rate(sum(c["detected"] for c in cs), sum(c["gt_palms"] for c in cs)) if cs else "–")
            rows.append([scale, bg] + cells)
    out.append(table(["scale", "background"] + POS_ORDER, rows))

    out.append("\n## T3. Falloff scales: CPU clean, CPU degraded, GPU clean (positions pooled)\n")
    sel = [c for c in image if c["scale"] in FALLOFF]
    g = pooled_image(sel, lambda c: (c["scale"], c["bg"], c["delegate"] + ((" degraded" + ("" if c["degraded"] == "all" else " (" + c["degraded"] + " only)")) if c["degraded"] else " clean")))
    out.append(table(["scale", "background", "run"] + IMAGE_HEADER, image_rows(g, by_cfg)))

    out.append("\n## T4. Every IMAGE-mode configuration\n")
    g = pooled_image(image, lambda c: (c["scale"], c["pos"], c["bg"], c["delegate"] + ((" degraded" + ("" if c["degraded"] == "all" else " (" + c["degraded"] + " only)")) if c["degraded"] else " clean")))
    out.append(table(["scale", "position", "background", "run"] + IMAGE_HEADER, image_rows(g, by_cfg)))

    SCALES_ALL = sorted({c["scale"] for c in image}, reverse=True)
    SCALE_PX = [(sc, next(c["palm_px_median"] for c in image if c["scale"] == sc)) for sc in SCALES_ALL]

    out.append("\n## C1. Detection rate against palm pixels, CPU clean, all scales\n")
    sec, xs1 = curve_section("CPU clean: black (1), room (2)", [
        ("CPU clean, black", [x for x in instances if x["delegate"] == "cpu" and not x["degraded"] and x["bg"] == "black"]),
        ("CPU clean, room", [x for x in instances if x["delegate"] == "cpu" and not x["degraded"] and x["bg"] == "room"]),
    ])
    out.append(sec)

    # Per-scale points: (median palm px of the scale, pooled detection rate).
    def scale_points(cs):
        pts = []
        for scale in sorted({c["scale"] for c in cs}, reverse=True):
            sc = [c for c in cs if c["scale"] == scale]
            palms = sum(c["gt_palms"] for c in sc)
            pts.append((scale, sc[0]["palm_px_median"], sum(c["detected"] for c in sc) / palms))
        return pts

    def scale_crossing(pts, level):
        for (s1, x1, y1), (s0, x0, y0) in zip(pts, pts[1:]):
            if y1 >= level > y0:
                return f(x0 + (x1 - x0) * (level - y0) / (y1 - y0))
        if pts and pts[0][2] < level:
            return f">{f(pts[0][1])} (not reached)"
        if pts and pts[-1][2] >= level:
            return f"<{f(pts[-1][1])} (not reached)"
        return "–"

    def series_configs(delegate, degraded, bg):
        return [c for c in image if c["delegate"] == delegate and c["degraded"] == degraded
                and (bg == "both" or c["bg"] == bg) and (c["scale"] < 1 or delegate == "cpu" and not degraded)]

    series = []
    for label, delegate, degraded in (("CPU clean", "cpu", ""), ("CPU degraded", "cpu", "all"), ("GPU clean", "gpu", "")):
        for bg in ("black", "room", "both"):
            cs = series_configs(delegate, degraded, bg)
            if cs:
                series.append((f"{label}, {bg}", scale_points(cs)))

    out.append("\n## C2. Detection rate per scale: CPU clean, CPU degraded, GPU clean (positions pooled)\n")
    common = sorted(set.intersection(*[{p[0] for p in pts} for _, pts in series if "both" not in _]), reverse=True)
    rows = []
    for name, pts in series:
        by = {p[0]: p for p in pts}
        rows.append([name] + [f"{100 * by[sc][2]:.1f}%" if sc in by else "–" for sc in SCALES_ALL])
    out.append(table(["series"] + [f"s={sc:g} ({f(px)} px)" for sc, px in SCALE_PX], rows))
    out.append("")
    chart = [(n, p) for n, p in series if "both" not in n]
    labels = [f"{sc:g} / {f(dict((a, b) for a, b in SCALE_PX)[sc], 1)} px" for sc in common]
    out.append("```mermaid\nxychart-beta\n    title \"Detection rate per scale (palm px median)\"\n"
               "    x-axis \"scale / palm px in the detector input\" [" + ", ".join(f'"{l}"' for l in labels) + "]\n"
               "    y-axis \"detection rate %\" 0 --> 100")
    for n, pts in chart:
        by = {p[0]: p for p in pts}
        out.append("    line [" + ", ".join(f"{100 * by[sc][2]:.1f}" for sc in common) + "]")
    out.append("```\n")
    out.append("Lines in order: " + "; ".join(f"{i + 1}. {n}" for i, (n, _) in enumerate(chart)) + ".")

    out.append("\n## T5. Palm px where the detection rate falls below 90% and 50%\n")
    out.append("Primary: per-scale points (median palm px of the scale, detection rate pooled over the 9 positions), "
               "linear between neighbouring scales. Secondary: the binned curve C1 (per GT palm, bins with n ≥ 30), "
               "which mixes frames of different scales in one bin.\n")
    rows = [[n, "per scale", scale_crossing(pts, 0.9), scale_crossing(pts, 0.5)] for n, pts in series]
    rows += [[a, "binned (C1)", b, c] for a, b, c in xs1]
    out.append(table(["series", "method", "below 90%", "below 50%"], rows))

    # Tiling translation: palm px of a far hand per layout, rates read off the per-scale points.
    def interp(pts, x):
        pts = sorted((p[1], p[2]) for p in pts)
        if x <= pts[0][0]:
            return None if x < pts[0][0] else pts[0][1]
        if x >= pts[-1][0]:
            return None if x > pts[-1][0] else pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    video_pts = []
    for scale in sorted({v["scale"] for v in res["video"]}, reverse=True):
        vs = [v for v in res["video"] if v["scale"] == scale]
        px = next(c["palm_px_median"] for c in image if c["scale"] == scale)
        video_pts.append((scale, px, sum(v["hands_found"] for v in vs) / sum(v["gt_hands"] for v in vs)))
    lookup = dict(series)
    layouts = [("none", "1280×720", 1.0), ("2 square tiles", "720×720, 160 px overlap", W_ / 720),
               ("2×2", "640×360", 2.0), ("4×4", "320×180", 4.0)]
    out.append("\n## T6. Tiling translation (derived, not measured)\n")
    out.append("Palm px of the same hand when each tile is letterboxed into 192×192 by its longer side "
               "(gain = 1280 / tile's longer side), and the detection rate at that palm px read off the per-scale "
               "points above (linear between scales; – outside the measured range). Untiled 5 px is the spec's "
               "figure for 2.5 m; 4 and 6 px bracket it.\n")
    rows = []
    for untiled in (4.0, 5.0, 6.0):
        for name, tile, gain in layouts:
            px = untiled * gain
            def pr(v):
                return "–" if v is None else f"{100 * v:.0f}%"
            rows.append([f(untiled, 1), name, tile, f"{gain:.2f}×", f(px, 1),
                         pr(interp(lookup["CPU clean, both"], px)),
                         pr(interp(lookup.get("CPU degraded, both", []) or [(0, 0, 0)], px)),
                         pr(interp(lookup.get("GPU clean, both", []) or [(0, 0, 0)], px)),
                         pr(interp(video_pts, px))])
    out.append(table(["untiled palm px", "layout", "tile", "gain", "palm px", "IMAGE CPU clean detection",
                      "IMAGE CPU degraded detection", "IMAGE GPU clean detection", "VIDEO CPU clean hand found"], rows))

    video = res["video"]
    if video:
        out.append("\n## T7. VIDEO mode, CPU clean (positions pooled; frame counts at 30 fps)\n")
        g = defaultdict(list)
        for v in video:
            g[(v["scale"], v["bg"])].append(v)
        rows = []
        for key in sorted(g, key=lambda k: (-k[0], k[1])):
            vs = g[key]
            acq = [a for v in vs for a in v["frames_to_first_detection_list"]]
            rq = [a for v in vs for a in v["frames_to_reacquire_list"]]
            first = [v["clip_first_found_frame"] - v["clip_first_gt_frame"] for v in vs
                     if v["clip_first_found_frame"] is not None and v["clip_first_gt_frame"] is not None]
            rows.append(list(key) + [
                rate(sum(v["hands_found"] for v in vs), sum(v["gt_hands"] for v in vs)),
                f"{f(pct(first, .5), 0)} / {max(first) if first else '–'} ({len(vs) - len(first)} runs never)",
                sum(v["appearances"] for v in vs), sum(v["appearances_acquired"] for v in vs),
                sum(v["appearances_never_acquired"] for v in vs),
                f"{f(pct(acq, .5), 0)} / {f(pct(acq, .9), 0)} / {max(acq) if acq else '–'}",
                sum(v["losses"] for v in vs), sum(v["losses_reacquired"] for v in vs),
                sum(v["losses_not_reacquired_before_hand_left"] for v in vs),
                f"{f(pct(rq, .5), 0)} / {f(pct(rq, .9), 0)} / {max(rq) if rq else '–'}",
            ])
        out.append(table(["scale", "background", "hand found rate",
                          "clip start → first hand, frames (median / max)",
                          "GT appearances", "acquired", "never acquired",
                          "frames to first detection (median / p90 / max)",
                          "losses", "re-acquired", "not re-acquired before the hand left GT",
                          "frames to re-acquire (median / p90 / max)"], rows))
    print("\n".join(out))


if __name__ == "__main__":
    main()
