#!/usr/bin/env python3
"""Builds the synthetic scenes of the palm search evaluation (phase 2).

    python p2_scenes.py render [scene ...]     # all scenes when none is named
    python p2_scenes.py list

Needs numpy and OpenCV. Each scene is written once, as a lossless H.264 RGB
file plus its reference, so that every configuration reads the same frames:

    $P2_OUT/scenes/<scene>.mkv        1280x720, libx264rgb -qp 0
    $P2_OUT/scenes/<scene>.gt.json    per frame: the hands that are in it

How a scene is made
- Source: two-hands-only.mov and its full-size IMAGE-mode CPU run (the
  reference of the falloff sweep). Hands of that run are linked into tracks by
  position (nearest palm centroid, at most 250 px, at most 8 frames apart).
- A layer is either the whole source frame or the cut-out of one track: a
  square of 1.7 x the track's median hand size around the centre of the hand's
  landmark box, clipped to the source frame.
- A layer is shrunk with an area filter (cv2.INTER_AREA) and pasted on the
  room still, or for "<scene>_wall" on source frame 620, the clip's own wall
  with no hand in it. Cut-outs are blended in over a 12 % ramp at their edges; whole
  frames are pasted with a hard edge, as in the falloff sweep.
- A layer plays a stretch of its track forwards and backwards (ping-pong), so
  the hand stays in view without a jump. A cut-out keeps the hand's own motion
  in the source frame, scaled, around the layer's anchor (times `motion`).
- Reference of a hand in a scene frame: the landmarks and palm box of its
  source frame, mapped through the same shrink and offset. A source frame in
  which the full-size run has no hand for the track still shows the cut-out
  (position interpolated); the hand is then listed with "pts": null.
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "test_projects", "shared"))
CLIP = os.path.join(SHARED, "two-hands-only.mov")
REFERENCE = os.environ.get("P2_REFERENCE", "/tmp/palmsweep/raw/two-hands_s1_mc_black_image_cpu.jsonl")
ROOM = os.environ.get("P2_ROOM", "/tmp/palmsweep/bg_room.png")
OUT = os.environ.get("P2_OUT", "/tmp/palmp2")
SCENES = os.path.join(OUT, "scenes")

W, H = 1280, 720
N = 645
LETTERBOX = 192 / max(W, H)
PALM_IDX = [0, 5, 9, 13, 17]
CUTOUT = 1.7      # cut-out side, in median hand sizes
FEATHER = 0.12    # blend ramp, share of the cut-out side


# ---------------------------------------------------------------- reference

def load_reference():
    with open(REFERENCE) as f:
        frames = [json.loads(line) for line in f]
    out = []
    for fr in frames:
        dets = [(d[1] * W, d[2] * H, (d[1] + d[3]) * W, (d[2] + d[4]) * H) for d in fr["det"]]
        used = set()
        hands = []
        for hand in fr["hands"]:
            lm = hand["lm"]
            pts = [(lm[2 * k] * W, lm[2 * k + 1] * H) for k in range(21)]
            c = (sum(pts[k][0] for k in PALM_IDX) / 5, sum(pts[k][1] for k in PALM_IDX) / 5)
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
            hands.append({"c": c, "pts": pts, "box": box})
        out.append(hands)
    return out


def build_tracks(ref):
    """track id -> {source frame: hand}. Ids in order of first appearance."""
    tracks, last = [], {}
    for i, hands in enumerate(ref):
        cand = sorted((math.dist(h["c"], last[t][1]), t, j) for j, h in enumerate(hands)
                      for t in last if i - last[t][0] <= 8)
        used_t, used_j = set(), set()
        for d, t, j in cand:
            if t in used_t or j in used_j or d > 250:
                continue
            used_t.add(t)
            used_j.add(j)
            tracks[t][i] = hands[j]
            last[t] = (i, hands[j]["c"])
        for j, h in enumerate(hands):
            if j not in used_j:
                tracks.append({i: h})
                last[len(tracks) - 1] = (i, h["c"])
    return tracks


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def hand_size(pts):
    x0, y0, x1, y1 = bbox(pts)
    return max(x1 - x0, y1 - y0)


# ------------------------------------------------------------------- layers

def pingpong(n, first, last):
    """Source frame for scene step n, playing first..last..first.."""
    span = last - first
    k = n % (2 * span)
    return first + (k if k <= span else 2 * span - k)


class Layer:
    """One pasted copy. `visible` is a list of [from, to) scene frame ranges."""

    def __init__(self, ident, track, scale, anchor, segment, visible=None, start=0,
                 motion=1.0, flip=False):
        self.id = ident
        self.track = track          # track id, or None for the whole frame
        self.scale = scale
        self.anchor = anchor        # function n -> (x, y), or a fixed (x, y)
        self.segment = segment      # (first, last) source frames
        self.visible = visible or [(0, N)]
        self.start = start          # offset into the ping-pong
        self.motion = motion
        self.flip = flip

    def source(self, n):
        for a, b in self.visible:
            if a <= n < b:
                return pingpong(n - a + self.start, *self.segment)
        return None

    def anchor_at(self, n):
        return self.anchor(n) if callable(self.anchor) else self.anchor


def track_geometry(track, segment):
    """Per source frame of the segment: centre of the landmark box, interpolated
    where the track has no hand. Also the mean centre and median hand size."""
    first, last = segment
    centres = {}
    for i in range(first, last + 1):
        if i in track:
            x0, y0, x1, y1 = bbox(track[i]["pts"])
            centres[i] = ((x0 + x1) / 2, (y0 + y1) / 2)
    known = sorted(centres)
    for i in range(first, last + 1):
        if i in centres:
            continue
        before = [k for k in known if k < i]
        after = [k for k in known if k > i]
        if before and after:
            a, b = before[-1], after[0]
            t = (i - a) / (b - a)
            centres[i] = tuple(centres[a][d] + (centres[b][d] - centres[a][d]) * t for d in (0, 1))
        else:
            centres[i] = centres[(before or after)[-1 if before else 0]]
    mean = tuple(sum(centres[i][d] for i in centres) / len(centres) for d in (0, 1))
    sizes = sorted(hand_size(track[i]["pts"]) for i in known)
    return centres, mean, sizes[len(sizes) // 2]


def feather_mask(h, w):
    ramp = max(1, int(round(FEATHER * max(h, w))))
    ax = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).clip(max=ramp) / ramp
    ay = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).clip(max=ramp) / ramp
    return np.minimum.outer(ay, ax).astype(np.float32)[..., None]


def place(layer, n, clip, tracks, geo):
    """Returns (patch rgb, alpha or None, x, y, mapper, hands) or None.
    mapper maps a source point to scene pixels."""
    src = layer.source(n)
    if src is None:
        return None
    s = layer.scale
    ax, ay = layer.anchor_at(n)
    if layer.track is None:
        tw, th = round(W * s), round(H * s)
        patch = cv2.resize(clip[src], (tw, th), interpolation=cv2.INTER_AREA)
        ox, oy = int(round(ax)), int(round(ay))       # anchor: top-left corner
        sx, sy = tw / W, th / H
        x0 = y0 = 0.0
        alpha = None
        source_w = W
        hands = [(f"{layer.id}/t{t}", tr[src]) for t, tr in enumerate(tracks) if src in tr]
    else:
        centres, mean, size = geo[layer.id]
        cx, cy = centres[src]
        half = CUTOUT * size / 2
        x0, y0 = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
        x1, y1 = min(W, int(round(cx + half))), min(H, int(round(cy + half)))
        tw, th = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
        patch = cv2.resize(clip[src][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA)
        sx, sy = tw / (x1 - x0), th / (y1 - y0)
        source_w = x1 - x0
        # Scene position of the hand's box centre: the anchor plus the hand's
        # own displacement in the source frame.
        dx = (cx - mean[0]) * s * layer.motion * (-1 if layer.flip else 1)
        dy = (cy - mean[1]) * s * layer.motion
        local_cx = ((x1 - x0) - (cx - x0)) if layer.flip else (cx - x0)
        ox = int(round(ax + dx - local_cx * sx))
        oy = int(round(ay + dy - (cy - y0) * sy))
        alpha = feather_mask(th, tw)
        hand = tracks[layer.track].get(src)
        hands = [(layer.id, hand)]
    if layer.flip:
        patch = patch[:, ::-1]

    def mapper(p):
        lx = p[0] - x0
        if layer.flip:
            lx = source_w - lx
        return (ox + lx * sx, oy + (p[1] - y0) * sy)

    return patch, alpha, ox, oy, mapper, hands


def paste(canvas, patch, alpha, ox, oy):
    th, tw = patch.shape[:2]
    cx0, cy0 = max(0, ox), max(0, oy)
    cx1, cy1 = min(W, ox + tw), min(H, oy + th)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    part = patch[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    if alpha is None:
        canvas[cy0:cy1, cx0:cx1] = part
        return
    a = alpha[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = np.rint(region * (1 - a) + part.astype(np.float32) * a).astype(np.uint8)


def reference_hand(ident, hand, mapper):
    if hand is None:
        return {"id": ident, "pts": None, "box": None}
    pts = [mapper(p) for p in hand["pts"]]
    box = None
    if hand["box"] is not None:
        a = mapper((hand["box"][0], hand["box"][1]))
        b = mapper((hand["box"][2], hand["box"][3]))
        box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
    entry = {"id": ident, "pts": [[round(x, 3), round(y, 3)] for x, y in pts],
             "box": [round(v, 3) for v in box] if box else None,
             "size": round(hand_size(pts), 3)}
    if box:
        entry["palm_px"] = round(math.sqrt((box[2] - box[0]) * (box[3] - box[1])) * LETTERBOX, 3)
    return entry


# ------------------------------------------------------------------- scenes

def triangle(n, period):
    k = (n % period) / period
    return 2 * k if k < 0.5 else 2 * (1 - k)


def border_scene(scale):
    """The whole source frame, shrunk, moving over the canvas: 3.5 passes left
    to right while it goes once from top to bottom."""
    w, h = round(W * scale), round(H * scale)

    def anchor(n):
        return (triangle(n, 184) * (W - w), n / (N - 1) * (H - h))
    return [Layer("copy", None, scale, anchor, (0, N - 1))]


SEG_A = (389, 563)   # track 12: the hand on the right, 174 of 175 frames
SEG_B = (85, 241)    # track 2: the hand on the right, earlier, 155 of 157 frames
TRACK_A, TRACK_B = 12, 2

FAR_SPOTS = [(760, 150), (1100, 180), (900, 360), (1180, 540),
             (700, 600), (960, 90), (1160, 360), (820, 480)]


def far_layers(scale):
    """One far hand that appears eight times, 60 frames each, 15 frames apart,
    at eight places."""
    return [Layer(f"far{k}", TRACK_B, scale, spot, SEG_B, visible=[(30 + 75 * k, 90 + 75 * k)],
                  start=17 * k) for k, spot in enumerate(FAR_SPOTS)]


def nearfar_scene(scale, near=True):
    layers = far_layers(scale)
    if near:
        layers.insert(0, Layer("near", TRACK_A, 0.8, (260, 400), SEG_A, motion=0.5))
    return layers


def crowd4_scene():
    return [
        Layer("n1", TRACK_A, 0.7, (290, 440), SEG_A, motion=0.5),
        Layer("n2", TRACK_B, 0.45, (980, 480), SEG_B, visible=[(0, 400), (520, N)]),
        Layer("f1", TRACK_A, 0.18, (640, 150), SEG_A, visible=[(60, N)], start=70, flip=True),
        Layer("f2", TRACK_B, 0.14, (1050, 170), SEG_B, visible=[(150, 500)], start=40, flip=True),
    ]


def crowd10_scene():
    """One near hand on the left, nine smaller ones in a 3 x 3 block on the
    right. (scale, track, visible, flip)"""
    layers = [Layer("h0", TRACK_A, 0.7, (225, 400), SEG_A, motion=0.3)]
    others = [
        (0.30, TRACK_B, [(0, N)], True), (0.18, TRACK_A, [(40, 560)], True),
        (0.25, TRACK_B, [(80, N)], False), (0.14, TRACK_A, [(120, 500)], False),
        (0.22, TRACK_B, [(0, 420), (540, N)], False), (0.18, TRACK_B, [(160, N)], True),
        (0.14, TRACK_B, [(100, 380), (470, N)], False), (0.12, TRACK_A, [(200, 440)], True),
        (0.16, TRACK_A, [(180, 600)], False),
    ]
    for k, (scale, track, visible, flip) in enumerate(others):
        anchor = (590 + 276 * (k % 3), 120 + 240 * (k // 3))
        segment = SEG_A if track == TRACK_A else SEG_B
        # The hand's own motion is scaled so that it stays inside its slot.
        motion = min(1.0, 40 / (250 * scale))
        layers.append(Layer(f"h{k + 1}", track, scale, anchor, segment, visible=visible,
                            start=23 * k, motion=motion, flip=flip))
    return layers


SCENE_BUILDERS = {
    "border_s018": lambda: border_scene(0.18),
    "border_s035": lambda: border_scene(0.35),
    "nearfar_s018": lambda: nearfar_scene(0.18),
    "nearfar_s025": lambda: nearfar_scene(0.25),
    "faronly_s018": lambda: nearfar_scene(0.18, near=False),
    "crowd4": crowd4_scene,
    "crowd10": crowd10_scene,
}


# ------------------------------------------------------------------- render

def load_clip():
    cmd = ["ffmpeg", "-v", "error", "-i", CLIP, "-vf",
           "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


WALL_FRAME = 620     # a source frame without hands: the clip's own wall


def render(name, clip, ref, tracks, room):
    # "<scene>_wall": the same layers on the clip's own empty wall.
    base = name[:-5] if name.endswith("_wall") else name
    if base != name:
        room = clip[WALL_FRAME]
    layers = SCENE_BUILDERS[base]()
    geo = {l.id: track_geometry(tracks[l.track], l.segment) for l in layers if l.track is not None}
    os.makedirs(SCENES, exist_ok=True)
    path = os.path.join(SCENES, name + ".mkv")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", "100000/3333", "-i", "-", "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast",
         "-pix_fmt", "rgb24", path], stdin=subprocess.PIPE)
    gt = []
    for n in range(N):
        canvas = room.copy()
        hands = []
        for layer in layers:
            placed = place(layer, n, clip, tracks, geo)
            if placed is None:
                continue
            patch, alpha, ox, oy, mapper, layer_hands = placed
            paste(canvas, patch, alpha, ox, oy)
            hands += [reference_hand(ident, hand, mapper) for ident, hand in layer_hands]
        gt.append(hands)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit(f"{name}: encoder failed")
    with open(os.path.join(SCENES, name + ".gt.json"), "w") as f:
        json.dump({"scene": name, "width": W, "height": H, "frames": gt}, f)
    shown = sum(len(h) for h in gt)
    print(f"{name}: {N} frames, {shown} hand instances, {os.path.getsize(path) / 1e6:.0f} MB", flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print("\n".join(SCENE_BUILDERS))
        return
    names = sys.argv[2:] or [n + suffix for suffix in ("", "_wall") for n in SCENE_BUILDERS]
    ref = load_reference()
    tracks = build_tracks(ref)
    clip = load_clip()
    room = cv2.cvtColor(cv2.imread(ROOM, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert room.shape == (H, W, 3), room.shape
    for name in names:
        render(name, clip, ref, tracks, room)


if __name__ == "__main__":
    main()
