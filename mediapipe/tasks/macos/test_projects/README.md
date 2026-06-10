# MediaPipe parity test projects

Cross-checks the native macOS `MediaPipeTasksMac` package against the official
MediaPipe **Web** Tasks-Vision library by running the same models on the same
image and comparing the results.

```
test_projects/
  shared/    download_models.sh, models/ (gitignored), test_image.jpg, output/ (gitignored)
  web/       Vite + TypeScript app using @mediapipe/tasks-vision  -> shared/output/web.json
  macos/     native .app depending on the local package           -> shared/output/macos.json
  compare/   compare.py: counts + x/y tolerance + handedness
```

Downloaded models and generated JSON are **not committed** (only `.gitkeep`
files and these READMEs are).

## Shared test image

`shared/test_image.jpg` is copied from a committed repo asset:
`mediapipe/model_maker/python/vision/gesture_recognizer/testdata/raw_data/call/17d804b5-7118-462d-8191-58d764f591b8.jpg`

It has one visible hand gesture, one visible face, and enough upper body for
pose — good for an initial smoke/parity test.

> **Pose caveat:** this image is upper-body / cropped, so pose parity here only
> exercises the visible upper-body landmarks. `download_models.sh` also fetches
> `shared/test_image_full_body.jpg` for thorough pose validation:
>
> ```bash
> # web: open with query params
> #   http://localhost:5173/?image=test_image_full_body.jpg&out=web_pose.json
> # macOS: point config.json's "image"/"output" at the full-body image + macos_pose.json
> python3 compare/compare.py --web shared/output/web_pose.json --macos shared/output/macos_pose.json --tasks pose
> ```
>
> Keep hand/face at their tight tolerances; pose defaults to a looser `0.04`
> while full-body/options parity is validated.

## Workflow

```bash
# 1. Shared models + image
./shared/download_models.sh

# 2. Web reference  ->  shared/output/web.json
cd web && npm install && npm run setup && npm run dev      # runs in the browser, saves JSON

# 3. Native macOS  ->  shared/output/macos.json
cd ../macos && ./build_app.sh && open ParitySmokeTest.app --args "$(pwd)/config.json"

# 4. Compare
cd ../compare && python3 compare.py
```

## Delegate / running mode

Both projects record the `delegate` and `runningMode` they used in the envelope
(`options.*`). Parity is validated for **CPU/IMAGE** and **CPU/VIDEO** first:

- **Web**: `?mode=image|video` and `?delegate=CPU|GPU` query params, e.g.
  `http://localhost:5173/?mode=video&out=web_video.json`.
- **macOS**: set `"runningMode": "VIDEO"` (and optionally `"delegate"`) in
  `config.json`; VIDEO runs a single frame at timestamp 0.

GPU is now supported on macOS (Metal): the default artifact is GPU-capable and
the native side accepts `.gpu`. For **GPU parity**, run web with
`?delegate=GPU` and macOS with `"delegate": "GPU"` in `config.json`, then
compare. (Web GPU uses WebGL; small numeric differences vs native Metal are
expected — keep them report-only at first.)

## Parity JSON envelope

Both projects emit the same shape (`source` is `"web"` or `"macos"`):

```jsonc
{
  "source": "web",                         // or "macos"
  "image": "test_image.jpg",
  "models":  { "hand": "<sha256>", "pose": "<sha256>", "face": "<sha256>" },
  "options": {                             // the exact options each engine used
    "hand": { "numHands": 2, "minHandDetectionConfidence": 0.5, …, "runningMode": "IMAGE", "delegate": "CPU" },
    "pose": { "numPoses": 1, …, "outputSegmentationMasks": false, "runningMode": "IMAGE", "delegate": "CPU" },
    "face": { "numFaces": 1, …, "outputFaceBlendshapes": true, "outputFacialTransformationMatrixes": true, … }
  },
  "hand": {
    "landmarks":      [[{ "x": .., "y": .., "z": .., "visibility": 0, "presence": 0 }, … ]],  // 21 per hand
    "worldLandmarks": [[ … ]],                                                                  // 21 per hand
    "handedness":     [[{ "index": .., "score": .., "categoryName": "Left"|"Right", "displayName": .. }]],
    "handednesses":   [[ … ]]                                                                   // deprecated alias
  },
  "pose": {
    "landmarks":         [[ … ]],   // 33 per pose
    "worldLandmarks":    [[ … ]],   // 33 per pose
    "segmentationMasks": null       // not produced yet
  },
  "face": {
    "faceLandmarks":               [[ … ]],   // 478 per face
    "faceBlendshapes":             [{ "categories": [ … ], "headIndex": 0, "headName": null }],  // 52 categories
    "facialTransformationMatrixes":[{ "rows": 4, "columns": 4, "data": [ …16… ] }]
  }
}
```

`visibility`/`presence` are always numbers (0 where the model omits them, e.g.
hand/face) for Web-compatible comparison. `compare.py` enforces detection +
landmark counts, normalized x/y per-task tolerances (hand 0.02, face 0.01, pose
0.04), model-SHA256 equality, and hand handedness labels; z, world landmarks,
blendshapes, and matrices are report-only.

See each subdirectory's README for details.
