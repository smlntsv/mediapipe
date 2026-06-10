# macOS parity smoke test (`.app`)

A minimal native macOS app that depends on the local root `MediaPipeTasksMac`
package (by relative path), runs HandLandmarker / PoseLandmarker /
FaceLandmarker on the shared test image, and writes
`../shared/output/macos.json` in the parity envelope
(see [../README.md](../README.md)).

## Prerequisites

- The `MediaPipeTasksC.xcframework` must be built first:
  `../../build_macos_xcframework.sh`.
- MacPorts OpenCV 3 must be installed (runtime dependency of the dylib):
  `sudo port install opencv3`.
- The shared models + image: `../shared/download_models.sh`.

## Build and run

```bash
./build_app.sh
open ParitySmokeTest.app --args "$(pwd)/config.json"
# or run the bundled binary directly (output goes to a file, not stdout):
./ParitySmokeTest.app/Contents/MacOS/ParitySmokeTest "$(pwd)/config.json"
```

`build_app.sh` compiles the SwiftPM executable, bundles it into a real `.app`
with `MediaPipeTasksC.framework` embedded (`@rpath` + ad-hoc signature), and
generates a `config.json` pointing at `../shared`. The app reads its config path
from `argv[1]` or the `PARITY_CONFIG` environment variable — required because a
`.app` launched via `open` runs with `cwd=/`. See `config.example.json` for the
shape.

The result is written to `../shared/output/macos.json`; a one-line summary with
detected counts is printed to stderr. The envelope includes each model's SHA256
and the exact options used (CPU delegate, IMAGE mode, confidences), plus face
blendshapes (52) and the 4×4 transformation matrix.

### Full-body pose run

Edit `config.json` to point `image` at `../shared/test_image_full_body.jpg` and
`output` at `../shared/output/macos_pose.json`, re-run, then compare with
`compare.py --tasks pose`. The config also accepts optional `numHands`/`numPoses`/
`numFaces` and `minDetectionConfidence`/`minPresenceConfidence`/
`minTrackingConfidence` overrides to match the web side exactly.

> If you change the package sources between runs, do a clean rebuild
> (`rm -rf .build` before `./build_app.sh`) so the path-dependency picks up new
> files.
