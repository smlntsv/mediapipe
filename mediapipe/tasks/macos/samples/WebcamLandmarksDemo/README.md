# WebcamLandmarksDemo

A native macOS **SwiftUI + AVFoundation** app that validates the live-video path
end to end:

```
AVCaptureSession → CVPixelBuffer (32BGRA) → MediaPipeTasksMac VIDEO mode
                → detectForVideo(pixelBuffer:timestampInMilliseconds:)
                → GPU (Metal) delegate → landmark overlay
```

It runs HandLandmarker, PoseLandmarker, and FaceLandmarker on the webcam feed
and draws the landmarks over the preview.

## Prerequisites

1. Build the GPU-capable, portable framework once (from the repo root):
   ```bash
   MP_BUNDLE_DEPS=1 ./mediapipe/tasks/macos/build_macos_xcframework.sh
   ```
   (Needs MacPorts `opencv3`, a JDK, Bazel 7.4.1, and a Python 3.9–3.12 — see
   `mediapipe/tasks/macos/README.md`.)
2. Download the models (gitignored):
   ```bash
   ./download_models.sh        # hand_landmarker, pose_landmarker_full, face_landmarker
   ```

## Build & run

```bash
./build_app.sh
open WebcamLandmarksDemo.app
```

`build_app.sh` compiles the SwiftUI executable, bundles it into a real `.app`
with `NSCameraUsageDescription`, embeds `MediaPipeTasksC.framework` (+ its
bundled OpenCV dylibs), and copies the models into `Resources`. On first launch
macOS prompts for camera access.

## UI

- **Task**: Hand / Pose / Face / All.
- **Delegate**: CPU / GPU. Default is **GPU** (Metal, ~2× faster) — memory is now
  bounded on both; see the memory note below.
- **Stage** (performance/debug): `Preview` / `Convert` / `Full` — see
  *Performance modes* below.
- **Mirror**: toggles the natural "mirror" view.
- **Low power**: capture at 640×480 instead of 1280×720.
- **Stats**: FPS, inference ms (smoothed), detection count, **resident memory
  (MB)**, frame count, delegate — published at ~4 Hz (not every frame) so the
  stats text doesn't add UI churn to the measurement.

Overlay: hand = yellow lines + red dots (21), pose = cyan lines + white dots
(33), face = faint green dots (478). The overlay Canvas is mounted **only** in
`Full`, so it never redraws in Preview/Convert.

## Performance modes

The capture format is fixed at a practical **1280×720 @ 30 fps** (640×480 in
*Low power*), not the camera's max resolution. The three stages isolate where CPU
goes so inference cost is attributable, not polluted by preview/UI overhead:

| Stage | Pipeline | What runs |
| --- | --- | --- |
| **Preview** | preview layer only | `AVCaptureVideoDataOutput` is **detached** — no delegate callback, conversion, inference, or overlay. Just the camera + `AVCaptureVideoPreviewLayer`. |
| **Convert** | + `CVPixelBuffer`→`MPCImage` | capture + RGBA conversion, no inference, no overlay. |
| **Full** | + inference + overlay | the real workload. |

Measured on an Apple-silicon Mac (All tasks, `pose_landmarker_full`), steady-state:

| Stage | CPU |
| --- | --- |
| Preview | **~4%** (was ~100% before this change) |
| Convert | ~20% (mostly capture/compositing; the vImage convert is a small slice) |
| Full · GPU | ~88% (subtract ~4% preview ⇒ ~84% attributable to inference + overlay) |
| Full · CPU | ~130% |

To measure a single mode without touching the UI, launch via `open` with env
overrides (they forward through LaunchServices):

```bash
WEBCAM_STAGE=Preview                 open WebcamLandmarksDemo.app   # Preview / Convert / Full
WEBCAM_STAGE=Full WEBCAM_DELEGATE=GPU open WebcamLandmarksDemo.app   # CPU / GPU
WEBCAM_LOWPOWER=1                    open WebcamLandmarksDemo.app   # 640×480
WEBCAM_DEBUG_LOG=1                   open WebcamLandmarksDemo.app   # stderr stats every 60 frames (off by default)
```

Then sample CPU with Activity Monitor or `ps -o %cpu= -p <pid>`.

## Memory: both delegates are bounded (a former GPU leak is fixed)

Earlier builds had a **per-frame IOSurface leak on the GPU (Metal) path** (~1 MB/
frame, inside MediaPipe's GPU inference — not this demo), which grew memory until
macOS force-quit the app. It is **fixed** by flushing the Metal and OpenGL texture
caches once per frame on the macOS CVPixelBuffer path (`mediapipe/gpu`). GPU memory
now ramps once to a small bounded working set and stays flat; CPU was always
leak-free. Both are safe for sustained runs, and **GPU is the default** because it
is ~2× faster. Root cause, evidence (`vmmap`, before/after numbers), and the
per-frame stress test are in [MEMORY_NOTES.md](MEMORY_NOTES.md).

## Orientation & mirroring (explicit)

- **Mirroring:** the frames MediaPipe processes are **not** mirrored. The preview
  and the landmark overlay are mirrored *together* (overlay maps `x → 1 - x`)
  so they stay aligned. The `Mirror` toggle flips both. Default is on (selfie
  feel). With it off, the preview and landmarks match the raw sensor orientation.
- **Orientation:** macOS webcams deliver landscape frames and
  `AVCaptureConnection` on macOS generally does **not** support
  `videoOrientation`/rotation, so no rotation is applied. The overlay uses an
  aspect-fit (`resizeAspect`) mapping that matches the preview letterboxing.

### Known limitations

- Front-camera mirroring is handled by flipping the overlay x-coordinate to match
  the mirrored preview; it is **not** applied to the pixels sent to MediaPipe, so
  detection is unaffected by the toggle (only the displayed alignment changes).
- Non-landscape / rotated capture devices are not handled (macOS rarely rotates).
- `All` mode runs three graphs per frame; with `pose_landmarker_full` on, some
  frames are dropped (`alwaysDiscardsLateVideoFrames`) — expected for a demo.
- The window's preview uses `AVCaptureVideoPreviewLayer`; the overlay assumes the
  same aspect-fit rect.

## Notes

- Models are loaded from the app bundle's `Resources`, or from a directory given
  by the `WEBCAM_MODELS_DIR` environment variable (handy during development).
- The embedded framework is the trimmed, portable artifact — the built `.app`
  has **no `/opt/local` dependencies** (verify with
  `otool -L WebcamLandmarksDemo.app/Contents/Frameworks/MediaPipeTasksC.framework/MediaPipeTasksC`).
