# MediaPipeIOSTest

Test bench for the fork's **iOS** support in the `MediaPipeTasksMac` Swift
package: live camera with hand / face / pose landmark overlays, a
CPU / GPU / Core ML (ANE) delegate switch, and per-task inference-time
sparklines.

It consumes the package **exactly the way an iOS app would** — as a local
Swift Package dependency (`packageReferences` → the repo root's
`Package.swift`, product `MediaPipeTasksMac`). The package is multiplatform:
its binary target is a universal `MediaPipeTasksC.xcframework` with macOS,
iOS-device and iOS-simulator slices, and the Swift API is platform-neutral
(`CVPixelBuffer` in, normalized landmarks out). The fork's `.coreML` delegate
and `coreMLModelCacheDirectory` flow through the shared C API.

## Building

0. Fetch the models on a fresh checkout: `./download_models.sh` (a thin wrapper
   around the shared downloader). It pulls the three `.task` files and the
   converted Core ML `<sha>.mlmodelc` models into the shared test-models dir
   the build phase reads from. Core ML download is best-effort — override
   `COREML_MODELS_URL` / `COREML_MODELS_SHA256` for your own converted set; a
   miss just means the Core ML delegate falls back to GPU/CPU.
1. Build the universal xcframework into the package's `Artifacts/` dir. On an
   Apple-Silicon Mac, from the repo root, build the iOS device + simulator
   dylibs and assemble the xcframework alongside the existing macOS slice
   (see `mediapipe/tasks/macos/build_macos_xcframework.sh` for the macOS
   slice and the session's `build_universal_xcframework.sh` for the iOS
   slices). The result must be at:
   `mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework`
   with macos-arm64, ios-arm64 and ios-arm64-simulator libraries.
2. Open `MediaPipeIOSTest.xcodeproj` and run on a device or simulator.

`Package.swift` auto-detects that local `Artifacts/` xcframework and links it
(no env var needed — the on-disk check works even though Xcode evaluates the
manifest in a sandbox that can't see shell/scheme environment variables).
With no local artifact it falls back to the released binary URL, so other
consumers are unaffected.

## Models

The `hand/face/pose_landmarker.task` files and the converted
`<sha256>.mlmodelc` Core ML models are copied into the app bundle by the
"Copy MediaPipe Models" build phase, from
`mediapipe/tasks/macos/test_projects/shared/models/`. The `.mlmodelc` bundles
sit **flat** next to the `.task` files — the Core ML delegate's default model
cache dir is the model asset's directory.

## Notes

- No `-force_load` / graph-library wiring is needed. The C API ships as a
  **dynamic** library, so all calculator/graph registrations are present at
  load — the same reason the macOS package works with a single binary target.
- A model without a converted `.mlmodelc` counterpart silently falls back to
  CPU/XNNPACK for that model only. The `.mlmodelc` are compiled on macOS
  (macOS 14 / iOS 17 share a Core ML version); if iOS refuses one, its timing
  collapses to CPU speed — that's the tell.
