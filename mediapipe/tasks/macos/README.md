# MediaPipe Tasks for native macOS (Swift Package)

`MediaPipeTasksMac` is a native macOS Swift package that runs MediaPipe Vision
Tasks — **HandLandmarker**, **PoseLandmarker**, **FaceLandmarker** — on
Apple-silicon Macs (`My Mac`, `macos-arm64`). It is **not** for iOS, the iOS
Simulator, or Mac Catalyst.

It wraps MediaPipe's existing **C Tasks API** (`mediapipe/tasks/c/...`) through a
thin Objective-C++ bridge, so it builds on the macOS-capable C/C++ layer rather
than the iOS Objective-C framework pipeline.

```
import MediaPipeTasksMac
  → Swift API (CGImage / CVPixelBuffer / NSImage conversion)
  → MediaPipeTasksObjC   (Obj-C++ shim; links the binary, owns native memory)
  → CMediaPipeTasksC      (MediaPipeTasksC.xcframework — binary runtime only)
  → MediaPipe runtime / TFLite
```

## Scope

macOS arm64 · model loaded from file path · `CGImage` input · landmarks for all
three tasks (Hand 21, Pose 33, Face 478) plus hand handedness, face blendshapes
(52) and the 4×4 transformation matrix.

- **Delegate**: `.cpu` works. `.gpu` throws `MediaPipeError.unsupportedDelegate`
  on the shipped artifact (see "GPU delegate" below) — it never silently falls
  back to CPU.
- **Running mode**: `.image` (`detect(cgImage:)`) and `.video`
  (`detectForVideo(cgImage:timestampInMilliseconds:)`). Calling the wrong method
  for the configured mode throws `MediaPipeError.invalidRunningMode`.

Live-stream mode, `CVPixelBuffer`/`NSImage` input, and segmentation masks are
future milestones.

```swift
// VIDEO mode example (timestamps must be monotonically increasing):
let options = PoseLandmarkerOptions()
options.modelPath = "/path/to/pose_landmarker_lite.task"
options.delegate = .cpu
options.runningMode = .video
let landmarker = try PoseLandmarker(options: options)
for (frame, tsMs) in frames {                       // frame: CGImage
    let result = try landmarker.detectForVideo(cgImage: frame, timestampInMilliseconds: tsMs)
}
```

### GPU delegate

The shipped macOS artifact is **CPU-only** (built with
`--define MEDIAPIPE_DISABLE_GPU=1`). A GPU-enabled build was attempted and
**does not compile** for desktop macOS in this MediaPipe revision — the Metal
path (`image_to_tensor_converter_metal.cc`) calls `MPPMetalHelper`
`metalTextureWithGpuBuffer:`, a selector not available for this target. Until
that is resolved upstream, `.gpu` is rejected with `unsupportedDelegate`. If a
GPU-capable artifact is ever produced, flip `mediaPipeGPUArtifactAvailable` in
`Modes.swift` to `true`.

### Pose model size (lite / full / heavy)

Model size is **not** part of the package API — choose it on the consumer side
by pointing `modelPath` at the desired `.task` file:

```swift
enum PoseModel {                                   // app-side helper, not in the package
    static func path(_ size: String) -> String {  // "lite" | "full" | "heavy"
        "/path/to/models/pose_landmarker_\(size).task"
    }
}
let options = PoseLandmarkerOptions()
options.modelPath = PoseModel.path("full")
```

## Prerequisites

- **macOS on Apple silicon** + **Xcode** (command-line tools).
- **Bazel 7.4.1** (pinned by `.bazelversion`). `brew install bazelisk` is the
  easiest way — `bazel` then auto-selects 7.4.1.
- **A JDK** on `PATH` with `JAVA_HOME` set. MediaPipe's Bazel build needs Java
  (protobuf / FlatBuffers codegen). Without it the build fails with
  `Cannot find Java binary` / `no such package '@@rules_java~//tools/jdk'`.

  ```bash
  brew install openjdk
  export JAVA_HOME="/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
  export PATH="$JAVA_HOME/bin:$PATH"
  ```

- **Python 3.9–3.12** as `python3`. MediaPipe's `WORKSPACE` only ships
  `requirements_lock` files for 3.9–3.12; a 3.13/3.14 default `python3` fails
  with `Could not find requirements_lock.txt file matching ... 3.14`. If your
  default `python3` is newer, put a 3.12 first on `PATH`:

  ```bash
  brew install python@3.12
  mkdir -p /tmp/pyshim && ln -sf "$(brew --prefix python@3.12)/bin/python3.12" /tmp/pyshim/python3
  export PATH="/tmp/pyshim:$PATH"
  ```

## Build the binary artifact

The package depends on a **locally-generated, gitignored** xcframework. You must
build it before `swift build`:

```bash
./mediapipe/tasks/macos/build_macos_xcframework.sh
```

This builds `//mediapipe/tasks/c:mediapipe_macos` (`-c opt`,
`--define MEDIAPIPE_DISABLE_GPU=1 --define OPENCV=source`, macOS arm64), verifies
the exported C symbols and the Mach-O platform (`MACOS`, not `IOS`), wraps the
dylib into `MediaPipeTasksC.framework`, and writes:

```
mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework
```

> The first build compiles OpenCV and MediaPipe from source and can take a long
> time. Subsequent builds are incremental.

## Build & run the sample

```bash
swift build
swift run mediapipe-macos-sample --hand /path/to/hand_landmarker.task --image /path/to/hand.jpg
# Expected: HandLandmarker: detected 1 hand, 21 landmarks
```

The sample accepts any subset of `--hand` / `--pose` / `--face`, so you can test
one model at a time, or all together:

```bash
swift run mediapipe-macos-sample \
  --hand hand_landmarker.task \
  --pose pose_landmarker.task \
  --face face_landmarker.task \
  --image person.jpg
# HandLandmarker: detected 1 hand, 21 landmarks
# PoseLandmarker: detected 1 pose, 33 landmarks
# FaceLandmarker: detected 1 face, 478 landmarks
```

Download `.task` models from the MediaPipe model index, e.g.:

```bash
curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Use from an app

After building the artifact, add the package by **local path**
(`File ▸ Add Package Dependencies… ▸ Add Local…`, point at the repo root), or in
another package's `Package.swift`:

```swift
.package(path: "/path/to/mediapipe")
```

```swift
import MediaPipeTasksMac

let options = HandLandmarkerOptions()
options.modelPath = "/path/to/hand_landmarker.task"
options.numHands = 2
let landmarker = try HandLandmarker(options: options)
let result = try landmarker.detect(cgImage: image)
print(result.landmarks.count)
```

> Because the xcframework is gitignored, the package cannot be resolved straight
> from a remote Git branch until the build script has produced the artifact.

## Result shapes (MediaPipe parity)

Result types mirror the MediaPipe Tasks Web/iOS shapes, with the nested
per-instance structure preserved (no flattening) so app code can share logic
with MediaPipe Web result handling:

- `HandLandmarkerResult`: `landmarks: [[NormalizedLandmark]]`,
  `worldLandmarks: [[Landmark]]`, `handedness: [[Category]]`
  (plus a deprecated `handednesses` alias).
- `PoseLandmarkerResult`: `landmarks: [[NormalizedLandmark]]`,
  `worldLandmarks: [[Landmark]]`, `segmentationMasks: [MPMask]?`
  (`nil` until masks are implemented).
- `FaceLandmarkerResult`: `faceLandmarks: [[NormalizedLandmark]]`,
  `faceBlendshapes: [Classifications]`, `facialTransformationMatrixes: [Matrix]`.
  Set `FaceLandmarkerOptions.outputFaceBlendshapes` /
  `.outputFacialTransformationMatrixes` to populate the last two (52 blendshapes
  and a 4×4 matrix per face); otherwise they are empty.

`NormalizedLandmark` and `Landmark` carry optional `visibility` / `presence`
(populated for pose; `nil` for hand/face where the model omits them).

## Tests

`swift test` runs result-shape/count assertions (Hand 21+21, Pose 33+33, Face
478). They require real models/images, supplied via environment variables, and
are **skipped** when unset:

```bash
MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=hand.jpg \
MP_POSE_MODEL=pose_landmarker.task MP_POSE_IMAGE=person.jpg \
MP_FACE_MODEL=face_landmarker.task MP_FACE_IMAGE=person.jpg \
swift test
```

## Future distribution (not done yet)

Once the artifact is stable: zip `MediaPipeTasksC.xcframework`, upload it to a
GitHub Release, run `swift package compute-checksum`, and switch `Package.swift`
from `.binaryTarget(path:)` to `.binaryTarget(url:checksum:)` so consumers don't
need to run the Bazel build locally.
