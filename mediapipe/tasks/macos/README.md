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

## Milestone 1 scope

macOS arm64 · CPU delegate · **image mode** · model loaded from file path ·
`CGImage` input · landmarks (+ hand handedness) for all three tasks
(Hand 21, Pose 33, Face 478). Video / live-stream / `CVPixelBuffer` / `NSImage` /
face blendshapes & transformation matrices are future milestones.

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

## Future distribution (not done yet)

Once the artifact is stable: zip `MediaPipeTasksC.xcframework`, upload it to a
GitHub Release, run `swift package compute-checksum`, and switch `Package.swift`
from `.binaryTarget(path:)` to `.binaryTarget(url:checksum:)` so consumers don't
need to run the Bazel build locally.
