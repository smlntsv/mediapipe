# MediaPipe Tasks for native macOS (Swift Package)

`MediaPipeTasksMac` is a native macOS Swift package that runs MediaPipe Vision
Tasks — **HandLandmarker**, **PoseLandmarker**, **FaceLandmarker** — on
Apple-silicon Macs (`My Mac`, `macos-arm64`). It is **not** for iOS, the iOS
Simulator, or Mac Catalyst.

> **Intel Macs:** the macOS slice can also be built for x86_64, or as a
> universal `macos-arm64_x86_64` slice. CPU and Core ML are validated natively
> on an Intel Mac, and so is the Metal GPU delegate on its AMD GPU. See
> [ARCHITECTURES.md](ARCHITECTURES.md)
> and [validation/VALIDATION_REPORT.md](validation/VALIDATION_REPORT.md).

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

macOS arm64 · model loaded from file path · landmarks for all three tasks
(Hand 21, Pose 33, Face 478) plus hand handedness, face blendshapes (52) and the
4×4 transformation matrix.

- **Input**: `CGImage` and `CVPixelBuffer` (`kCVPixelFormatType_32BGRA`).
  `CVPixelBuffer` is converted BGRA→tightly-packed RGBA via vImage (the same
  RGBA path as `CGImage`), locking the buffer read-only.
- **Delegate**: both `.cpu` and `.gpu` (Metal) work with the default
  GPU-capable artifact (see "GPU delegate" below). With a CPU-only artifact,
  `.gpu` throws `MediaPipeError.unsupportedDelegate` — it never silently falls
  back to CPU.
- **Running mode**: `.image` (`detect(cgImage:)` / `detect(pixelBuffer:)`) and
  `.video` (`detectForVideo(…:timestampInMilliseconds:)`). Calling the wrong
  method for the configured mode throws `MediaPipeError.invalidRunningMode`.

Live-stream mode, `NSImage` input, other pixel formats, and segmentation masks
are future milestones.

> **Distribution:** the default artifact links MacPorts OpenCV from `/opt/local`
> (fine for local dev, not portable). Build with **`MP_BUNDLE_DEPS=1`** to bundle
> the non-system dylibs into the framework for a portable release. OpenCV is
> trimmed to `core` + `imgproc`, so a bundled release is just **3 dylibs / ~4.7 MB**
> (`libopencv_core`, `libopencv_imgproc`, `libz`) — **no GPL/LGPL codecs**, ~29 MB
> xcframework. See [PACKAGING.md](PACKAGING.md).

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

### GPU delegate (Metal)

The default macOS artifact is **GPU-capable** and runs on Apple-silicon Metal
(verified: `renderer: Apple M1 Pro`). It supports **both** `.cpu` and `.gpu`
delegates; on a pose VIDEO benchmark the GPU delegate ran ~2× faster than CPU.

> ✅ **Sustained GPU VIDEO is bounded (a former leak is fixed).** Earlier builds
> leaked ~1 MB/frame on the `.gpu` VIDEO path (IOSurfaces pinned by the Metal/GL
> texture caches and never flushed), so long runs grew without bound. This is fixed
> by flushing those caches once per frame on the macOS CVPixelBuffer path
> (`MPPMetalHelper.cc`, `gpu_buffer_storage_cv_pixel_buffer.cc`). GPU memory now
> ramps once to a small bounded working set and stays flat — the package's
> `MemoryStressTests` assert a near-zero steady-state rate for `.cpu` **and**
> `.gpu` (hand/pose/face). Full write-up:
> `samples/WebcamLandmarksDemo/MEMORY_NOTES.md`.

Enabling GPU required one build-config fix: upstream defines
`MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER` only for `!TARGET_OS_OSX`, so on macOS
`MPPMetalHelper`'s `metalTextureWithGpuBuffer:` is not declared even though
`image_to_tensor_converter_metal.cc` calls it. `build_macos_xcframework.sh`
force-defines the macro for every C++/ObjC translation unit (`--copt` +
`--objccopt`) and drops `MEDIAPIPE_DISABLE_GPU`. Required frameworks (Metal,
MetalKit, CoreVideo, CoreMedia, AVFoundation, OpenGL) are linked automatically
by MediaPipe's GPU targets.

`mediaPipeGPUArtifactAvailable` in `Modes.swift` is `true` to match this
artifact. To ship a smaller CPU-only build instead, run the build script with
`MP_ENABLE_GPU=0` and set that flag to `false` (then `.gpu` is rejected with
`unsupportedDelegate` — never a silent CPU fallback).

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

### Hand pipeline stage outputs (debugging)

Set `HandLandmarkerOptions.stageOutputs = true` to get
`HandLandmarkerResult.stages` on every result, including frames with no hand.
It is a copy of the hand graph's intermediate streams for that frame: the
rects carried in from the previous frame, whether the palm detector ran, its
detections and rects, the regions the landmark model was given with their
presence scores, the landmarks and next-frame rects before deduplication, and
the keep-alive state. See `HandLandmarkerStages` for each field. Default off:
the graph, the results and the timings are then unchanged and `stages` is nil.

Things the record shows that are easy to misread:

- All coordinates are normalized to the image as passed in, whatever
  `rotationDegrees` is (the same frame as `landmarks`).
- When the detector runs and finds no palm, the rect arrays hold one all-zero
  rect instead of being empty; the landmark model skips it.
- `StageRect.rectId` is reassigned on every frame. It is not a track id.

### Model input tensors (debugging)

Set `HandLandmarkerOptions.tensorCapture = true` and pass
`captureTensors: true` to a detect call to get `HandLandmarkerResult.tensorCapture`
for that frame: a copy of the tensor the palm detection model was given and of
the tensor the landmark model was given for each region of interest.

```swift
let options = HandLandmarkerOptions(modelPath: path, runningMode: .video, tensorCapture: true)
let landmarker = try HandLandmarker(options: options)
let result = try landmarker.detectForVideo(pixelBuffer: frame,
                                           timestampInMilliseconds: ms,
                                           captureTensors: wanted)
if let capture = result.tensorCapture {
    capture.palmDetectorInput        // CapturedTensor?, nil: the detector did not run
    capture.landmarkInputs[0].tensor // CapturedTensor?, nil: the model skipped that region
}
```

- **Which frames.** Only the calls that pass `captureTensors: true`. On every
  other call nothing reads the tensors (on the GPU path that means no
  GPU-to-CPU readback), and `tensorCapture` is nil. Passing it without the
  option throws `MediaPipeError.invalidArgument`.
- **Nil has two meanings, at two levels.** `result.tensorCapture == nil`: there
  is no capture for this frame. A nil tensor inside a capture: that model did
  not run on this frame (the palm detector while hands are tracked; the
  landmark model for an all-zero region).
- **What a tensor is.** The `mediapipe::Tensor` handed to the inference step,
  after crop, rotation, resize and value normalization: `shape`, `elementType`
  and the raw bytes. The palm input is 192 x 192, the landmark input 224 x 224,
  both `float32`. On CPU and Core ML they have 3 channels; on the GPU delegate
  the Metal preprocessing produces 4 channels per pixel, of which the first
  three are RGB.
- **Landmark inputs line up with regions.** `landmarkInputs[i]` belongs to
  `stages.landmarkROIs[i]`; `roiIndex` is that `i`.
- The option is independent of `stageOutputs`.

### Palm candidates (debugging)

Set `HandLandmarkerOptions.palmCandidateLimit = N` (N > 0) to get
`HandLandmarkerResult.palmCandidates` on every result. It shows where the palm
detector's candidates went on that frame:

| Field | Meaning |
|---|---|
| `palmDetectorRan` | false: the detector was skipped, the rest is empty |
| `candidates` | the N highest scored decoded boxes, highest first, with no threshold and before suppression |
| `aboveThresholdCount` | boxes with a score of at least `minHandDetectionConfidence` |
| `afterSuppressionCount` | detections left after non-max suppression |

The first `numHands` detections after suppression become hand rects. So a best
candidate score under the threshold means the scores fell short;
`aboveThresholdCount` greater than `afterSuppressionCount` is the number merged
by suppression; `afterSuppressionCount` greater than `numHands` is the number
cut by the output limit. Which box was merged into which is not reported.
Candidates are in the same coordinates as `stages.palmDetections`.

### Hand Landmarker debug API reference

Everything the three debugging aids add to the public Swift API. All of it is
opt-in; with the defaults the pipeline, the results and the timings are what
they were before. The records are built by the package, so the structs have no
public initializers.

**Options and calls**

```swift
public final class HandLandmarkerOptions {
    public var stageOutputs: Bool        // default false -> result.stages
    public var tensorCapture: Bool       // default false -> allows captureTensors: true
    public var palmCandidateLimit: Int   // default 0     -> result.palmCandidates
    public init(modelPath: String = "", numHands: Int = 1, /* ...existing parameters... */
                stageOutputs: Bool = false,
                tensorCapture: Bool = false,
                palmCandidateLimit: Int = 0)
}

public final class HandLandmarker {
    // Each of the four detect methods has a trailing `captureTensors` parameter.
    public func detect(cgImage: CGImage,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                       captureTensors: Bool = false) throws -> HandLandmarkerResult
    public func detect(pixelBuffer: CVPixelBuffer,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                       captureTensors: Bool = false) throws -> HandLandmarkerResult
    public func detectForVideo(cgImage: CGImage, timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                               captureTensors: Bool = false) throws -> HandLandmarkerResult
    public func detectForVideo(pixelBuffer: CVPixelBuffer, timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                               captureTensors: Bool = false) throws -> HandLandmarkerResult
}
```

`captureTensors: true` on a landmarker created without `tensorCapture` throws
`MediaPipeError.invalidArgument`.

**Result**

```swift
public struct HandLandmarkerResult {
    public var stages: HandLandmarkerStages?                  // every frame, with stageOutputs
    public var tensorCapture: HandLandmarkerTensorCapture?    // only the frames that asked
    public var palmCandidates: HandLandmarkerPalmCandidates?  // every frame, with palmCandidateLimit > 0
}
```

**Stage record** (`stageOutputs`)

```swift
public struct StageRect: Sendable, Equatable {
    public var xCenter: Float     // normalized by the image width
    public var yCenter: Float     // normalized by the image height
    public var width: Float
    public var height: Float
    public var rotation: Float    // radians, clockwise
    public var rectId: Int64?     // reassigned every frame; not a track id
}

public struct StagePoint: Sendable, Equatable {
    public var x: Float
    public var y: Float
}

public struct StageDetection: Sendable, Equatable {
    public var score: Float
    public var xMin: Float
    public var yMin: Float
    public var width: Float
    public var height: Float
    public var keypoints: [StagePoint]   // 7 for the palm model
}

public struct HandLandmarkerStages: Sendable, Equatable {
    public var timestampMicroseconds: Int64
    public var previousRects: [StageRect]
    public var palmDetectorRan: Bool
    public var palmDetections: [StageDetection]
    public var palmRects: [StageRect]
    public var detectorHandRects: [StageRect]
    public var landmarkROIs: [StageRect]
    public var presenceScores: [Float]
    public var presencePassed: [Bool]
    public var landmarksBeforeDeduplication: [[NormalizedLandmark]]
    public var handRectsBeforeDeduplication: [StageRect]
    public var handRectsNextFrame: [StageRect]
    public var keepAliveRects: [StageRect]
    public var keepAliveMissCounts: [Int]
}
```

| Field | What it is | When it is empty |
|---|---|---|
| `timestampMicroseconds` | Graph timestamp the record was read from. In `.video` mode `timestampInMilliseconds * 1000` of the call | never (synthetic in `.image` mode) |
| `previousRects` | Rects carried in from the previous frame: its `handRectsNextFrame`, or its `keepAliveRects` when `trackingGraceFrames` > 0 | `.image` mode; nothing was tracked |
| `palmDetectorRan` | Whether the palm detector ran. In `.video` mode it is skipped when `previousRects.count >= numHands` | |
| `palmDetections` | Palm detections after non-max suppression; can be more than `numHands` | detector skipped or found nothing |
| `palmRects` | One rect per palm detection | detector skipped (one all-zero rect when it found nothing) |
| `detectorHandRects` | `palmRects` expanded to whole hands, clipped to `numHands` | same as `palmRects` |
| `landmarkROIs` | What the landmark model is given: `previousRects`, then the detector rects that do not overlap them, clipped to `numHands` | never in practice |
| `presenceScores`, `presencePassed` | Per ROI the model ran on, in ROI order: the hand presence score and whether it passed `minHandPresenceConfidence` | no ROI was run (only an all-zero ROI) |
| `landmarksBeforeDeduplication`, `handRectsBeforeDeduplication` | Per ROI that passed presence: its landmarks and its next-frame rect | no ROI passed |
| `handRectsNextFrame` | Next-frame rects after deduplication, one per hand in the result | no hand |
| `keepAliveRects`, `keepAliveMissCounts` | `.video` with `trackingGraceFrames` > 0: the rects handed to the next frame and, per rect, 0 if live or the number of frames it has been missing | no grace; nothing to hand on |

**Tensor capture** (`tensorCapture` + `captureTensors: true`)

```swift
public enum TensorElementType: Int, Sendable {
    case none = 0, float16, float32, uint8, int8, int32, int64, char, bool
}

public struct CapturedTensor: Sendable, Equatable {
    public var shape: [Int]                   // [1, height, width, channels]
    public var elementType: TensorElementType
    public var data: Data                     // row-major, native byte order
    public func floatValues() -> [Float]?     // nil unless elementType == .float32
}

public struct LandmarkModelInput: Sendable, Equatable {
    public var roiIndex: Int                  // index into stages.landmarkROIs
    public var tensor: CapturedTensor?        // nil: the model did not run on this ROI
}

public struct HandLandmarkerTensorCapture: Sendable, Equatable {
    public var timestampMicroseconds: Int64
    public var palmDetectorInput: CapturedTensor?    // nil: the palm detector did not run
    public var landmarkInputs: [LandmarkModelInput]  // one per ROI, in ROI order
}
```

**Palm candidates** (`palmCandidateLimit`)

```swift
public struct HandLandmarkerPalmCandidates: Sendable, Equatable {
    public var timestampMicroseconds: Int64
    public var palmDetectorRan: Bool
    public var candidates: [StageDetection]   // highest score first, no threshold, before suppression
    public var aboveThresholdCount: Int       // score >= minHandDetectionConfidence
    public var afterSuppressionCount: Int     // == stages.palmDetections.count
}
```

**Platforms.** macOS. On iOS the options are accepted, and the three records
stay nil until the iOS slices are rebuilt from a version that has them.

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
