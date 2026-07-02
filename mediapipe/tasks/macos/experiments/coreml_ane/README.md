# Core ML / Apple Neural Engine feasibility benchmark

De-risking experiment for a potential Core ML (ANE) inference backend in
MediaPipeTasksMac, answering two questions **before** committing to an
integration:

1. **Op coverage** — does the exact bundled `hand_landmarks_detector.tflite`
   (224×224) convert to Core ML and produce numerically-equivalent outputs?
2. **Placement + latency** — do its layers actually run on the Neural Engine
   (per `MLComputePlan`), and how does ANE latency compare to GPU/CPU for a
   model this small?

Context: MediaPipe's TFLite path on macOS supports CPU (XNNPACK) and GPU
(Metal) only. The ANE is reachable solely through Core ML, and Core ML decides
placement itself — hence the explicit per-op placement report here.

## Layout
- `models/` — tflite models extracted from `hand_landmarker.task` (zip).
- `convert.py` — tflite → onnx (`tf2onnx --tflite`) → torch (`onnx2torch`) →
  mlpackage (`coremltools`), validating max-abs-diff against the TFLite
  interpreter at each hop. Emits FP32 (validation) + FP16 (benchmark) packages
  into `out/`, plus a TFLite-XNNPACK CPU latency reference.
- `bench/` — SwiftPM executable: for each `MLComputeUnits` config, prints the
  per-op preferred-device placement (CPU/GPU/ANE counts) and prediction
  latency mean/p50/p90.

## Run
```bash
python3.12 -m venv venv && venv/bin/pip install numpy "tensorflow>=2.16,<2.20" tf2onnx onnx onnx2torch torch coremltools
venv/bin/python convert.py models/hand_landmarks_detector.tflite out
cd bench && swift run -c release ane-bench ../out/hand_landmarks_detector_fp16.mlpackage
```

## Results (2026-07-02, M1 Pro, macOS 26, 300 iters, fp16 mlprogram)

Landmark model (224×224, 103 MIL ops — converts cleanly):

| computeUnits | placement | mean / p50 / p90 ms |
|---|---|---|
| cpuOnly | CPU:103 | 4.08 / 4.05 / 4.28 |
| cpuAndGPU | GPU:103 | 2.51 / 2.49 / 2.70 |
| cpuAndNeuralEngine | **ANE:103** | **0.79 / 0.77 / 0.87** |
| all | ANE:103 | 0.77 / 0.75 / 0.86 |

Palm detector (192×192) — required a PReLU fix: onnx2torch lowers ONNX `PRelu`
via boolean-mask assignment → `non_zero`/`gather_nd` (dynamic-shape, ANE-
ineligible, 712 bloated ops, 63–96 ms). Rewriting as elementwise
`relu(x) − slope·relu(−x)` (see history) yields a static graph:

| computeUnits | placement | mean / p50 / p90 ms |
|---|---|---|
| cpuOnly | CPU:273 | 4.77 / 4.68 / 5.03 |
| cpuAndGPU | GPU:273 | 4.29 / 4.12 / 5.51 |
| cpuAndNeuralEngine | **ANE:273** | 1.40 / 1.18 / 2.06 |
| all | ANE:273 | **1.05 / 1.03 / 1.13** |

TFLite XNNPACK(4T) references: landmark 4.65 ms, palm 7.24 ms.
FP16-vs-TFLite max|Δ|: landmark 0.17 px (of 224 scale); palm 0.49 on pixel-scale
box regressors, 0.089 on score logits (validate detection-level agreement
before production).

**Verdict: ANE is a decisive win here — ~3–4× faster than the Metal GPU path
per model, all ops ANE-resident, and it frees the GPU (shared with WebRTC
encode/render in ScreenBar) while cutting power. The earlier hypothesis that
per-call dispatch overhead would erase the win for small models was wrong on
this hardware.**

## End-to-end pipeline A/B (pipeline/, 2026-07-02)

`pipeline/` reimplements the full HandLandmarkerGraph VIDEO dataflow with Core
ML inference (letterbox -> palm -> anchor decode -> weighted NMS -> ROI x2.6 ->
rotated crop -> landmarks -> presence gate -> ROI x2.0 loop) with ALL marshaling
on the clock, and runs it against the MediaPipeTasksMac baseline on the same
clip (two-hands-only.mov, 1280x720, 645 frames, numHands=2):

| pipeline | per-frame ms mean/p50/p90 | >=1 hand | both hands |
|---|---|---|---|
| MediaPipe TFLite **Metal** (baseline) | 7.86 / 7.61 / 11.01 | 77.8% | 42.9% |
| Core ML **ANE** (.all) | **3.77 / 2.80 / 6.38** | 76.7% | 68.2% |
| Core ML GPU (.cpuAndGPU) | 10.01 / 10.56 / 12.31 | 77.1% | 67.4% |

ANE stage means: palm pre 0.49 + infer 2.00 + post 0.79 (ran on 32% of frames —
tracking/detector-skip works); land pre 1.14 + infer 2.24 (sum over 2 hands) +
post 0.04. Fidelity vs baseline on matched hands: p50 29.6 px (mean inflated by
hand-count-mismatch frames; the CoreML pipeline holds BOTH hands more often).

**Verdict: 2.1x faster end-to-end than the production TFLite/Metal path, with
all marshaling counted, GPU left free, comparable detection quality.** CoreML-
GPU is SLOWER than TFLite-Metal — the win is specifically the ANE.

### Landmines found (must be handled by any production integration)
1. **ANE output strides**: MLMultiArray outputs are padded non-contiguously
   (e.g. [1,2016,1] with strides [64512,32,1]) — a linear read yields zeros.
   Honor `.strides`.
2. **sigmoid(0)=0.5 meets `score >= 0.5`**: zero-garbage passes the default
   threshold; symptom was 1953 constant "detections".
3. **vImageAffineWarp is BOTTOM-UP** (CoreGraphics space): conjugate top-down
   transforms with y-flips (symmetric letterboxes mask the bug; rotated crops
   break silently). Fixed in Warp.swift.
4. onnx2torch PReLU -> non_zero (see above) and tf2onnx output name mapping
   (var_NNN -> Identity_N by shape+value matching).

## Production integration (2026-07-02): InferenceCalculatorCoreMl

The findings above are productionized as a proper MediaPipe inference backend
at the TENSORS -> TENSORS boundary, covering hand, pose, AND face (any task,
in fact — the delegate plugs into the shared InferenceCalculator machinery):

- `mediapipe/calculators/tensor/inference_calculator_coreml.cc` — ObjC++
  calculator. Looks up the compiled Core ML model by the SHA-256 of the TFLite
  flatbuffer it replaces (`<hex>.mlmodelc` in `model_cache_dir`), feeds inputs
  zero-copy as MLMultiArray, copies outputs stride-aware (landmine #1) into
  tensors shaped by the TFLite I/O contract. Falls back to TFLite CPU/XNNPACK
  per-model when no converted model exists (e.g. an unconverted bundle member).
- Proto/plumbing: `InferenceCalculatorOptions.Delegate.CoreMl`
  (model_cache_dir, compute_units) -> tasks `Acceleration.coreml` ->
  C++ `BaseOptions::COREML` + `CoreMlOptions` -> C `MP_DELEGATE_COREML` +
  `coreml_model_cache_dir` -> Swift `MediaPipeDelegate.coreML` +
  `coreMLModelCacheDirectory` (defaults to the `.task` file's directory).
- `convert_delegate.py` — converts every .tflite inside .task bundles to
  `<sha256>.mlmodelc` with the `input_<i>`/`output_<i>` naming convention the
  calculator expects (exact I/O-order mapping via preserved tensor names).
  Additional landmines handled: pose_detector's sparse weights (38 DENSIFY ops
  crash tf2onnx — densified via interpreter + flatbuffer surgery), TFLite's
  DCR DepthToSpace (onnx2torch only has CRD), fp16 boundary fuzz on the pose
  segmentation mask (tolerated: fp32 chain is exact, the macOS wrapper does
  not expose masks, and the production TFLite-GPU path is fp16 anyway).
- Benchmark: `test_projects/macos` -> `CoreMLDelegateBench` compares
  .cpu/.gpu/.coreML for all three landmarkers.

Convert models for a bundle:
```bash
venv/bin/python convert_delegate.py <models-dir>/coreml_models \
    <models-dir>/hand_landmarker.task <models-dir>/pose_landmarker.task \
    <models-dir>/face_landmarker.task
```
Then in Swift: `options.delegate = .coreML` (plus
`options.coreMLModelCacheDirectory` if the .mlmodelc files are not next to the
.task file).

### Integrated results (CoreMLDelegateBench, M1 Pro, 2026-07-02)

All three tasks through the REAL MediaPipe graphs, VIDEO mode. Hand runs the
645-frame two-hands clip (numHands=2); pose/face run a repeated still (steady
state = tracked path; "first" = detector frame). Agreement is mean px error vs
the .cpu baseline. Numbers below include the two post-integration fixes:
row-wise strided output copy (6.3x on pose's 160k-element heatmap: 0.60 ->
0.095 ms) and pruning the unused pose segmentation mask at conversion
(--drop-output; the ANE never computes the mask decoder branch, the
calculator zero-fills the tensor).

| task | .cpu ms | .gpu (Metal) ms | .coreML ms (mean/p50/p90) | quality |
|---|---|---|---|---|
| hand | 23.18 | 7.19 / 6.79 / 10.85 | **4.39 / 4.37 / 5.30** | >=1 77.7% vs 77.8%; both 42.3% vs 42.9%; 0.70 px |
| pose | 10.41 | 4.00 / 3.73 / 4.98 | **2.81 / 2.69 / 3.31** | detected 100%; 0.63 px; first frame 6.7 vs GPU 16.8 |
| face (+blendshapes) | 7.22 | 4.47 / 4.06 / 5.83 | **3.80 / 3.76 / 3.98** | detected 100%; 0.02 px; first frame 5.9 vs GPU 19.7 |

Pre-fix .coreML for reference: hand 4.21, pose 4.09, face 4.63 ms — the fixes
took pose from GPU-parity to a 1.4x win and face from parity to 1.2x.

Verdict: **.coreML is the fastest delegate on all three tasks** — hand 1.6x,
pose 1.4x, face 1.2x vs Metal on mean (face 1.5x on p90), first frame
2.5-3.4x — while leaving the GPU completely free. Landmark agreement vs CPU
is sub-pixel everywhere. Under system load the gap widens further (the ANE is
uncontended while Metal fights WindowServer et al. — measured pose 3.9 vs GPU
6.0-7.8 ms at load average 30+). The XNNPACK fallback was verified by hiding
the .mlmodelc dir: per-model warnings fire and every task keeps working at
CPU speed with identical detection rates.

## Interpretation
- The benchmark measures the **landmark model alone** (one 224×224 crop). The
  full MediaPipe pipeline adds palm detection (192×192, skipped while
  tracking), pre/post-processing, and per-hand looping.
- ANE's advantage is **power/thermal**, not necessarily latency; for models
  this small, per-call dispatch overhead can erase the latency win. If
  `cpuAndNeuralEngine` does not clearly beat `cpuAndGPU` AND placement shows
  most ops on ANE, the integration (a custom Core ML InferenceCalculator) is
  not worth its complexity for speed alone.
