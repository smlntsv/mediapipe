# Validation report: Intel (x86_64) macOS support, universal macOS slice

Date: 2026-10-26 · Source: `b55fb6819` + this change. Every file compiled
into the binary is identical to `6cecfe31b`, the revision StaxelPointer pins
(the only commit in between adds `bump_package_pin.sh`) · Artifact: local,
**unreleased** `MediaPipeTasksC.xcframework` (`ios-arm64`,
`ios-arm64-simulator`, `macos-arm64_x86_64`), swiftpm-checksum
`95f727c247c10d569002633f26289ff7ef6916756e85ec8506a7ffa49c962c7f`.
Nothing was published or pushed, and no downstream app was changed.

## Verdict

| | CPU (XNNPACK) | GPU (Metal) | Core ML |
|---|---|---|---|
| **Native arm64**, M1 Pro, macOS 26.3.1 | ✅ validated: bit-identical to `v1.0.0-apple.3` | ✅ validated: bit-identical to `apple.3` | ✅ validated: bit-identical to `apple.3` |
| **Rosetta x86_64**, same M1 Pro (not Intel validation) | ✅ smoke-tested, within tolerance | ✅ smoke-tested (Apple GPU) | ✅ smoke-tested |
| **Native x86_64**, real Intel Mac: MacBookPro16,1, i9-9880H, AMD Radeon Pro 5500M + Intel UHD 630, macOS 26.7.1 | ✅ **validated natively** | ❌ **unsupported**: Metal runs but returns no detections. Now rejected with `MediaPipeError.unsupportedDelegate` | ✅ **validated natively** (no Neural Engine on Intel) |

Not covered:

- **Intel Macs on macOS 13–15.** The only Intel host runs macOS 26.7.1. The x86_64 slice is built for `minos 13.0` throughout, but has not been *run* on an older OS.
- **The integrated Intel UHD 630 for Metal.** It would require changing the Mac's graphics-switching system setting.
- **Pose and Face on Intel.** They share the same runtime, and the Swift suite covers them under Rosetta, but the Intel matrix ran Hand only.

## What was run

| Check | Native arm64 | Rosetta x86_64 | Native Intel |
|---|---|---|---|
| Build: each arch independently, then lipo-merge, audit and dSYM UUIDs | ✅ | (same artifact) | (same artifact) |
| Existing Swift suite (`swift test`; 29 tests: hand/pose/face, CPU+GPU, IMAGE+VIDEO, memory stress) | ✅ 23 passed, 6 env-gated skips, 0 failures. Same set as against `apple.3` | ✅ 23 passed, 6 skips, 0 failures (`arch -x86_64 xctest`; includes the 7 GPU tests on the Apple GPU) | — (no Xcode on the Intel host) |
| `mp-arch-validate` matrix: 4 fixed stills × 23 runs, plus 5 × 645-frame sustained VIDEO, × cpu/gpu/coreml | ✅ all PASS, 0 crashes | ✅ all PASS, 0 crashes | ✅ cpu + coreml PASS, gpu UNSUPPORTED, 0 crashes |
| 20-loop soak (12,900 frames) | gpu (new and `apple.3`), cpu | — | cpu, coreml (+ 60-loop cpu soak, below) |

Inputs were identical everywhere, with SHA-256 recorded in every report:
`hand_landmarker.task`, `test_image.jpg`, and three lossless PNG stills
extracted once on the build Mac (two-hands-only.mov @ 1.5 s, 8 s and 12 s,
with 1, 2 and 1 hands). Video runs decode `two-hands-only.mov` on each
machine (1280×720, 645 frames).

## Accuracy: cross-architecture agreement (tolerance gates)

Gates: same hand count and handedness labels; max |dx|,|dy| ≤ 0.02
(normalized) over all 21 landmarks; handedness score delta ≤ 0.05; video
hand-rate within 3 percentage points, frame-level hand-count agreement ≥ 95%,
p90 per-frame max |dxy| ≤ 0.02.

| Pair | Delegate | Image: max \|dxy\| (worst still) | Handedness | Video: hand-rate base / cand | Frame agreement | p90 \|dxy\| |
|---|---|---|---|---|---|---|
| `apple.3` arm64 vs new arm64 | cpu / gpu / coreml | **0.00000** (all 12 still×delegate pairs) | identical | identical every loop | **100%** | **0.0000** |
| native arm64 vs **native Intel** | cpu | 0.00004 | identical (max score delta 0.0006) | 77.8% / 77.7% | 99.2% | 0.0068 |
| native arm64 vs **native Intel** | coreml | 0.00252 | identical (≤ 0.0078) | 77.7% / 77.5% | 98.8% | 0.0066 |
| native arm64 vs **native Intel** | gpu | — UNSUPPORTED (see GPU below) | | | | |
| native arm64 vs Rosetta | cpu | 0.00018 | identical | 77.8% / 77.8% | 99.7% | 0.0038 |
| native arm64 vs Rosetta | gpu | 0.00069 | identical | 77.8% / 77.8% | 99.8% | 0.0038 |
| native arm64 vs Rosetta | coreml | 0.00239 | identical (≤ 0.0225) | 77.7% / 77.7% | 99.4% | 0.0038 |

All 12 image results on every platform also passed the anatomical
plausibility checks: 21 landmarks per hand, all inside the image, bone lengths
non-degenerate, handedness score ≥ 0.5, world palm length 3–20 cm. IMAGE
mode was deterministic across 23 repeats on every platform (repeat Δ = 0).
Single-frame video outliers (max |dxy| up to about 0.48 on Intel CPU) are
tracking-association flips on individual frames. The p90 stays at 0.007.

## Performance and memory: sustained VIDEO, hand, 2 hands max

5 loops × 645 frames (3,225 frames). Latency is the `detectForVideo` call
only. Memory is the process `phys_footprint`, with the slope fitted after
loop 0.

| Platform | Delegate | Mean / p50 / p90 / p99 ms | Throughput | First frame | Footprint (after loop 0 → end, peak) |
|---|---|---|---|---|---|
| native arm64 (M1 Pro) | cpu | 23.5 / 23.9 / 26.9 / 38.5 | 42.5 fps | 16.6 ms | 106 → 104 MB (110) |
| native arm64 | gpu | 6.4 / 6.1 / 8.7 / 12.4 | 155.5 fps | 12.7 ms | 1246 → 1325 MB (1339)\* |
| native arm64 | coreml | 3.6 / 3.3 / 4.8 / 6.1 | 281.1 fps | 6.7 ms | 76 → 76 MB (82) |
| Rosetta x86_64 (M1 Pro) | cpu | 41.2 / 40.8 / 48.7 / 88.8 | 24.3 fps | 29.0 ms | 208 → 234 MB (236)† |
| Rosetta x86_64 | gpu | 13.4 / 12.4 / 17.7 / 32.9 | 74.4 fps | 25.6 ms | 710 → 830 MB (853)† |
| Rosetta x86_64 | coreml | 9.6 / 8.5 / 14.1 / 21.6 | 104.2 fps | 52.7 ms | 222 → 229 MB (234)† |
| **native Intel** (i9-9880H) | cpu | 23.7 / 22.9 / 28.1 / 37.5 | 42.3 fps | 23.7 ms | 61 → 61 MB (64) |
| **native Intel** | coreml | 14.1 / 14.3 / 16.6 / 22.0 | 70.4 fps | 325.6 ms (Core ML model load) | 108 → 109 MB (112) |
| **native Intel** | gpu | UNSUPPORTED | | | |

Soak runs (20 loops = 12,900 frames, unless noted):

| Platform / delegate | Mean ms | Footprint after loop 0 → end (peak) | Slope MB / 1k frames |
|---|---|---|---|
| native arm64 gpu, **new** | 6.92 | 1317 → 1013 MB (1330) | −5.2 |
| native arm64 gpu, **`apple.3`** | 8.33 | 1346 → 1072 MB (1479) | −23.6 |
| native arm64 cpu | 24.24 | 104 → 106 MB (110) | +0.16 |
| native Intel cpu | 23.79 | 59 → 75 MB (77) | +0.72 |
| native Intel coreml | 14.10 | 107 → 109 MB (112) | +0.14 |
| native Intel cpu, 60 loops (38,700 frames) | 23.29 (p99 36.9) | 59 → 78 MB (80) | +0.42 overall; about +0.16 in the middle third |

Intel CPU memory shape over 38,700 frames, by thirds:

- First third: 57 → 64 MB (warm-up).
- Middle third: 63 → 65 MB, the same rate as arm64.
- Final third: 64.5 → 79.5 MB, almost all of it **one 11.9 MB step at frame
  ~31,025**. That is 65 frames after a loop boundary, where the harness
  creates a new AVAssetReader and video decoder.

The step pattern, rather than a linear ramp, points to an allocator or
decoder high-water mark, not a per-frame leak in MediaPipe. A soak with a
persistent decoder would separate the two if it ever matters.

\* The GPU working set of about 1.0–1.6 GB at stress rates (no pacing,
140–190 fps) is **pre-existing and unchanged**. The released `apple.3`
binary shows the same plateau, and both versions *shrink* over 12,900 frames,
so it is bounded (the texture-cache working set described in
[MEMORY_NOTES.md](../samples/WebcamLandmarksDemo/MEMORY_NOTES.md)); it is not
a leak. It would be smaller at camera rates (30 fps), which was not
measured here.
† Rosetta numbers include translation overhead (about +100 MB of translated
code, and CPU about 1.75× slower). They are not representative of Intel
hardware.

Run-to-run noise: the `apple.3` and new arm64 binaries execute byte-identical
code, yet their means differ by up to ±20% between runs, in both directions
(5-loop GPU: 5.25 vs 6.43 ms; 20-loop GPU: 8.33 vs 6.92 ms). Treat
single-run latency deltas below about 20% as noise on this laptop.

Crashes: **none** across all runs. The runner records signal exits and
collects DiagnosticReports, and every `crash/` directory is empty.

## GPU (Metal) on Intel: investigation

- **What happens:** with `.gpu` on the Intel host, MediaPipe selects the
  system default Metal device, the discrete **AMD Radeon Pro 5500M**
  (`hasUnifiedMemory = false`, GPU family `mac2`, not `apple*`). OpenGL
  (`ATI-7.1.8`) also lands on the AMD GPU. The TFLite Metal delegate
  initializes ("Created TensorFlow Lite delegate for Metal") and each frame
  takes about 15 ms. **Every result is empty:** 0 hands on all four stills and
  on all 1,290 probe video frames. There is no error and no crash. This is a
  silent wrong result, which is the worst failure mode for consumers.
- **Not an x86_64 build problem:** the identical x86_64 machine code runs the
  Metal path correctly on the Apple GPU under Rosetta (within 0.0007 of
  native arm64, and all GPU tests in the Swift suite pass).
- **Root cause: not isolated in this increment.** Candidates, in order of
  suspicion:
  1. Shared-storage `MTLBuffer` tensors (`tensor.cc`,
     `newBufferWithBytesNoCopy … StorageModeShared`), or
     CVPixelBuffer/IOSurface-backed textures, on a discrete
     non-unified-memory GPU.
  2. TFLite Metal kernels on AMD.
  3. Half-precision conversion in `image_to_tensor_converter_metal`.

  The `MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1` probe reproduces the failure on
  demand; see the next-steps section below.
- **Untested:** the integrated Intel UHD 630 (unified memory). Selecting it
  needs the system's automatic graphics switching turned off, a system
  setting that was deliberately not changed.
- **Mitigation shipped in this change:** the Swift wrapper's `checkDelegate`
  now rejects `.gpu` with `MediaPipeError.unsupportedDelegate` in **x86_64
  builds only**, when the default Metal device is not an Apple GPU
  (`!supportsFamily(.apple1)`). The message reads: *"The GPU delegate is not
  supported on this Intel Mac's GPU (AMD Radeon Pro 5500M) … Use .cpu (or
  .coreML …)"*. Consumers whose delegate selection already falls back on
  `unsupportedDelegate` (an auto chain of `[gpu, cpu]`) therefore land on CPU
  automatically instead of silently detecting nothing. Under Rosetta (Apple
  GPU) `.gpu` stays enabled. In arm64 builds the check is compiled out
  (`#if arch(x86_64)`).
- **Core ML on Intel** works. It runs on CPU/GPU, since there is no ANE. It
  logs `MPSGraph doesn't support padded tensors on Intel macs` (Core ML then
  takes another path; results are within 0.0025 of arm64). It is about 1.7×
  faster than XNNPACK on this i9, and the first frame pays a one-time Core ML
  load of about 300 ms.

## Architecture audit (native runtime, ObjC++ bridge, Swift, dependencies)

- **Swift API / ObjC++ bridge / C API:** no architecture-conditional code
  (grepped for `arm64`, `x86`, `__aarch64__`, `__x86_64__`, `arch(`,
  `TARGET_CPU`). Struct layouts are the same LP64 on both architectures. The
  only new conditional is the Intel GPU guard above.
- **Native runtime:** the full `//mediapipe/tasks/c:mediapipe_macos` target
  compiles for `darwin_x86_64` without source changes. XNNPACK selects x86
  kernels at runtime (cpuinfo). The i9-9880H reports AVX2/FMA and no AVX-512;
  `xnn_enable_avx512amx=false` stays as before.
- **OpenCV / zlib:** the only non-system dependencies. MacPorts on Apple
  silicon is arm64-only, so they are source-built for x86_64 at the same
  versions (`build_macos_deps.sh`). OpenCV's CMake initially configured a
  **NEON** baseline for the x86_64 build, because CMake reads the host CPU,
  which silently dropped every SSE/AVX path. This is fixed by declaring the
  target processor, and the script now fails if the SIMD baseline doesn't
  match the arch.
- **Deployment targets:** arm64 dylib `minos 11.0` (unchanged). x86_64 dylib
  and its bundled libs `minos 13.0`, matching the package floor.
  **Pre-existing:** the arm64 bundled MacPorts libs are `minos 26.0` (now
  surfaced as an audit warning; see the ARM impact section below).
- **Packaging/gates:** two existing gates parsed `otool -L` as if every
  binary were thin: the iOS script's portability gate, and the consumer
  `verify_archive.sh`. Both would have misread the universal slice (the
  per-arch header line contains the build path), and both ignored
  `Libraries/*.dylib`. Both now check every Mach-O for every architecture,
  and `verify_archive.sh` fails on a bundled dylib missing an architecture
  (tested with a planted thin `libz`).

## Every distributed Mach-O (`audit_xcframework.sh`, PASSED)

| Slice / file | Archs | minos | Non-system deps | Signature | dSYM UUID |
|---|---|---|---|---|---|
| macos `MediaPipeTasksC` | x86_64 + arm64 | 13.0 / 11.0 | `@rpath/libopencv_{core,imgproc}.3.4.20.dylib` | ad-hoc, valid | `132D101A…` (x86_64), `9D468F6A…` (arm64): match |
| macos `Libraries/libopencv_core.3.4.20.dylib` | x86_64 + arm64 | 13.0 / 26.0 ⚠ | `@rpath/libz.1.3.2.dylib` | valid | — |
| macos `Libraries/libopencv_imgproc.3.4.20.dylib` | x86_64 + arm64 | 13.0 / 26.0 ⚠ | `@rpath/libopencv_core…` | valid | — |
| macos `Libraries/libz.1.3.2.dylib` | x86_64 + arm64 | 13.0 / 26.0 ⚠ | — | valid | — |
| ios-arm64 `MediaPipeTasksC` | arm64 | 17.0 | none | valid | `FA1D48B2…` match (byte-identical to `apple.3`) |
| ios-arm64-simulator `MediaPipeTasksC` | arm64 | 17.0 | none | valid | `310A422E…` match (byte-identical to `apple.3`) |

No file references `/opt/local`, `/opt/homebrew`, `/usr/local` or any
build-machine path, for either architecture. Every `@rpath` dependency
resolves inside `Libraries/`. The x86_64 OpenCV build is about 3.3 MB
larger (AVX2/AVX-512 dispatch code).

## ARM consumer impact (assessment)

**Unchanged for ARM:**

- Default build (`build_macos_xcframework.sh`, no new env vars): slice
  `macos-arm64`, **byte-identical to `apple.3`**, including binary,
  signature, bundled libs and Info.plist. The only difference is a random temp
  path inside the dSYM's relocation YAML, which already differed between any
  two builds.
- **Universal build, arm64 half:**
  - The framework binary has the same LC_UUID and is `cmp`-identical after
    stripping the ad-hoc signature. Only the resource-seal hash in the
    signature changes, because the sealed `Libraries/` are now fat.
  - The bundled dylibs' arm64 halves are byte-identical.
  - Inference outputs are **bit-identical** on CPU, GPU and Core ML, in both
    IMAGE and VIDEO mode.
  - The Swift test suite gives the same results, and ARM GPU memory behavior
    is the same.
- The Swift API surface is unchanged: no new or removed public symbols. The C
  ABI is unchanged.

**Changes an ARM consumer could notice (only if a universal release is
published):**

1. **Size:**
   - xcframework zip 27.8 → 40.8 MB.
   - macOS slice on disk 35 → 78 MB: the fat framework binary is
     54.5 MB (25.5 MB arm64), and the bundled libs are 13 MB (4.7 MB arm64
     + 8.0 MB x86_64).
   - macOS apps aren't thinned, so embedding apps grow by about 33 MB unless
     they `lipo -thin` at build time.
   - Two full-DWARF assets instead of one (`…macos-x86_64.dSYM.zip` is new;
     `…macos-arm64.dSYM.zip` keeps its name).
2. **Slice identifier** `macos-arm64` → `macos-arm64_x86_64`. SwiftPM and
   Xcode resolve it transparently. Only scripts that hardcode the path break;
   in this repo, `build_ios_xcframework.sh` did and is fixed.
3. **Embedded dSYM** carries two UUIDs. App Store Connect, Xcode and
   `verify_archive.sh` compare UUID sets, so this is fine.
4. **Stricter gates:** the iOS packaging and `verify_archive.sh` now also
   audit `Libraries/*.dylib`. A release that previously slipped through with a
   non-portable bundled lib would now fail, which is intended.
5. **Intel GPU guard:** compiled only into x86_64, so ARM is unaffected. Intel
   consumers selecting `.gpu` get an error instead of empty results.

**Recommended follow-up (not done here, because it changes ARM bits):** the
arm64 bundled MacPorts libs have `minos 26.0` while the package claims macOS
13. Apple-silicon Macs on macOS 13–15 may fail to load them if they use newer
libc++/libSystem symbols. This is untested; ScreenBar's fleet presumably runs
macOS 26. `MP_ARM64_DEPS=source` builds arm64 OpenCV/zlib from the same
pinned sources with `minos 13.0`. It replaces the arm64 OpenCV binaries, so it
needs its own validation pass: rerun this matrix, then compare against
`apple.3` with tolerances instead of bit-identity.

## Remaining blockers before calling Intel "released"

1. **GPU on Intel/AMD:** root-cause the empty results, or keep the guard.
   Test the integrated GPU on a machine where it is the default device.
2. **macOS 13–15 on Intel:** run the bundle on an Intel Mac with an older
   macOS. This is the realistic Intel fleet, since few Intel Macs run macOS
   26.
3. **Pose and Face on native Intel** (the hand matrix only). The runtime is
   shared, and the Swift suite covers both under Rosetta.
4. **Release itself:** publish the zip, update `Package.swift`'s
   URL/checksum, and repin StaxelPointer. This was out of scope by request.

## Reproduce

See [README.md](README.md). Raw comparison outputs, per-machine summaries
and environment captures from these runs are in
[`results/2026-10-26/`](results/2026-10-26/).
