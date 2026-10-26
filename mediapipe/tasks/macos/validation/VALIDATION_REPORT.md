# Validation report: Intel (x86_64) macOS support, universal macOS slice

Date: 2026-10-26 · Source: `b55fb6819` + this change. Every file compiled
into the binary is identical to `6cecfe31b`, the revision StaxelPointer pins
(the only commit in between adds `bump_package_pin.sh`) · Artifact: local,
**unreleased** `MediaPipeTasksC.xcframework` (`ios-arm64`,
`ios-arm64-simulator`, `macos-arm64_x86_64`), swiftpm-checksum
`95f727c247c10d569002633f26289ff7ef6916756e85ec8506a7ffa49c962c7f`.
Nothing was published or pushed, and no downstream app was changed.

**Update (same day): Metal on Intel/AMD GPUs fixed.** The root cause of the
empty Intel GPU results is in MediaPipe's CPU→GPU image upload (see
[GPU (Metal) on Intel](#gpu-metal-on-intel-investigation)). With the fix, the
Intel GPU matrix passes against native arm64. arm64 results stay
bit-identical to the pre-fix build, and the Swift Intel-GPU guard is removed.
The new local artifact's swiftpm-checksum is
`1934fad6576b153b693f19004c922beffcdec5b9ca5d4e7daa218a37253b38e0`.
Results are in
[`results/2026-10-26/intel-gpu-fix/`](results/2026-10-26/intel-gpu-fix/).
Rows below without a fix note are from the pre-fix build. That build's
arm64 results are identical to the fixed build, and the fix does not touch
the CPU or Core ML paths. The fix changes two compiled source files, so
"identical to `6cecfe31b`" above describes only the pre-fix build.

## Verdict

| | CPU (XNNPACK) | GPU (Metal) | Core ML |
|---|---|---|---|
| **Native arm64**, M1 Pro, macOS 26.3.1 | ✅ validated: bit-identical to `v1.0.0-apple.3` | ✅ validated: bit-identical to `apple.3` | ✅ validated: bit-identical to `apple.3` |
| **Rosetta x86_64**, same M1 Pro (not Intel validation) | ✅ smoke-tested, within tolerance | ✅ smoke-tested (Apple GPU) | ✅ smoke-tested |
| **Native x86_64**, real Intel Mac: MacBookPro16,1, i9-9880H, AMD Radeon Pro 5500M + Intel UHD 630, macOS 26.7.1 | ✅ **validated natively** | ✅ **validated natively** on the AMD Radeon Pro 5500M, after the upload fix below | ✅ **validated natively** (no Neural Engine on Intel) |

Not covered:

- **Intel Macs on macOS 13–15.** The only Intel host runs macOS 26.7.1. The x86_64 slice is built for `minos 13.0` throughout, but has not been *run* on an older OS.
- **MediaPipe on the integrated Intel UHD 630.** Selecting it would require changing the Mac's graphics-switching system setting. The standalone repros (see GPU below) do run on it and show the same root cause and the same fix.
- **Pose and Face on Intel.** They share the same runtime, and the Swift suite covers them under Rosetta, but the Intel matrix ran Hand only.

## What was run

| Check | Native arm64 | Rosetta x86_64 | Native Intel |
|---|---|---|---|
| Build: each arch independently, then lipo-merge, audit and dSYM UUIDs | ✅ | (same artifact) | (same artifact) |
| Existing Swift suite (`swift test`; 29 tests: hand/pose/face, CPU+GPU, IMAGE+VIDEO, memory stress) | ✅ 23 passed, 6 env-gated skips, 0 failures. Same set as against `apple.3` | ✅ 23 passed, 6 skips, 0 failures (`arch -x86_64 xctest`; includes the 7 GPU tests on the Apple GPU) | — (no Xcode on the Intel host) |
| `mp-arch-validate` matrix: 4 fixed stills × 23 runs, plus 5 × 645-frame sustained VIDEO, × cpu/gpu/coreml | ✅ all PASS, 0 crashes | ✅ all PASS, 0 crashes | ✅ all PASS (gpu after the fix), 0 crashes |
| 20-loop soak (12,900 frames) | gpu (new and `apple.3`), cpu | — | cpu, coreml, gpu after the fix (+ 60-loop cpu soak, below) |

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
| native arm64 vs **native Intel** (after the fix) | gpu | 0.00550 | identical (≤ 0.0068) | 77.8% / 77.8% | 99.7% | 0.0074 |
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
| **native Intel** (after the fix) | gpu | 21.1 / 20.6 / 26.2 / 32.8 | 47.4 fps | 27.5 ms | 522 → 492 MB (522) |

Soak runs (20 loops = 12,900 frames, unless noted):

| Platform / delegate | Mean ms | Footprint after loop 0 → end (peak) | Slope MB / 1k frames |
|---|---|---|---|
| native arm64 gpu, **new** | 6.92 | 1317 → 1013 MB (1330) | −5.2 |
| native arm64 gpu, **`apple.3`** | 8.33 | 1346 → 1072 MB (1479) | −23.6 |
| native arm64 cpu | 24.24 | 104 → 106 MB (110) | +0.16 |
| native Intel cpu | 23.79 | 59 → 75 MB (77) | +0.72 |
| native Intel coreml | 14.10 | 107 → 109 MB (112) | +0.14 |
| native Intel gpu (after the fix) | 21.36 (p99 33.5) | 508 → 531 MB (532), mean 402 / 403 / 403 MB by thirds | +0.19 |
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

**Symptom (before the fix).** With `.gpu` on the Intel host, MediaPipe uses
the system default Metal device, the discrete **AMD Radeon Pro 5500M**
(`hasUnifiedMemory = false`, family `mac2`). OpenGL (`ATI-7.1.8`) also lands
on the AMD GPU. The TFLite Metal delegate initializes and each frame takes
about 15 ms, but **every result is empty**: 0 hands on all four stills and on
all 1,290 probe video frames, with no error and no crash. The same x86_64 code
works on the Apple GPU under Rosetta, so this was not an x86_64 build issue.

**Root cause: the CPU→GPU image upload writes under a read-only lock.**
On the GPU path, `ImageCloneCalculator` moves the input `ImageFrame` to the GPU
through `GpuBufferStorageCvPixelBuffer`'s converter, which calls
`CreateCVPixelBufferForImageFrame` (`mediapipe/objc/util.cc`). For SRGBA input
it creates an IOSurface-backed `CVPixelBuffer`, then locks it with
`kCVPixelBufferLock_ReadOnly` and **writes** the R/B-swapped pixels into it
with the CPU. It also unlocks with `kCVPixelBufferLock_ReadOnly`. The read-only
flag tells IOSurface that the CPU did not modify the surface.

- Apple GPUs see the written pixels anyway.
- Intel and AMD GPUs never do. `CVMetalTextureCache` vends a texture whose
  contents are all zero, so the palm detector ran on a black image, found
  nothing, and the graph returned empty results without any error.

`CreateCVPixelBufferCopyingImageFrame` in the same file had the same pattern.

**Evidence.** Standalone repros, sources and output are in
[`results/2026-10-26/intel-gpu-fix/repro/`](results/2026-10-26/intel-gpu-fix/repro/).
They run on every Metal device of the machine, which also covers the
integrated GPU that MediaPipe cannot select without changing system settings.

| Repro | Apple M1 Pro | Intel UHD 630 (unified memory) | AMD Radeon Pro 5500M (discrete) |
|---|---|---|---|
| `iosurface_lock_repro`: CPU writes under a **read-only** lock, GPU reads via `CVMetalTextureCache` | 196,608 / 196,608 px correct | **0 / 196,608** (all zero) | **0 / 196,608** (all zero) |
| same, lock flags `0` (read-write) | 196,608 / 196,608 | 196,608 / 196,608 | 196,608 / 196,608 |
| `linear_rt_repro`: render into a texture aliasing an `MTLBuffer` (the image→tensor pattern) | correct | correct | correct |

The failure is therefore not specific to discrete memory or to AMD: it
affects both Intel-Mac GPUs. A build with only the image→tensor change below
still returned 0 hands. Adding the lock fix restored detections.

The earlier suspects were ruled out:

- Shared-storage `MTLBuffer` tensors work: the whole pipeline passes after the
  fix with them unchanged.
- TFLite Metal kernels on AMD work: results match arm64 within tolerance.
- The half-precision path is not involved: the converter uses `kF32C4`.

**Fix (two changes in shared MediaPipe code):**

1. `mediapipe/objc/util.cc`: lock and unlock the new pixel buffer read-write
   (flags `0`) in `CreateCVPixelBufferForImageFrame` and
   `CreateCVPixelBufferCopyingImageFrame`. This is the root-cause fix. It is
   unconditional, because the read-only flag was wrong on every GPU and only
   happened to work on Apple GPUs.
2. `mediapipe/calculators/tensor/image_to_tensor_converter_metal.cc`:
   defensive, not the root cause. With `MTL_DEBUG_LAYER=1` on the AMD GPU, the
   first assertion is "linear textures cannot be created with usage
   MTLTextureUsageRenderTarget": the image→tensor crop renders into a texture
   aliasing the tensor's `MTLBuffer`. It works today (see the repro), but it is
   outside the API contract on non-Apple GPUs. On GPUs without
   `MTLGPUFamilyApple1`, the converter now renders into a private texture and
   blits it into the buffer (same values, same bytes per row). Apple GPUs keep
   the original direct path. With both changes, a GPU run on the AMD GPU
   under `MTL_DEBUG_LAYER=1` completes with no assertion and correct results.

**Results after the fix:**

- **Intel, AMD 5500M:** the full matrix passes `compare_reports.py` against
  native arm64, with no `--allow-unsupported`. Image max |dxy| is 0.0055 on
  the worst still, 0.0004–0.0027 on the others. Video hand-rate is 77.8% /
  77.8% in every loop, with 99.7% frame agreement and p90 |dxy| 0.0074.
- **Intel performance:** about 21 ms/frame (47 fps). That is only slightly
  faster than XNNPACK on this i9 (23.7 ms) and slower than Core ML (14.1 ms).
  The CPU→IOSurface upload and the PCIe round trips dominate. For Intel
  consumers, `.coreML` remains the fastest delegate.
- **Intel memory:** the 20-loop soak (12,900 frames) is flat: 508 → 531 MB
  (peak 532), slope +0.19 MB/1k frames, and the mean footprint is
  402 / 403 / 403 MB across thirds.
- **arm64:** `compare_reports.py` of the pre-fix and fixed arm64 builds shows
  **0.00000** on every still and every landmark, and identical video for
  cpu/gpu/coreml (100% agreement, p90 0.0000). The arm64 GPU footprint
  differs between runs because of machine state. Back to back, old and new
  peak at 1537/1541 MB and 1543/1540 MB. The Swift suite on arm64 passes
  with 0 failures, including the hand/pose GPU parity, smoke and
  memory-stress tests. Face tests were skipped because the repo has no face
  test image.
- **Rosetta:** passes against native arm64 (gpu image max |dxy| 0.0007).

**Guard removed.** The x86_64-only `checkIntelMacGPU` in `Modes.swift`
rejected `.gpu` on non-Apple GPUs. It is deleted, along with
`MEDIAPIPE_ALLOW_UNVALIDATED_GPU` and the runner's
`--allow-unvalidated-gpu`. `Modes.swift` is back to its pre-guard content.
The guard existed only because of the empty results, and the root cause
applies to all Intel-Mac GPUs, as does the fix. The guard ships in the same
Swift package as the binary it protected, so no released consumer depends
on it.

**Still not run:** MediaPipe on the integrated UHD 630 as its Metal device
(only the repros cover it), other Intel-Mac GPUs (Iris, Radeon 5xx/Vega), and
Intel Macs on macOS 13–15.

- **Core ML on Intel** works. It runs on CPU/GPU, since there is no ANE. It
  logs `MPSGraph doesn't support padded tensors on Intel macs` (Core ML then
  takes another path; results are within 0.0025 of arm64). It is about 1.7×
  faster than XNNPACK on this i9, and the first frame pays a one-time Core ML
  load of about 300 ms.

## Architecture audit (native runtime, ObjC++ bridge, Swift, dependencies)

- **Swift API / ObjC++ bridge / C API:** no architecture-conditional code
  (grepped for `arm64`, `x86`, `__aarch64__`, `__x86_64__`, `arch(`,
  `TARGET_CPU`). Struct layouts are the same LP64 on both architectures. The
  Intel GPU guard that briefly added one was removed with the Metal fix.
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
5. **Metal fix for Intel GPUs:** two shared source files change
   (`mediapipe/objc/util.cc`, `image_to_tensor_converter_metal.cc`), so the
   arm64 binary is no longer built from sources identical to `6cecfe31b`.
   Its results are still bit-identical: 0.00000 on cpu/gpu/coreml against
   the pre-fix build. Apple GPUs keep the direct image→tensor path. The only
   change they run is the read-write lock during the CPU→GPU upload. The
   short-lived x86_64 guard is gone, so Intel consumers selecting `.gpu` now
   get Metal.

**Recommended follow-up (not done here, because it changes ARM bits):** the
arm64 bundled MacPorts libs have `minos 26.0` while the package claims macOS
13. Apple-silicon Macs on macOS 13–15 may fail to load them if they use newer
libc++/libSystem symbols. This is untested; ScreenBar's fleet presumably runs
macOS 26. `MP_ARM64_DEPS=source` builds arm64 OpenCV/zlib from the same
pinned sources with `minos 13.0`. It replaces the arm64 OpenCV binaries, so it
needs its own validation pass: rerun this matrix, then compare against
`apple.3` with tolerances instead of bit-identity.

## Remaining blockers before calling Intel "released"

1. **GPU on other Intel-Mac GPUs:** fixed and validated on the AMD 5500M.
   MediaPipe has not run on an integrated Intel GPU as its Metal device (the
   repros cover the UHD 630) or on other AMD generations.
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
