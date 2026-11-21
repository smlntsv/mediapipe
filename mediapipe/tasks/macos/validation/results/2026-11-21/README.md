# Validation: upstream merge release (`v1.3.0-apple.1`)

Date: 2026-11-21/22 · Source: branch `upstream-merge-2026-11` (upstream
`3ad0cf160` merged into `v1.0.0-apple.4`, `b4d072932`) · Artifact:
`MediaPipeTasksC.xcframework.zip`, swiftpm-checksum
`e0f8da05738d662a40f0135cb058973113faee0779830f0f3a6a28e4bba24c40`.
macOS UUIDs `5277D921-B07D-3D12-B5F6-C19E06E53E98` (arm64) and
`1806272A-3855-33CC-B5F3-66DC537E7196` (x86_64): the binaries measured below
are byte-identical to the ones in that zip.

Baselines are the published `v1.0.0-apple.3` and `v1.0.0-apple.4` zips. Their
validator bundles are built from the apple.4 sources: the C options structs
changed upstream, so the new Swift bridge cannot drive the old binaries.
apple.3 and apple.4 share the same C ABI.

## Hand Landmarker, `mp-arch-validate` (`run_validation.sh`, 5 loops)

Video: `two-hands-only.mov`, 645 frames × 5 = 3,225 frames, VIDEO mode, two
hands. Stills: 4 images, IMAGE mode.

### Native arm64, MacBookPro18,1, M1 Pro, macOS 26.3.1

| Delegate | Landmarks vs apple.3 and apple.4 | apple.3 mean / p90 | apple.4 mean / p90 | new mean / p90 |
|---|---|---|---|---|
| `.cpu` | bit-identical (max \|dxy\| 0.00000) | 22.50 / 25.91 ms | 22.66 / 25.89 ms | 22.94 / 26.12 ms |
| `.gpu` (Metal) | bit-identical | 5.74 / 8.01 ms | 5.02 / 6.39 ms | 5.22 / 6.72 ms |
| `.coreML` | bit-identical | 3.14 / 4.17 ms | 3.00 / 3.99 ms | 3.10 / 4.16 ms |

Hand rates are identical (77.8% / 77.7% / 77.7% for CPU / GPU / Core ML).
Memory is flat on CPU and Core ML. The GPU working set (peak about 1.6 GB at
about 200 fps) is the same as apple.4's bounded texture-cache working set.

### Native x86_64, MacBookPro16,1, i9-9880H, AMD Radeon Pro 5500M, macOS 26.7.1

| Delegate | Landmarks vs apple.4 | apple.4 mean / p90 | new mean / p90 |
|---|---|---|---|
| `.cpu` | bit-identical | 24.03 / 28.29 ms | 24.16 / 28.44 ms |
| `.gpu` (Metal, AMD) | bit-identical | 20.38 / 24.27 ms | 20.30 / 24.40 ms |
| `.coreML` (no ANE) | bit-identical | 14.10 / 16.62 ms | 13.92 / 16.65 ms |

Full comparisons: [`apple3_vs_arm64.txt`](apple3_vs_arm64.txt),
[`apple4_vs_arm64.txt`](apple4_vs_arm64.txt),
[`apple4_vs_intel.txt`](apple4_vs_intel.txt).

## Regressions found during the merge and fixed before release

1. **GPU delegate failed to start.** Upstream builds the hand and pose
   landmarker graphs with `use_litert=true`, which routes `.gpu` to LiteRT's
   closed-source GPU accelerator ("No GPU accelerator is registered in the
   LiteRT environment"). Apple builds keep the TFLite path (`844bc4d5d`).
2. **GPU 1.9× slower** (9.66 vs 5.09 ms) with bit-identical output. Upstream
   hard-wired `MEDIAPIPE_GPU_BUFFER_POOL_USE_CV_PIXEL_BUFFER=0` on macOS, so
   every input frame took two CPU texture uploads. The macOS build sets it to
   1 again (`939848b7e`).

## Other checks

- `swift test` with real models: 29 tests, 0 failures (6 skipped: the 5
  env-gated `VerifyRotationBenefit` tests, run separately below, and the
  CPU-only-artifact test). This includes `MemoryStressTests`: GPU memory stays
  bounded for hand, pose and face, so the rebased texture-cache flush patch
  works.
- `MP_VERIFY=1` rotation, `roiScale` (2.5 on the 90°-rotated clip) and
  `trackingGraceFrames` runs: output identical to apple.4. With randomized
  hashing the rotation sweep's tie-break between equally good angles varies
  from run to run; with `SWIFT_DETERMINISTIC_HASHING=1` both builds print the
  same numbers.
- Bazel `hand_rects_keep_alive_calculator_test`: 4/4 pass.
- Web/macOS parity (`test_projects/compare/compare.py`, tasks-vision 0.10.18):
  CPU IMAGE, CPU VIDEO and full-body pose pass. GPU: hand and face pass; pose
  on the cropped upper-body image is at 0.041 / 0.050 against the 0.04
  tolerance. apple.4 gives exactly the same numbers, and the new build's GPU
  output is bit-identical to apple.4 for all three tasks (WebGL vs Metal;
  report-only per the parity README).
- iOS: `MediaPipeIOSTest` builds for device and arm64 simulator. The iOS
  slices were rebuilt from the merged sources. Inference on iOS was not run:
  the app needs a camera, and no device was available.

## Not covered

- Pose and Face Landmarker timings (only smoke, parity and memory tests).
- Intel Macs on macOS 13–15, and Apple-silicon Macs below macOS 26: the
  bundled arm64 OpenCV/zlib dylibs still carry `minos 26.0` (unchanged since
  apple.3).
