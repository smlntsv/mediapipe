# macOS architectures: Apple silicon, Intel, universal

The macOS slice of `MediaPipeTasksC.xcframework` can be built for **arm64**
(Apple silicon, the historical default), **x86_64** (Intel), or both as one
**universal** slice (`macos-arm64_x86_64`). The Swift API, the ObjC++ bridge
and the C API are identical on every architecture — no source changes, no
`#if arch(...)` anywhere in the package. Only the build configuration differs,
and those differences are confined to `build_arch()` in
[`build_macos_xcframework.sh`](build_macos_xcframework.sh).

Validation status lives in
[`validation/VALIDATION_REPORT.md`](validation/VALIDATION_REPORT.md). Intel
support counts as **validated only once it has run natively on a real Intel
Mac.** Rosetta runs exercise the x86_64 code but on Apple's CPU and GPU, so
they do not count.

## What differs per architecture (and what does not)

| | arm64 (default, unchanged) | x86_64 |
|---|---|---|
| Bazel config | `--config=macos --config=darwin_arm64` | `--config=macos --config=darwin_x86_64 --macos_minimum_os=13.0` |
| Deployment target of the MediaPipe dylib | 11.0 (from `darwin_arm64`'s `--macos_minimum_os=10.16`) | 13.0 (`MACOS_MIN_VERSION`, = package floor). `darwin_x86_64`'s own 10.12 predates the C++17 library availability MediaPipe needs |
| OpenCV 3.4.20 + zlib 1.3.2 | MacPorts `/opt/local` (as before). `MP_ARM64_DEPS=source` opts into the source-built deps | built from pinned, sha256-verified sources by [`build_macos_deps.sh`](build_macos_deps.sh) and swapped in via `--override_repository=macos_opencv=<prefix>` (the same `third_party/opencv_macos.BUILD` file) |
| OpenCV SIMD | NEON (MacPorts) | baseline SSE…SSE4.1, runtime dispatch SSE4.2/AVX/AVX2/AVX-512 |
| Compiler defines / copts | `MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER=1` (`--copt` + `--objccopt`), `-c opt` | identical |
| In-tree patches (GPU texture-cache flush) | applied | identical |
| Delegates compiled in | CPU (XNNPACK), GPU (Metal), Core ML | identical (runtime support on Intel: see the report) |
| XNNPACK kernels | ARM NEON/dotprod/fp16 | x86 SSE/AVX/AVX2/AVX-512 (selected at runtime by cpuinfo; `xnn_enable_avx512amx=false` as before) |

## Delegate support by platform

| | `.cpu` | `.gpu` (Metal) | `.coreML` |
|---|---|---|---|
| Apple silicon (arm64) | ✅ | ✅ | ✅ (ANE) |
| x86_64 under Rosetta (Apple GPU) | ✅ | ✅ | ✅ |
| Intel Mac, Intel/AMD GPU | ✅ | ❌ throws `MediaPipeError.unsupportedDelegate` | ✅ (CPU/GPU, no ANE) |

On a real Intel Mac (MacBookPro16,1, AMD Radeon Pro 5500M as the default
Metal device), MediaPipe's Metal path runs without errors but returns **no
detections**. To avoid silently empty results, the Swift wrapper rejects
`.gpu` when the default Metal device is not an Apple GPU. The check lives in
`checkDelegate` in `Modes.swift`, is compiled only into x86_64 (`#if
arch(x86_64)`), and leaves arm64 untouched. Consumers that fall back on
`unsupportedDelegate`, such as a `[gpu, cpu]` auto chain, land on CPU. To
re-measure new hardware or drivers, set `MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1`
(the validation runner's `--allow-unvalidated-gpu`). Details and the open
root-cause candidates are in the validation report.

Why the x86_64 OpenCV is source-built: on an Apple-silicon build machine,
MacPorts `opencv3`/`zlib` are arm64-only, so the x86_64 link fails with
`found architecture 'arm64', required architecture 'x86_64'`. `port install
opencv3 +universal` would rebuild (and change) the arm64 libraries too, and it
needs root. `build_macos_deps.sh` reproduces the **same versions** with the
MacPorts Portfile options that matter for `core`/`imgproc` (shared libs,
Accelerate LAPACK, GCD `parallel_for`, no IPP/TBB/OpenCL/OpenMP), trimmed to
those two modules. It verifies arch, `minos` and dependencies of each output.
It reuses the MacPorts distfile cache for the OpenCV tarball when present (the
exact tarball the arm64 libs were built from), so the sha256 pins match the
MacPorts Portfiles.

Known pre-existing issue (arm64, **not changed** here): the MacPorts libraries
bundled in the arm64 slice carry `minos 26.0` (MacPorts stamps the build host's
OS), while the framework binary says 11.0 and the package says macOS 13. They
may fail to load on macOS 13–15 Apple-silicon Macs if they reference newer
libSystem/libc++ symbols. `audit_xcframework.sh` now prints this as a warning.
Fixing it is a one-flag change (`MP_ARM64_DEPS=source`), but that replaces the
arm64 OpenCV binaries, so it is left as an explicit decision (see the report's
"ARM consumer impact").

## Build commands

Toolchain prerequisites are unchanged (see [README.md](README.md#prerequisites)).
Additionally, `cmake` is needed for the x86_64 dependency build.

Use **one fixed environment** for every Bazel invocation. Bazel keys its
action cache on the client `PATH`, so reordering `PATH` silently rebuilds
everything:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
mkdir -p /tmp/mp_build/pyshim
ln -sf /opt/homebrew/bin/python3.12 /tmp/mp_build/pyshim/python3
ln -sf /opt/homebrew/bin/python3.12 /tmp/mp_build/pyshim/python
export PATH="/tmp/mp_build/pyshim:$JAVA_HOME/bin:$PATH"
```

ARM only (default, unchanged behavior, slice `macos-arm64`):

```bash
./mediapipe/tasks/macos/build_macos_xcframework.sh
```

Intel only (slice `macos-x86_64`):

```bash
MP_MACOS_ARCHS=x86_64 ./mediapipe/tasks/macos/build_macos_xcframework.sh
```

Universal (slice `macos-arm64_x86_64`). Each architecture is built, bundled and
portability-checked independently. The framework binary and every bundled
dylib are then `lipo`-merged, after verifying that both architectures bundle
the same file set with identical install names and dependency lists:

```bash
MP_MACOS_ARCHS="arm64 x86_64" ./mediapipe/tasks/macos/build_macos_xcframework.sh
```

The x86_64 dependency prefix is built on first use (about 2 minutes) into
`~/Library/Caches/mediapipe-macos-deps/x86_64` (`MP_MACOS_DEPS_DIR` overrides
the root). To build it explicitly:

```bash
./mediapipe/tasks/macos/build_macos_deps.sh x86_64
```

### Release recipe (universal macOS + unchanged iOS slices)

```bash
MP_MACOS_ARCHS="arm64 x86_64" MP_BUNDLE_DEPS=1 MPP_BUILD_VERSION=1.0.0 \
MP_PRESERVE_SLICES_FROM=/path/to/previous/MediaPipeTasksC.xcframework.zip \
  ./mediapipe/tasks/macos/build_macos_xcframework.sh
```

`MP_PRESERVE_SLICES_FROM` copies every non-macOS slice and its embedded dSYM
byte-for-byte from an existing xcframework or release zip. The iOS slices are
therefore not rebuilt or perturbed. It then writes
`MediaPipeTasksC.xcframework.zip` and prints the SwiftPM checksum.
Alternatively, rebuild iOS from source as before. `build_ios_xcframework.sh`
now picks up whichever `macos-*` slice exists:

```bash
MP_MACOS_ARCHS="arm64 x86_64" MP_BUNDLE_DEPS=1 MPP_BUILD_VERSION=1.0.0 ./mediapipe/tasks/macos/build_macos_xcframework.sh
MPP_BUILD_VERSION=1.0.0 ./mediapipe/tasks/macos/build_ios_xcframework.sh
```

Outputs in `mediapipe/tasks/macos/swift/Artifacts/`:

- `MediaPipeTasksC.xcframework`. Its universal slice contains a fat framework
  binary and fat `Libraries/libopencv_core.3.4.20.dylib`,
  `libopencv_imgproc.3.4.20.dylib` and `libz.1.3.2.dylib`, plus an embedded
  Tier-A dSYM with one UUID per architecture.
- `MediaPipeTasksC.macos-arm64.dSYM.zip` (same name as before) and
  `MediaPipeTasksC.macos-x86_64.dSYM.zip`: optional full-DWARF symbolication
  assets, one per architecture, generated from each thin binary before the
  merge.

## Gates (run on every build)

1. Per arch: exported C symbols, platform `MACOS`, `lipo -archs` equals the
   requested arch, and portability of the framework binary and every bundled
   dylib (`MP_BUNDLE_DEPS=1`).
2. Merge: identical bundled file set, install names and dependency lists
   across architectures.
3. Slice: every Mach-O carries exactly the requested architectures and is
   portable for every architecture: no `/opt/local`, `/opt/homebrew`,
   `/usr/local` or build-machine path in any load command.
4. dSYM UUIDs equal the binary's UUIDs for every architecture.
5. [`audit_xcframework.sh`](audit_xcframework.sh) re-checks the finished
   xcframework: all of the above for **every slice**, every `@rpath` dependency
   present in `Libraries/`, a valid code signature, plus `minos` per arch with a
   warning when a bundled dylib needs a newer OS than the framework. Run it on
   any artifact or release zip:

   ```bash
   ./mediapipe/tasks/macos/audit_xcframework.sh mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework.zip
   ```

`build_ios_xcframework.sh` now uses the same audit. Its previous inline
check parsed `otool -L` as if every binary were thin, so it would have
misread a universal macOS slice.

## Validation (headless, local and remote)

[`validation/`](validation/) contains a SwiftPM CLI (`mp-arch-validate`) and
scripts that run the same Hand Landmarker matrix on any Mac. The matrix covers
IMAGE mode on fixed stills and sustained VIDEO mode, for each of CPU, GPU and
Core ML. It needs no camera, no Accessibility permission, no Xcode and no
Python on the target, so it runs over SSH.

```bash
cd mediapipe/tasks/macos/validation

# 1. Bundle the current Artifacts/ xcframework (universal validator binary).
./package_bundle.sh                       # -> /tmp/mp_build/mp-validation.tar.gz

# 2a. Native ARM + Rosetta on this Apple-silicon Mac.
mkdir -p /tmp/mp_build/local && tar -xzf /tmp/mp_build/mp-validation.tar.gz -C /tmp/mp_build/local
/tmp/mp_build/local/mp-validation/run_validation.sh --out /tmp/mp_build/results
/tmp/mp_build/local/mp-validation/run_validation.sh --rosetta --out /tmp/mp_build/results

# 2b. Native Intel over SSH (only the bundle is copied, into ~/mp-validation-run).
./remote_validate.sh <user@intel-mac>     # or a Host alias from ~/.ssh/config

# 3. Compare with tolerances (exit 1 on any gate failure).
python3 -I compare_reports.py /tmp/mp_build/results/native-arm64 /tmp/mp_build/results/native-x86_64
python3 -I compare_reports.py /tmp/mp_build/results/native-arm64 /tmp/mp_build/results/rosetta-x86_64
```

Results land in `results/<execution>/`, where `<execution>` is detected at
runtime: `native-arm64`, `native-x86_64`, or `rosetta-x86_64` (via
`sysctl.proc_translated`). Each report records the CPU, the Metal devices,
the OS and the SHA-256 of every input. Each test runs in its own process. A
crash is recorded as `CRASH(signal N)`, and its DiagnosticReport is copied
into `crash/`. An unsupported delegate is recorded as
`UNSUPPORTED(init failed)` and never silently falls back.

Tolerances used by `compare_reports.py`:

- **Image:** same hand count and handedness labels; max |dx|,|dy| ≤ 0.02 over
  all 21 landmarks (the Web-parity hand tolerance); handedness score delta
  ≤ 0.05; anatomical plausibility checks on both sides.
- **Video:** hand-detection rate within 3 percentage points per loop; at least
  95% frame-level hand-count agreement; p90 per-frame max |dxy| ≤ 0.02.
  Each machine decodes the video itself, so small decoder differences are
  expected.
- **Not gated:** latency, throughput and memory are printed side by side,
  since the machines are different hardware.
