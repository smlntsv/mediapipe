# Headless architecture validation

Repeatable, permission-free validation of `MediaPipeTasksC.xcframework` on
Apple silicon (native and Rosetta) and on Intel Macs (natively, over SSH). See
[../ARCHITECTURES.md](../ARCHITECTURES.md) for the build side and
[VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the latest results.

| File | Runs on | Purpose |
|---|---|---|
| `Package.swift`, `Sources/MPArchValidate/main.swift` | build Mac | `mp-arch-validate`: Hand Landmarker through the **public Swift API**. Subcommands: `env`, `image` (fixed stills, repeat-stability, plausibility), `video` (sustained VIDEO mode: latency percentiles, throughput, `phys_footprint` samples and slope, detection rates), `frames` (extracts PNG stills) |
| `package_bundle.sh` | build Mac | builds the validator (universal by default), bundles it with the framework copied verbatim from `Artifacts/`, the model, the Core ML hand models, the stills and the clip → `/tmp/mp_build/mp-validation.tar.gz` |
| `remote/run_validation.sh` | any Mac ≥ 13 | runs the matrix (image + video × cpu/gpu/coreml), one process per test; `--rosetta`, `--quick`, `--loops N`, `--delegates` |
| `remote_validate.sh` | build Mac | `scp` bundle → verify checksums → run → fetch `results/` (`./remote_validate.sh intel-mac`) |
| `compare_reports.py` | build Mac | tolerance comparison of two result dirs (stdlib Python; `python3 -I`) |

No camera, Accessibility permission, Xcode, Swift toolchain or Python is
needed on the target. Video frames come from a file through AVAssetReader.

Exit codes of `mp-arch-validate`: `0` ok · `1` implausible (e.g. no hands, or
hand rate below `--min-hand-rate` in video) · `3` landmarker creation failed
(`UNSUPPORTED`, e.g. `.gpu` against a CPU-only artifact) · `4` inference error. A signal
exit (crash) is recorded by the runner, which also collects crash reports.

Typical session:

```bash
cd mediapipe/tasks/macos/validation
./package_bundle.sh
mkdir -p /tmp/mp_build/local && tar -xzf /tmp/mp_build/mp-validation.tar.gz -C /tmp/mp_build/local
/tmp/mp_build/local/mp-validation/run_validation.sh --out /tmp/mp_build/results            # native-arm64
/tmp/mp_build/local/mp-validation/run_validation.sh --rosetta --out /tmp/mp_build/results  # rosetta-x86_64
./remote_validate.sh intel-mac                                                              # native-x86_64
python3 -I compare_reports.py /tmp/mp_build/results/native-arm64 /tmp/mp_build/results/native-x86_64 --allow-unsupported gpu
python3 -I compare_reports.py /tmp/mp_build/results/native-arm64 /tmp/mp_build/results/rosetta-x86_64
```

ARM regression check against a published release: build a bundle around
that release's xcframework with `./package_bundle.sh --archs arm64 --out
…`, after placing the release's xcframework in `Artifacts/`. Run it, then
compare its `native-arm64` results with the new build's. For identical arm64
binaries every delta is exactly 0.

Re-measure only Metal on an Intel Mac (for a new GPU or driver):

```bash
RESULTS=/tmp/mp_build/results-gpu-probe ./remote_validate.sh intel-mac --delegates gpu --loops 2
```

If Metal misbehaves on a new GPU, rerun a single still with Apple's Metal API
validation layer on the target. It needs no Xcode, and it aborts with the
first API misuse it finds:

```bash
ssh intel-mac 'cd ~/mp-validation-run/mp-validation && MTL_DEBUG_LAYER=1 ./bin/mp-arch-validate image --model models/hand_landmarker.task --image media/test_image.jpg --delegate gpu --repeat 1 --out /tmp/x.json'
```
