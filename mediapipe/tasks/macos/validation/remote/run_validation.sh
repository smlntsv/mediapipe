#!/bin/bash
# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Runs the full headless validation matrix from an extracted mp-validation
# bundle (see ../package_bundle.sh). Needs only /bin/bash and the bundle: no
# Xcode, no Swift toolchain, no Python, no camera, no Accessibility permission.
#
#   ./run_validation.sh                 # native architecture of this Mac
#   ./run_validation.sh --rosetta       # x86_64 slice under Rosetta (Apple silicon only)
#   ./run_validation.sh --quick         # short video pass (smoke test)
#   ./run_validation.sh --delegates "cpu gpu"   # default: cpu gpu [coreml if bundled]
#   ./run_validation.sh --loops 10      # sustained video loops (default 5)
#   ./run_validation.sh --allow-unvalidated-gpu --delegates gpu
#                                       # bypass the Swift wrapper's Intel-GPU guard
#                                       # (MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1) to
#                                       # re-measure Metal on Intel/AMD GPUs
#
# Writes results/<execution>/ (execution = native-arm64 | native-x86_64 |
# rosetta-x86_64): env.json, image_<delegate>.json, video_<delegate>.json,
# logs/*.log, summary.tsv, and crash/ (any new DiagnosticReports for the
# validator). Each test runs in its own process so a crash in one delegate
# cannot hide results from another. Exit status is non-zero if any CPU test
# failed; GPU/Core ML failures are reported but do not fail the run (their
# support on a given machine is exactly what this matrix measures).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${HERE}/bin/mp-arch-validate"
MODEL="${HERE}/models/hand_landmarker.task"
COREML_DIR="${HERE}/models/coreml"
VIDEO="${HERE}/media/two-hands-only.mov"

ROSETTA=0
QUICK=0
LOOPS=5
DELEGATES=""
OUT_ROOT="${HERE}/results"
while (( $# )); do
  case "$1" in
    --rosetta) ROSETTA=1 ;;
    --quick) QUICK=1 ;;
    --loops) LOOPS="$2"; shift ;;
    --delegates) DELEGATES="$2"; shift ;;
    --out) OUT_ROOT="$2"; shift ;;
    --allow-unvalidated-gpu) export MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ -z "${DELEGATES}" ]]; then
  DELEGATES="cpu gpu"
  [[ -d "${COREML_DIR}" ]] && DELEGATES+=" coreml"
fi

HOST_ARCH="$(uname -m)"
# On Apple silicon `uname -m` is arm64 for native shells; force the slice so a
# Rosetta-launched shell cannot silently change what is being measured.
RUN=("${BIN}")
if [[ "${ROSETTA}" == 1 ]]; then
  if ! sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
    echo "error: --rosetta only makes sense on an Apple-silicon Mac" >&2; exit 2
  fi
  RUN=(arch -x86_64 "${BIN}")
elif sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
  RUN=(arch -arm64 "${BIN}")
fi

ENV_JSON="$("${RUN[@]}" env)" || { echo "error: validator failed to start (dyld?)" >&2; exit 1; }
EXECUTION="$(sed -n 's/.*"execution" : "\([^"]*\)".*/\1/p' <<< "${ENV_JSON}")"
OUT="${OUT_ROOT}/${EXECUTION:-unknown}"
rm -rf "${OUT}"
mkdir -p "${OUT}/logs" "${OUT}/crash"
echo "${ENV_JSON}" > "${OUT}/env.json"
if [[ "${MEDIAPIPE_ALLOW_UNVALIDATED_GPU:-0}" == 1 ]]; then
  echo "note: MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1 (Intel-GPU guard bypassed)" | tee "${OUT}/UNVALIDATED_GPU_ALLOWED"
fi
printf 'test\tdelegate\texit\tresult\n' > "${OUT}/summary.tsv"

echo "== MediaPipe validation: ${EXECUTION} on $(sysctl -n hw.model) ($(sw_vers -productVersion)), host uname ${HOST_ARCH}"
grep -E '"(cpu_brand|metal_default_device|os_version)"' "${OUT}/env.json" | sed 's/^ */   /'
# macOS /bin/bash is 3.2: every possibly-empty array below is expanded with the
# ${a[@]+"${a[@]}"} idiom, which `set -u` accepts there.

STAMP="$(mktemp)"
FAILED_CPU=0

record() {  # name delegate exit_code
  local name="$1" del="$2" code="$3" result
  if (( code == 0 )); then result="PASS"
  elif (( code == 1 )); then result="IMPLAUSIBLE"
  elif (( code == 3 )); then result="UNSUPPORTED(init failed)"
  elif (( code == 4 )); then result="INFERENCE_FAILED"
  elif (( code > 128 )); then result="CRASH(signal $((code - 128)))"
  else result="ERROR(${code})"; fi
  printf '%s\t%s\t%s\t%s\n' "${name}" "${del}" "${code}" "${result}" >> "${OUT}/summary.tsv"
  printf '   %-6s %-7s %s\n' "${name}" "${del}" "${result}"
  if [[ "${del}" == "cpu" && "${code}" -ne 0 ]]; then FAILED_CPU=1; fi
}

images=()
for f in "${HERE}"/media/*.jpg "${HERE}"/media/*.png; do
  [[ -f "${f}" ]] && images+=(--image "${f}")
done

for del in ${DELEGATES}; do
  extra=()
  [[ "${del}" == "coreml" ]] && extra=(--coreml-cache "${COREML_DIR}")
  echo "-- image ${del}"
  "${RUN[@]}" image --model "${MODEL}" "${images[@]}" --delegate "${del}" \
    --repeat 20 --warmup 3 "${extra[@]+"${extra[@]}"}" --out "${OUT}/image_${del}.json" \
    > "${OUT}/logs/image_${del}.log" 2>&1
  code=$?
  sed 's/^/   | /' "${OUT}/logs/image_${del}.log" | tail -6
  record image "${del}" "${code}"
done

for del in ${DELEGATES}; do
  extra=()
  [[ "${del}" == "coreml" ]] && extra=(--coreml-cache "${COREML_DIR}")
  if [[ "${QUICK}" == 1 ]]; then extra+=(--max-frames 150); loops=1; else loops="${LOOPS}"; fi
  echo "-- video ${del} (${loops} loop(s))"
  "${RUN[@]}" video --model "${MODEL}" --video "${VIDEO}" --delegate "${del}" \
    --loops "${loops}" "${extra[@]+"${extra[@]}"}" --out "${OUT}/video_${del}.json" \
    > "${OUT}/logs/video_${del}.log" 2>&1
  code=$?
  sed 's/^/   | /' "${OUT}/logs/video_${del}.log" | tail -3
  record video "${del}" "${code}"
done

# Collect crash reports written during this run.
find "${HOME}/Library/Logs/DiagnosticReports" -newer "${STAMP}" -name 'mp-arch-validate*' \
  -exec cp {} "${OUT}/crash/" \; 2>/dev/null || true
rm -f "${STAMP}"
n_crash="$(ls "${OUT}/crash" | wc -l | tr -d ' ')"
echo "== results: ${OUT} (crash reports: ${n_crash})"
exit "${FAILED_CPU}"
