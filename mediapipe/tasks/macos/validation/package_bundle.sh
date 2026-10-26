#!/usr/bin/env bash
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
# Packages a self-contained validation bundle (mp-validation.tar.gz) that runs
# on any Mac with macOS >= 13 — no Xcode/Swift/Python needed on the target:
#
#   mp-validation/
#     bin/mp-arch-validate            validator (universal unless --archs says otherwise)
#     bin/MediaPipeTasksC.framework   copied VERBATIM from the xcframework's macOS slice
#     models/hand_landmarker.task     + models/coreml/<sha>.mlmodelc for .coreML (if available)
#     media/                          fixed stills (PNG extracted once, here) + test video
#     run_validation.sh               the matrix runner
#     MANIFEST.txt                    git revision, slice UUIDs, sha256 of every input
#
#   ./package_bundle.sh [--archs "arm64 x86_64"] [--out /tmp/mp_build/mp-validation.tar.gz]
#
# Uses whatever mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework
# currently holds — build it first with build_macos_xcframework.sh.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "${HERE}" rev-parse --show-toplevel)"
SHARED="${ROOT}/mediapipe/tasks/macos/test_projects/shared"
XCF="${ROOT}/mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework"
ARCHS="arm64 x86_64"
OUT="/tmp/mp_build/mp-validation.tar.gz"
STILLS_AT_MS="1500,8000,12000"   # 8000 ms has both hands; the others one each
while (( $# )); do
  case "$1" in
    --archs) ARCHS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

for f in "${SHARED}/models/hand_landmarker.task" "${SHARED}/test_image.jpg" "${SHARED}/two-hands-only.mov"; do
  [[ -f "${f}" ]] || { echo "error: missing ${f} (run test_projects/shared/download_models.sh)" >&2; exit 1; }
done
SLICE="$(ls -d "${XCF}"/macos-* 2>/dev/null | head -1)"
[[ -n "${SLICE}" ]] || { echo "error: no macOS slice in ${XCF}" >&2; exit 1; }

echo "==> Building mp-arch-validate (${ARCHS})..."
arch_flags=()
for a in ${ARCHS}; do arch_flags+=(--arch "${a}"); done
swift build -c release --package-path "${HERE}" "${arch_flags[@]}" 2>&1 | grep -vE 'deprecated|^\s*\||^\s*[0-9]+ \|' || true
BIN_DIR="$(swift build -c release --package-path "${HERE}" "${arch_flags[@]}" --show-bin-path)"
[[ -x "${BIN_DIR}/mp-arch-validate" ]] || { echo "error: build failed" >&2; exit 1; }

STAGE="$(mktemp -d)/mp-validation"
mkdir -p "${STAGE}/bin" "${STAGE}/models" "${STAGE}/media"
cp "${BIN_DIR}/mp-arch-validate" "${STAGE}/bin/"
cp -R "${SLICE}/MediaPipeTasksC.framework" "${STAGE}/bin/"
# Single-arch SwiftPM builds use an @loader_path rpath, multi-arch (xcbuild)
# builds @loader_path/../lib; pin the framework lookup to bin/ explicitly.
if ! otool -l "${STAGE}/bin/mp-arch-validate" | grep -q 'path @executable_path '; then
  install_name_tool -add_rpath @executable_path "${STAGE}/bin/mp-arch-validate"
  codesign --force --sign - "${STAGE}/bin/mp-arch-validate" >/dev/null 2>&1
fi
cp "${SHARED}/models/hand_landmarker.task" "${STAGE}/models/"
cp "${SHARED}/test_image.jpg" "${SHARED}/two-hands-only.mov" "${STAGE}/media/"
cp "${HERE}/remote/run_validation.sh" "${STAGE}/"

# Core ML models for the hand task only: the .coreML delegate looks models up
# by the SHA-256 of each TFLite flatbuffer inside the .task bundle.
if [[ -d "${SHARED}/models/coreml_models" ]]; then
  TMPX="$(mktemp -d)"
  unzip -q "${SHARED}/models/hand_landmarker.task" -d "${TMPX}"
  while IFS= read -r tfl; do
    sha="$(shasum -a 256 "${tfl}" | awk '{print $1}')"
    if [[ -d "${SHARED}/models/coreml_models/${sha}.mlmodelc" ]]; then
      mkdir -p "${STAGE}/models/coreml"
      cp -R "${SHARED}/models/coreml_models/${sha}.mlmodelc" "${STAGE}/models/coreml/"
    fi
  done < <(find "${TMPX}" -name '*.tflite')
  rm -rf "${TMPX}"
fi

# Stills are extracted HERE, once, so every target gets bit-identical pixels
# (each Mac's H.264 decoder may differ in the last bit).
echo "==> Extracting fixed stills..."
"${STAGE}/bin/mp-arch-validate" frames --video "${SHARED}/two-hands-only.mov" \
  --at-ms "${STILLS_AT_MS}" --out-dir "${STAGE}/media" > /dev/null

{
  echo "mp-validation bundle"
  echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git: $(git -C "${ROOT}" rev-parse HEAD)$(git -C "${ROOT}" diff --quiet HEAD -- . ':!*.patch' || echo ' (+uncommitted changes)')"
  echo "slice: $(basename "${SLICE}")"
  echo "validator archs: $(lipo -archs "${STAGE}/bin/mp-arch-validate")"
  dwarfdump --uuid "${STAGE}/bin/MediaPipeTasksC.framework/Versions/A/MediaPipeTasksC" | awk '{print "framework uuid:", $2, $3}'
  echo
  ( cd "${STAGE}" && find . -type f ! -name MANIFEST.txt | sort | xargs shasum -a 256 )
} > "${STAGE}/MANIFEST.txt"

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"
COPYFILE_DISABLE=1 tar -czf "${OUT}" -C "$(dirname "${STAGE}")" mp-validation
rm -rf "$(dirname "${STAGE}")"
echo "==> ${OUT} ($(du -h "${OUT}" | awk '{print $1}'))"
