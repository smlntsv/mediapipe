#!/usr/bin/env bash
# Copyright 2025 The MediaPipe Authors.
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
# Downloads the three .task models used by the parity test projects into
# shared/models/, and copies the shared test image to shared/test_image.jpg.
# None of these outputs are committed (see ../.gitignore).

set -euo pipefail

SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SHARED_DIR}/models"
REPO_ROOT="$(git -C "${SHARED_DIR}" rev-parse --show-toplevel)"

# Shared test image: one visible hand gesture, one visible face, upper body for
# pose. NOTE: pose is upper-body/cropped here; supplement with a full-body image
# for thorough pose parity later.
TEST_IMAGE_SRC="${REPO_ROOT}/mediapipe/model_maker/python/vision/gesture_recognizer/testdata/raw_data/call/17d804b5-7118-462d-8191-58d764f591b8.jpg"

mkdir -p "${MODELS_DIR}"

download() {
  local name="$1" url="$2"
  if [[ -f "${MODELS_DIR}/${name}" ]]; then
    echo "    have ${name}"
  else
    echo "    downloading ${name}..."
    curl -sSL -o "${MODELS_DIR}/${name}" "${url}"
  fi
}

echo "==> Downloading models into ${MODELS_DIR}"
download hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
download pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
download face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# --- Core ML (ANE) models for the "coreml" delegate ------------------------
# Converted counterparts of the .task models above: one <sha256>.mlmodelc per
# embedded tflite, keyed by the tflite's content hash, so they match only
# THESE exact .task versions (see experiments/coreml_ane/convert_delegate.py).
# The test projects copy them flat next to the .task files (the Core ML
# delegate's default lookup dir). Best-effort: a missing/mismatched archive
# just means the "coreml" delegate falls back to GPU/CPU, so this warns
# instead of failing. Override COREML_MODELS_URL for your own converted set.
COREML_DIR="${MODELS_DIR}/coreml_models"
COREML_MODELS_URL="${COREML_MODELS_URL:-https://s3.staxel.ai/staxel-bar/staxel-bar/coreml_models.zip}"
COREML_MODELS_SHA256="${COREML_MODELS_SHA256:-78dbd5f151d6308a08a7a074f72b042ccdf952ab80085c92653567639bf02269}"

if compgen -G "${COREML_DIR}/*.mlmodelc" > /dev/null 2>&1; then
  echo "    have coreml models"
else
  echo "==> Downloading Core ML models (best-effort) from ${COREML_MODELS_URL}"
  tmp_zip="${MODELS_DIR}/coreml_models.zip.download"
  rm -f "${tmp_zip}"
  if curl -sSL --fail --retry 3 --retry-all-errors -o "${tmp_zip}" "${COREML_MODELS_URL}"; then
    got="$(shasum -a 256 "${tmp_zip}" | cut -d' ' -f1)"
    if [[ -n "${COREML_MODELS_SHA256}" && "${got}" != "${COREML_MODELS_SHA256}" ]]; then
      echo "    ! coreml zip sha256 mismatch (got ${got}); skipping — delegate will fall back." >&2
      rm -f "${tmp_zip}"
    else
      # The archive may hold the .mlmodelc bundles at its root or nested one
      # level down (e.g. coreml_models/); find wherever they actually live and
      # copy from there, so both layouts land flat in COREML_DIR.
      rm -rf "${COREML_DIR}.unzip"
      unzip -oq "${tmp_zip}" -d "${COREML_DIR}.unzip"
      mkdir -p "${COREML_DIR}"
      first_model="$(find "${COREML_DIR}.unzip" -maxdepth 2 -type d -name '*.mlmodelc' | head -1)"
      if [[ -n "${first_model}" ]]; then
        cp -Rf "$(dirname "${first_model}")/." "${COREML_DIR}/"
      else
        echo "    ! no .mlmodelc found in archive; delegate will fall back." >&2
      fi
      rm -rf "${COREML_DIR}.unzip" "${tmp_zip}"
      echo "    coreml models: $(find "${COREML_DIR}" -maxdepth 1 -type d -name '*.mlmodelc' | wc -l | tr -d ' ') .mlmodelc"
    fi
  else
    echo "    ! could not download coreml models; delegate will fall back to GPU/CPU." >&2
    rm -f "${tmp_zip}"
  fi
fi

echo "==> Copying shared test image to ${SHARED_DIR}/test_image.jpg"
cp -f "${TEST_IMAGE_SRC}" "${SHARED_DIR}/test_image.jpg"

# Second, full-body image for thorough pose parity (the primary image above is
# upper-body/cropped). Downloaded from the public MediaPipe assets bucket.
if [[ -f "${SHARED_DIR}/test_image_full_body.jpg" ]]; then
  echo "    have test_image_full_body.jpg"
else
  echo "==> Downloading full-body pose image to ${SHARED_DIR}/test_image_full_body.jpg"
  curl -sSL -o "${SHARED_DIR}/test_image_full_body.jpg" \
    "https://storage.googleapis.com/mediapipe-assets/pose.jpg"
fi

echo "==> Done. Models in shared/models/, images at shared/test_image*.jpg"
