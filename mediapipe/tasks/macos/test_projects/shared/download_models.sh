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
