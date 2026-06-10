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
# Downloads the .task models used by the webcam demo into ./models (gitignored).

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "${DIR}"

download() {
  local name="$1" url="$2"
  if [[ -f "${DIR}/${name}" ]]; then
    echo "    have ${name}"
  else
    echo "    downloading ${name}..."
    curl -sSL -o "${DIR}/${name}" "${url}"
  fi
}

echo "==> Downloading models into ${DIR}"
download hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
download pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
download face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

echo "==> Done. Now run ./build_app.sh"
