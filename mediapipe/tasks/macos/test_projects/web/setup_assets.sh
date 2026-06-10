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
# Copies the shared models, test image, and the tasks-vision WASM runtime into
# public/ so the Vite dev server can serve them locally (no CDN needed).
# Run `npm install` and `../shared/download_models.sh` first.

set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "${WEB_DIR}/../shared" && pwd)"
ASSETS_DIR="${WEB_DIR}/public/assets"
WASM_DIR="${WEB_DIR}/public/wasm"
WASM_SRC="${WEB_DIR}/node_modules/@mediapipe/tasks-vision/wasm"

if [[ ! -d "${WASM_SRC}" ]]; then
  echo "error: ${WASM_SRC} not found. Run 'npm install' first." >&2
  exit 1
fi
if [[ ! -f "${SHARED_DIR}/test_image.jpg" ]]; then
  echo "error: ${SHARED_DIR}/test_image.jpg not found. Run ../shared/download_models.sh first." >&2
  exit 1
fi

mkdir -p "${ASSETS_DIR}" "${WASM_DIR}"
cp -f "${SHARED_DIR}/models/"*.task "${ASSETS_DIR}/"
cp -f "${SHARED_DIR}/test_image.jpg" "${ASSETS_DIR}/test_image.jpg"
cp -f "${WASM_SRC}/"* "${WASM_DIR}/"

echo "==> Assets ready in public/assets and public/wasm"
