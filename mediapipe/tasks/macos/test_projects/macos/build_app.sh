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
# Builds the ParitySmokeTest SwiftPM executable and bundles it into a real
# macOS .app with the MediaPipeTasksC framework embedded, so it runs via
# `open ParitySmokeTest.app` (not just `swift run`). Also generates a config.json
# pointing at ../shared.

set -euo pipefail

APP_NAME="ParitySmokeTest"
MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "${MACOS_DIR}/../shared" && pwd)"
REPO_ROOT="$(git -C "${MACOS_DIR}" rev-parse --show-toplevel)"
FRAMEWORK_SRC="${REPO_ROOT}/mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework/macos-arm64/MediaPipeTasksC.framework"

if [[ ! -d "${FRAMEWORK_SRC}" ]]; then
  echo "error: ${FRAMEWORK_SRC} not found. Build it first with" >&2
  echo "  ${REPO_ROOT}/mediapipe/tasks/macos/build_macos_xcframework.sh" >&2
  exit 1
fi

cd "${MACOS_DIR}"

echo "==> swift build -c release"
swift build -c release
BIN_DIR="$(swift build -c release --show-bin-path)"
EXE="${BIN_DIR}/${APP_NAME}"
[[ -x "${EXE}" ]] || { echo "error: built executable not found at ${EXE}" >&2; exit 1; }

echo "==> Assembling ${APP_NAME}.app"
APP="${MACOS_DIR}/${APP_NAME}.app"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Frameworks"

cp -f "${EXE}" "${APP}/Contents/MacOS/${APP_NAME}"
cp -Rf "${FRAMEWORK_SRC}" "${APP}/Contents/Frameworks/"

cat > "${APP}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key><string>com.google.mediapipe.${APP_NAME}</string>
  <key>CFBundleName</key><string>${APP_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.0.1</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSBackgroundOnly</key><true/>
</dict>
</plist>
EOF

# Let the embedded framework resolve via @rpath at runtime.
install_name_tool -add_rpath "@executable_path/../Frameworks" \
  "${APP}/Contents/MacOS/${APP_NAME}" 2>/dev/null || true

# Ad-hoc sign so macOS will launch the bundle locally.
codesign --force --deep --sign - "${APP}" >/dev/null 2>&1 || \
  echo "    (codesign skipped/failed; the app may still run locally)"

# Generate a turnkey config.json pointing at ../shared (gitignored).
cat > "${MACOS_DIR}/config.json" <<EOF
{
  "handModel": "${SHARED_DIR}/models/hand_landmarker.task",
  "poseModel": "${SHARED_DIR}/models/pose_landmarker.task",
  "faceModel": "${SHARED_DIR}/models/face_landmarker.task",
  "image": "${SHARED_DIR}/test_image.jpg",
  "output": "${SHARED_DIR}/output/macos.json"
}
EOF

echo ""
echo "==> Built ${APP}"
echo "    Run it (writes ${SHARED_DIR}/output/macos.json):"
echo "      open \"${APP}\" --args \"${MACOS_DIR}/config.json\""
echo "    or directly:"
echo "      \"${APP}/Contents/MacOS/${APP_NAME}\" \"${MACOS_DIR}/config.json\""
