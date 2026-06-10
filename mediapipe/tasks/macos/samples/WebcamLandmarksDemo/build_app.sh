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
# Builds the WebcamLandmarksDemo SwiftUI executable and bundles it into a real
# macOS .app with: a camera usage description (required to access the webcam),
# the MediaPipeTasksC.framework embedded (+ its bundled Libraries/ if present),
# and the .task models copied into Resources. Run download_models.sh first.

set -euo pipefail

APP_NAME="WebcamLandmarksDemo"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${DEMO_DIR}" rev-parse --show-toplevel)"
FRAMEWORK_SRC="${REPO_ROOT}/mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework/macos-arm64/MediaPipeTasksC.framework"
MODELS_DIR="${DEMO_DIR}/models"

if [[ ! -d "${FRAMEWORK_SRC}" ]]; then
  echo "error: ${FRAMEWORK_SRC} not found." >&2
  echo "       Build it first (GPU + portable):" >&2
  echo "       MP_BUNDLE_DEPS=1 ${REPO_ROOT}/mediapipe/tasks/macos/build_macos_xcframework.sh" >&2
  exit 1
fi

cd "${DEMO_DIR}"
echo "==> swift build -c release"
swift build -c release
BIN_DIR="$(swift build -c release --show-bin-path)"
EXE="${BIN_DIR}/${APP_NAME}"
[[ -x "${EXE}" ]] || { echo "error: built executable not found at ${EXE}" >&2; exit 1; }

echo "==> Assembling ${APP_NAME}.app"
APP="${DEMO_DIR}/${APP_NAME}.app"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Frameworks" "${APP}/Contents/Resources"

cp -f "${EXE}" "${APP}/Contents/MacOS/${APP_NAME}"
cp -Rf "${FRAMEWORK_SRC}" "${APP}/Contents/Frameworks/"

# Bundle the models into Resources (so Bundle.main finds them). Gitignored.
if compgen -G "${MODELS_DIR}/*.task" >/dev/null; then
  cp -f "${MODELS_DIR}"/*.task "${APP}/Contents/Resources/"
else
  echo "    warning: no models in ${MODELS_DIR}. Run ./download_models.sh." >&2
fi

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
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSCameraUsageDescription</key><string>This demo uses the camera to detect hand, pose, and face landmarks on the live video.</string>
</dict>
</plist>
EOF

# Resolve the embedded framework (and its bundled OpenCV dylibs) at runtime.
install_name_tool -add_rpath "@executable_path/../Frameworks" \
  "${APP}/Contents/MacOS/${APP_NAME}" 2>/dev/null || true

# Ad-hoc sign so macOS launches the bundle and the camera TCC prompt works.
codesign --force --deep --sign - "${APP}" >/dev/null 2>&1 || \
  echo "    (codesign skipped/failed; the app may still run locally)"

echo ""
echo "==> Built ${APP}"
echo "    Launch it:  open \"${APP}\""
echo "    (Grant camera access at the first-launch prompt.)"
