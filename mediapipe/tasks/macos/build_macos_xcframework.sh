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
# Builds the MediaPipe Tasks C library for native macOS (arm64) and packages it
# as MediaPipeTasksC.xcframework (a framework-based xcframework) for the
# MediaPipeTasksMac Swift package.
#
# Output (gitignored):
#   mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework
#
# Prerequisites (see README.md): Bazel 7.4.1, Xcode, a JDK (JAVA_HOME), and a
# Python 3.9-3.12 interpreter as `python3` on PATH (MediaPipe has no 3.13/3.14
# requirements lock).

set -euo pipefail

# --- Configuration ----------------------------------------------------------
FRAMEWORK_NAME="MediaPipeTasksC"
BAZEL="${BAZEL:-$(command -v bazel)}"
MACOS_MIN_VERSION="${MACOS_MIN_VERSION:-13.0}"
BUILD_VERSION="${MPP_BUILD_VERSION:-0.0.1-dev}"

MPP_ROOT_DIR="$(git rev-parse --show-toplevel)"
ARTIFACTS_DIR="${MPP_ROOT_DIR}/mediapipe/tasks/macos/swift/Artifacts"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

BAZEL_TARGET="//mediapipe/tasks/c:mediapipe_macos"
# `mediapipe_macos` genrule emits libmediapipe.dylib next to the target package.
DYLIB_BAZEL_PATH="bazel-bin/mediapipe/tasks/c/libmediapipe.dylib"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "error: this script only runs on macOS." >&2
  exit 1
fi
if [[ -z "${BAZEL}" || ! -x "${BAZEL}" ]]; then
  echo "error: bazel not found on PATH (set BAZEL=/path/to/bazel)." >&2
  exit 1
fi

cd "${MPP_ROOT_DIR}"

# --- 1. Build the dylib -----------------------------------------------------
# By default this builds a GPU-capable (Metal) artifact that supports BOTH the
# .cpu and .gpu delegates. The GPU build drops MEDIAPIPE_DISABLE_GPU and force-
# defines MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER (upstream gates it to
# !TARGET_OS_OSX, but the CVPixelBuffer-backed Metal path is needed on macOS).
# The macro MUST be defined for every C++/ObjC TU (it changes MPPMetalHelper's
# interface), hence both --copt and --objccopt.
#
# Set MP_ENABLE_GPU=0 to build a smaller CPU-only artifact instead (then set
# `mediaPipeGPUArtifactAvailable = false` in Modes.swift).
if [[ "${MP_ENABLE_GPU:-1}" == "1" ]]; then
  echo "==> Building ${BAZEL_TARGET} for macOS arm64 (GPU/Metal, MacPorts OpenCV 3)..."
  "${BAZEL}" build -c opt \
    --config=macos --config=darwin_arm64 \
    --copt=-DMEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER=1 \
    --objccopt=-DMEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER=1 \
    "${BAZEL_TARGET}"
else
  echo "==> Building ${BAZEL_TARGET} for macOS arm64 (CPU, MacPorts OpenCV 3)..."
  "${BAZEL}" build -c opt \
    --config=macos --config=darwin_arm64 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    "${BAZEL_TARGET}"
fi

if [[ ! -f "${DYLIB_BAZEL_PATH}" ]]; then
  echo "error: expected dylib not found at ${DYLIB_BAZEL_PATH}" >&2
  exit 1
fi

DYLIB="${WORK_DIR}/libmediapipe.dylib"
cp -f "${DYLIB_BAZEL_PATH}" "${DYLIB}"
chmod u+w "${DYLIB}"

# --- 2. Verify exported C symbols -------------------------------------------
# Dump the symbol table to a file once, then grep the file. (Piping `nm` into
# `grep -q` would let grep close the pipe early, killing nm with SIGPIPE; under
# `set -o pipefail` that makes a *found* symbol look like a failure.)
echo "==> Verifying exported MediaPipe C symbols..."
SYMBOLS="${WORK_DIR}/symbols.txt"
nm -gU "${DYLIB}" > "${SYMBOLS}" 2>/dev/null || true
missing=0
for sym in MpHandLandmarkerCreate MpPoseLandmarkerCreate MpFaceLandmarkerCreate MpImageCreateFromUint8Data; do
  if grep -q "${sym}" "${SYMBOLS}"; then
    echo "    found: ${sym}"
  else
    echo "    MISSING: ${sym}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "error: required C symbols are not exported by the dylib. Aborting." >&2
  exit 1
fi

# --- 3. Verify the binary really targets macOS (not iOS) --------------------
echo "==> Verifying Mach-O build platform..."
LOADCMDS="${WORK_DIR}/loadcmds.txt"
otool -l "${DYLIB}" > "${LOADCMDS}" 2>/dev/null || true
if grep -A3 LC_BUILD_VERSION "${LOADCMDS}" | grep -qE "platform 1$|platform MACOS"; then
  echo "    platform MACOS confirmed"
else
  echo "    note: parsing platform from otool (expecting MACOS):"
  grep -A3 LC_BUILD_VERSION "${LOADCMDS}" | head -8 || true
fi

# --- 4. Assemble MediaPipeTasksC.framework (macOS versioned bundle) ---------
echo "==> Assembling ${FRAMEWORK_NAME}.framework..."
FW="${WORK_DIR}/${FRAMEWORK_NAME}.framework"
VERSION_DIR="${FW}/Versions/A"
mkdir -p "${VERSION_DIR}/Headers" "${VERSION_DIR}/Resources"

cp -f "${DYLIB}" "${VERSION_DIR}/${FRAMEWORK_NAME}"
# The framework's load command must resolve via @rpath so the consuming app can
# embed and sign it predictably.
install_name_tool -id "@rpath/${FRAMEWORK_NAME}.framework/Versions/A/${FRAMEWORK_NAME}" \
  "${VERSION_DIR}/${FRAMEWORK_NAME}"

# Placeholder public header. The MediaPipeTasksObjC shim includes the real
# MediaPipe C headers straight from the repo source tree; this framework is the
# binary runtime only and should NOT be imported directly.
cat > "${VERSION_DIR}/Headers/${FRAMEWORK_NAME}.h" <<EOF
// ${FRAMEWORK_NAME} — binary runtime for MediaPipe Tasks C on macOS.
//
// Do NOT import this framework directly. It carries only the compiled MediaPipe
// C symbols (MpHandLandmarker*, MpImage*, ...). The Swift-facing bridge is the
// MediaPipeTasksObjC target, which includes the MediaPipe C headers from the
// repo source tree and links this binary. App code should import
// MediaPipeTasksMac.
EOF

cat > "${VERSION_DIR}/Resources/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>${FRAMEWORK_NAME}</string>
  <key>CFBundleIdentifier</key><string>com.google.mediapipe.${FRAMEWORK_NAME}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>${FRAMEWORK_NAME}</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>${BUILD_VERSION}</string>
  <key>CFBundleVersion</key><string>${BUILD_VERSION}</string>
  <key>CFBundleSupportedPlatforms</key><array><string>MacOSX</string></array>
  <key>LSMinimumSystemVersion</key><string>${MACOS_MIN_VERSION}</string>
</dict>
</plist>
EOF

# Standard macOS framework symlinks.
ln -sfn "A" "${FW}/Versions/Current"
ln -sfn "Versions/Current/${FRAMEWORK_NAME}" "${FW}/${FRAMEWORK_NAME}"
ln -sfn "Versions/Current/Headers" "${FW}/Headers"
ln -sfn "Versions/Current/Resources" "${FW}/Resources"

# --- 4b. (Opt-in) Bundle non-system dylibs for a portable release -----------
# MP_BUNDLE_DEPS=1 recursively copies every non-system dynamic dependency (the
# MacPorts OpenCV tree) into Versions/A/Libraries and rewrites install names so
# the framework no longer references /opt/local at runtime. Bundled libs and the
# framework binary reference each other via @rpath, anchored by @loader_path
# rpaths (main binary: @loader_path/Libraries; each bundled lib: @loader_path).
if [[ "${MP_BUNDLE_DEPS:-0}" == "1" ]]; then
  echo "==> Bundling non-system dependencies (MP_BUNDLE_DEPS=1)..."
  if ! command -v dylibbundler >/dev/null 2>&1; then
    echo "error: dylibbundler is required for MP_BUNDLE_DEPS=1 but was not found." >&2
    echo "       Install it with:  brew install dylibbundler" >&2
    echo "                    or:  sudo port install dylibbundler" >&2
    exit 1
  fi

  LIB_SUBDIR="Libraries"
  # dylibbundler resolves -x/-d relative to cwd; run inside the version dir.
  (
    cd "${VERSION_DIR}"
    dylibbundler \
      --fix-file "${FRAMEWORK_NAME}" \
      --bundle-deps \
      --dest-dir "${LIB_SUBDIR}" \
      --install-path "@rpath/" \
      --search-path /opt/local/lib \
      --search-path /opt/local/lib/opencv3 \
      --overwrite-files --create-dir --no-codesign
  )

  # dylibbundler rewrites the binary's pre-existing (Bazel) rpaths to a bogus
  # "@rpath/" entry — sometimes more than once, which yields a duplicate
  # LC_RPATH that the linker rejects. Strip every "@rpath/" rpath, then add the
  # ones we actually want.
  strip_rpath_all() {
    local bin="$1" rp="$2" n
    while :; do
      n="$(otool -l "${bin}" 2>/dev/null \
        | awk -v r="${rp}" '/LC_RPATH/{f=1;next} f&&/ path /{if($2==r)c++;f=0} END{print c+0}')"
      [[ "${n:-0}" -gt 0 ]] || break
      install_name_tool -delete_rpath "${rp}" "${bin}" 2>/dev/null || break
    done
  }

  # Anchor @rpath: the main binary finds bundled libs in ./Libraries; each
  # bundled lib finds its siblings in its own directory.
  strip_rpath_all "${VERSION_DIR}/${FRAMEWORK_NAME}" "@rpath/"
  install_name_tool -add_rpath "@loader_path/${LIB_SUBDIR}" \
    "${VERSION_DIR}/${FRAMEWORK_NAME}" 2>/dev/null || true
  if [[ -d "${VERSION_DIR}/${LIB_SUBDIR}" ]]; then
    for lib in "${VERSION_DIR}/${LIB_SUBDIR}"/*.dylib; do
      strip_rpath_all "${lib}" "@rpath/"
      install_name_tool -add_rpath "@loader_path" "${lib}" 2>/dev/null || true
    done
  fi

  # Re-sign (install_name_tool invalidates signatures): libs first, then binary.
  if [[ -d "${VERSION_DIR}/${LIB_SUBDIR}" ]]; then
    for lib in "${VERSION_DIR}/${LIB_SUBDIR}"/*.dylib; do
      codesign --force --sign - "${lib}" >/dev/null 2>&1 || true
    done
  fi
  codesign --force --sign - "${VERSION_DIR}/${FRAMEWORK_NAME}" >/dev/null 2>&1 || true

  # Verify portability: no dependency may point outside the allowed prefixes
  # (/System, /usr/lib, @rpath, @loader_path, @executable_path).
  echo "==> Verifying no non-portable (/opt/local etc.) dependencies remain..."
  verify_portable() {
    local f="$1" bad
    bad="$(otool -L "${f}" | tail -n +2 | sed 's/^[[:space:]]*//' | awk '{print $1}' \
      | grep -vE '^/System/|^/usr/lib/|^@rpath|^@loader_path|^@executable_path' || true)"
    if [[ -n "${bad}" ]]; then
      echo "error: non-portable dependency in ${f}:" >&2
      echo "${bad}" | sed 's/^/    /' >&2
      return 1
    fi
  }
  verify_portable "${VERSION_DIR}/${FRAMEWORK_NAME}" || exit 1
  bundled_count=0
  if [[ -d "${VERSION_DIR}/${LIB_SUBDIR}" ]]; then
    for lib in "${VERSION_DIR}/${LIB_SUBDIR}"/*.dylib; do
      verify_portable "${lib}" || exit 1
      bundled_count=$((bundled_count + 1))
    done
  fi
  bundled_size="$(du -sh "${VERSION_DIR}/${LIB_SUBDIR}" 2>/dev/null | awk '{print $1}')"
  echo "    OK: framework + ${bundled_count} bundled dylibs (${bundled_size:-?}) are portable."
fi

# --- 5. Create the xcframework ----------------------------------------------
echo "==> Creating ${FRAMEWORK_NAME}.xcframework..."
OUT_XCFRAMEWORK="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.xcframework"
mkdir -p "${ARTIFACTS_DIR}"
rm -rf "${OUT_XCFRAMEWORK}"
xcodebuild -create-xcframework \
  -framework "${FW}" \
  -output "${OUT_XCFRAMEWORK}"

# --- 6. Verify the xcframework ----------------------------------------------
echo "==> Verifying xcframework..."
plutil -p "${OUT_XCFRAMEWORK}/Info.plist"
echo "--- Mach-O platform of the packaged slice ---"
SLICE_BIN="$(find "${OUT_XCFRAMEWORK}" -name "${FRAMEWORK_NAME}" -type f | head -1)"
otool -l "${SLICE_BIN}" | grep -A3 LC_BUILD_VERSION | head -8 || true

echo ""
echo "==> Done: ${OUT_XCFRAMEWORK}"
echo "    You can now run: swift build   (and: swift run mediapipe-macos-sample ...)"
