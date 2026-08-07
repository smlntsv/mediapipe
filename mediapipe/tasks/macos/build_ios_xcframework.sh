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
# Builds the MediaPipe Tasks C library as an iOS DYNAMIC framework (device +
# simulator, arm64) and folds those slices into the same
# MediaPipeTasksC.xcframework the MediaPipeTasksMac Swift package consumes, so
# one Swift package serves macOS and iOS from a single binary target.
#
# The C API ships as a dynamic library exactly like the macOS build: every
# calculator/graph registration is present at load, so no -force_load or graph
# static-library dance is needed (that is the CocoaPods/ObjC-framework path).
# The fork's .coreML delegate flows through the shared C plumbing.
#
# The macOS slice is reused from an already-built MediaPipeTasksC.xcframework
# (run build_macos_xcframework.sh first); pass MP_IOS_ONLY=1 to emit an
# iOS-only xcframework instead.
#
# Prerequisites (see README.md): Bazel 7.4.1, Xcode, a JDK (JAVA_HOME), a
# Python 3.9-3.12 as `python3` AND `python` on PATH (the OpenCV iOS source
# build shells out to `python`), and `setuptools` installed for that Python
# (the OpenCV build imports distutils).

set -euo pipefail

FRAMEWORK_NAME="MediaPipeTasksC"
BAZEL="${BAZEL:-$(command -v bazel)}"
IOS_MIN_VERSION="${IOS_MIN_VERSION:-17.0}"
# CFBundleShortVersionString/CFBundleVersion must be 1-3 period-separated
# non-negative integers, or App Store Connect rejects the app embedding this
# framework (ITMS-90060/90058). v0.10.35-apple.1/.2 shipped with the old
# "0.0.1-dev" fallback and hit exactly that. Validate anything provided (the
# git tag, e.g. "0.10.35-apple.2", is NOT valid — pass plain "0.10.35"), and
# stamp unset dev builds with a valid placeholder.
BUILD_VERSION="${MPP_BUILD_VERSION:-}"
if [[ -z "${BUILD_VERSION}" ]]; then
  BUILD_VERSION="0.0.1"
  echo "warning: MPP_BUILD_VERSION is unset; stamping dev version ${BUILD_VERSION}." >&2
  echo "         Release builds must set MPP_BUILD_VERSION (plain integers, e.g. 0.10.35)." >&2
elif [[ ! "${BUILD_VERSION}" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
  echo "error: MPP_BUILD_VERSION='${BUILD_VERSION}' is not a valid CFBundle version:" >&2
  echo "       Apple requires 1-3 period-separated non-negative integers (e.g. 0.10.35)." >&2
  echo "       Do not pass the git tag; the -apple.N suffix lives only in the tag/release name." >&2
  exit 1
fi

MPP_ROOT_DIR="$(git rev-parse --show-toplevel)"
ARTIFACTS_DIR="${MPP_ROOT_DIR}/mediapipe/tasks/macos/swift/Artifacts"
XCFRAMEWORK="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.xcframework"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# The C dynamic library target (shared by every platform).
BAZEL_TARGET="//mediapipe/tasks/c:mediapipe_source"
DYLIB_BAZEL_PATH="bazel-bin/mediapipe/tasks/c/libmediapipe_source.dylib"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "error: this script only runs on macOS." >&2
  exit 1
fi
if [[ -z "${BAZEL}" || ! -x "${BAZEL}" ]]; then
  echo "error: bazel not found on PATH (set BAZEL=/path/to/bazel)." >&2
  exit 1
fi

cd "${MPP_ROOT_DIR}"

# Build one iOS slice and wrap the dylib as a flat iOS dynamic framework.
#   $1 bazel --cpu   (ios_arm64 | ios_sim_arm64)
#   $2 platform      (iPhoneOS | iPhoneSimulator)
#   $3 out framework dir
build_slice() {
  local cpu="$1" platform="$2" out_fw="$3"
  echo "==> Building ${BAZEL_TARGET} (${cpu}, min-os ${IOS_MIN_VERSION})..."
  # NOTE: --config=ios_arm64 is broken with the pinned apple_support (its
  # --platforms points at a package that module lacks); --config=ios --cpu=…
  # is the working form. Genrule=local because the OpenCV source build's
  # try_compile probes fail inside the Bazel sandbox on modern Xcode.
  "${BAZEL}" build -c opt --strategy=Genrule=local \
    --config=ios --cpu="${cpu}" --ios_minimum_os="${IOS_MIN_VERSION}" \
    --apple_generate_dsym=false --define OPENCV=source \
    "${BAZEL_TARGET}"

  rm -rf "${out_fw}"
  mkdir -p "${out_fw}/Headers"
  cp -f "${DYLIB_BAZEL_PATH}" "${out_fw}/${FRAMEWORK_NAME}"
  chmod u+w "${out_fw}/${FRAMEWORK_NAME}"
  install_name_tool -id "@rpath/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}" \
    "${out_fw}/${FRAMEWORK_NAME}"

  cat > "${out_fw}/Headers/${FRAMEWORK_NAME}.h" <<EOF
// ${FRAMEWORK_NAME} — binary runtime for MediaPipe Tasks C on iOS.
// Do NOT import directly; app code imports MediaPipeTasksMac (the Swift package).
EOF

  cat > "${out_fw}/Info.plist" <<EOF
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
  <key>CFBundleSupportedPlatforms</key><array><string>${platform}</string></array>
  <key>MinimumOSVersion</key><string>${IOS_MIN_VERSION}</string>
</dict>
</plist>
EOF

  codesign --force --sign - "${out_fw}/${FRAMEWORK_NAME}" >/dev/null 2>&1 || true
  echo "    assembled ${out_fw} (${platform})"
}

build_slice ios_arm64     iPhoneOS        "${WORK_DIR}/device/${FRAMEWORK_NAME}.framework"
build_slice ios_sim_arm64 iPhoneSimulator "${WORK_DIR}/sim/${FRAMEWORK_NAME}.framework"

# Companion dSYMs (Tier A): a UUID-matched dSYM per slice, embedded via
# -debug-symbols so it rides in <slice>/dSYMs/ inside the xcframework. Xcode
# auto-copies them into consumer archives; App Store Connect's "Upload Symbols
# Failed ... expected UUIDs" check then passes (StaxelLauncher hit exactly
# that on v1.0.0-apple.2, which shipped without dSYMs). The -c opt binaries
# carry no DWARF, so these are symbol-table-only dSYMs (function-level
# symbolication; file/line would need a -g build).
make_dsym() {
  local fw_bin="$1" out_dsym="$2"
  dsymutil "${fw_bin}" -o "${out_dsym}" 2> >(grep -v "no debug symbols" >&2 || true)
  local bin_uuids dsym_uuids
  bin_uuids="$(dwarfdump --uuid "${fw_bin}" | awk '{print $2}' | sort)"
  dsym_uuids="$(dwarfdump --uuid "${out_dsym}" | awk '{print $2}' | sort)"
  if [[ "${bin_uuids}" != "${dsym_uuids}" ]]; then
    echo "error: dSYM UUIDs (${dsym_uuids}) do not match ${fw_bin} (${bin_uuids})" >&2
    exit 1
  fi
}

echo "==> Generating UUID-matched dSYMs for the iOS slices..."
make_dsym "${WORK_DIR}/device/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}" \
          "${WORK_DIR}/device/${FRAMEWORK_NAME}.framework.dSYM"
make_dsym "${WORK_DIR}/sim/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}" \
          "${WORK_DIR}/sim/${FRAMEWORK_NAME}.framework.dSYM"

# Assemble the universal xcframework. Reuse the existing macOS slice unless
# MP_IOS_ONLY=1.
CREATE_ARGS=(
  -framework "${WORK_DIR}/device/${FRAMEWORK_NAME}.framework"
  -debug-symbols "${WORK_DIR}/device/${FRAMEWORK_NAME}.framework.dSYM"
  -framework "${WORK_DIR}/sim/${FRAMEWORK_NAME}.framework"
  -debug-symbols "${WORK_DIR}/sim/${FRAMEWORK_NAME}.framework.dSYM"
)
if [[ "${MP_IOS_ONLY:-0}" != "1" ]]; then
  MACOS_FW="${XCFRAMEWORK}/macos-arm64/${FRAMEWORK_NAME}.framework"
  if [[ ! -d "${MACOS_FW}" ]]; then
    echo "error: macOS slice not found at ${MACOS_FW}." >&2
    echo "       Run build_macos_xcframework.sh first, or set MP_IOS_ONLY=1." >&2
    exit 1
  fi
  # Reuse the macOS slice's embedded dSYM when build_macos_xcframework.sh
  # produced one (it does since the Tier-A change); otherwise generate it here
  # from the slice binary — the UUID is fixed at link time, so a late dsymutil
  # still matches. Stage it in WORK_DIR: -create-xcframework reads inputs
  # before the old xcframework is deleted below, but keeping inputs out of the
  # output's parent is cleaner.
  MACOS_DSYM="${WORK_DIR}/macos/${FRAMEWORK_NAME}.framework.dSYM"
  mkdir -p "${WORK_DIR}/macos"
  if [[ -d "${XCFRAMEWORK}/macos-arm64/dSYMs/${FRAMEWORK_NAME}.framework.dSYM" ]]; then
    cp -R "${XCFRAMEWORK}/macos-arm64/dSYMs/${FRAMEWORK_NAME}.framework.dSYM" "${MACOS_DSYM}"
    bin_uuids="$(dwarfdump --uuid "${MACOS_FW}/Versions/A/${FRAMEWORK_NAME}" | awk '{print $2}' | sort)"
    dsym_uuids="$(dwarfdump --uuid "${MACOS_DSYM}" | awk '{print $2}' | sort)"
    if [[ "${bin_uuids}" != "${dsym_uuids}" ]]; then
      echo "    embedded macOS dSYM is stale (UUID mismatch); regenerating..."
      rm -rf "${MACOS_DSYM}"
      make_dsym "${MACOS_FW}/Versions/A/${FRAMEWORK_NAME}" "${MACOS_DSYM}"
    fi
  else
    echo "    no embedded macOS dSYM found; generating..."
    make_dsym "${MACOS_FW}/Versions/A/${FRAMEWORK_NAME}" "${MACOS_DSYM}"
  fi
  CREATE_ARGS+=(-framework "${MACOS_FW}" -debug-symbols "${MACOS_DSYM}")
fi

echo "==> Creating universal ${FRAMEWORK_NAME}.xcframework..."
mkdir -p "${ARTIFACTS_DIR}"
xcodebuild -create-xcframework "${CREATE_ARGS[@]}" -output "${WORK_DIR}/out.xcframework"
rm -rf "${XCFRAMEWORK}"
mv "${WORK_DIR}/out.xcframework" "${XCFRAMEWORK}"

# Package a distributable zip for the SwiftPM release with Info-ZIP `zip`, NOT
# `ditto`. `ditto -c -k` serializes every entry's extended attributes as an
# AppleDouble `._name` sidecar (and __MACOSX/ dirs with --sequesterRsrc) — most
# problematically `com.apple.provenance`, a machine-local xattr macOS stamps on
# written files that `xattr -cr` cannot strip (it is protected/re-applied). That
# noise has no business in a distributed artifact. `zip` never writes
# AppleDouble at all: -X drops extra file attributes, -y stores symlinks as
# symlinks (the macOS slice is a versioned framework bundle with 4 symlinks).
# Prints the SwiftPM checksum to drop into Package.swift + checksums.txt.
# Portability gate: the zip below IS the release asset, so every slice must be
# loadable on a clean machine — no /opt/local, /opt/homebrew or /usr/local
# dylib references (iOS slices are static-linked; the macOS slice is the one
# that can regress when built without MP_BUNDLE_DEPS=1, which is what shipped
# broken in v1.0.0-apple.1 and dyld-crashed ScreenBar build 27 fleet-wide).
echo "==> Auditing slice portability before packaging..."
while IFS= read -r slice_bin; do
  bad="$(otool -L "${slice_bin}" | tail -n +2 | sed 's/^[[:space:]]*//' | awk '{print $1}' \
    | grep -vE '^/System/|^/usr/lib/|^@rpath|^@loader_path|^@executable_path' || true)"
  if [[ -n "${bad}" ]]; then
    if [[ "${MP_ALLOW_NONPORTABLE:-0}" == "1" ]]; then
      echo "warning: NON-PORTABLE slice ${slice_bin} (MP_ALLOW_NONPORTABLE=1); do NOT release this zip:" >&2
      echo "${bad}" | sed 's/^/    /' >&2
    else
      echo "error: non-portable dylib references in ${slice_bin}:" >&2
      echo "${bad}" | sed 's/^/    /' >&2
      echo "       Rebuild the macOS slice with MP_BUNDLE_DEPS=1 (see PACKAGING.md)," >&2
      echo "       or set MP_ALLOW_NONPORTABLE=1 for a local-only artifact." >&2
      exit 1
    fi
  fi
done < <(find "${XCFRAMEWORK}" -type f -name "${FRAMEWORK_NAME}" -not -path "*/Libraries/*" -not -path "*/dSYMs/*")

echo "==> Packaging ${FRAMEWORK_NAME}.xcframework.zip (clean, no AppleDouble)..."
ZIP="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.xcframework.zip"
xattr -cr "${XCFRAMEWORK}" 2>/dev/null || true
rm -f "${ZIP}"
( cd "${ARTIFACTS_DIR}" && zip -r -X -y -q "${FRAMEWORK_NAME}.xcframework.zip" "${FRAMEWORK_NAME}.xcframework" )
echo "    swiftpm-checksum: $(swift package compute-checksum "${ZIP}")"

echo "==> Done. Slices:"
plutil -p "${XCFRAMEWORK}/Info.plist" | grep -iE "LibraryIdentifier|SupportedPlatform" || true
