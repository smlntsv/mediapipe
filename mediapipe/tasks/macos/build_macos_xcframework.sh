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
# Builds the MediaPipe Tasks C library for native macOS and packages it as
# MediaPipeTasksC.xcframework (a framework-based xcframework) for the
# MediaPipeTasks Swift package.
#
#   ./build_macos_xcframework.sh                          # arm64 (default, as before)
#   MP_MACOS_ARCHS="arm64 x86_64" ./build_macos_xcframework.sh   # universal macOS slice
#   MP_MACOS_ARCHS=x86_64 ./build_macos_xcframework.sh    # Intel-only slice
#
# Release builds additionally need MP_BUNDLE_DEPS=1 and MPP_BUILD_VERSION; see
# PACKAGING.md ("Universal macOS slice") for the full release recipe, including
# MP_PRESERVE_SLICES_FROM to keep the published iOS slices byte-identical.
#
# Output (gitignored):
#   mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework
#     (slice macos-arm64, macos-x86_64 or macos-arm64_x86_64)
#   mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.macos-<arch>.dSYM.zip
#     (optional full-DWARF symbolication asset, one per architecture)
#
# Prerequisites (see README.md): Bazel 7.4.1, Xcode, a JDK (JAVA_HOME), and a
# Python 3.9-3.12 interpreter as `python3` on PATH (MediaPipe has no 3.13/3.14
# requirements lock).

set -euo pipefail

# --- Configuration ----------------------------------------------------------
FRAMEWORK_NAME="MediaPipeTasksC"
BAZEL="${BAZEL:-$(command -v bazel)}"
MACOS_MIN_VERSION="${MACOS_MIN_VERSION:-13.0}"
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
WORK_DIR="$(mktemp -d)"

# In-tree source patches this run applied (see step 0). They are reversed on exit
# so the working tree is left exactly as we found it — the BUILT dylib contains
# the patched code, but `git status` stays clean.
APPLIED_TREE_PATCHES=()

cleanup() {
  if (( ${#APPLIED_TREE_PATCHES[@]} > 0 )); then
    cd "${MPP_ROOT_DIR}" || true
    local p
    for p in "${APPLIED_TREE_PATCHES[@]}"; do
      if git apply --reverse --check "${p}" >/dev/null 2>&1; then
        git apply --reverse "${p}" \
          && echo "==> restored upstream source (un-applied $(basename "${p}"))"
      fi
    done
  fi
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

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

# --- 0. Apply in-tree source patches ----------------------------------------
# Patches that live in mediapipe/tasks/macos/patches/ and target files in the
# mediapipe ROOT repo (not external Bazel modules) are applied here, idempotently,
# before the Bazel build. The MODULE.bazel single_version_override mechanism (used
# for apple_support_lc_uuid.patch) only patches FETCHED EXTERNAL modules, so it
# can't touch the root repo — hence this step. A clean checkout therefore builds
# the patched (bounded-memory) dylib deterministically.
#
# mediapipe_macos_gpu_texture_cache_flush.patch: flushes the Metal + OpenGL
# texture caches once per frame on the macOS CVPixelBuffer GPU path so sustained
# VIDEO inference doesn't leak IOSurfaces (see the patch header for the full why).
#
# The patch is applied only for the duration of the build and is REVERSED on exit
# by cleanup() (the two upstream files are committed pristine), so a release build
# never leaves mediapipe/gpu/*.cc showing as modified in `git status`.
apply_tree_patch() {
  local patch="$1"
  if [[ ! -f "${patch}" ]]; then
    echo "error: patch not found: ${patch}" >&2
    exit 1
  fi
  if git apply --reverse --check "${patch}" >/dev/null 2>&1; then
    # Already applied before this run (e.g. left over from an interrupted build):
    # leave it as-is and don't schedule a reversal we didn't cause.
    echo "==> patch already applied: $(basename "${patch}")"
  elif git apply --check "${patch}" >/dev/null 2>&1; then
    git apply "${patch}"
    APPLIED_TREE_PATCHES+=("${patch}")
    echo "==> applied patch (will be reversed on exit): $(basename "${patch}")"
  else
    echo "error: ${patch} neither applies cleanly nor is already applied." >&2
    echo "       (the target source may have diverged — reconcile the patch.)" >&2
    exit 1
  fi
}
apply_tree_patch "mediapipe/tasks/macos/patches/mediapipe_macos_gpu_texture_cache_flush.patch"

# --- 1. Per-architecture build ----------------------------------------------
# MP_MACOS_ARCHS selects the macOS architectures (space- or comma-separated):
#   arm64          (default) — byte-for-byte the historical Apple-silicon build:
#                  --config=darwin_arm64, MacPorts OpenCV from /opt/local.
#   x86_64         Intel-only slice.
#   "arm64 x86_64" universal slice (macos-arm64_x86_64): each arch is built
#                  independently, bundled + verified on its own, then the
#                  framework binary and every bundled dylib are lipo-merged.
#
# Architecture-specific settings are scoped HERE and nowhere else:
#   arm64  : --config=darwin_arm64 (minos 11.0 via its --macos_minimum_os=10.16)
#            OpenCV/zlib from MacPorts (/opt/local) unless MP_ARM64_DEPS=source.
#   x86_64 : --config=darwin_x86_64 --macos_minimum_os=${MACOS_MIN_VERSION}
#            (that config's own 10.12 floor predates the C++17 library
#            availability MediaPipe needs, and the package floor is 13.0);
#            OpenCV/zlib built from pinned sources by build_macos_deps.sh
#            (MacPorts on Apple silicon is arm64-only), swapped in with
#            --override_repository=macos_opencv=<prefix>.
# Compiler flags, defines (GPU/Metal, CV_PIXEL_BUFFER), sources and patches are
# identical for both architectures.
MACOS_ARCHS=()
for a in ${MP_MACOS_ARCHS:-arm64}; do
  IFS=',' read -r -a parts <<< "${a}"
  for p in "${parts[@]}"; do
    case "${p}" in
      arm64|x86_64) MACOS_ARCHS+=("${p}") ;;
      "") ;;
      *) echo "error: unsupported MP_MACOS_ARCHS entry '${p}' (arm64|x86_64)" >&2; exit 1 ;;
    esac
  done
done
if (( ${#MACOS_ARCHS[@]} == 0 )); then
  echo "error: MP_MACOS_ARCHS selected no architectures" >&2
  exit 1
fi
# Canonical order (arm64 first) so the slice identifier is deterministic.
SORTED_ARCHS=()
for a in arm64 x86_64; do
  for b in "${MACOS_ARCHS[@]}"; do
    if [[ "${a}" == "${b}" ]]; then SORTED_ARCHS+=("${a}"); break; fi
  done
done
MACOS_ARCHS=("${SORTED_ARCHS[@]}")
echo "==> macOS architectures: ${MACOS_ARCHS[*]}"

DEPS_ROOT="${MP_MACOS_DEPS_DIR:-${HOME}/Library/Caches/mediapipe-macos-deps}"

# Prints the deps prefix to override the macos_opencv repository with, or
# nothing to use the WORKSPACE default (MacPorts /opt/local).
deps_prefix_for() {
  local arch="$1"
  if [[ "${arch}" == "x86_64" || "${MP_ARM64_DEPS:-macports}" == "source" ]]; then
    echo "${DEPS_ROOT}/${arch}"
  fi
}

# Library search paths dylibbundler uses to resolve the OpenCV/zlib install
# names recorded in the linked binary.
deps_search_paths_for() {
  local prefix
  prefix="$(deps_prefix_for "$1")"
  if [[ -n "${prefix}" ]]; then
    echo "${prefix}/local/lib ${prefix}/local/lib/opencv3"
  else
    echo "/opt/local/lib /opt/local/lib/opencv3"
  fi
}

# Every Mach-O dependency must resolve inside the bundle or to the OS.
nonportable_deps() {
  local f="$1" a
  for a in $(lipo -archs "${f}"); do
    otool -arch "${a}" -L "${f}" | tail -n +2 | sed 's/^[[:space:]]*//' | awk '{print $1}'
  done | grep -vE '^/System/|^/usr/lib/|^@rpath|^@loader_path|^@executable_path' | sort -u || true
}

minos_of() {
  otool -arch "$2" -l "$1" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}'
}

strip_rpath_all() {
  local bin="$1" rp="$2" n
  while :; do
    n="$(otool -l "${bin}" 2>/dev/null \
      | awk -v r="${rp}" '/LC_RPATH/{f=1;next} f&&/ path /{if($2==r)c++;f=0} END{print c+0}')"
    [[ "${n:-0}" -gt 0 ]] || break
    install_name_tool -delete_rpath "${rp}" "${bin}" 2>/dev/null || break
  done
}

# build_arch <arch>: builds, verifies and (MP_BUNDLE_DEPS=1) bundles one thin
# framework at ${WORK_DIR}/<arch>/${FRAMEWORK_NAME}.framework.
build_arch() {
  local arch="$1"
  local -a bazel_args=(-c opt --config=macos --config="darwin_${arch}")
  local deps_prefix
  deps_prefix="$(deps_prefix_for "${arch}")"
  if [[ "${arch}" == "x86_64" ]]; then
    bazel_args+=(--macos_minimum_os="${MACOS_MIN_VERSION}")
  fi
  if [[ -n "${deps_prefix}" ]]; then
    "${MPP_ROOT_DIR}/mediapipe/tasks/macos/build_macos_deps.sh" "${arch}"
    bazel_args+=(--override_repository=macos_opencv="${deps_prefix}")
  fi
  local deps_label="MacPorts OpenCV 3"
  [[ -z "${deps_prefix}" ]] || deps_label="source-built OpenCV 3 (${deps_prefix})"

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
    echo "==> Building ${BAZEL_TARGET} for macOS ${arch} (GPU/Metal, ${deps_label})..."
    "${BAZEL}" build "${bazel_args[@]}" \
      --copt=-DMEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER=1 \
      --objccopt=-DMEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER=1 \
      "${BAZEL_TARGET}"
  else
    echo "==> Building ${BAZEL_TARGET} for macOS ${arch} (CPU, ${deps_label})..."
    "${BAZEL}" build "${bazel_args[@]}" \
      --define MEDIAPIPE_DISABLE_GPU=1 \
      "${BAZEL_TARGET}"
  fi

  if [[ ! -f "${DYLIB_BAZEL_PATH}" ]]; then
    echo "error: expected dylib not found at ${DYLIB_BAZEL_PATH}" >&2
    exit 1
  fi

  local arch_dir="${WORK_DIR}/${arch}"
  mkdir -p "${arch_dir}"
  local dylib="${arch_dir}/libmediapipe.dylib"
  # bazel-bin points at the LAST build's configuration; copy it out right away.
  cp -f "${DYLIB_BAZEL_PATH}" "${dylib}"
  chmod u+w "${dylib}"

  local got_archs
  got_archs="$(lipo -archs "${dylib}")"
  if [[ "${got_archs}" != "${arch}" ]]; then
    echo "error: built dylib has architectures '${got_archs}', expected '${arch}'" >&2
    exit 1
  fi

  # --- Verify exported C symbols ---
  # Dump the symbol table to a file once, then grep the file. (Piping `nm` into
  # `grep -q` would let grep close the pipe early, killing nm with SIGPIPE; under
  # `set -o pipefail` that makes a *found* symbol look like a failure.)
  echo "==> Verifying exported MediaPipe C symbols (${arch})..."
  local symbols="${arch_dir}/symbols.txt"
  nm -gU "${dylib}" > "${symbols}" 2>/dev/null || true
  local missing=0 sym
  for sym in MpHandLandmarkerCreate MpPoseLandmarkerCreate MpFaceLandmarkerCreate MpImageCreateFromUint8Data; do
    if grep -q "${sym}" "${symbols}"; then
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

  # --- Verify the binary really targets macOS (not iOS) ---
  echo "==> Verifying Mach-O build platform (${arch})..."
  local loadcmds="${arch_dir}/loadcmds.txt"
  otool -l "${dylib}" > "${loadcmds}" 2>/dev/null || true
  if grep -A3 LC_BUILD_VERSION "${loadcmds}" | grep -qE "platform 1$|platform MACOS"; then
    echo "    platform MACOS confirmed (minos $(minos_of "${dylib}" "${arch}"))"
  else
    echo "    note: parsing platform from otool (expecting MACOS):"
    grep -A3 LC_BUILD_VERSION "${loadcmds}" | head -8 || true
  fi

  # --- Assemble MediaPipeTasksC.framework (macOS versioned bundle) ---
  echo "==> Assembling ${FRAMEWORK_NAME}.framework (${arch})..."
  local fw="${arch_dir}/${FRAMEWORK_NAME}.framework"
  local version_dir="${fw}/Versions/A"
  mkdir -p "${version_dir}/Headers" "${version_dir}/Resources"

  cp -f "${dylib}" "${version_dir}/${FRAMEWORK_NAME}"
  # The framework's load command must resolve via @rpath so the consuming app can
  # embed and sign it predictably.
  install_name_tool -id "@rpath/${FRAMEWORK_NAME}.framework/Versions/A/${FRAMEWORK_NAME}" \
    "${version_dir}/${FRAMEWORK_NAME}"

  # Placeholder public header. The MediaPipeTasksObjC shim includes the real
  # MediaPipe C headers straight from the repo source tree; this framework is the
  # binary runtime only and should NOT be imported directly.
  cat > "${version_dir}/Headers/${FRAMEWORK_NAME}.h" <<EOF
// ${FRAMEWORK_NAME} — binary runtime for MediaPipe Tasks C on macOS.
//
// Do NOT import this framework directly. It carries only the compiled MediaPipe
// C symbols (MpHandLandmarker*, MpImage*, ...). The Swift-facing bridge is the
// MediaPipeTasksObjC target, which includes the MediaPipe C headers from the
// repo source tree and links this binary. App code should import
// MediaPipeTasksMac.
EOF

  cat > "${version_dir}/Resources/Info.plist" <<EOF
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
  ln -sfn "A" "${fw}/Versions/Current"
  ln -sfn "Versions/Current/${FRAMEWORK_NAME}" "${fw}/${FRAMEWORK_NAME}"
  ln -sfn "Versions/Current/Headers" "${fw}/Headers"
  ln -sfn "Versions/Current/Resources" "${fw}/Resources"

  # --- (Opt-in) Bundle non-system dylibs for a portable release ---
  # MP_BUNDLE_DEPS=1 recursively copies every non-system dynamic dependency (the
  # OpenCV tree) into Versions/A/Libraries and rewrites install names so the
  # framework no longer references the build machine's prefix at runtime.
  # Bundled libs and the framework binary reference each other via @rpath,
  # anchored by @loader_path rpaths (main binary: @loader_path/Libraries; each
  # bundled lib: @loader_path).
  if [[ "${MP_BUNDLE_DEPS:-0}" == "1" ]]; then
    echo "==> Bundling non-system dependencies (${arch}, MP_BUNDLE_DEPS=1)..."
    if ! command -v dylibbundler >/dev/null 2>&1; then
      echo "error: dylibbundler is required for MP_BUNDLE_DEPS=1 but was not found." >&2
      echo "       Install it with:  brew install dylibbundler" >&2
      echo "                    or:  sudo port install dylibbundler" >&2
      exit 1
    fi

    local lib_subdir="Libraries" sp
    local -a search_args=()
    for sp in $(deps_search_paths_for "${arch}"); do
      search_args+=(--search-path "${sp}")
    done
    # dylibbundler resolves -x/-d relative to cwd; run inside the version dir.
    (
      cd "${version_dir}"
      dylibbundler \
        --fix-file "${FRAMEWORK_NAME}" \
        --bundle-deps \
        --dest-dir "${lib_subdir}" \
        --install-path "@rpath/" \
        "${search_args[@]}" \
        --overwrite-files --create-dir --no-codesign
    )

    # dylibbundler rewrites the binary's pre-existing (Bazel) rpaths to a bogus
    # "@rpath/" entry — sometimes more than once, which yields a duplicate
    # LC_RPATH that the linker rejects. Strip every "@rpath/" rpath, then add the
    # ones we actually want.
    # Anchor @rpath: the main binary finds bundled libs in ./Libraries; each
    # bundled lib finds its siblings in its own directory.
    strip_rpath_all "${version_dir}/${FRAMEWORK_NAME}" "@rpath/"
    install_name_tool -add_rpath "@loader_path/${lib_subdir}" \
      "${version_dir}/${FRAMEWORK_NAME}" 2>/dev/null || true
    local lib
    if [[ -d "${version_dir}/${lib_subdir}" ]]; then
      for lib in "${version_dir}/${lib_subdir}"/*.dylib; do
        strip_rpath_all "${lib}" "@rpath/"
        install_name_tool -add_rpath "@loader_path" "${lib}" 2>/dev/null || true
      done
    fi

    # Verify portability: no dependency may point outside the allowed prefixes
    # (/System, /usr/lib, @rpath, @loader_path, @executable_path).
    echo "==> Verifying no non-portable (/opt/local etc.) dependencies remain (${arch})..."
    local bad f bundled_count=0
    for f in "${version_dir}/${FRAMEWORK_NAME}" "${version_dir}/${lib_subdir}"/*.dylib; do
      [[ -f "${f}" ]] || continue
      bad="$(nonportable_deps "${f}")"
      if [[ -n "${bad}" ]]; then
        echo "error: non-portable dependency in ${f}:" >&2
        echo "${bad}" | sed 's/^/    /' >&2
        exit 1
      fi
      [[ "${f}" == *.dylib ]] && bundled_count=$((bundled_count + 1))
    done
    local bundled_size
    bundled_size="$(du -sh "${version_dir}/${lib_subdir}" 2>/dev/null | awk '{print $1}')"
    echo "    OK: framework + ${bundled_count} bundled dylibs (${bundled_size:-?}) are portable."
  fi

  # Full file/line DWARF (optional release asset, per architecture). The
  # Bazel-linked dylib carries a debug map (N_OSO) into bazel-out's cached .o
  # files, so dsymutil can link a FULL DWARF dSYM (~370MB). It must be
  # generated from the THIN binary while that arch's bazel-out objects are the
  # ones on disk, and is packaged as ${FRAMEWORK_NAME}.macos-<arch>.dSYM.zip
  # for offline crash symbolication — NOT embedded (see step 3).
  echo "==> Generating full ${FRAMEWORK_NAME}.framework.dSYM (${arch})..."
  local dsym_full_dir="${arch_dir}/full"
  mkdir -p "${dsym_full_dir}"
  dsymutil "${version_dir}/${FRAMEWORK_NAME}" -o "${dsym_full_dir}/${FRAMEWORK_NAME}.framework.dSYM" \
    2> >(grep -v "no debug symbols" >&2 || true)
  # Real DWARF shows a DW_TAG_compile_unit immediately; a symbol-table-only
  # dSYM has an empty .debug_info. NOTE: must not be written as
  # `dwarfdump | head | grep -q` — under `set -o pipefail` head's early exit
  # SIGPIPEs dwarfdump and the pipeline reports failure even when grep matched.
  local debug_info_head asset
  debug_info_head="$(dwarfdump --debug-info "${dsym_full_dir}/${FRAMEWORK_NAME}.framework.dSYM" 2>/dev/null | head -12 || true)"
  asset="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.macos-${arch}.dSYM.zip"
  mkdir -p "${ARTIFACTS_DIR}"
  rm -f "${asset}"
  if grep -q 'DW_TAG' <<< "${debug_info_head}"; then
    echo "==> Full DWARF present; packaging optional symbolication asset..."
    ( cd "${dsym_full_dir}" && zip -r -X -q "${asset}" "${FRAMEWORK_NAME}.framework.dSYM" )
    echo "    $(du -sh "${asset}" | awk '{print $1}') $(basename "${asset}")"
  fi
  # Free the disk early; the asset zip (if any) is what we keep.
  rm -rf "${dsym_full_dir}"
}

for arch in "${MACOS_ARCHS[@]}"; do
  build_arch "${arch}"
done

# --- 2. Merge the per-arch frameworks ----------------------------------------
# One arch: the thin framework IS the slice. Several: start from the first
# arch's bundle (headers/plist/symlinks are arch-independent) and lipo-merge the
# framework binary and every bundled dylib. The per-arch bundles must agree on
# the bundled file set AND on each Mach-O's install name + dependency list, or
# one architecture would dyld-fail looking for a lib that only the other has.
FW="${WORK_DIR}/${FRAMEWORK_NAME}.framework"
VERSION_DIR="${FW}/Versions/A"
SLICE_ARCHS="$(IFS=_; echo "${MACOS_ARCHS[*]}")"
cp -R "${WORK_DIR}/${MACOS_ARCHS[0]}/${FRAMEWORK_NAME}.framework" "${FW}"
if (( ${#MACOS_ARCHS[@]} > 1 )); then
  echo "==> Merging ${MACOS_ARCHS[*]} into a universal ${FRAMEWORK_NAME}.framework..."
  list_machos() {
    ( cd "$1/Versions/A" && { echo "${FRAMEWORK_NAME}"; ls Libraries/*.dylib 2>/dev/null || true; } | sort )
  }
  # Install name (first entry, exact) + the SET of dependency names — compared
  # across arches. Load-command order may legitimately differ between
  # independently linked libraries (e.g. libSystem vs libc++) and is irrelevant
  # to dyld; a dependency present on only one arch is not.
  macho_signature() {
    local deps
    deps="$(otool -L "$1" | tail -n +2 | awk '{print $1}')"
    head -1 <<< "${deps}"
    tail -n +2 <<< "${deps}" | sort
  }
  base_set="$(list_machos "${FW}")"
  for arch in "${MACOS_ARCHS[@]:1}"; do
    other_fw="${WORK_DIR}/${arch}/${FRAMEWORK_NAME}.framework"
    other_set="$(list_machos "${other_fw}")"
    if [[ "${base_set}" != "${other_set}" ]]; then
      echo "error: per-arch bundles differ (${MACOS_ARCHS[0]} vs ${arch}):" >&2
      diff <(echo "${base_set}") <(echo "${other_set}") >&2 || true
      exit 1
    fi
  done
  while IFS= read -r rel; do
    inputs=()
    base_sig="$(macho_signature "${WORK_DIR}/${MACOS_ARCHS[0]}/${FRAMEWORK_NAME}.framework/Versions/A/${rel}")"
    for arch in "${MACOS_ARCHS[@]}"; do
      f="${WORK_DIR}/${arch}/${FRAMEWORK_NAME}.framework/Versions/A/${rel}"
      sig="$(macho_signature "${f}")"
      if [[ "${sig}" != "${base_sig}" ]]; then
        echo "error: ${rel} links differently on ${arch} than on ${MACOS_ARCHS[0]}:" >&2
        diff <(echo "${base_sig}") <(echo "${sig}") >&2 || true
        exit 1
      fi
      inputs+=("${f}")
    done
    lipo -create "${inputs[@]}" -output "${VERSION_DIR}/${rel}"
    echo "    ${rel}: $(lipo -archs "${VERSION_DIR}/${rel}")"
  done <<< "${base_set}"
fi

# Re-sign (install_name_tool and lipo invalidate signatures): libs first, then
# the binary.
for lib in "${VERSION_DIR}"/Libraries/*.dylib; do
  [[ -f "${lib}" ]] && codesign --force --sign - "${lib}" >/dev/null 2>&1 || true
done
codesign --force --sign - "${VERSION_DIR}/${FRAMEWORK_NAME}" >/dev/null 2>&1 || true

# --- 3. Embedded (Tier-A) dSYM ----------------------------------------------
# Companion dSYM: embedded via -debug-symbols it lands in <slice>/dSYMs/
# inside the xcframework; Xcode auto-copies it into consumer archives,
# satisfying App Store Connect's "Upload Symbols Failed ... expected UUIDs"
# check. It is kept Tier A (symbol-table only, a few MB per arch) so the
# SPM-downloaded zip stays small: strip the debug map + stabs from a throwaway
# copy of the (possibly universal) binary — strip does not touch LC_UUID — so
# dsymutil falls back to the symbol table. A universal binary yields one dSYM
# carrying one UUID per architecture.
echo "==> Generating Tier-A (symbol-table) dSYM for embedding..."
mkdir -p "${WORK_DIR}/tiera"
cp "${VERSION_DIR}/${FRAMEWORK_NAME}" "${WORK_DIR}/tiera/${FRAMEWORK_NAME}"
strip -S "${WORK_DIR}/tiera/${FRAMEWORK_NAME}" 2>/dev/null || strip -S "${WORK_DIR}/tiera/${FRAMEWORK_NAME}"
DSYM_OUT="${WORK_DIR}/${FRAMEWORK_NAME}.framework.dSYM"
dsymutil "${WORK_DIR}/tiera/${FRAMEWORK_NAME}" -o "${DSYM_OUT}" 2> >(grep -v "no debug symbols" >&2 || true)

BIN_UUIDS="$(dwarfdump --uuid "${VERSION_DIR}/${FRAMEWORK_NAME}" | awk '{print $2, $3}' | sort)"
DSYM_UUIDS="$(dwarfdump --uuid "${DSYM_OUT}" | awk '{print $2, $3}' | sort)"
if [[ "${BIN_UUIDS}" != "${DSYM_UUIDS}" ]]; then
  echo "error: dSYM UUIDs (${DSYM_UUIDS}) do not match binary UUIDs (${BIN_UUIDS})" >&2
  exit 1
fi
echo "${BIN_UUIDS}" | sed 's/^/    UUID /'

# --- 4. Slice gates (ALWAYS run) --------------------------------------------
# Every Mach-O that ships in the slice — the framework binary AND each bundled
# dylib, for EVERY architecture — must:
#   * carry exactly the requested architectures (a thin bundled lib inside a
#     universal framework dyld-fails on the missing arch), and
#   * be portable: a slice that references /opt/local, /opt/homebrew or
#     /usr/local dylibs only loads on machines with that package manager
#     installed. v1.0.0-apple.1 shipped such a slice (built without
#     MP_BUNDLE_DEPS=1) and crashed every fleet Mac at dyld time before
#     main() — so a non-portable artifact is a hard error unless explicitly
#     allowed for local development on this machine.
echo "==> Auditing every Mach-O in the macOS slice..."
EXPECTED_ARCHS="$(printf '%s\n' "${MACOS_ARCHS[@]}" | sort | xargs)"
NONPORTABLE_REPORT=""
while IFS= read -r f; do
  got="$(lipo -archs "${f}" | tr ' ' '\n' | sort | xargs)"
  rel="${f#"${VERSION_DIR}"/}"
  if [[ "${got}" != "${EXPECTED_ARCHS}" ]]; then
    echo "error: ${rel} has architectures '${got}', expected '${EXPECTED_ARCHS}'" >&2
    exit 1
  fi
  minos_list=""
  for a in ${got}; do minos_list+="${a}:$(minos_of "${f}" "${a}") "; done
  printf '    %-36s %s\n' "${rel}" "${minos_list}"
  bad="$(nonportable_deps "${f}")"
  [[ -z "${bad}" ]] || NONPORTABLE_REPORT+="${rel}:"$'\n'"$(echo "${bad}" | sed 's/^/  /')"$'\n'
done < <(find "${VERSION_DIR}" -type f \( -name "${FRAMEWORK_NAME}" -o -name '*.dylib' \))
if [[ -n "${NONPORTABLE_REPORT}" ]]; then
  if [[ "${MP_ALLOW_NONPORTABLE:-0}" == "1" ]]; then
    echo "warning: artifact is NOT portable (MP_ALLOW_NONPORTABLE=1); local use only:" >&2
    echo "${NONPORTABLE_REPORT}" | sed 's/^/    /' >&2
  else
    echo "error: the packaged framework references non-portable dylibs:" >&2
    echo "${NONPORTABLE_REPORT}" | sed 's/^/    /' >&2
    echo "       These paths exist only on this build machine; the framework would" >&2
    echo "       crash at dyld time everywhere else (this is what broke ScreenBar" >&2
    echo "       build 27). Rebuild with MP_BUNDLE_DEPS=1 for any distributable" >&2
    echo "       artifact, or set MP_ALLOW_NONPORTABLE=1 for a local-only build." >&2
    exit 1
  fi
fi

# --- 5. Create the xcframework ----------------------------------------------
# By default the output holds ONLY the macOS slice (build_ios_xcframework.sh
# folds the iOS slices in afterwards). MP_PRESERVE_SLICES_FROM=<xcframework or
# .zip> instead carries every NON-macOS slice (with its embedded dSYMs) over
# byte-for-byte from an existing artifact — e.g. the published release zip —
# so the macOS slice can be replaced without rebuilding or perturbing iOS. That
# mode also packages ${FRAMEWORK_NAME}.xcframework.zip + its SwiftPM checksum.
CREATE_ARGS=(-framework "${FW}" -debug-symbols "${DSYM_OUT}")
if [[ -n "${MP_PRESERVE_SLICES_FROM:-}" ]]; then
  SRC_XCF="${MP_PRESERVE_SLICES_FROM}"
  if [[ "${SRC_XCF}" == *.zip ]]; then
    mkdir -p "${WORK_DIR}/preserve"
    unzip -q "${SRC_XCF}" -d "${WORK_DIR}/preserve"
    SRC_XCF="$(find "${WORK_DIR}/preserve" -maxdepth 2 -name "${FRAMEWORK_NAME}.xcframework" -type d | head -1)"
  fi
  if [[ ! -f "${SRC_XCF}/Info.plist" ]]; then
    echo "error: MP_PRESERVE_SLICES_FROM does not contain ${FRAMEWORK_NAME}.xcframework" >&2
    exit 1
  fi
  # Stage outside the output directory (the output is deleted below).
  cp -R "${SRC_XCF}" "${WORK_DIR}/preserved.xcframework"
  SRC_XCF="${WORK_DIR}/preserved.xcframework"
  n_libs="$(/usr/libexec/PlistBuddy -c 'Print :AvailableLibraries' "${SRC_XCF}/Info.plist" | grep -c 'LibraryIdentifier')"
  for ((i = 0; i < n_libs; i++)); do
    plat="$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries:${i}:SupportedPlatform" "${SRC_XCF}/Info.plist")"
    lib_id="$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries:${i}:LibraryIdentifier" "${SRC_XCF}/Info.plist")"
    lib_path="$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries:${i}:LibraryPath" "${SRC_XCF}/Info.plist")"
    if [[ "${plat}" == "macos" ]]; then
      echo "    replacing preserved slice ${lib_id}"
      continue
    fi
    echo "    preserving slice ${lib_id}"
    CREATE_ARGS+=(-framework "${SRC_XCF}/${lib_id}/${lib_path}")
    if [[ -d "${SRC_XCF}/${lib_id}/dSYMs/${FRAMEWORK_NAME}.framework.dSYM" ]]; then
      CREATE_ARGS+=(-debug-symbols "${SRC_XCF}/${lib_id}/dSYMs/${FRAMEWORK_NAME}.framework.dSYM")
    fi
  done
fi

echo "==> Creating ${FRAMEWORK_NAME}.xcframework (macos-${SLICE_ARCHS})..."
OUT_XCFRAMEWORK="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.xcframework"
mkdir -p "${ARTIFACTS_DIR}"
rm -rf "${OUT_XCFRAMEWORK}"
xcodebuild -create-xcframework "${CREATE_ARGS[@]}" -output "${OUT_XCFRAMEWORK}"

# --- 6. Verify the xcframework ----------------------------------------------
echo "==> Verifying xcframework..."
plutil -p "${OUT_XCFRAMEWORK}/Info.plist"
echo "--- Mach-O platform of the packaged macOS slice ---"
SLICE_BIN="${OUT_XCFRAMEWORK}/macos-${SLICE_ARCHS}/${FRAMEWORK_NAME}.framework/Versions/A/${FRAMEWORK_NAME}"
if [[ ! -f "${SLICE_BIN}" ]]; then
  echo "error: expected macOS slice binary at ${SLICE_BIN}" >&2
  exit 1
fi
otool -l "${SLICE_BIN}" | grep -A3 LC_BUILD_VERSION | head -8 || true
"${MPP_ROOT_DIR}/mediapipe/tasks/macos/audit_xcframework.sh" "${OUT_XCFRAMEWORK}"

if [[ -n "${MP_PRESERVE_SLICES_FROM:-}" ]]; then
  # Same packaging as build_ios_xcframework.sh: Info-ZIP (no AppleDouble
  # sidecars), symlinks preserved (-y) for the versioned macOS bundle.
  echo "==> Packaging ${FRAMEWORK_NAME}.xcframework.zip (clean, no AppleDouble)..."
  ZIP="${ARTIFACTS_DIR}/${FRAMEWORK_NAME}.xcframework.zip"
  xattr -cr "${OUT_XCFRAMEWORK}" 2>/dev/null || true
  rm -f "${ZIP}"
  ( cd "${ARTIFACTS_DIR}" && zip -r -X -y -q "${FRAMEWORK_NAME}.xcframework.zip" "${FRAMEWORK_NAME}.xcframework" )
  echo "    swiftpm-checksum: $(swift package compute-checksum "${ZIP}")"
fi

echo ""
echo "==> Done: ${OUT_XCFRAMEWORK}"
echo "    You can now run: swift build   (and: swift run mediapipe-macos-sample ...)"
