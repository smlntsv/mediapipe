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
# Builds the macOS runtime dependencies of MediaPipeTasksC (OpenCV core +
# imgproc and the zlib they link) for ONE architecture from pinned, checksum-
# verified sources, into a self-contained prefix that build_macos_xcframework.sh
# feeds to Bazel in place of MacPorts:
#
#   ./build_macos_deps.sh x86_64          # -> ${MP_MACOS_DEPS_DIR}/x86_64
#
# Why this exists: the arm64 slice links the build machine's MacPorts
# opencv3/zlib (/opt/local), which on an Apple-silicon Mac are arm64-only — an
# x86_64 link against them fails. This script reproduces the SAME versions
# (OpenCV 3.4.20, zlib 1.3.2 — exactly what `port installed` reports for the
# arm64 release) with an equivalent configuration (MacPorts Portfile options
# that affect core/imgproc: shared libs, Accelerate LAPACK, GCD parallel_for,
# no IPP/TBB/OpenCL/OpenMP/pthreads-pf), trimmed to the two modules MediaPipe
# links (third_party/opencv_macos.BUILD), and with an explicit deployment
# target (default 13.0 = the package's macOS floor; MacPorts stamps its libs
# with the build host's OS, e.g. minos 26.0).
#
# Layout mirrors MacPorts so third_party/opencv_macos.BUILD is reused verbatim
# (PREFIX "local", OPENCV_SUBDIR "opencv3"):
#   <out>/local/lib/opencv3/libopencv_{core,imgproc}.3.4.20.dylib (+ symlinks)
#   <out>/local/include/opencv3/opencv2/...
#   <out>/local/lib/libz.1.3.2.dylib (+ symlinks), <out>/local/include/zlib.h
#   <out>/BUILD.bazel + WORKSPACE  (for --override_repository=macos_opencv=<out>)
#
# Install names are absolute paths inside <out>; build_macos_xcframework.sh's
# MP_BUNDLE_DEPS=1 step rewrites them to @rpath exactly as it does for
# MacPorts, and its portability gate rejects anything left behind.
#
# Env:
#   MP_MACOS_DEPS_DIR   root for per-arch prefixes
#                       (default ~/Library/Caches/mediapipe-macos-deps)
#   MACOS_MIN_VERSION   deployment target (default 13.0)
#   MP_DEPS_FORCE=1     rebuild even if the prefix is already complete

set -euo pipefail

ARCH="${1:-}"
case "${ARCH}" in
  arm64|x86_64) ;;
  *) echo "usage: $0 <arm64|x86_64>" >&2; exit 2 ;;
esac

MACOS_MIN_VERSION="${MACOS_MIN_VERSION:-13.0}"
DEPS_ROOT="${MP_MACOS_DEPS_DIR:-${HOME}/Library/Caches/mediapipe-macos-deps}"
OUT="${DEPS_ROOT}/${ARCH}"
PREFIX="${OUT}/local"
SRC_CACHE="${DEPS_ROOT}/src"
MPP_ROOT_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
JOBS="$(sysctl -n hw.ncpu)"

OPENCV_VERSION="3.4.20"
OPENCV_SHA256="b9eda448a08ba7b10bfd5bd45697056569ebdf7a02070947e1c1f3e8e69280cd"  # = MacPorts opencv3 Portfile
OPENCV_URL="https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz"
ZLIB_VERSION="1.3.2"
ZLIB_SHA256="d7a0654783a4da529d1bb793b7ad9c3318020af77667bcae35f95d0e42a792f3"    # = MacPorts zlib Portfile
ZLIB_URLS=(
  "https://zlib.net/zlib-${ZLIB_VERSION}.tar.xz"
  "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.xz"
)

STAMP="${OUT}/.complete-opencv${OPENCV_VERSION}-zlib${ZLIB_VERSION}-min${MACOS_MIN_VERSION}"
if [[ -f "${STAMP}" && "${MP_DEPS_FORCE:-0}" != "1" ]]; then
  echo "==> ${ARCH} deps already built: ${OUT}"
  exit 0
fi

command -v cmake >/dev/null || { echo "error: cmake not found (brew/port install cmake)" >&2; exit 1; }

# fetch <dest> <sha256> <url-or-local-path>... — reuses a verified cached copy,
# then tries each source in order; only a sha256-verified file is accepted.
fetch() {
  local dest="$1" sha="$2"; shift 2
  local url got
  if [[ -f "${dest}" ]] && [[ "$(shasum -a 256 "${dest}" | awk '{print $1}')" == "${sha}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  for url in "$@"; do
    if [[ "${url}" == /* ]]; then
      [[ -f "${url}" ]] || continue
      cp "${url}" "${dest}.part"
    else
      echo "    downloading ${url}"
      curl -fL --retry 3 -o "${dest}.part" "${url}" || { rm -f "${dest}.part"; continue; }
    fi
    got="$(shasum -a 256 "${dest}.part" | awk '{print $1}')"
    if [[ "${got}" == "${sha}" ]]; then
      mv "${dest}.part" "${dest}"
      return 0
    fi
    echo "    sha256 mismatch for ${url}: ${got}" >&2
    rm -f "${dest}.part"
  done
  echo "error: could not fetch a verified copy of $(basename "${dest}")" >&2
  exit 1
}

echo "==> Building macOS ${ARCH} deps (OpenCV ${OPENCV_VERSION}, zlib ${ZLIB_VERSION}, min ${MACOS_MIN_VERSION}) into ${OUT}"
rm -rf "${OUT}"
mkdir -p "${PREFIX}" "${SRC_CACHE}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

# The MacPorts distfile cache (present when opencv3 was built from source on
# this machine) is the exact tarball the arm64 release's OpenCV came from.
fetch "${SRC_CACHE}/opencv-${OPENCV_VERSION}.tar.gz" "${OPENCV_SHA256}" \
  "/opt/local/var/macports/distfiles/opencv3/${OPENCV_VERSION}.tar.gz" "${OPENCV_URL}"
fetch "${SRC_CACHE}/zlib-${ZLIB_VERSION}.tar.xz" "${ZLIB_SHA256}" "${ZLIB_URLS[@]}"

ARCH_FLAGS="-arch ${ARCH} -mmacosx-version-min=${MACOS_MIN_VERSION}"
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT

# --- zlib -------------------------------------------------------------------
echo "==> zlib ${ZLIB_VERSION} (${ARCH})"
tar -xf "${SRC_CACHE}/zlib-${ZLIB_VERSION}.tar.xz" -C "${BUILD_DIR}"
(
  cd "${BUILD_DIR}/zlib-${ZLIB_VERSION}"
  CC="$(xcrun -f clang)" CFLAGS="-O3 ${ARCH_FLAGS}" LDFLAGS="${ARCH_FLAGS}" \
    ./configure --prefix="${PREFIX}" >/dev/null
  make -j"${JOBS}" >/dev/null
  make install >/dev/null
)
rm -f "${PREFIX}/lib/libz.a"

# --- OpenCV (core + imgproc) ------------------------------------------------
echo "==> OpenCV ${OPENCV_VERSION} core+imgproc (${ARCH})"
tar -xzf "${SRC_CACHE}/opencv-${OPENCV_VERSION}.tar.gz" -C "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/opencv-build"
# CMAKE_IGNORE_PREFIX_PATH keeps CMake from opportunistically picking up the
# host's (arm64-only, non-portable) MacPorts/Homebrew packages.
# OpenCV picks its SIMD baseline/dispatch set from CMAKE_SYSTEM_PROCESSOR, which
# CMake takes from the HOST — building x86_64 on Apple silicon would otherwise
# configure NEON (and silently drop all SSE/AVX paths). Declaring the target
# processor requires CMAKE_SYSTEM_NAME (CMake's cross-compiling mode).
CROSS_ARGS=()
if [[ "$(uname -m)" != "${ARCH}" ]]; then
  CROSS_ARGS=(-DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR="${ARCH}")
fi
cmake -S "${BUILD_DIR}/opencv-${OPENCV_VERSION}" -B "${BUILD_DIR}/opencv-build" \
  ${CROSS_ARGS[@]+"${CROSS_ARGS[@]}"} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="${ARCH}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOS_MIN_VERSION}" \
  -DCMAKE_OSX_SYSROOT="${SDKROOT}" \
  -DCMAKE_IGNORE_PREFIX_PATH="/opt/local;/opt/homebrew;/usr/local" \
  -DCMAKE_FIND_FRAMEWORK=LAST \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DOPENCV_LIB_INSTALL_PATH=lib/opencv3 \
  -DOPENCV_INCLUDE_INSTALL_PATH=include/opencv3 \
  -DOPENCV_CONFIG_INSTALL_PATH=lib/opencv3/cmake \
  -DOPENCV_OTHER_INSTALL_PATH=share/opencv3 \
  -DOPENCV_LICENSES_INSTALL_PATH=share/opencv3/licenses \
  -DCMAKE_INSTALL_NAME_DIR="${PREFIX}/lib/opencv3" \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_WITH_DEBUG_INFO=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DOPENCV_PYTHON_SKIP_DETECTION=ON \
  -DBUILD_ZLIB=OFF -DZLIB_LIBRARY="${PREFIX}/lib/libz.dylib" -DZLIB_INCLUDE_DIR="${PREFIX}/include" \
  -DLAPACK_IMPL=Apple -DLAPACK_LIBRARIES="-framework Accelerate" \
  -DLAPACK_CBLAS_H=Accelerate/Accelerate.h -DLAPACK_LAPACKE_H=Accelerate/Accelerate.h \
  -DWITH_IPP=OFF -DBUILD_WITH_DYNAMIC_IPP=OFF -DWITH_ITT=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF \
  -DWITH_PTHREADS_PF=OFF -DWITH_CSTRIPES=OFF -DWITH_OPENCL=OFF -DWITH_OPENCL_SVM=OFF \
  -DWITH_CUDA=OFF -DWITH_EIGEN=OFF -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF \
  -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF \
  -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
  -DWITH_AVFOUNDATION=OFF -DWITH_OPENGL=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_VTK=OFF \
  -DWITH_1394=OFF -DWITH_GDAL=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=ON \
  -DOPENCV_GENERATE_PKGCONFIG=OFF -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DENABLE_CONFIG_VERIFICATION=OFF \
  > "${BUILD_DIR}/opencv-cmake.log" 2>&1 \
  || { tail -40 "${BUILD_DIR}/opencv-cmake.log" >&2; exit 1; }
grep -A12 "CPU/HW features" "${BUILD_DIR}/opencv-cmake.log" | sed 's/^/    /' || true
grep -E "To be built|Lapack|Parallel framework" "${BUILD_DIR}/opencv-cmake.log" | sed 's/^/    /' || true
# Fail loudly if the SIMD baseline does not match the target (see CROSS_ARGS).
case "${ARCH}" in
  x86_64) want="SSE" ;;
  arm64)  want="NEON" ;;
esac
if ! grep -A3 "CPU/HW features" "${BUILD_DIR}/opencv-cmake.log" | grep -q "${want}"; then
  echo "error: OpenCV CPU baseline is not ${want}-based for ${ARCH}" >&2
  exit 1
fi
cmake --build "${BUILD_DIR}/opencv-build" -j"${JOBS}" > "${BUILD_DIR}/opencv-build.log" 2>&1 \
  || { tail -40 "${BUILD_DIR}/opencv-build.log" >&2; exit 1; }
cmake --install "${BUILD_DIR}/opencv-build" > /dev/null

# --- Bazel repository shim --------------------------------------------------
# Same BUILD file the MacPorts repo uses, so the arm64/x86_64 link sets are
# identical by construction.
cp "${MPP_ROOT_DIR}/third_party/opencv_macos.BUILD" "${OUT}/BUILD.bazel"
echo 'workspace(name = "macos_opencv")' > "${OUT}/WORKSPACE"

# --- Verify -----------------------------------------------------------------
echo "==> Verifying ${ARCH} deps"
fail=0
for lib in "${PREFIX}/lib/libz.${ZLIB_VERSION}.dylib" \
           "${PREFIX}/lib/opencv3/libopencv_core.${OPENCV_VERSION}.dylib" \
           "${PREFIX}/lib/opencv3/libopencv_imgproc.${OPENCV_VERSION}.dylib"; do
  [[ -f "${lib}" ]] || { echo "    MISSING ${lib}" >&2; fail=1; continue; }
  archs="$(lipo -archs "${lib}")"
  minos="$(otool -l "${lib}" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
  bad="$(otool -L "${lib}" | tail -n +2 | awk '{print $1}' \
    | grep -vE "^/System/|^/usr/lib/|^${PREFIX}/" || true)"
  printf '    %-34s archs=%s minos=%s\n' "$(basename "${lib}")" "${archs}" "${minos}"
  [[ "${archs}" == "${ARCH}" ]] || { echo "    wrong arch" >&2; fail=1; }
  [[ "${minos}" == "${MACOS_MIN_VERSION}" ]] || { echo "    wrong minos" >&2; fail=1; }
  [[ -z "${bad}" ]] || { echo "    unexpected deps: ${bad}" >&2; fail=1; }
done
[[ "${fail}" == 0 ]] || exit 1
touch "${STAMP}"
echo "==> Done: ${OUT}"
echo "    Use with: --override_repository=macos_opencv=${OUT}"
