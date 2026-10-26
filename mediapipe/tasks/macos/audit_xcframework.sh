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
# Audits EVERY distributed Mach-O in a MediaPipeTasksC.xcframework (or its
# release .zip): each slice's framework binary AND each bundled dylib, for
# EVERY architecture in each file.
#
#   ./audit_xcframework.sh path/to/MediaPipeTasksC.xcframework[.zip]
#
# Per slice (from the xcframework Info.plist):
#   ARCH      every Mach-O carries exactly the slice's SupportedArchitectures
#             (a thin bundled dylib inside a universal framework dyld-fails on
#             the missing architecture).
#   PORTABLE  every load command resolves to the OS (/System, /usr/lib) or
#             inside the bundle (@rpath/@loader_path/@executable_path) — never
#             /opt/local, /opt/homebrew, /usr/local or a build-machine path.
#   RPATH     every @rpath dependency of a bundled dylib exists in the bundle.
#   DSYM      the embedded dSYM's UUIDs equal the framework binary's, per arch.
# Informational: minos per arch, code signature validity.
#
# Exit 0 = all gates pass. MP_ALLOW_NONPORTABLE=1 downgrades PORTABLE to a
# warning (local development only — never release such an artifact).

set -euo pipefail

INPUT="${1:?usage: $0 path/to/MediaPipeTasksC.xcframework[.zip]}"
FRAMEWORK_NAME="MediaPipeTasksC"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

XCF="${INPUT}"
if [[ "${INPUT}" == *.zip ]]; then
  unzip -q "${INPUT}" -d "${TMP}"
  XCF="$(find "${TMP}" -maxdepth 2 -name "${FRAMEWORK_NAME}.xcframework" -type d | head -1)"
fi
[[ -f "${XCF}/Info.plist" ]] || { echo "error: no xcframework Info.plist under ${INPUT}" >&2; exit 1; }

FAIL=0
fail() { echo "  FAIL: $*" >&2; FAIL=1; }

PB=/usr/libexec/PlistBuddy
n_libs="$(${PB} -c 'Print :AvailableLibraries' "${XCF}/Info.plist" | grep -c 'LibraryIdentifier')"
for ((i = 0; i < n_libs; i++)); do
  lib_id="$(${PB} -c "Print :AvailableLibraries:${i}:LibraryIdentifier" "${XCF}/Info.plist")"
  lib_path="$(${PB} -c "Print :AvailableLibraries:${i}:LibraryPath" "${XCF}/Info.plist")"
  bin_path="$(${PB} -c "Print :AvailableLibraries:${i}:BinaryPath" "${XCF}/Info.plist" 2>/dev/null || echo "${lib_path}/${FRAMEWORK_NAME}")"
  want_archs="$(${PB} -c "Print :AvailableLibraries:${i}:SupportedArchitectures" "${XCF}/Info.plist" \
    | sed -n 's/^ *\([a-z0-9_]*\)$/\1/p' | sort | xargs)"
  slice="${XCF}/${lib_id}"
  bin="${slice}/${bin_path}"
  echo "== ${lib_id} (Info.plist archs: ${want_archs})"
  [[ -f "${bin}" ]] || { fail "missing binary ${bin_path}"; continue; }

  bundle_root="$(dirname "${bin}")"
  machos=("${bin}")
  if [[ -d "${bundle_root}/Libraries" ]]; then
    while IFS= read -r f; do machos+=("${f}"); done \
      < <(find "${bundle_root}/Libraries" -type f -name '*.dylib' | sort)
  fi

  # The framework binary's minos per arch is the slice's effective floor; a
  # bundled dylib requiring a NEWER OS can fail to load (missing libc++/libSystem
  # symbols) on machines between the two. Reported as a warning, not a gate.
  bin_minos=""
  for f in "${machos[@]}"; do
    rel="${f#"${slice}"/}"
    got="$(lipo -archs "${f}" 2>/dev/null | tr ' ' '\n' | sort | xargs)"
    line="$(printf '  %-58s' "${rel}")"
    [[ "${got}" == "${want_archs}" ]] || fail "${rel}: archs '${got}' != '${want_archs}'"
    for a in ${got}; do
      minos="$(otool -arch "${a}" -l "${f}" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
      line+=" ${a}:minos=${minos}"
      if [[ "${f}" == "${bin}" ]]; then
        bin_minos+=" ${a}=${minos}"
      else
        floor="$(tr ' ' '\n' <<< "${bin_minos}" | sed -n "s/^${a}=//p")"
        if [[ -n "${floor}" && "$(printf '%s\n%s\n' "${floor}" "${minos}" | sort -V | tail -1)" != "${floor}" ]]; then
          echo "  WARNING: ${rel} [${a}] minos ${minos} > framework minos ${floor}" >&2
        fi
      fi
      deps="$(otool -arch "${a}" -L "${f}" | tail -n +2 | awk '{print $1}')"
      bad="$(grep -vE '^/System/|^/usr/lib/|^@rpath/|^@loader_path/|^@executable_path/' <<< "${deps}" || true)"
      if [[ -n "${bad}" ]]; then
        if [[ "${MP_ALLOW_NONPORTABLE:-0}" == "1" ]]; then
          echo "  WARNING (MP_ALLOW_NONPORTABLE=1): ${rel} [${a}] non-portable: $(echo ${bad})" >&2
        else
          fail "${rel} [${a}] non-portable deps: $(echo ${bad})"
        fi
      fi
      # @rpath deps (other than the framework's own id) must ship in Libraries/.
      while IFS= read -r d; do
        [[ -z "${d}" || "${d}" == "@rpath/${FRAMEWORK_NAME}.framework/"* ]] && continue
        [[ -f "${bundle_root}/Libraries/${d#@rpath/}" ]] || fail "${rel} [${a}] needs ${d}, not bundled"
      done < <(grep '^@rpath/' <<< "${deps}" || true)
    done
    if codesign --verify "${f}" 2>/dev/null; then line+=" signed"; else line+=" UNSIGNED/INVALID-SIG"; fi
    echo "${line}"
  done

  dsym="${slice}/dSYMs/${FRAMEWORK_NAME}.framework.dSYM"
  if [[ -d "${dsym}" ]]; then
    bu="$(dwarfdump --uuid "${bin}" | awk '{print $2, $3}' | sort)"
    du="$(dwarfdump --uuid "${dsym}" | awk '{print $2, $3}' | sort)"
    if [[ "${bu}" == "${du}" ]]; then
      echo "  dSYM UUIDs match: $(echo ${bu})"
    else
      fail "dSYM UUID mismatch: binary [$(echo ${bu})] vs dSYM [$(echo ${du})]"
    fi
  else
    fail "no embedded dSYM at dSYMs/${FRAMEWORK_NAME}.framework.dSYM"
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "AUDIT FAILED" >&2
  exit 1
fi
echo "AUDIT PASSED"
