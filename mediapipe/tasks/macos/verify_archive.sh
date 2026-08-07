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
# Consumer-side release gate for apps embedding MediaPipeTasksC.framework.
# Run it against a local .xcarchive BEFORE uploading to App Store Connect /
# TestFlight / Sparkle:
#
#   xcodebuild archive -scheme <Scheme> -destination 'generic/platform=iOS' \
#     -archivePath /tmp/app-test.xcarchive
#   ./verify_archive.sh /tmp/app-test.xcarchive
#
# Checks (each one maps to a real shipped incident or App Store check):
#   1. PORTABILITY — the embedded framework must not reference /opt/local,
#      /opt/homebrew or /usr/local dylibs. A framework built without
#      MP_BUNDLE_DEPS=1 references the build machine's MacPorts OpenCV and
#      dyld-crashes on every machine without it (ScreenBar build 27,
#      fleet-wide, died before main()).
#   2. dSYM — the archive must contain MediaPipeTasksC.framework.dSYM whose
#      UUIDs exactly match the embedded binary. This is the same check App
#      Store Connect runs ("Upload Symbols Failed ... did not include a dSYM
#      ... with the UUIDs"); presence alone is not enough, a dSYM from a
#      different build fails on UUID mismatch.
#
# Exit status: 0 = all checks pass, 1 = at least one failed.

set -euo pipefail

ARCHIVE="${1:?usage: $0 /path/to/App.xcarchive}"
FRAMEWORK_NAME="MediaPipeTasksC"
FAIL=0

if [[ ! -d "${ARCHIVE}/Products" ]]; then
  echo "error: ${ARCHIVE} does not look like an .xcarchive (no Products/)" >&2
  exit 1
fi

# The framework binary inside the archived app. iOS apps embed it flat
# (Frameworks/X.framework/X); macOS apps embed the versioned bundle
# (Contents/Frameworks/X.framework/Versions/A/X). -type f skips the
# macOS-framework symlinks; bundled OpenCV dylibs under Libraries/ are not
# the framework binary.
BIN="$(find "${ARCHIVE}/Products" -type f -name "${FRAMEWORK_NAME}" \
  -path "*/${FRAMEWORK_NAME}.framework/*" -not -path "*/Libraries/*" | head -1)"
if [[ -z "${BIN}" ]]; then
  echo "FAIL: no embedded ${FRAMEWORK_NAME}.framework binary found in ${ARCHIVE}/Products" >&2
  exit 1
fi
echo "embedded binary: ${BIN#"${ARCHIVE}"/}"

# --- 1. Portability ----------------------------------------------------------
NONPORTABLE="$(otool -L "${BIN}" | tail -n +2 | sed 's/^[[:space:]]*//' | awk '{print $1}' \
  | grep -vE '^/System/|^/usr/lib/|^@rpath|^@loader_path|^@executable_path' || true)"
if [[ -n "${NONPORTABLE}" ]]; then
  echo "FAIL: non-portable dylib references (crashes at dyld time off the build machine):" >&2
  echo "${NONPORTABLE}" | sed 's/^/    /' >&2
  FAIL=1
else
  echo "PASS: portability — no /opt/local, /opt/homebrew or /usr/local references"
fi

# --- 2. dSYM presence + UUID match ------------------------------------------
DSYM="${ARCHIVE}/dSYMs/${FRAMEWORK_NAME}.framework.dSYM"
BIN_UUIDS="$(dwarfdump --uuid "${BIN}" | awk '{print $2}' | sort)"
if [[ ! -d "${DSYM}" ]]; then
  echo "FAIL: no ${FRAMEWORK_NAME}.framework.dSYM in ${ARCHIVE}/dSYMs/" >&2
  echo "      (App Store Connect: 'Upload Symbols Failed'. The xcframework release" >&2
  echo "      must embed per-slice dSYMs; binary UUIDs: $(echo "${BIN_UUIDS}" | tr '\n' ' '))" >&2
  FAIL=1
else
  DSYM_UUIDS="$(dwarfdump --uuid "${DSYM}" 2>/dev/null | awk '{print $2}' | sort)"
  if [[ "${BIN_UUIDS}" != "${DSYM_UUIDS}" ]]; then
    echo "FAIL: dSYM UUID mismatch (dSYM is from a different build):" >&2
    echo "    binary: $(echo "${BIN_UUIDS}"  | tr '\n' ' ')" >&2
    echo "    dSYM:   $(echo "${DSYM_UUIDS}" | tr '\n' ' ')" >&2
    FAIL=1
  else
    echo "PASS: dSYM present, UUIDs match ($(echo "${BIN_UUIDS}" | tr '\n' ' '))"
    # Informational: does the dSYM carry line tables (file:line symbolication)
    # or only a remapped symbol table (function names only)?
    if dwarfdump --debug-line "${DSYM}" 2>/dev/null | head -50 | grep -q 'file_names\|include_directories'; then
      echo "info: dSYM has line tables (file:line symbolication available)"
    else
      echo "info: dSYM is symbol-only (function-level symbolication; no file:line)"
    fi
  fi
fi

exit "${FAIL}"
