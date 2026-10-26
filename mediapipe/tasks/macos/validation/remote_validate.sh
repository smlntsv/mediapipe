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
# Deploys a validation bundle to a remote Mac over SSH, runs the matrix there,
# and copies the results back next to the local ones:
#
#   ./remote_validate.sh <ssh-destination> [run_validation.sh args...]
#   ./remote_validate.sh intel-mac                 # Host alias from ~/.ssh/config
#   ./remote_validate.sh dima@10.0.0.42 --quick
#
# Env: BUNDLE (default /tmp/mp_build/mp-validation.tar.gz), RESULTS (local
# results root, default /tmp/mp_build/results), REMOTE_DIR (default
# mp-validation-run, relative to the remote home).
#
# Only the bundle is copied; nothing is installed on the remote machine and
# nothing outside REMOTE_DIR is modified. Uses your normal ssh/scp config.

set -euo pipefail

DEST="${1:?usage: $0 <ssh-destination> [run_validation.sh args...]}"
shift
BUNDLE="${BUNDLE:-/tmp/mp_build/mp-validation.tar.gz}"
RESULTS="${RESULTS:-/tmp/mp_build/results}"
REMOTE_DIR="${REMOTE_DIR:-mp-validation-run}"
[[ -f "${BUNDLE}" ]] || { echo "error: ${BUNDLE} not found (run package_bundle.sh)" >&2; exit 1; }

echo "==> Remote host facts"
ssh "${DEST}" 'sw_vers; uname -m; sysctl -n machdep.cpu.brand_string hw.model; system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Metal" || true'

echo "==> Copying bundle ($(du -h "${BUNDLE}" | awk '{print $1}'))"
ssh "${DEST}" "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}'"
scp -q "${BUNDLE}" "${DEST}:${REMOTE_DIR}/bundle.tar.gz"
ssh "${DEST}" "cd '${REMOTE_DIR}' && tar -xzf bundle.tar.gz && cd mp-validation && shasum -a 256 -c <(sed -n 's|^\([0-9a-f]\{64\}\)  \./|\1  |p' MANIFEST.txt) >/dev/null && echo 'bundle integrity OK'"

echo "==> Running validation on ${DEST}"
set +e
# shellcheck disable=SC2029  # args are intentionally expanded locally
ssh "${DEST}" "cd '${REMOTE_DIR}/mp-validation' && ./run_validation.sh $*"
status=$?
set -e

echo "==> Fetching results"
mkdir -p "${RESULTS}"
scp -q -r "${DEST}:${REMOTE_DIR}/mp-validation/results/*" "${RESULTS}/"
echo "    results in ${RESULTS}: $(ls "${RESULTS}" | xargs)"
echo "    compare: python3 -I $(dirname "$0")/compare_reports.py ${RESULTS}/native-arm64 ${RESULTS}/native-x86_64"
exit "${status}"
