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
# Bump every consumer app's MediaPipe SwiftPM pin to a release tag — WITHOUT
# opening Xcode. Point it at one or more directories containing .xcodeproj
# bundles (it scans recursively):
#
#   ./bump_package_pin.sh v1.0.0-apple.3 ~/Documents/projects/staxel-third ~/Documents/projects/staxel-4/ScreenBar
#
# Why this exists: Package.resolved is a LOCKFILE — Xcode rewrites it to
# satisfy the authoritative pin in project.pbxproj (XCRemoteSwiftPackageReference
# → requirement). Editing Package.resolved by hand silently reverts on the next
# resolution; the pbxproj is what must change. This script:
#   1. resolves the tag to its commit SHA via `git ls-remote` (annotated tags
#      are peeled to the commit),
#   2. patches the mediapipe requirement in each project.pbxproj that
#      references the repo — `kind = revision` pins get the new SHA,
#      `kind = exactVersion` pins get the new version string,
#   3. regenerates each project's Package.resolved with
#      `xcodebuild -resolvePackageDependencies` (downloads the release zip and
#      validates the binary-target checksum in the process).
# Commit the changed project.pbxproj AND Package.resolved together.
#
# Flags: --no-resolve skips step 3 (e.g. offline; resolution then happens on
# next build/open, from the already-correct pbxproj pin).

set -euo pipefail

REPO_URL="https://github.com/smlntsv/mediapipe.git"

RESOLVE=1
if [[ "${1:-}" == "--no-resolve" ]]; then RESOLVE=0; shift; fi
TAG="${1:?usage: $0 [--no-resolve] <tag, e.g. v1.0.0-apple.3> <dir>...}"
shift
[[ $# -ge 1 ]] || { echo "usage: $0 [--no-resolve] <tag> <dir>..." >&2; exit 1; }

VERSION="${TAG#v}"

echo "==> Resolving ${TAG} on ${REPO_URL}..."
LS="$(git ls-remote "${REPO_URL}" "refs/tags/${TAG}" "refs/tags/${TAG}^{}")"
# Prefer the peeled (^{}) commit of an annotated tag; fall back to the tag ref.
SHA="$(awk '/\^\{\}$/{print $1; found=1} END{exit !found}' <<< "${LS}" || awk 'NR==1{print $1}' <<< "${LS}")"
if [[ -z "${SHA}" ]]; then
  echo "error: tag ${TAG} not found on ${REPO_URL}" >&2
  exit 1
fi
echo "    ${TAG} -> ${SHA} (version ${VERSION})"

PATCHED=0
while IFS= read -r pbxproj; do
  proj_dir="$(dirname "${pbxproj}")"
  before="$(shasum "${pbxproj}")"
  # Scope both substitutions to the requirement block that follows the
  # mediapipe repositoryURL, so pins of other packages are never touched.
  perl -0pi -e '
    s{(repositoryURL = "\Q'"${REPO_URL}"'\E";\s*requirement = \{\s*kind = revision;\s*revision = )[0-9a-f]{40}}{${1}'"${SHA}"'}s;
    s{(repositoryURL = "\Q'"${REPO_URL}"'\E";\s*requirement = \{\s*kind = exactVersion;\s*version = ")[^"]+}{${1}'"${VERSION}"'}s;
  ' "${pbxproj}"
  if [[ "$(shasum "${pbxproj}")" == "${before}" ]]; then
    echo "--  ${proj_dir}: pin already current (or unrecognized requirement kind) — skipped"
    continue
  fi
  echo "==> ${proj_dir}: pin updated"
  PATCHED=$((PATCHED + 1))
  if [[ "${RESOLVE}" == "1" ]]; then
    ( cd "$(dirname "${proj_dir}")" \
      && xcodebuild -resolvePackageDependencies -project "$(basename "${proj_dir}")" >/dev/null \
      && echo "    Package.resolved regenerated" ) \
      || { echo "error: resolution failed for ${proj_dir}" >&2; exit 1; }
  fi
done < <(grep -rl "smlntsv/mediapipe" "$@" --include=project.pbxproj 2>/dev/null)

if [[ "${PATCHED}" == "0" ]]; then
  echo "warning: no project.pbxproj was changed — check the paths given" >&2
fi
echo "==> Done: ${PATCHED} project(s) bumped to ${TAG}. Commit project.pbxproj + Package.resolved together."
