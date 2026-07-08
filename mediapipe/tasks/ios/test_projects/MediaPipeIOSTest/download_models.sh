#!/usr/bin/env bash
# Fetches the models the app bundles at build time (.task + converted Core ML
# <sha>.mlmodelc) into the SHARED test-models dir the "Copy MediaPipe Models"
# build phase reads from. Run once on a fresh checkout before building.
#
# The models live in the shared macOS/iOS test dir (not duplicated here); this
# is a thin wrapper around that shared downloader. Override COREML_MODELS_URL /
# COREML_MODELS_SHA256 to point at your own converted Core ML set.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../../../macos/test_projects/shared/download_models.sh" "$@"
