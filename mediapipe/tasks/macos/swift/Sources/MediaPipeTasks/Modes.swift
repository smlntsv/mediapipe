// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
#if arch(x86_64) && os(macOS)
import Metal
#endif

/// The accelerator a task runs on.
public enum MediaPipeDelegate: String, Sendable, Codable {
    case cpu = "CPU"
    case gpu = "GPU"
    /// Core ML inference — the only route to the Apple Neural Engine. Requires
    /// pre-converted `<sha256-of-tflite>.mlmodelc` models (see
    /// `mediapipe/tasks/macos/experiments/coreml_ane/convert_delegate.py`) in the
    /// options' `coreMLModelCacheDirectory` (defaults to the `.task` file's
    /// directory). Models without a converted counterpart transparently fall
    /// back to TFLite CPU/XNNPACK with a logged warning.
    case coreML = "COREML"

    /// The MediaPipe C `MpDelegate` integer value.
    var cValue: Int32 {
        switch self {
        case .cpu: return 0     // MP_DELEGATE_CPU
        case .gpu: return 1     // MP_DELEGATE_GPU
        case .coreML: return 3  // MP_DELEGATE_COREML
        }
    }
}

/// The running mode a task is configured for.
public enum RunningMode: String, Sendable, Codable {
    case image = "IMAGE"
    case video = "VIDEO"

    /// The MediaPipe C `MpRunningMode` integer value.
    var cValue: Int32 { self == .image ? 1 : 2 }  // MP_RUNNING_MODE_IMAGE / _VIDEO
}

/// Whether the linked macOS binary artifact supports the GPU delegate.
///
/// The default macOS `MediaPipeTasksC.xcframework` is built CPU-only
/// (`--define MEDIAPIPE_DISABLE_GPU=1`), so requesting `.gpu` throws
/// `MediaPipeError.unsupportedDelegate` rather than silently running on CPU.
/// Flip this to `true` only when shipping a GPU-capable artifact (built with
/// `MP_ENABLE_GPU=1 build_macos_xcframework.sh`).
let mediaPipeGPUArtifactAvailable = true

/// Throws `unsupportedDelegate` if `.gpu` is requested against a CPU-only build,
/// or (Intel Macs only) on a GPU the Metal path has not been validated on.
func checkDelegate(_ delegate: MediaPipeDelegate) throws {
    if delegate == .gpu && !mediaPipeGPUArtifactAvailable {
        throw MediaPipeError.unsupportedDelegate(
            "The GPU delegate is not available in this macOS artifact, which is "
            + "built CPU-only (MEDIAPIPE_DISABLE_GPU=1). Use .cpu, or build and "
            + "ship a GPU-capable MediaPipeTasksC.xcframework.")
    }
    #if arch(x86_64) && os(macOS)
    if delegate == .gpu {
        try checkIntelMacGPU()
    }
    #endif
}

#if arch(x86_64) && os(macOS)
/// The MediaPipe Metal pipeline (CVPixelBuffer→Metal image conversion + the
/// TFLite Metal delegate) is validated on Apple GPUs only. On a real Intel Mac
/// (MacBookPro16,1: AMD Radeon Pro 5500M, the system default Metal device) it
/// runs without any error yet returns NO detections, so an Intel/AMD GPU is
/// rejected up front instead of silently producing empty results. Apple GPUs
/// (an x86_64 process under Rosetta) are unaffected and keep working.
/// Setting MEDIAPIPE_ALLOW_UNVALIDATED_GPU=1 skips this check (for validating
/// new hardware/driver combinations, see validation/README.md).
///
/// arm64 builds compile this out entirely (Apple-silicon Macs always have an
/// Apple GPU), so Apple-silicon behavior is unchanged.
private func checkIntelMacGPU() throws {
    if ProcessInfo.processInfo.environment["MEDIAPIPE_ALLOW_UNVALIDATED_GPU"] == "1" { return }
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw MediaPipeError.unsupportedDelegate(
            "The GPU delegate needs a Metal device, and none is available. Use .cpu.")
    }
    if !device.supportsFamily(.apple1) {
        throw MediaPipeError.unsupportedDelegate(
            "The GPU delegate is not supported on this Intel Mac's GPU "
            + "(\(device.name)): MediaPipe's Metal path is validated on Apple GPUs "
            + "only and returns no detections on Intel/AMD GPUs. Use .cpu (or "
            + ".coreML, which runs on Intel Macs without the Neural Engine).")
    }
}
#endif

/// Resolves the Core ML model-cache directory for a landmarker: the explicit
/// option when set, otherwise the directory containing the `.task` model (the
/// natural place to ship converted `<sha256>.mlmodelc` models alongside it).
/// Returns nil unless the delegate is `.coreML`.
func resolveCoreMLModelCacheDir(delegate: MediaPipeDelegate,
                                explicit: String?,
                                modelPath: String) -> String? {
    guard delegate == .coreML else { return nil }
    return explicit ?? (modelPath as NSString).deletingLastPathComponent
}

/// Throws `invalidRunningMode` if a detection method is called in the wrong mode.
func requireRunningMode(_ required: RunningMode, actual: RunningMode, method: String) throws {
    if actual != required {
        throw MediaPipeError.invalidRunningMode(
            "\(method) requires runningMode == .\(required.rawValue.lowercased()), "
            + "but the landmarker was created with .\(actual.rawValue.lowercased()).")
    }
}
