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

/// Throws `unsupportedDelegate` if `.gpu` is requested against a CPU-only build.
func checkDelegate(_ delegate: MediaPipeDelegate) throws {
    if delegate == .gpu && !mediaPipeGPUArtifactAvailable {
        throw MediaPipeError.unsupportedDelegate(
            "The GPU delegate is not available in this macOS artifact, which is "
            + "built CPU-only (MEDIAPIPE_DISABLE_GPU=1). Use .cpu, or build and "
            + "ship a GPU-capable MediaPipeTasksC.xcframework.")
    }
}

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
