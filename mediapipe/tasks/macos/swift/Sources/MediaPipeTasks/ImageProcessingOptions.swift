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

/// Per-call image preprocessing applied before inference. Mirrors the rotation
/// field of the MediaPipe C `MpImageProcessingOptions`.
///
/// The default value (no rotation) reproduces the previous behaviour, so
/// existing call sites are unaffected.
///
/// Note: the `region_of_interest` field of the underlying C struct is *not*
/// exposed here. The HandLandmarker, FaceLandmarker and PoseLandmarker tasks all
/// reject a region of interest (`roi_allowed = false` in the native task), so a
/// region-of-interest knob would always fail. Only rotation is supported.
public struct ImageProcessingOptions: Sendable, Equatable {
    /// Clockwise rotation, in degrees, to apply to the image before running the
    /// model. Must be a multiple of 90 (e.g. `0`, `90`, `180`, `270`, `-90`).
    ///
    /// Set this when the camera is physically mounted at an angle and frames
    /// arrive rotated: the model then runs on an internally-uprighted image. The
    /// returned landmarks remain in the supplied image's coordinate system.
    public var rotationDegrees: Int

    public init(rotationDegrees: Int = 0) {
        self.rotationDegrees = rotationDegrees
    }

    /// Throws `MediaPipeError.invalidArgument` if the options are malformed.
    func validated() throws {
        if rotationDegrees % 90 != 0 {
            throw MediaPipeError.invalidArgument(
                "rotationDegrees must be a multiple of 90 (got \(rotationDegrees)).")
        }
    }
}
