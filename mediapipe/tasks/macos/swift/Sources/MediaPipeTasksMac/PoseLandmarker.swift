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

import CoreGraphics
import Foundation
import MediaPipeTasksObjC

/// Configuration for `PoseLandmarker`. Milestone 1: image mode, CPU delegate.
public final class PoseLandmarkerOptions {
    /// Absolute path to a `pose_landmarker_*.task` model file.
    public var modelPath: String
    /// Maximum number of poses to detect.
    public var numPoses: Int
    public var minPoseDetectionConfidence: Float
    public var minPosePresenceConfidence: Float
    public var minTrackingConfidence: Float

    public init(modelPath: String = "",
                numPoses: Int = 1,
                minPoseDetectionConfidence: Float = 0.5,
                minPosePresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5) {
        self.modelPath = modelPath
        self.numPoses = numPoses
        self.minPoseDetectionConfidence = minPoseDetectionConfidence
        self.minPosePresenceConfidence = minPosePresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
    }
}

/// Result of pose landmark detection. Each top-level element corresponds to one
/// detected pose.
public struct PoseLandmarkerResult: Sendable {
    /// Pose landmarks in normalized image coordinates, per pose.
    public var landmarks: [[NormalizedLandmark]]
    /// Pose landmarks in world coordinates (meters), per pose.
    public var worldLandmarks: [[Landmark]]

    init(_ result: MPCPoseLandmarkerResult) {
        landmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        worldLandmarks = result.worldLandmarks.map { $0.map(Landmark.init) }
    }
}

/// Detects pose landmarks on still images.
public final class PoseLandmarker {
    private let impl: MPCPoseLandmarker

    public init(options: PoseLandmarkerOptions) throws {
        impl = try MPCPoseLandmarker(
            modelPath: options.modelPath,
            numPoses: options.numPoses,
            minPoseDetectionConfidence: options.minPoseDetectionConfidence,
            minPosePresenceConfidence: options.minPosePresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence)
    }

    /// Runs pose landmark detection on a `CGImage`.
    public func detect(cgImage: CGImage) throws -> PoseLandmarkerResult {
        let image = try MPCImage(cgImage: cgImage)
        let result = try impl.detect(image)
        return PoseLandmarkerResult(result)
    }
}
