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

/// Errors raised by the Swift layer of MediaPipeTasksMac. Errors originating in
/// the native MediaPipe runtime surface as `NSError` from throwing calls.
public enum MediaPipeError: Error {
    case invalidImage(String)
}

/// Configuration for `HandLandmarker`. Milestone 1: image mode, CPU delegate.
public final class HandLandmarkerOptions {
    /// Absolute path to a `hand_landmarker.task` model file.
    public var modelPath: String
    /// Maximum number of hands to detect.
    public var numHands: Int
    public var minHandDetectionConfidence: Float
    public var minHandPresenceConfidence: Float
    public var minTrackingConfidence: Float

    public init(modelPath: String = "",
                numHands: Int = 1,
                minHandDetectionConfidence: Float = 0.5,
                minHandPresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5) {
        self.modelPath = modelPath
        self.numHands = numHands
        self.minHandDetectionConfidence = minHandDetectionConfidence
        self.minHandPresenceConfidence = minHandPresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
    }
}

/// Result of hand landmark detection. Each top-level element corresponds to one
/// detected hand. Mirrors the MediaPipe Tasks `HandLandmarkerResult` shape.
public struct HandLandmarkerResult: Sendable {
    /// Hand landmarks in normalized image coordinates, per hand.
    public var landmarks: [[NormalizedLandmark]]
    /// Hand landmarks in world coordinates (meters), per hand.
    public var worldLandmarks: [[Landmark]]
    /// Handedness classification, per hand.
    public var handedness: [[Category]]

    /// Deprecated alias for `handedness`, kept for parity with older MediaPipe
    /// result shapes.
    @available(*, deprecated, renamed: "handedness")
    public var handednesses: [[Category]] { handedness }

    init(_ result: MPCHandLandmarkerResult) {
        landmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        worldLandmarks = result.worldLandmarks.map { $0.map(Landmark.init) }
        handedness = result.handedness.map { $0.map(Category.init) }
    }
}

/// Detects hand landmarks on still images.
public final class HandLandmarker {
    private let impl: MPCHandLandmarker

    public init(options: HandLandmarkerOptions) throws {
        impl = try MPCHandLandmarker(
            modelPath: options.modelPath,
            numHands: options.numHands,
            minHandDetectionConfidence: options.minHandDetectionConfidence,
            minHandPresenceConfidence: options.minHandPresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence)
    }

    /// Runs hand landmark detection on a `CGImage`.
    public func detect(cgImage: CGImage) throws -> HandLandmarkerResult {
        let image = try MPCImage(cgImage: cgImage)
        let result = try impl.detect(image)
        return HandLandmarkerResult(result)
    }
}
