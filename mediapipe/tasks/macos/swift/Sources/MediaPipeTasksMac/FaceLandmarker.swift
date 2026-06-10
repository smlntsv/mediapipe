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

/// Configuration for `FaceLandmarker`. Milestone 1: image mode, CPU delegate,
/// landmarks only (blendshapes / transformation matrices are not yet exposed).
public final class FaceLandmarkerOptions {
    /// Absolute path to a `face_landmarker.task` model file.
    public var modelPath: String
    /// Maximum number of faces to detect.
    public var numFaces: Int
    public var minFaceDetectionConfidence: Float
    public var minFacePresenceConfidence: Float
    public var minTrackingConfidence: Float

    public init(modelPath: String = "",
                numFaces: Int = 1,
                minFaceDetectionConfidence: Float = 0.5,
                minFacePresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5) {
        self.modelPath = modelPath
        self.numFaces = numFaces
        self.minFaceDetectionConfidence = minFaceDetectionConfidence
        self.minFacePresenceConfidence = minFacePresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
    }
}

/// Result of face landmark detection. Each top-level element corresponds to one
/// detected face (478 landmarks each for the standard model). Mirrors the
/// MediaPipe Tasks `FaceLandmarkerResult` shape.
public struct FaceLandmarkerResult: Sendable {
    /// Face landmarks in normalized image coordinates, per face.
    public var faceLandmarks: [[NormalizedLandmark]]
    /// Face blendshapes, per face. Empty unless blendshape output is enabled
    /// (not yet supported in milestone 1).
    public var faceBlendshapes: [Classifications]
    /// Facial transformation matrices, per face. Empty unless matrix output is
    /// enabled (not yet supported in milestone 1).
    ///
    /// Note the MediaPipe Web spelling "Matrixes" is preserved intentionally.
    public var facialTransformationMatrixes: [Matrix]

    init(_ result: MPCFaceLandmarkerResult) {
        faceLandmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        // Blendshapes / transformation matrices are not requested in milestone 1.
        faceBlendshapes = []
        facialTransformationMatrixes = []
    }
}

/// Detects face landmarks on still images.
public final class FaceLandmarker {
    private let impl: MPCFaceLandmarker

    public init(options: FaceLandmarkerOptions) throws {
        impl = try MPCFaceLandmarker(
            modelPath: options.modelPath,
            numFaces: options.numFaces,
            minFaceDetectionConfidence: options.minFaceDetectionConfidence,
            minFacePresenceConfidence: options.minFacePresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence)
    }

    /// Runs face landmark detection on a `CGImage`.
    public func detect(cgImage: CGImage) throws -> FaceLandmarkerResult {
        let image = try MPCImage(cgImage: cgImage)
        let result = try impl.detect(image)
        return FaceLandmarkerResult(result)
    }
}
