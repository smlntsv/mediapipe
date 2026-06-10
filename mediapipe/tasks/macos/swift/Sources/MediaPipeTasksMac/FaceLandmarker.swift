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
import CoreVideo
import Foundation
import MediaPipeTasksObjC

/// Configuration for `FaceLandmarker`.
public final class FaceLandmarkerOptions {
    /// Absolute path to a `face_landmarker.task` model file.
    public var modelPath: String
    /// Maximum number of faces to detect.
    public var numFaces: Int
    public var minFaceDetectionConfidence: Float
    public var minFacePresenceConfidence: Float
    public var minTrackingConfidence: Float
    /// Whether to output face blendshapes.
    public var outputFaceBlendshapes: Bool
    /// Whether to output facial transformation matrices.
    public var outputFacialTransformationMatrixes: Bool
    /// Accelerator to run on. Default `.cpu`.
    public var delegate: MediaPipeDelegate
    /// Running mode. Default `.image`.
    public var runningMode: RunningMode

    public init(modelPath: String = "",
                numFaces: Int = 1,
                minFaceDetectionConfidence: Float = 0.5,
                minFacePresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5,
                outputFaceBlendshapes: Bool = false,
                outputFacialTransformationMatrixes: Bool = false,
                delegate: MediaPipeDelegate = .cpu,
                runningMode: RunningMode = .image) {
        self.modelPath = modelPath
        self.numFaces = numFaces
        self.minFaceDetectionConfidence = minFaceDetectionConfidence
        self.minFacePresenceConfidence = minFacePresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.outputFaceBlendshapes = outputFaceBlendshapes
        self.outputFacialTransformationMatrixes = outputFacialTransformationMatrixes
        self.delegate = delegate
        self.runningMode = runningMode
    }
}

/// Result of face landmark detection. Each top-level element corresponds to one
/// detected face (478 landmarks each for the standard model). Mirrors the
/// MediaPipe Tasks `FaceLandmarkerResult` shape.
public struct FaceLandmarkerResult: Sendable {
    /// Face landmarks in normalized image coordinates, per face.
    public var faceLandmarks: [[NormalizedLandmark]]
    /// Face blendshapes, per face. Empty unless
    /// `FaceLandmarkerOptions.outputFaceBlendshapes` is set.
    public var faceBlendshapes: [Classifications]
    /// Facial transformation matrices, per face. Empty unless
    /// `FaceLandmarkerOptions.outputFacialTransformationMatrixes` is set.
    ///
    /// Note the MediaPipe Web spelling "Matrixes" is preserved intentionally.
    public var facialTransformationMatrixes: [Matrix]

    init(_ result: MPCFaceLandmarkerResult) {
        faceLandmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        faceBlendshapes = result.blendshapes.map {
            Classifications(categories: $0.categories.map(Category.init),
                            headIndex: $0.headIndex,
                            headName: $0.headName)
        }
        facialTransformationMatrixes = result.transformationMatrixes.map {
            Matrix(rows: $0.rows, columns: $0.columns, data: $0.data.map(\.floatValue))
        }
    }
}

/// Detects face landmarks on images (IMAGE mode) or video frames (VIDEO mode).
public final class FaceLandmarker {
    private let impl: MPCFaceLandmarker
    private let runningMode: RunningMode

    public init(options: FaceLandmarkerOptions) throws {
        try checkDelegate(options.delegate)
        runningMode = options.runningMode
        impl = try MPCFaceLandmarker(
            modelPath: options.modelPath,
            numFaces: options.numFaces,
            minFaceDetectionConfidence: options.minFaceDetectionConfidence,
            minFacePresenceConfidence: options.minFacePresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence,
            outputFaceBlendshapes: options.outputFaceBlendshapes,
            outputFacialTransformationMatrixes: options.outputFacialTransformationMatrixes,
            delegate: options.delegate.cValue,
            runningMode: options.runningMode.cValue)
    }

    /// Runs face landmark detection on a still `CGImage`. Requires `.image` mode.
    public func detect(cgImage: CGImage) throws -> FaceLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(cgImage:)")
        let image = try MPCImage(cgImage: cgImage)
        return FaceLandmarkerResult(try impl.detect(image))
    }

    /// Runs face landmark detection on a video frame. Requires `.video` mode.
    /// Timestamps must be monotonically increasing.
    public func detectForVideo(cgImage: CGImage,
                               timestampInMilliseconds: Int) throws -> FaceLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(cgImage:timestampInMilliseconds:)")
        let image = try MPCImage(cgImage: cgImage)
        return FaceLandmarkerResult(
            try impl.detect(forVideoImage: image, timestampMs: Int64(timestampInMilliseconds)))
    }

    /// Runs face landmark detection on a `CVPixelBuffer` (`kCVPixelFormatType_32BGRA`).
    /// Requires `.image` mode.
    public func detect(pixelBuffer: CVPixelBuffer) throws -> FaceLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(pixelBuffer:)")
        return FaceLandmarkerResult(try impl.detect(try MPCImage(pixelBuffer: pixelBuffer)))
    }

    /// Runs face landmark detection on a `CVPixelBuffer` video frame
    /// (`kCVPixelFormatType_32BGRA`). Requires `.video` mode.
    public func detectForVideo(pixelBuffer: CVPixelBuffer,
                               timestampInMilliseconds: Int) throws -> FaceLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(pixelBuffer:timestampInMilliseconds:)")
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return FaceLandmarkerResult(
            try impl.detect(forVideoImage: image, timestampMs: Int64(timestampInMilliseconds)))
    }
}
