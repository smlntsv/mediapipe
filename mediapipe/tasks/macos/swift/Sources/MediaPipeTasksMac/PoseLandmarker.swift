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

/// Configuration for `PoseLandmarker`. Milestone 1: image mode, CPU delegate.
public final class PoseLandmarkerOptions {
    /// Absolute path to a `pose_landmarker_*.task` model file.
    public var modelPath: String
    /// Maximum number of poses to detect.
    public var numPoses: Int
    public var minPoseDetectionConfidence: Float
    public var minPosePresenceConfidence: Float
    public var minTrackingConfidence: Float
    /// Accelerator to run on. Default `.cpu`.
    public var delegate: MediaPipeDelegate
    /// Running mode. Default `.image`.
    public var runningMode: RunningMode

    public init(modelPath: String = "",
                numPoses: Int = 1,
                minPoseDetectionConfidence: Float = 0.5,
                minPosePresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5,
                delegate: MediaPipeDelegate = .cpu,
                runningMode: RunningMode = .image) {
        self.modelPath = modelPath
        self.numPoses = numPoses
        self.minPoseDetectionConfidence = minPoseDetectionConfidence
        self.minPosePresenceConfidence = minPosePresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.delegate = delegate
        self.runningMode = runningMode
    }
}

/// Result of pose landmark detection. Each top-level element corresponds to one
/// detected pose. Mirrors the MediaPipe Tasks `PoseLandmarkerResult` shape.
public struct PoseLandmarkerResult: Sendable {
    /// Pose landmarks in normalized image coordinates, per pose.
    public var landmarks: [[NormalizedLandmark]]
    /// Pose landmarks in world coordinates (meters), per pose.
    public var worldLandmarks: [[Landmark]]
    /// Segmentation masks, per pose. `nil` until mask output is implemented
    /// (not yet supported in milestone 1).
    public var segmentationMasks: [MPMask]?

    init(_ result: MPCPoseLandmarkerResult) {
        landmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        worldLandmarks = result.worldLandmarks.map { $0.map(Landmark.init) }
        // Masks are not requested in milestone 1.
        segmentationMasks = nil
    }
}

/// Detects pose landmarks on images (IMAGE mode) or video frames (VIDEO mode).
public final class PoseLandmarker {
    private let impl: MPCPoseLandmarker
    private let runningMode: RunningMode

    public init(options: PoseLandmarkerOptions) throws {
        try checkDelegate(options.delegate)
        runningMode = options.runningMode
        impl = try MPCPoseLandmarker(
            modelPath: options.modelPath,
            numPoses: options.numPoses,
            minPoseDetectionConfidence: options.minPoseDetectionConfidence,
            minPosePresenceConfidence: options.minPosePresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence,
            delegate: options.delegate.cValue,
            runningMode: options.runningMode.cValue)
    }

    /// Runs pose landmark detection on a still `CGImage`. Requires `.image` mode.
    public func detect(cgImage: CGImage,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> PoseLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(cgImage:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(cgImage: cgImage)
        return PoseLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees)))
    }

    /// Runs pose landmark detection on a video frame. Requires `.video` mode.
    /// Timestamps must be monotonically increasing.
    public func detectForVideo(cgImage: CGImage,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> PoseLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(cgImage:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(cgImage: cgImage)
        return PoseLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds)))
    }

    /// Runs pose landmark detection on a `CVPixelBuffer` (`kCVPixelFormatType_32BGRA`).
    /// Requires `.image` mode.
    public func detect(pixelBuffer: CVPixelBuffer,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> PoseLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(pixelBuffer:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return PoseLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees)))
    }

    /// Runs pose landmark detection on a `CVPixelBuffer` video frame
    /// (`kCVPixelFormatType_32BGRA`). Requires `.video` mode.
    public func detectForVideo(pixelBuffer: CVPixelBuffer,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> PoseLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(pixelBuffer:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return PoseLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds)))
    }
}
