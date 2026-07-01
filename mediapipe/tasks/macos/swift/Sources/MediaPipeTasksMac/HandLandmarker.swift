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

/// Errors raised by the Swift layer of MediaPipeTasksMac. Errors originating in
/// the native MediaPipe runtime surface as `NSError` from throwing calls.
public enum MediaPipeError: Error, Equatable {
    case invalidImage(String)
    /// The requested delegate is not available in the linked artifact (e.g.
    /// `.gpu` against the CPU-only macOS build). The wrapper never silently
    /// falls back to CPU.
    case unsupportedDelegate(String)
    /// A detection method was called that does not match the configured
    /// `runningMode` (e.g. `detect(cgImage:)` on a `.video` landmarker).
    case invalidRunningMode(String)
    /// A caller-supplied argument was invalid (e.g. a `rotationDegrees` that is
    /// not a multiple of 90 in `ImageProcessingOptions`).
    case invalidArgument(String)
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
    /// Scale factor for the frame-to-frame region of interest derived from the
    /// previous frame's hand landmarks. Larger values (e.g. 2.5–3.0) expand the
    /// tracking crop so a fast-moving hand stays inside it between frames, at the
    /// cost of a looser crop. Default 2.0 (MediaPipe's historical value).
    public var roiScale: Float
    /// VIDEO mode only ("frames without a hand" grace): number of consecutive
    /// frames a vanished hand's region of interest is kept in the tracking loop
    /// before its track is dropped. While held, the palm detector stays skipped
    /// and the landmark model keeps retrying the last-known region, so a hand
    /// lost to momentary motion blur is re-acquired quickly and cheaply —
    /// without lowering `minHandPresenceConfidence`. Keep small (2–3 frames,
    /// ~70–100 ms at 30 fps): a truly departed hand occupies a tracking slot
    /// for this many frames before the detector resumes looking for new hands.
    /// Default 0 (drop immediately, MediaPipe's original behavior).
    public var trackingGraceFrames: Int
    /// Accelerator to run on. Default `.cpu`.
    public var delegate: MediaPipeDelegate
    /// Running mode. Default `.image`.
    public var runningMode: RunningMode

    public init(modelPath: String = "",
                numHands: Int = 1,
                minHandDetectionConfidence: Float = 0.5,
                minHandPresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5,
                roiScale: Float = 2.0,
                trackingGraceFrames: Int = 0,
                delegate: MediaPipeDelegate = .cpu,
                runningMode: RunningMode = .image) {
        self.modelPath = modelPath
        self.numHands = numHands
        self.minHandDetectionConfidence = minHandDetectionConfidence
        self.minHandPresenceConfidence = minHandPresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.roiScale = roiScale
        self.trackingGraceFrames = trackingGraceFrames
        self.delegate = delegate
        self.runningMode = runningMode
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

/// Detects hand landmarks on images (IMAGE mode) or video frames (VIDEO mode).
public final class HandLandmarker {
    private let impl: MPCHandLandmarker
    private let runningMode: RunningMode

    public init(options: HandLandmarkerOptions) throws {
        try checkDelegate(options.delegate)
        runningMode = options.runningMode
        impl = try MPCHandLandmarker(
            modelPath: options.modelPath,
            numHands: options.numHands,
            minHandDetectionConfidence: options.minHandDetectionConfidence,
            minHandPresenceConfidence: options.minHandPresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence,
            roiScale: options.roiScale,
            trackingGraceFrames: options.trackingGraceFrames,
            delegate: options.delegate.cValue,
            runningMode: options.runningMode.cValue)
    }

    /// Runs hand landmark detection on a still `CGImage`. Requires `.image` mode.
    public func detect(cgImage: CGImage,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> HandLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(cgImage:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(cgImage: cgImage)
        return HandLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees)))
    }

    /// Runs hand landmark detection on a video frame. Requires `.video` mode.
    /// Timestamps must be monotonically increasing.
    public func detectForVideo(cgImage: CGImage,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> HandLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(cgImage:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(cgImage: cgImage)
        return HandLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds)))
    }

    /// Runs hand landmark detection on a `CVPixelBuffer` (`kCVPixelFormatType_32BGRA`).
    /// Requires `.image` mode.
    public func detect(pixelBuffer: CVPixelBuffer,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> HandLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(pixelBuffer:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return HandLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees)))
    }

    /// Runs hand landmark detection on a `CVPixelBuffer` video frame
    /// (`kCVPixelFormatType_32BGRA`). Requires `.video` mode.
    public func detectForVideo(pixelBuffer: CVPixelBuffer,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions())
        throws -> HandLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(pixelBuffer:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return HandLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds)))
    }
}
