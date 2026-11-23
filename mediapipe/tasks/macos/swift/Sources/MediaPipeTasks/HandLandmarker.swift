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

/// Errors raised by the Swift layer of MediaPipeTasks. Errors originating in
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
    /// `.coreML` only: directory containing pre-converted
    /// `<sha256-of-tflite>.mlmodelc` models. Defaults to the directory of
    /// `modelPath` when nil.
    public var coreMLModelCacheDirectory: String?
    /// Running mode. Default `.image`.
    public var runningMode: RunningMode
    /// Debugging aid. When true, every result carries `stages`: the per-stage
    /// outputs of the hand pipeline for that frame (see `HandLandmarkerStages`).
    /// Default false, which leaves the pipeline and the results unchanged and
    /// `stages` nil.
    public var stageOutputs: Bool
    /// Debugging aid. When true, the detect methods accept
    /// `captureTensors: true`, and the result of such a call carries
    /// `tensorCapture`: the tensors given to the palm detection model and to
    /// the landmark model on that frame. Calls without `captureTensors` read no
    /// tensor. Default false, which leaves the pipeline and the results
    /// unchanged.
    public var tensorCapture: Bool
    /// Debugging aid. When positive, every result carries `palmCandidates`
    /// with this many of the palm detector's highest scored candidates (see
    /// `HandLandmarkerPalmCandidates`). Default 0, which leaves the pipeline
    /// and the results unchanged.
    public var palmCandidateLimit: Int
    /// Palm search regions: runs the palm detector on regions of the frame
    /// as well as, or instead of, the whole frame (see `PalmSearchOptions`).
    /// With `stageOutputs` every result then carries `palmSearch`. Default
    /// nil, which leaves the pipeline and the results unchanged.
    public var palmSearch: PalmSearchOptions?

    public init(modelPath: String = "",
                numHands: Int = 1,
                minHandDetectionConfidence: Float = 0.5,
                minHandPresenceConfidence: Float = 0.5,
                minTrackingConfidence: Float = 0.5,
                roiScale: Float = 2.0,
                trackingGraceFrames: Int = 0,
                delegate: MediaPipeDelegate = .cpu,
                coreMLModelCacheDirectory: String? = nil,
                runningMode: RunningMode = .image,
                stageOutputs: Bool = false,
                tensorCapture: Bool = false,
                palmCandidateLimit: Int = 0,
                palmSearch: PalmSearchOptions? = nil) {
        self.modelPath = modelPath
        self.numHands = numHands
        self.minHandDetectionConfidence = minHandDetectionConfidence
        self.minHandPresenceConfidence = minHandPresenceConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.roiScale = roiScale
        self.trackingGraceFrames = trackingGraceFrames
        self.delegate = delegate
        self.coreMLModelCacheDirectory = coreMLModelCacheDirectory
        self.runningMode = runningMode
        self.stageOutputs = stageOutputs
        self.tensorCapture = tensorCapture
        self.palmCandidateLimit = palmCandidateLimit
        self.palmSearch = palmSearch
    }
}

/// A rotated rectangle in normalized image coordinates, as used for the hand
/// pipeline's regions of interest.
public struct StageRect: Sendable, Equatable {
    /// Center, normalized by the image width / height.
    public var xCenter: Float
    public var yCenter: Float
    /// Size, normalized by the image width / height.
    public var width: Float
    public var height: Float
    /// Rotation around the center in radians, clockwise.
    public var rotation: Float
    /// Id assigned by the tracking loop, nil if the rect has none. Only
    /// `HandLandmarkerStages.landmarkROIs` in `.video` mode carry one, and it
    /// is reassigned on every frame: it does not identify a hand across frames.
    public var rectId: Int64?

    init(_ rect: MPCRotatedRect) {
        xCenter = rect.xCenter
        yCenter = rect.yCenter
        width = rect.width
        height = rect.height
        rotation = rect.rotation
        rectId = rect.hasRectId ? rect.rectId : nil
    }
}

/// A point in normalized image coordinates.
public struct StagePoint: Sendable, Equatable {
    public var x: Float
    public var y: Float
}

/// A palm detection: a float bounding box, its score and keypoints, in
/// normalized image coordinates.
public struct StageDetection: Sendable, Equatable {
    public var score: Float
    public var xMin: Float
    public var yMin: Float
    public var width: Float
    public var height: Float
    public var keypoints: [StagePoint]

    init(_ detection: MPCNormalizedDetection) {
        score = detection.score
        xMin = detection.xMin
        yMin = detection.yMin
        width = detection.width
        height = detection.height
        let coordinates = detection.keypointCoordinates
        keypoints = stride(from: 0, to: coordinates.count - 1, by: 2).map {
            StagePoint(x: coordinates[$0].floatValue, y: coordinates[$0 + 1].floatValue)
        }
    }
}

/// Per-stage outputs of the hand pipeline for one frame. Present on a result
/// only when `HandLandmarkerOptions.stageOutputs` is true.
///
/// Every array is the content of one internal graph stream, copied unchanged.
/// An array is empty when its stream produced nothing on this frame.
///
/// Coordinates are normalized to the image as passed in, whatever
/// `ImageProcessingOptions.rotationDegrees` is: the same frame as
/// `HandLandmarkerResult.landmarks`. A rect's `rotation` is in that frame too.
public struct HandLandmarkerStages: Sendable, Equatable {
    /// Timestamp, in microseconds, of the graph output this record was read
    /// from. In `.video` mode it equals `timestampInMilliseconds * 1000` of
    /// the call that returned it.
    public var timestampMicroseconds: Int64
    /// `.video` mode only. Rects carried in from the previous frame: that
    /// frame's `handRectsNextFrame`, or its `keepAliveRects` when
    /// `trackingGraceFrames` > 0.
    public var previousRects: [StageRect]
    /// Whether the palm detector ran on this frame. In `.video` mode it is
    /// skipped when `previousRects` holds at least `numHands` rects. In
    /// `.image` mode it always runs.
    public var palmDetectorRan: Bool
    /// Palm detections after non-max suppression.
    public var palmDetections: [StageDetection]
    /// Palm rects derived from `palmDetections`. When the detector ran and
    /// found no palm, this and `detectorHandRects` each hold one all-zero rect.
    public var palmRects: [StageRect]
    /// `palmRects` expanded to whole-hand rects, clipped to `numHands`.
    public var detectorHandRects: [StageRect]
    /// Rects handed to the landmark model, in order: `previousRects` followed
    /// by the `detectorHandRects` that do not overlap them, clipped to
    /// `numHands`. In `.image` mode these are the `detectorHandRects`.
    public var landmarkROIs: [StageRect]
    /// Hand presence score per `landmarkROIs` entry, in the same order. The
    /// landmark model skips an all-zero rect, which has no entry here or in
    /// `presencePassed`; such a rect is always the last one.
    public var presenceScores: [Float]
    /// Whether the score passed `minHandPresenceConfidence`, per
    /// `landmarkROIs` entry, in the same order.
    public var presencePassed: [Bool]
    /// Landmarks per ROI that passed presence, in `landmarkROIs` order, before
    /// deduplication. The landmarks after deduplication are
    /// `HandLandmarkerResult.landmarks`.
    public var landmarksBeforeDeduplication: [[NormalizedLandmark]]
    /// Next-frame rect per ROI that passed presence, before deduplication.
    public var handRectsBeforeDeduplication: [StageRect]
    /// Next-frame rects after deduplication, one per hand in the result.
    public var handRectsNextFrame: [StageRect]
    /// `.video` mode with `trackingGraceFrames` > 0 only. The rects handed to
    /// the next frame: `handRectsNextFrame` first, held rects appended.
    public var keepAliveRects: [StageRect]
    /// Per `keepAliveRects` entry: 0 if the rect is live, otherwise the number
    /// of consecutive frames it has been missing.
    public var keepAliveMissCounts: [Int]

    init(_ stages: MPCHandLandmarkerStages) {
        timestampMicroseconds = stages.timestampUs
        previousRects = stages.previousRects.map(StageRect.init)
        palmDetectorRan = stages.palmDetectorRan
        palmDetections = stages.palmDetections.map(StageDetection.init)
        palmRects = stages.palmRects.map(StageRect.init)
        detectorHandRects = stages.detectorHandRects.map(StageRect.init)
        landmarkROIs = stages.landmarkRois.map(StageRect.init)
        presenceScores = stages.presenceScores.map(\.floatValue)
        presencePassed = stages.presencePassed.map(\.boolValue)
        landmarksBeforeDeduplication = stages.landmarksBeforeDedup.map { $0.map(NormalizedLandmark.init) }
        handRectsBeforeDeduplication = stages.handRectsBeforeDedup.map(StageRect.init)
        handRectsNextFrame = stages.handRectsNextFrame.map(StageRect.init)
        keepAliveRects = stages.keepAliveRects.map(StageRect.init)
        keepAliveMissCounts = stages.keepAliveMissCounts.map(\.intValue)
    }
}

/// Element type of a captured tensor.
public enum TensorElementType: Int, Sendable {
    case none = 0
    case float16 = 1
    case float32 = 2
    case uint8 = 3
    case int8 = 4
    case int32 = 5
    case int64 = 6
    case char = 7
    case bool = 8
}

/// A copy of one tensor as it was given to a model: the image region after
/// crop, rotation, resize and value normalization.
public struct CapturedTensor: Sendable, Equatable {
    /// Dimensions, e.g. `[1, 224, 224, 3]`.
    public var shape: [Int]
    public var elementType: TensorElementType
    /// The tensor's bytes: the product of `shape` elements of `elementType`,
    /// in row-major order and native byte order.
    public var data: Data

    /// The elements as floats, or nil if `elementType` is not `.float32`.
    public func floatValues() -> [Float]? {
        guard elementType == .float32 else { return nil }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    init?(_ capture: MPCTensorCapture) {
        guard capture.hasTensor, let data = capture.data else { return nil }
        shape = capture.shape.map(\.intValue)
        elementType = TensorElementType(rawValue: capture.elementType) ?? .none
        self.data = data
    }
}

/// The landmark model's input for one region of interest.
public struct LandmarkModelInput: Sendable, Equatable {
    /// Index of the region in the list the landmark model was given on this
    /// frame (`HandLandmarkerStages.landmarkROIs`).
    public var roiIndex: Int
    /// The input tensor, or nil if the model was not run on this region (an
    /// all-zero rect).
    public var tensor: CapturedTensor?
}

/// The tensors given to the two models on one frame. Present on a result only
/// when that detect call passed `captureTensors: true` (and
/// `HandLandmarkerOptions.tensorCapture` is true). A nil
/// `HandLandmarkerResult.tensorCapture` therefore means "nothing was captured
/// for this frame"; a nil tensor inside a capture means "that model did not
/// run".
public struct HandLandmarkerTensorCapture: Sendable, Equatable {
    /// Timestamp, in microseconds, of the graph output this record was read
    /// from. In `.video` mode it equals `timestampInMilliseconds * 1000` of
    /// the call that returned it.
    public var timestampMicroseconds: Int64
    /// Input of the palm detection model, or nil if the palm detector did not
    /// run on this frame.
    public var palmDetectorInput: CapturedTensor?
    /// Input of the landmark model per region of interest, in order. One entry
    /// per region, whether or not the model ran on it.
    public var landmarkInputs: [LandmarkModelInput]

    init(_ capture: MPCHandLandmarkerTensorCapture) {
        timestampMicroseconds = capture.timestampUs
        palmDetectorInput = CapturedTensor(capture.palmDetectorInput)
        landmarkInputs = capture.landmarkInputs.enumerated().map {
            LandmarkModelInput(roiIndex: $0.offset, tensor: CapturedTensor($0.element))
        }
    }
}

/// What became of the palm detector's candidates on one frame. Present on
/// every result when `HandLandmarkerOptions.palmCandidateLimit` is positive.
///
/// The palm model scores 2,016 boxes. Boxes scoring at least
/// `minHandDetectionConfidence` go to non-max suppression, which merges
/// overlapping ones; the first `numHands` of its output become hand rects.
public struct HandLandmarkerPalmCandidates: Sendable, Equatable {
    /// Timestamp, in microseconds, of the graph output this record was read
    /// from.
    public var timestampMicroseconds: Int64
    /// Whether the palm detector ran on this frame. The other fields are empty
    /// or 0 when it did not.
    public var palmDetectorRan: Bool
    /// The `palmCandidateLimit` highest scored boxes, highest first, whatever
    /// their score and before suppression. Same coordinates as
    /// `HandLandmarkerStages.palmDetections`.
    public var candidates: [StageDetection]
    /// Number of boxes whose score passed `minHandDetectionConfidence`.
    public var aboveThresholdCount: Int
    /// Number of detections left after non-max suppression. The first
    /// `numHands` of them become hand rects.
    public var afterSuppressionCount: Int

    init(_ candidates: MPCHandLandmarkerPalmCandidates) {
        timestampMicroseconds = candidates.timestampUs
        palmDetectorRan = candidates.palmDetectorRan
        self.candidates = candidates.candidates.map(StageDetection.init)
        aboveThresholdCount = candidates.aboveThresholdCount
        afterSuppressionCount = candidates.afterSuppressionCount
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
    /// Per-stage debug outputs for this frame. Nil unless
    /// `HandLandmarkerOptions.stageOutputs` is true.
    public var stages: HandLandmarkerStages?
    /// The model input tensors of this frame. Nil unless this detect call
    /// passed `captureTensors: true`.
    public var tensorCapture: HandLandmarkerTensorCapture?
    /// Palm candidate diagnostics for this frame. Nil unless
    /// `HandLandmarkerOptions.palmCandidateLimit` is positive.
    public var palmCandidates: HandLandmarkerPalmCandidates?
    /// Palm search record for this frame. Nil unless
    /// `HandLandmarkerOptions.palmSearch` is set and `stageOutputs` is true.
    public var palmSearch: HandLandmarkerPalmSearch?

    /// Deprecated alias for `handedness`, kept for parity with older MediaPipe
    /// result shapes.
    @available(*, deprecated, renamed: "handedness")
    public var handednesses: [[Category]] { handedness }

    init(_ result: MPCHandLandmarkerResult) {
        landmarks = result.landmarks.map { $0.map(NormalizedLandmark.init) }
        worldLandmarks = result.worldLandmarks.map { $0.map(Landmark.init) }
        handedness = result.handedness.map { $0.map(Category.init) }
        stages = result.stages.map(HandLandmarkerStages.init)
        tensorCapture = result.tensorCapture.map(HandLandmarkerTensorCapture.init)
        palmCandidates = result.palmCandidates.map(HandLandmarkerPalmCandidates.init)
        palmSearch = result.palmSearch.map(HandLandmarkerPalmSearch.init)
    }
}

/// Detects hand landmarks on images (IMAGE mode) or video frames (VIDEO mode).
public final class HandLandmarker {
    private let impl: MPCHandLandmarker
    private let runningMode: RunningMode
    private let tensorCaptureEnabled: Bool
    private let externalRegionsEnabled: Bool

    public init(options: HandLandmarkerOptions) throws {
        try checkDelegate(options.delegate)
        runningMode = options.runningMode
        tensorCaptureEnabled = options.tensorCapture
        externalRegionsEnabled = options.palmSearch?.acceptsExternalRegions ?? false
        impl = try MPCHandLandmarker(
            modelPath: options.modelPath,
            numHands: options.numHands,
            minHandDetectionConfidence: options.minHandDetectionConfidence,
            minHandPresenceConfidence: options.minHandPresenceConfidence,
            minTrackingConfidence: options.minTrackingConfidence,
            roiScale: options.roiScale,
            trackingGraceFrames: options.trackingGraceFrames,
            delegate: options.delegate.cValue,
            coreMLModelCacheDir: resolveCoreMLModelCacheDir(
                delegate: options.delegate,
                explicit: options.coreMLModelCacheDirectory,
                modelPath: options.modelPath),
            runningMode: options.runningMode.cValue,
            stageOutputs: options.stageOutputs,
            tensorCapture: options.tensorCapture,
            palmCandidateLimit: options.palmCandidateLimit,
            palmSearch: options.palmSearch?.bridged)
    }

    /// Checks the per-call regions and flattens them to {left, top, right,
    /// bottom} quadruples for the bridge.
    private func bridgedSearchRegions(_ regions: [CGRect]) throws -> [NSNumber]? {
        if regions.isEmpty { return nil }
        guard externalRegionsEnabled else {
            throw MediaPipeError.invalidArgument(
                "searchRegions requires HandLandmarkerOptions.palmSearch.acceptsExternalRegions")
        }
        var flat: [NSNumber] = []
        flat.reserveCapacity(regions.count * 4)
        for region in regions {
            guard region.width > 0, region.height > 0 else {
                throw MediaPipeError.invalidArgument(
                    "searchRegions must have a positive width and height")
            }
            flat.append(NSNumber(value: Float(region.minX)))
            flat.append(NSNumber(value: Float(region.minY)))
            flat.append(NSNumber(value: Float(region.maxX)))
            flat.append(NSNumber(value: Float(region.maxY)))
        }
        return flat
    }

    private func requireTensorCapture(_ captureTensors: Bool) throws {
        if captureTensors && !tensorCaptureEnabled {
            throw MediaPipeError.invalidArgument(
                "captureTensors requires HandLandmarkerOptions.tensorCapture")
        }
    }

    /// Runs hand landmark detection on a still `CGImage`. Requires `.image` mode.
    ///
    /// `captureTensors` (on every detect method) makes the result carry
    /// `tensorCapture`. It requires `HandLandmarkerOptions.tensorCapture`.
    ///
    /// `searchRegions` (on every detect method) are regions of this frame for
    /// the palm detector to look at, normalized to the image as passed in
    /// (origin top-left). They require
    /// `HandLandmarkerOptions.palmSearch.acceptsExternalRegions`, and are not
    /// used on a video frame on which the palm detector is skipped.
    public func detect(cgImage: CGImage,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                       captureTensors: Bool = false,
                       searchRegions: [CGRect] = [])
        throws -> HandLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(cgImage:)")
        try imageProcessingOptions.validated()
        try requireTensorCapture(captureTensors)
        let regions = try bridgedSearchRegions(searchRegions)
        let image = try MPCImage(cgImage: cgImage)
        return HandLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            captureTensors: captureTensors,
            searchRegions: regions))
    }

    /// Runs hand landmark detection on a video frame. Requires `.video` mode.
    /// Timestamps must be monotonically increasing.
    public func detectForVideo(cgImage: CGImage,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                               captureTensors: Bool = false,
                               searchRegions: [CGRect] = [])
        throws -> HandLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(cgImage:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        try requireTensorCapture(captureTensors)
        let regions = try bridgedSearchRegions(searchRegions)
        let image = try MPCImage(cgImage: cgImage)
        return HandLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds),
            captureTensors: captureTensors,
            searchRegions: regions))
    }

    /// Runs hand landmark detection on a `CVPixelBuffer` (`kCVPixelFormatType_32BGRA`).
    /// Requires `.image` mode.
    public func detect(pixelBuffer: CVPixelBuffer,
                       imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                       captureTensors: Bool = false,
                       searchRegions: [CGRect] = [])
        throws -> HandLandmarkerResult {
        try requireRunningMode(.image, actual: runningMode, method: "detect(pixelBuffer:)")
        try imageProcessingOptions.validated()
        try requireTensorCapture(captureTensors)
        let regions = try bridgedSearchRegions(searchRegions)
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return HandLandmarkerResult(try impl.detect(
            image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            captureTensors: captureTensors,
            searchRegions: regions))
    }

    /// Runs hand landmark detection on a `CVPixelBuffer` video frame
    /// (`kCVPixelFormatType_32BGRA`). Requires `.video` mode.
    public func detectForVideo(pixelBuffer: CVPixelBuffer,
                               timestampInMilliseconds: Int,
                               imageProcessingOptions: ImageProcessingOptions = ImageProcessingOptions(),
                               captureTensors: Bool = false,
                               searchRegions: [CGRect] = [])
        throws -> HandLandmarkerResult {
        try requireRunningMode(.video, actual: runningMode, method: "detectForVideo(pixelBuffer:timestampInMilliseconds:)")
        try imageProcessingOptions.validated()
        try requireTensorCapture(captureTensors)
        let regions = try bridgedSearchRegions(searchRegions)
        let image = try MPCImage(pixelBuffer: pixelBuffer)
        return HandLandmarkerResult(try impl.detect(
            forVideoImage: image,
            rotationDegrees: Int32(imageProcessingOptions.rotationDegrees),
            timestampMs: Int64(timestampInMilliseconds),
            captureTensors: captureTensors,
            searchRegions: regions))
    }
}
