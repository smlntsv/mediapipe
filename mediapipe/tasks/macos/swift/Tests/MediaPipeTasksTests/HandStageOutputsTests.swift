// Copyright 2026 The MediaPipe Authors.
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
import ImageIO
import XCTest

@testable import MediaPipeTasks

// Tests for `HandLandmarkerOptions.stageOutputs` / `HandLandmarkerResult.stages`.
//
// The sequences are synthetic: the test image drawn on a black canvas (a hand
// in frame) and plain black frames (no hand). Every VIDEO-mode frame asserts
// that `stages.timestampMicroseconds` is the timestamp of the call that
// returned it. That is the check for the one way this feature can go wrong
// silently: a stage stream whose timestamp bound does not settle on some frame
// stops the task runner's output callback from firing, and the call then
// returns the previous frame's packets.
//
// Skipped unless MP_HAND_MODEL and MP_HAND_IMAGE are set:
//   MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=test_image.jpg swift test \
//     --filter HandStageOutputsTests
final class HandStageOutputsTests: XCTestCase {

    private func env(_ key: String) -> String? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return value
    }

    private struct Fixture {
        let model: String
        let source: CGImage
        /// The test image at 0.45 scale in the middle of a black canvas.
        let hand: CGImage
        /// A black canvas of the same size.
        let black: CGImage
    }

    private func fixture() throws -> Fixture {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let url = URL(fileURLWithPath: imagePath) as CFURL
        guard let imageSource = CGImageSourceCreateWithURL(url, nil),
              let source = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            throw XCTSkip("Could not load test image at \(imagePath)")
        }
        guard let hand = canvas(source, placements: [(0.45, 0, 0)]),
              let black = canvas(source, placements: []) else {
            throw XCTSkip("Could not compose frames")
        }
        return Fixture(model: model, source: source, hand: hand, black: black)
    }

    /// Draws `src` onto a black canvas of its own size once per placement:
    /// scaled by `scale` and shifted from the center by (`dx`, `dy`), given as
    /// fractions of the canvas size.
    private func canvas(_ src: CGImage,
                        placements: [(scale: CGFloat, dx: CGFloat, dy: CGFloat)]) -> CGImage? {
        let w = src.width, h = src.height
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: bitmapInfo) else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        for placement in placements {
            let sw = CGFloat(w) * placement.scale, sh = CGFloat(h) * placement.scale
            ctx.draw(src, in: CGRect(x: placement.dx * CGFloat(w) + (CGFloat(w) - sw) / 2,
                                     y: placement.dy * CGFloat(h) + (CGFloat(h) - sh) / 2,
                                     width: sw, height: sh))
        }
        return ctx.makeImage()
    }

    /// Rotates the image a quarter turn counter-clockwise (so that
    /// `rotationDegrees: 90`, clockwise, brings it upright again).
    private func rotatedCounterClockwise(_ image: CGImage) -> CGImage? {
        let w = image.width, h = image.height
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: nil, width: h, height: w, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: bitmapInfo) else { return nil }
        // CoreGraphics has y up, so a positive angle is counter-clockwise on
        // screen.
        ctx.translateBy(x: CGFloat(h), y: 0)
        ctx.rotate(by: .pi / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    private func makeLandmarker(_ fixture: Fixture,
                                numHands: Int = 1,
                                grace: Int = 0,
                                mode: RunningMode = .video,
                                delegate: MediaPipeDelegate = .cpu,
                                coreMLCache: String? = nil,
                                stageOutputs: Bool = true,
                                tensorCapture: Bool = false,
                                palmCandidates: Int = 0) throws -> HandLandmarker {
        let options = HandLandmarkerOptions()
        options.delegate = delegate
        options.coreMLModelCacheDirectory = coreMLCache
        options.modelPath = fixture.model
        options.numHands = numHands
        options.trackingGraceFrames = grace
        options.runningMode = mode
        options.stageOutputs = stageOutputs
        options.tensorCapture = tensorCapture
        options.palmCandidateLimit = palmCandidates
        return try HandLandmarker(options: options)
    }

    private let frameIntervalMs = 33

    /// Runs the frames through VIDEO mode and returns one result per frame,
    /// asserting on the way that every frame's stage record belongs to it.
    private func run(_ landmarker: HandLandmarker, _ frames: [CGImage],
                     file: StaticString = #filePath, line: UInt = #line) throws
        -> [HandLandmarkerResult] {
        var results: [HandLandmarkerResult] = []
        for (index, frame) in frames.enumerated() {
            let timestampMs = index * frameIntervalMs
            let result = try landmarker.detectForVideo(
                cgImage: frame, timestampInMilliseconds: timestampMs)
            let stages = try XCTUnwrap(result.stages, "frame \(index): no stages",
                                       file: file, line: line)
            XCTAssertEqual(stages.timestampMicroseconds, Int64(timestampMs) * 1000,
                           "frame \(index): result belongs to another frame",
                           file: file, line: line)
            results.append(result)
        }
        return results
    }

    private func describe(_ rect: StageRect) -> String {
        String(format: "(%.3f,%.3f %.3fx%.3f r=%.2f id=%@)", rect.xCenter, rect.yCenter,
               rect.width, rect.height, rect.rotation,
               rect.rectId.map(String.init) ?? "-")
    }

    private func describe(_ index: Int, _ result: HandLandmarkerResult) -> String {
        guard let s = result.stages else { return "frame \(index): stages nil" }
        func rects(_ r: [StageRect]) -> String { "[" + r.map(describe).joined(separator: " ") + "]" }
        return """
            frame \(index) ts=\(s.timestampMicroseconds) hands=\(result.landmarks.count)
              previousRects      \(rects(s.previousRects))
              palmDetectorRan    \(s.palmDetectorRan)  palmDetections=\(s.palmDetections.count) \
            scores=\(s.palmDetections.map { String(format: "%.2f", $0.score) })
              palmRects          \(rects(s.palmRects))
              detectorHandRects  \(rects(s.detectorHandRects))
              landmarkROIs       \(rects(s.landmarkROIs))
              presence           \(s.presenceScores.map { String(format: "%.3f", $0) }) \(s.presencePassed)
              beforeDedup        landmarks=\(s.landmarksBeforeDeduplication.count) \
            rects=\(rects(s.handRectsBeforeDeduplication))
              handRectsNextFrame \(rects(s.handRectsNextFrame))
              keepAlive          \(rects(s.keepAliveRects)) misses=\(s.keepAliveMissCounts)
            """
    }

    private func assertSameGeometry(_ a: [StageRect], _ b: [StageRect], _ message: String,
                                    file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, message, file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.xCenter, y.xCenter, message, file: file, line: line)
            XCTAssertEqual(x.yCenter, y.yCenter, message, file: file, line: line)
            XCTAssertEqual(x.width, y.width, message, file: file, line: line)
            XCTAssertEqual(x.height, y.height, message, file: file, line: line)
            XCTAssertEqual(x.rotation, y.rotation, message, file: file, line: line)
        }
    }

    /// What the detector graph emits when it ran and found no palm: no
    /// detection, one all-zero palm rect, one all-zero hand rect, and so one
    /// all-zero ROI that the landmark model skips.
    private func assertDetectorFoundNothing(_ stages: HandLandmarkerStages, _ message: String,
                                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(stages.palmDetectorRan, message, file: file, line: line)
        XCTAssertTrue(stages.palmDetections.isEmpty, message, file: file, line: line)
        for rects in [stages.palmRects, stages.detectorHandRects, stages.landmarkROIs] {
            XCTAssertEqual(rects.count, 1, message, file: file, line: line)
            XCTAssertEqual(rects.first?.width, 0, message, file: file, line: line)
            XCTAssertEqual(rects.first?.height, 0, message, file: file, line: line)
        }
        XCTAssertTrue(stages.presenceScores.isEmpty, message, file: file, line: line)
        XCTAssertTrue(stages.presencePassed.isEmpty, message, file: file, line: line)
        XCTAssertTrue(stages.landmarksBeforeDeduplication.isEmpty, message, file: file, line: line)
        XCTAssertTrue(stages.handRectsBeforeDeduplication.isEmpty, message, file: file, line: line)
        XCTAssertTrue(stages.handRectsNextFrame.isEmpty, message, file: file, line: line)
    }

    // MARK: - Option off

    func testStagesAreNilWhenOff() throws {
        let fixture = try fixture()
        let video = try makeLandmarker(fixture, stageOutputs: false)
        for (index, frame) in [fixture.hand, fixture.black, fixture.hand].enumerated() {
            let result = try video.detectForVideo(
                cgImage: frame, timestampInMilliseconds: index * frameIntervalMs)
            XCTAssertNil(result.stages)
        }
        let image = try makeLandmarker(fixture, mode: .image, stageOutputs: false)
        XCTAssertNil(try image.detect(cgImage: fixture.hand).stages)
        XCTAssertFalse(HandLandmarkerOptions().stageOutputs)
    }

    /// Turning the option on must not change what the pipeline computes.
    func testResultsAreIdenticalWithOptionOnAndOff() throws {
        let fixture = try fixture()
        let frames = Array(repeating: fixture.hand, count: 4)
            + Array(repeating: fixture.black, count: 3)
            + Array(repeating: fixture.hand, count: 4)
        for grace in [0, 2] {
            let off = try makeLandmarker(fixture, numHands: 2, grace: grace, stageOutputs: false)
            let on = try makeLandmarker(fixture, numHands: 2, grace: grace, stageOutputs: true)
            for (index, frame) in frames.enumerated() {
                let timestampMs = index * frameIntervalMs
                let a = try off.detectForVideo(cgImage: frame, timestampInMilliseconds: timestampMs)
                let b = try on.detectForVideo(cgImage: frame, timestampInMilliseconds: timestampMs)
                XCTAssertEqual(a.landmarks, b.landmarks, "grace \(grace) frame \(index)")
                XCTAssertEqual(a.worldLandmarks, b.worldLandmarks, "grace \(grace) frame \(index)")
                XCTAssertEqual(a.handedness, b.handedness, "grace \(grace) frame \(index)")
            }
        }
    }

    // MARK: - Frame kinds

    func testNoHandInFrame() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture)
        let results = try run(landmarker, Array(repeating: fixture.black, count: 4))
        for (index, result) in results.enumerated() {
            let stages = try XCTUnwrap(result.stages)
            XCTAssertTrue(result.landmarks.isEmpty, "frame \(index)")
            XCTAssertTrue(stages.previousRects.isEmpty, "frame \(index)")
            assertDetectorFoundNothing(stages, "frame \(index)")
        }
    }

    func testFirstDetectionThenTrackedFrames() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture)
        let results = try run(landmarker, Array(repeating: fixture.hand, count: 4))

        // First detection: the detector runs and its output feeds the landmark
        // model.
        let first = try XCTUnwrap(results[0].stages)
        XCTAssertEqual(results[0].landmarks.count, 1)
        XCTAssertTrue(first.previousRects.isEmpty)
        XCTAssertTrue(first.palmDetectorRan)
        XCTAssertGreaterThanOrEqual(first.palmDetections.count, 1)
        XCTAssertEqual(first.palmRects.count, first.palmDetections.count)
        let detection = try XCTUnwrap(first.palmDetections.first)
        XCTAssertGreaterThan(detection.score, 0.5)
        XCTAssertEqual(detection.keypoints.count, 7)
        XCTAssertGreaterThan(detection.width, 0)
        XCTAssertGreaterThan(detection.height, 0)
        XCTAssertEqual(first.detectorHandRects.count, 1, "clipped to numHands")
        XCTAssertEqual(first.landmarkROIs.count, 1)
        XCTAssertNotNil(first.landmarkROIs[0].rectId)
        assertSameGeometry(first.landmarkROIs, first.detectorHandRects, "frame 0 ROI source")
        XCTAssertEqual(first.presenceScores.count, 1)
        XCTAssertEqual(first.presencePassed, [true])
        XCTAssertGreaterThan(first.presenceScores[0], 0.5)
        XCTAssertEqual(first.landmarksBeforeDeduplication.count, 1)
        XCTAssertEqual(first.landmarksBeforeDeduplication[0].count, 21)
        XCTAssertEqual(first.landmarksBeforeDeduplication, results[0].landmarks,
                       "nothing to deduplicate with one hand")
        XCTAssertEqual(first.handRectsBeforeDeduplication.count, 1)
        XCTAssertEqual(first.handRectsNextFrame.count, 1)
        XCTAssertTrue(first.keepAliveRects.isEmpty, "no keep-alive without a grace")
        XCTAssertTrue(first.keepAliveMissCounts.isEmpty)

        // Tracked frames: the detector is skipped and the ROI is the rect the
        // previous frame predicted.
        for index in 1..<results.count {
            let stages = try XCTUnwrap(results[index].stages)
            let previous = try XCTUnwrap(results[index - 1].stages)
            XCTAssertEqual(results[index].landmarks.count, 1, "frame \(index)")
            XCTAssertEqual(stages.previousRects, previous.handRectsNextFrame, "frame \(index)")
            XCTAssertFalse(stages.palmDetectorRan, "frame \(index)")
            XCTAssertTrue(stages.palmDetections.isEmpty, "frame \(index)")
            XCTAssertTrue(stages.palmRects.isEmpty, "frame \(index)")
            XCTAssertTrue(stages.detectorHandRects.isEmpty, "frame \(index)")
            assertSameGeometry(stages.landmarkROIs, stages.previousRects, "frame \(index) ROI source")
            XCTAssertEqual(stages.presencePassed, [true], "frame \(index)")
            XCTAssertEqual(stages.handRectsNextFrame.count, 1, "frame \(index)")
        }
    }

    func testHandLost() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture)
        let frames = Array(repeating: fixture.hand, count: 3)
            + Array(repeating: fixture.black, count: 3)
        let results = try run(landmarker, frames)

        // Frame 3 is the first black frame: the tracked ROI is still tried, and
        // fails presence. The failing ROI and its score are reported.
        let lost = try XCTUnwrap(results[3].stages)
        XCTAssertTrue(results[3].landmarks.isEmpty)
        XCTAssertEqual(lost.previousRects.count, 1)
        XCTAssertFalse(lost.palmDetectorRan)
        XCTAssertEqual(lost.landmarkROIs.count, 1)
        XCTAssertEqual(lost.presenceScores.count, 1)
        XCTAssertEqual(lost.presencePassed, [false])
        XCTAssertLessThan(lost.presenceScores[0], 0.5)
        XCTAssertTrue(lost.landmarksBeforeDeduplication.isEmpty)
        XCTAssertTrue(lost.handRectsBeforeDeduplication.isEmpty)
        XCTAssertTrue(lost.handRectsNextFrame.isEmpty)

        // Next frame: nothing was carried over, so the detector runs again.
        for index in 4...5 {
            let stages = try XCTUnwrap(results[index].stages)
            XCTAssertTrue(stages.previousRects.isEmpty, "frame \(index)")
            assertDetectorFoundNothing(stages, "frame \(index)")
        }
    }

    /// Two hands tracked, then one disappears while the other stays: the
    /// keep-alive holds the vanished hand's rect for `trackingGraceFrames`
    /// frames and reports the miss count next to it.
    func testKeepAliveHoldsVanishedHand() throws {
        let fixture = try fixture()
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        let left = try XCTUnwrap(canvas(fixture.source, placements: [(0.45, -0.25, 0)]))
        let landmarker = try makeLandmarker(fixture, numHands: 2, grace: 2)
        let results = try run(landmarker, Array(repeating: two, count: 3)
            + Array(repeating: left, count: 5))
        let stages = try results.map { try XCTUnwrap($0.stages) }

        // What a frame carries in is what the keep-alive handed over.
        for index in 1..<stages.count {
            XCTAssertEqual(stages[index].previousRects, stages[index - 1].keepAliveRects,
                           "frame \(index)")
        }
        // Both hands live.
        for index in 0...2 {
            XCTAssertEqual(results[index].landmarks.count, 2, "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveRects, stages[index].handRectsNextFrame,
                           "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveMissCounts, [0, 0], "frame \(index)")
        }
        // Frames 3 and 4: the right hand is gone. Its ROI is still tried and
        // fails presence; its rect is held unchanged, the detector stays off.
        let held = try XCTUnwrap(stages[2].keepAliveRects.last)
        for (index, misses) in [(3, 1), (4, 2)] {
            XCTAssertEqual(results[index].landmarks.count, 1, "frame \(index)")
            XCTAssertFalse(stages[index].palmDetectorRan, "frame \(index)")
            XCTAssertEqual(stages[index].landmarkROIs.count, 2, "frame \(index)")
            XCTAssertEqual(stages[index].presencePassed, [true, false], "frame \(index)")
            XCTAssertEqual(stages[index].presenceScores.count, 2, "frame \(index)")
            XCTAssertEqual(stages[index].handRectsNextFrame.count, 1, "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveRects.count, 2, "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveRects.first,
                           stages[index].handRectsNextFrame.first, "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveRects.last, held, "frame \(index)")
            XCTAssertEqual(stages[index].keepAliveMissCounts, [0, misses], "frame \(index)")
        }
        // Frame 5: the held rect was carried in and tried once more, then
        // dropped after exceeding the grace.
        XCTAssertFalse(stages[5].palmDetectorRan)
        XCTAssertEqual(stages[5].landmarkROIs.count, 2)
        XCTAssertEqual(stages[5].keepAliveRects.count, 1)
        XCTAssertEqual(stages[5].keepAliveMissCounts, [0])
        // Frame 6: one rect carried in, fewer than numHands, so the detector
        // runs again.
        XCTAssertEqual(stages[6].previousRects.count, 1)
        XCTAssertTrue(stages[6].palmDetectorRan)
    }

    /// Stage coordinates stay in the frame of the image as passed in, whatever
    /// `rotationDegrees` is.
    func testRectsAreInInputImageFrameUnderRotation() throws {
        let fixture = try fixture()
        let rotated = try XCTUnwrap(rotatedCounterClockwise(fixture.hand))
        let landmarker = try makeLandmarker(fixture, mode: .image)
        let upright = try XCTUnwrap(
            try landmarker.detect(cgImage: fixture.hand).stages?.landmarkROIs.first)
        // A point (x, y) of the upright image sits at (y, 1 - x) once the
        // image is turned a quarter counter-clockwise.
        let expectedX = upright.yCenter, expectedY = 1 - upright.xCenter
        for degrees in [0, 90] {
            let result = try landmarker.detect(
                cgImage: rotated,
                imageProcessingOptions: ImageProcessingOptions(rotationDegrees: degrees))
            let stages = try XCTUnwrap(result.stages)
            let roi = try XCTUnwrap(stages.landmarkROIs.first, "rotationDegrees \(degrees)")
            XCTAssertEqual(roi.xCenter, expectedX, accuracy: 0.03, "rotationDegrees \(degrees)")
            XCTAssertEqual(roi.yCenter, expectedY, accuracy: 0.03, "rotationDegrees \(degrees)")
            // The ROI and the landmarks it produced share one frame.
            let hand = try XCTUnwrap(result.landmarks.first, "rotationDegrees \(degrees)")
            let meanX = hand.reduce(0) { $0 + $1.x } / Float(hand.count)
            let meanY = hand.reduce(0) { $0 + $1.y } / Float(hand.count)
            let next = try XCTUnwrap(stages.handRectsNextFrame.first)
            XCTAssertEqual(next.xCenter, meanX, accuracy: 0.05, "rotationDegrees \(degrees)")
            XCTAssertEqual(next.yCenter, meanY, accuracy: 0.05, "rotationDegrees \(degrees)")
        }
    }

    /// The frame kinds again on the GPU and Core ML delegates, whose graphs
    /// use different preprocessing and inference calculators: every frame's
    /// record must still belong to that frame (asserted in `run`).
    func testEveryFrameKindOnGPUAndCoreML() throws {
        let fixture = try fixture()
        var delegates: [(String, MediaPipeDelegate, String?)] = []
        if mediaPipeGPUArtifactAvailable { delegates.append(("gpu", .gpu, nil)) }
        // Core ML needs the pre-converted models: MP_COREML_CACHE, or a
        // coreml_models directory next to the .task file.
        let defaultCache = URL(fileURLWithPath: fixture.model)
            .deletingLastPathComponent().appendingPathComponent("coreml_models").path
        if let cache = env("MP_COREML_CACHE")
            ?? (FileManager.default.fileExists(atPath: defaultCache) ? defaultCache : nil) {
            delegates.append(("coreml", .coreML, cache))
        }
        try XCTSkipIf(delegates.isEmpty, "No GPU artifact and no Core ML models.")
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        let left = try XCTUnwrap(canvas(fixture.source, placements: [(0.45, -0.25, 0)]))
        let frames = Array(repeating: fixture.black, count: 2)   // no hand
            + Array(repeating: two, count: 3)                     // first detection, tracked
            + Array(repeating: left, count: 4)                    // one lost, held, dropped
            + Array(repeating: fixture.black, count: 3)           // all lost
        for (name, delegate, cache) in delegates {
            let landmarker = try makeLandmarker(fixture, numHands: 2, grace: 2,
                                                delegate: delegate, coreMLCache: cache)
            let results = try run(landmarker, frames)
            let stages = try results.map { try XCTUnwrap($0.stages) }
            let timeline = zip(results, stages).map {
                "\($0.landmarks.count)\($1.palmDetectorRan ? "D" : "-")"
            }.joined(separator: " ")
            print("[\(name)] hands+detector per frame: \(timeline)")
            assertDetectorFoundNothing(stages[0], "\(name) frame 0")
            assertDetectorFoundNothing(stages[1], "\(name) frame 1")
            XCTAssertEqual(results[2].landmarks.count, 2, "\(name) first detection")
            XCTAssertTrue(stages[2].palmDetectorRan, "\(name) first detection")
            XCTAssertGreaterThanOrEqual(stages[2].palmDetections.count, 2,
                                        "\(name) first detection")
            for index in 3...4 {
                XCTAssertFalse(stages[index].palmDetectorRan, "\(name) frame \(index)")
                XCTAssertEqual(stages[index].previousRects, stages[index - 1].keepAliveRects,
                               "\(name) frame \(index)")
                XCTAssertEqual(stages[index].presencePassed, [true, true], "\(name) frame \(index)")
            }
            XCTAssertEqual(stages[5].presencePassed, [true, false], "\(name) hand lost")
            XCTAssertEqual(stages[5].keepAliveMissCounts, [0, 1], "\(name) hand lost")
            XCTAssertEqual(stages[6].keepAliveMissCounts, [0, 2], "\(name) held")
            XCTAssertEqual(stages[7].keepAliveMissCounts, [0], "\(name) dropped")
            XCTAssertTrue(results[9].landmarks.isEmpty, "\(name) all lost")
        }
    }

    func testImageMode() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture, mode: .image)
        let result = try landmarker.detect(cgImage: fixture.hand)
        let stages = try XCTUnwrap(result.stages)
        XCTAssertEqual(result.landmarks.count, 1)
        XCTAssertTrue(stages.palmDetectorRan)
        XCTAssertTrue(stages.previousRects.isEmpty)
        XCTAssertGreaterThanOrEqual(stages.palmDetections.count, 1)
        XCTAssertEqual(stages.landmarkROIs, stages.detectorHandRects)
        XCTAssertNil(stages.landmarkROIs.first?.rectId, "no association in image mode")
        XCTAssertEqual(stages.presencePassed, [true])
        XCTAssertEqual(stages.handRectsNextFrame.count, 1)
        XCTAssertTrue(stages.keepAliveRects.isEmpty)

        let empty = try landmarker.detect(cgImage: fixture.black)
        let emptyStages = try XCTUnwrap(empty.stages, "stages must be filled when no hand is found")
        XCTAssertTrue(empty.landmarks.isEmpty)
        assertDetectorFoundNothing(emptyStages, "black image")
        XCTAssertNotEqual(emptyStages.timestampMicroseconds, stages.timestampMicroseconds,
                          "the record must come from this call")
    }

    func testPixelBufferPathCarriesStages() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture)
        let buffer = try makeBGRAPixelBuffer(from: fixture.hand)
        for index in 0..<3 {
            let result = try landmarker.detectForVideo(
                pixelBuffer: buffer, timestampInMilliseconds: index * frameIntervalMs)
            let stages = try XCTUnwrap(result.stages)
            XCTAssertEqual(stages.timestampMicroseconds, Int64(index * frameIntervalMs) * 1000)
            XCTAssertEqual(stages.palmDetectorRan, index == 0)
            XCTAssertEqual(stages.landmarkROIs.count, 1)
        }
    }

    private func makeBGRAPixelBuffer(from cgImage: CGImage) throws -> CVPixelBuffer {
        let w = cgImage.width, h = cgImage.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw XCTSkip("CVPixelBufferCreate failed (\(status))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo) else {
            throw XCTSkip("CGContext for CVPixelBuffer failed")
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    // MARK: - Exploration (prints, no assertions)

    /// Prints the full stage record of every frame of a few sequences.
    ///   MP_VERIFY=1 MP_HAND_MODEL=... MP_HAND_IMAGE=... swift test \
    ///     --filter HandStageOutputsTests/testPrintStageTimelines
    func testPrintStageTimelines() throws {
        guard env("MP_VERIFY") != nil else { throw XCTSkip("Set MP_VERIFY=1 to run.") }
        let fixture = try fixture()
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        let left = try XCTUnwrap(canvas(fixture.source, placements: [(0.45, -0.25, 0)]))
        let sequences: [(String, Int, Int, [CGImage])] = [
            ("one hand, lost, grace 0", 1, 0,
             Array(repeating: fixture.hand, count: 3) + Array(repeating: fixture.black, count: 3)),
            ("one hand, lost, grace 2", 1, 2,
             Array(repeating: fixture.hand, count: 3) + Array(repeating: fixture.black, count: 5)),
            ("two copies, right one lost, grace 2, numHands 2", 2, 2,
             Array(repeating: two, count: 3) + Array(repeating: left, count: 5)),
            ("full image, numHands 4", 4, 0, Array(repeating: fixture.source, count: 2)),
        ]
        for (name, numHands, grace, frames) in sequences {
            print("====== \(name) ======")
            let landmarker = try makeLandmarker(fixture, numHands: numHands, grace: grace)
            for (index, frame) in frames.enumerated() {
                let result = try landmarker.detectForVideo(
                    cgImage: frame, timestampInMilliseconds: index * frameIntervalMs)
                print(describe(index, result))
            }
        }
        print("====== rotation ======")
        let rotated = try XCTUnwrap(rotatedCounterClockwise(fixture.hand))
        let image = try makeLandmarker(fixture, mode: .image)
        let upright = try image.detect(cgImage: fixture.hand)
        print("upright image, rotationDegrees 0")
        print(describe(0, upright))
        for degrees in [0, 90, -90] {
            let result = try image.detect(
                cgImage: rotated,
                imageProcessingOptions: ImageProcessingOptions(rotationDegrees: degrees))
            print("image rotated 90 counter-clockwise, rotationDegrees \(degrees)")
            print(describe(0, result))
            if let wrist = result.landmarks.first?.first {
                print(String(format: "  wrist (%.3f, %.3f)", wrist.x, wrist.y))
            }
        }
        if let wrist = upright.landmarks.first?.first {
            print(String(format: "  upright wrist (%.3f, %.3f)", wrist.x, wrist.y))
        }
    }

    /// Per-frame cost of the option: VIDEO mode, two tracked hands, the same
    /// pixel buffer every frame, option off and on in alternating rounds.
    ///   MP_VERIFY=1 MP_HAND_MODEL=... MP_HAND_IMAGE=... [MP_COREML_CACHE=dir] \
    ///     swift test -c release --filter HandStageOutputsTests/testStageOutputsOverhead
    func testStageOutputsOverhead() throws {
        guard env("MP_VERIFY") != nil else { throw XCTSkip("Set MP_VERIFY=1 to run.") }
        let fixture = try fixture()
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        let buffer = try makeBGRAPixelBuffer(from: two)
        let frames = Int(env("MP_BENCH_FRAMES") ?? "") ?? 400
        let rounds = Int(env("MP_BENCH_ROUNDS") ?? "") ?? 5
        var delegates: [(String, MediaPipeDelegate, String?)] = [("cpu", .cpu, nil), ("gpu", .gpu, nil)]
        if let cache = env("MP_COREML_CACHE") { delegates.append(("coreml", .coreML, cache)) }

        func meanMs(_ delegate: MediaPipeDelegate, _ cache: String?, stages: Bool) throws
            -> (ms: Double, hands: Int) {
            let options = HandLandmarkerOptions()
            options.modelPath = fixture.model
            options.numHands = 2
            options.runningMode = .video
            options.delegate = delegate
            options.coreMLModelCacheDirectory = cache
            options.stageOutputs = stages
            let landmarker = try HandLandmarker(options: options)
            for i in 0..<30 {
                _ = try landmarker.detectForVideo(pixelBuffer: buffer, timestampInMilliseconds: i)
            }
            var hands = 0
            let start = ProcessInfo.processInfo.systemUptime
            for i in 0..<frames {
                let result = try landmarker.detectForVideo(
                    pixelBuffer: buffer, timestampInMilliseconds: 30 + i)
                hands += result.landmarks.count
            }
            return ((ProcessInfo.processInfo.systemUptime - start) / Double(frames) * 1000, hands)
        }
        func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

        print("====== STAGE OUTPUTS OVERHEAD (\(frames) frames x \(rounds) rounds, 2 hands tracked) ======")
        for (name, delegate, cache) in delegates {
            var off: [Double] = [], on: [Double] = []
            var hands = 0
            for _ in 0..<rounds {
                let a = try meanMs(delegate, cache, stages: false)
                let b = try meanMs(delegate, cache, stages: true)
                off.append(a.ms); on.append(b.ms); hands = b.hands
            }
            print(String(format: "%@: off %.3f ms  on %.3f ms  delta %+.3f ms/frame  (rounds off %@ on %@, hands/frame %.2f)",
                         name, median(off), median(on), median(on) - median(off),
                         off.map { String(format: "%.3f", $0) }.joined(separator: " "),
                         on.map { String(format: "%.3f", $0) }.joined(separator: " "),
                         Double(hands) / Double(frames)))
        }
        print("=================================================================================")
    }
}

// MARK: - Tensor capture and palm candidates

extension HandStageOutputsTests {

    /// GPU and Core ML when this build and machine have them, after CPU.
    private func availableDelegates(_ fixture: Fixture) -> [(String, MediaPipeDelegate, String?)] {
        var delegates: [(String, MediaPipeDelegate, String?)] = [("cpu", .cpu, nil)]
        if mediaPipeGPUArtifactAvailable { delegates.append(("gpu", .gpu, nil)) }
        let defaultCache = URL(fileURLWithPath: fixture.model)
            .deletingLastPathComponent().appendingPathComponent("coreml_models").path
        if let cache = env("MP_COREML_CACHE")
            ?? (FileManager.default.fileExists(atPath: defaultCache) ? defaultCache : nil) {
            delegates.append(("coreml", .coreML, cache))
        }
        return delegates
    }

    /// Checks a captured model input and returns its RGB values. The CPU and
    /// Core ML paths give the model a 3-channel tensor; the GPU path's Metal
    /// preprocessing gives it 4 channels per pixel, of which the first three
    /// are RGB.
    private func assertModelInput(_ tensor: CapturedTensor?, side: Int, _ message: String,
                                  file: StaticString = #filePath, line: UInt = #line) -> [Float] {
        guard let tensor else {
            XCTFail("\(message): no tensor", file: file, line: line)
            return []
        }
        XCTAssertEqual(tensor.shape.count, 4, message, file: file, line: line)
        guard tensor.shape.count == 4 else { return [] }
        let channels = tensor.shape[3]
        XCTAssertEqual(Array(tensor.shape.prefix(3)), [1, side, side], message, file: file, line: line)
        XCTAssertTrue(channels == 3 || channels == 4, "\(message): \(channels) channels",
                      file: file, line: line)
        XCTAssertEqual(tensor.elementType, .float32, message, file: file, line: line)
        XCTAssertEqual(tensor.data.count, side * side * channels * 4, message, file: file, line: line)
        let values = tensor.floatValues() ?? []
        XCTAssertEqual(values.count, side * side * channels, message, file: file, line: line)
        guard values.count == side * side * channels else { return [] }
        var rgb: [Float] = []
        rgb.reserveCapacity(side * side * 3)
        for pixel in 0..<(side * side) {
            rgb.append(contentsOf: values[(pixel * channels)..<(pixel * channels + 3)])
        }
        XCTAssertTrue(rgb.allSatisfy { $0.isFinite && $0 >= -1.001 && $0 <= 1.001 },
                      "\(message): values outside the normalized range", file: file, line: line)
        return rgb
    }

    private func spread(_ values: [Float]) -> Float {
        (values.max() ?? 0) - (values.min() ?? 0)
    }

    func testTensorCaptureIsOffByDefault() throws {
        let fixture = try fixture()
        XCTAssertFalse(HandLandmarkerOptions().tensorCapture)
        XCTAssertEqual(HandLandmarkerOptions().palmCandidateLimit, 0)

        // Without the option a capture request is an error, and plain calls
        // carry neither record.
        let plain = try makeLandmarker(fixture, stageOutputs: false)
        XCTAssertThrowsError(try plain.detectForVideo(
            cgImage: fixture.hand, timestampInMilliseconds: 0, captureTensors: true)) { error in
            guard case MediaPipeError.invalidArgument = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
        let result = try plain.detectForVideo(cgImage: fixture.hand, timestampInMilliseconds: 0)
        XCTAssertNil(result.tensorCapture)
        XCTAssertNil(result.palmCandidates)

        // With the option, a call that does not ask gets no capture.
        let capturing = try makeLandmarker(fixture, stageOutputs: false, tensorCapture: true)
        for index in 0..<3 {
            let result = try capturing.detectForVideo(
                cgImage: fixture.hand, timestampInMilliseconds: index * frameIntervalMs)
            XCTAssertNil(result.tensorCapture, "frame \(index)")
            XCTAssertNil(result.stages, "tensor capture does not turn the stage record on")
        }
    }

    /// One capture per frame kind: what was given to each model, and which
    /// model did not run.
    func testTensorCaptureOnEveryFrameKind() throws {
        let fixture = try fixture()
        let frames = [fixture.black, fixture.hand, fixture.hand, fixture.hand,
                      fixture.black, fixture.black]
        for (name, delegate, cache) in availableDelegates(fixture) {
            let landmarker = try makeLandmarker(fixture, delegate: delegate, coreMLCache: cache,
                                                tensorCapture: true)
            var results: [HandLandmarkerResult] = []
            for (index, frame) in frames.enumerated() {
                let timestampMs = index * frameIntervalMs
                let result = try landmarker.detectForVideo(
                    cgImage: frame, timestampInMilliseconds: timestampMs, captureTensors: true)
                let capture = try XCTUnwrap(result.tensorCapture, "\(name) frame \(index)")
                XCTAssertEqual(capture.timestampMicroseconds, Int64(timestampMs) * 1000,
                               "\(name) frame \(index): capture belongs to another frame")
                XCTAssertEqual(try XCTUnwrap(result.stages).timestampMicroseconds,
                               Int64(timestampMs) * 1000, "\(name) frame \(index)")
                XCTAssertEqual(capture.landmarkInputs.map(\.roiIndex),
                               Array(capture.landmarkInputs.indices), "\(name) frame \(index)")
                results.append(result)
            }
            let captures = try results.map { try XCTUnwrap($0.tensorCapture) }
            let stages = try results.map { try XCTUnwrap($0.stages) }

            // Frame 0, no hand: the detector ran on a black image; its one
            // all-zero ROI never reached the landmark model.
            let blackPalm = assertModelInput(captures[0].palmDetectorInput, side: 192,
                                             "\(name) frame 0 palm")
            XCTAssertLessThan(spread(blackPalm), 0.01, "\(name): a black frame is a flat tensor")
            XCTAssertEqual(captures[0].landmarkInputs.count, 1, "\(name) frame 0")
            XCTAssertNil(captures[0].landmarkInputs.first?.tensor, "\(name) frame 0")

            // Frame 1, first detection: both models ran.
            let handPalm = assertModelInput(captures[1].palmDetectorInput, side: 192,
                                            "\(name) frame 1 palm")
            XCTAssertGreaterThan(spread(handPalm), 0.5, "\(name): the hand frame has content")
            XCTAssertEqual(captures[1].landmarkInputs.count, 1, "\(name) frame 1")
            let firstCrop = assertModelInput(captures[1].landmarkInputs.first?.tensor, side: 224,
                                             "\(name) frame 1 landmark")
            XCTAssertGreaterThan(spread(firstCrop), 0.5, "\(name) frame 1 landmark")

            // Frames 2 and 3, tracked: the detector is skipped, the landmark
            // model runs on the carried-over ROI.
            for index in 2...3 {
                XCTAssertFalse(stages[index].palmDetectorRan, "\(name) frame \(index)")
                XCTAssertNil(captures[index].palmDetectorInput, "\(name) frame \(index)")
                XCTAssertEqual(captures[index].landmarkInputs.count, 1, "\(name) frame \(index)")
                _ = assertModelInput(captures[index].landmarkInputs.first?.tensor, side: 224,
                                     "\(name) frame \(index) landmark")
            }
            // The same image and the same ROI give the same crop.
            XCTAssertEqual(stages[3].landmarkROIs.count, 1)
            if stages[2].handRectsNextFrame == stages[3].handRectsNextFrame,
               stages[2].previousRects == stages[3].previousRects {
                XCTAssertEqual(captures[2].landmarkInputs.first?.tensor,
                               captures[3].landmarkInputs.first?.tensor, "\(name)")
            }

            // Frame 4, hand lost: the tracked ROI is still tried, on black.
            XCTAssertNil(captures[4].palmDetectorInput, "\(name) frame 4")
            let lostCrop = assertModelInput(captures[4].landmarkInputs.first?.tensor, side: 224,
                                            "\(name) frame 4 landmark")
            XCTAssertLessThan(spread(lostCrop), 0.01, "\(name) frame 4: crop of a black frame")
            XCTAssertEqual(stages[4].presencePassed, [false], "\(name) frame 4")

            // Frame 5: back to the detector, nothing for the landmark model.
            XCTAssertNotNil(captures[5].palmDetectorInput, "\(name) frame 5")
            XCTAssertNil(captures[5].landmarkInputs.first?.tensor, "\(name) frame 5")
        }
    }

    /// Capture on some frames only, with two hands; the pipeline's results do
    /// not depend on it.
    func testCaptureOnSelectedFramesDoesNotChangeResults() throws {
        let fixture = try fixture()
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        let frames = Array(repeating: two, count: 4) + Array(repeating: fixture.black, count: 2)
            + Array(repeating: fixture.hand, count: 3)
        let selected: Set<Int> = [0, 2, 3, 5, 7]
        for (name, delegate, cache) in availableDelegates(fixture) {
            let plain = try makeLandmarker(fixture, numHands: 2, delegate: delegate,
                                           coreMLCache: cache, stageOutputs: false)
            let debug = try makeLandmarker(fixture, numHands: 2, delegate: delegate,
                                           coreMLCache: cache, tensorCapture: true,
                                           palmCandidates: 16)
            for (index, frame) in frames.enumerated() {
                let timestampMs = index * frameIntervalMs
                let a = try plain.detectForVideo(cgImage: frame, timestampInMilliseconds: timestampMs)
                let b = try debug.detectForVideo(cgImage: frame, timestampInMilliseconds: timestampMs,
                                                 captureTensors: selected.contains(index))
                XCTAssertEqual(a.landmarks, b.landmarks, "\(name) frame \(index)")
                XCTAssertEqual(a.worldLandmarks, b.worldLandmarks, "\(name) frame \(index)")
                XCTAssertEqual(a.handedness, b.handedness, "\(name) frame \(index)")
                let stages = try XCTUnwrap(b.stages)
                XCTAssertEqual(stages.timestampMicroseconds, Int64(timestampMs) * 1000,
                               "\(name) frame \(index)")
                guard selected.contains(index) else {
                    XCTAssertNil(b.tensorCapture, "\(name) frame \(index): not requested")
                    continue
                }
                let capture = try XCTUnwrap(b.tensorCapture, "\(name) frame \(index)")
                XCTAssertEqual(capture.timestampMicroseconds, Int64(timestampMs) * 1000,
                               "\(name) frame \(index)")
                XCTAssertEqual(capture.palmDetectorInput != nil, stages.palmDetectorRan,
                               "\(name) frame \(index)")
                // One entry per ROI; a tensor for exactly the ROIs the model
                // scored.
                XCTAssertEqual(capture.landmarkInputs.count, stages.landmarkROIs.count,
                               "\(name) frame \(index)")
                XCTAssertEqual(capture.landmarkInputs.filter { $0.tensor != nil }.count,
                               stages.presenceScores.count, "\(name) frame \(index)")
            }
        }
    }

    /// The CPU and GPU preprocessing paths give the palm model the same image
    /// up to resampling differences, i.e. the GPU tensor really is read back.
    /// (The landmark inputs are crops of each delegate's own detection, so
    /// they are only printed.)
    func testCapturedTensorsAgreeAcrossDelegates() throws {
        let fixture = try fixture()
        let delegates = availableDelegates(fixture)
        try XCTSkipIf(delegates.count < 2, "Only the CPU delegate is available.")
        var palm: [String: [Float]] = [:]
        var crop: [String: [Float]] = [:]
        for (name, delegate, cache) in delegates {
            let landmarker = try makeLandmarker(fixture, mode: .image, delegate: delegate,
                                                coreMLCache: cache, tensorCapture: true)
            let result = try landmarker.detect(cgImage: fixture.hand, captureTensors: true)
            let capture = try XCTUnwrap(result.tensorCapture, name)
            print("[\(name)] palm input shape \(capture.palmDetectorInput?.shape ?? []), "
                  + "landmark input shape \(capture.landmarkInputs.first?.tensor?.shape ?? [])")
            palm[name] = assertModelInput(capture.palmDetectorInput, side: 192, "\(name) palm")
            crop[name] = assertModelInput(capture.landmarkInputs.first?.tensor, side: 224,
                                          "\(name) landmark")
            print("[\(name)] value range: palm input \(palm[name]?.min() ?? .nan)...\(palm[name]?.max() ?? .nan), "
                  + "landmark input \(crop[name]?.min() ?? .nan)...\(crop[name]?.max() ?? .nan)")
            if let raw = capture.palmDetectorInput?.floatValues(),
               let channels = capture.palmDetectorInput?.shape.last, channels == 4 {
                let fourth = stride(from: 3, to: raw.count, by: 4).map { raw[$0] }
                print("[\(name)] fourth channel range: \(fourth.min() ?? .nan)...\(fourth.max() ?? .nan)")
            }
        }
        func meanAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(max(a.count, 1))
        }
        let cpuPalm = try XCTUnwrap(palm["cpu"]), cpuCrop = try XCTUnwrap(crop["cpu"])
        for (name, _, _) in delegates.dropFirst() {
            let palmDifference = meanAbsDifference(cpuPalm, try XCTUnwrap(palm[name]))
            let cropDifference = meanAbsDifference(cpuCrop, try XCTUnwrap(crop[name]))
            print("[\(name)] mean |cpu - \(name)|: palm input \(palmDifference), "
                  + "landmark input \(cropDifference)")
            XCTAssertLessThan(palmDifference, 0.03, "\(name) palm input")
        }
    }

    /// Writes the captured model inputs of the test image as PNG files, one
    /// per delegate and model, to look at what each model was given:
    ///   MP_DUMP_DIR=/tmp/tensors MP_HAND_MODEL=... MP_HAND_IMAGE=... swift test \
    ///     --filter HandStageOutputsTests/testDumpCapturedTensors
    func testDumpCapturedTensors() throws {
        guard let directory = env("MP_DUMP_DIR") else { throw XCTSkip("Set MP_DUMP_DIR to run.") }
        let fixture = try fixture()
        try FileManager.default.createDirectory(atPath: directory,
                                                withIntermediateDirectories: true)
        func write(_ tensor: CapturedTensor?, _ name: String) throws {
            guard let tensor, tensor.shape.count == 4, let values = tensor.floatValues() else { return }
            let height = tensor.shape[1], width = tensor.shape[2], channels = tensor.shape[3]
            var pixels = [UInt8](repeating: 255, count: width * height * 4)
            for pixel in 0..<(width * height) {
                for channel in 0..<3 {
                    let value = values[pixel * channels + channel]
                    pixels[pixel * 4 + channel] = UInt8(max(0, min(255, (value * 255).rounded())))
                }
            }
            let context = try XCTUnwrap(pixels.withUnsafeMutableBytes {
                CGContext(data: $0.baseAddress, width: width, height: height, bitsPerComponent: 8,
                          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
            })
            let url = URL(fileURLWithPath: directory).appendingPathComponent(name + ".png")
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
                url as CFURL, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, context, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
        }
        for (name, delegate, cache) in availableDelegates(fixture) {
            let landmarker = try makeLandmarker(fixture, mode: .image, delegate: delegate,
                                                coreMLCache: cache, tensorCapture: true)
            let result = try landmarker.detect(cgImage: fixture.hand, captureTensors: true)
            let capture = try XCTUnwrap(result.tensorCapture)
            let stages = try XCTUnwrap(result.stages)
            print("[\(name)] landmark ROI \(stages.landmarkROIs.map(describe))")
            try write(capture.palmDetectorInput, "\(name)_palm_input")
            for input in capture.landmarkInputs {
                try write(input.tensor, "\(name)_landmark_input_\(input.roiIndex)")
            }
        }
    }

    func testTensorCaptureInImageMode() throws {
        let fixture = try fixture()
        let landmarker = try makeLandmarker(fixture, mode: .image, tensorCapture: true)
        XCTAssertNil(try landmarker.detect(cgImage: fixture.hand).tensorCapture)
        let result = try landmarker.detect(cgImage: fixture.hand, captureTensors: true)
        let capture = try XCTUnwrap(result.tensorCapture)
        _ = assertModelInput(capture.palmDetectorInput, side: 192, "palm")
        XCTAssertEqual(capture.landmarkInputs.count, 1)
        _ = assertModelInput(capture.landmarkInputs.first?.tensor, side: 224, "landmark")
        let buffer = try makeBGRAPixelBuffer(from: fixture.hand)
        XCTAssertNotNil(try landmarker.detect(pixelBuffer: buffer, captureTensors: true).tensorCapture)
    }

    func testPalmCandidates() throws {
        let fixture = try fixture()
        let limit = 32
        let two = try XCTUnwrap(canvas(fixture.source,
                                       placements: [(0.45, -0.25, 0), (0.45, 0.25, 0)]))
        // Black: nothing above the threshold. Two hands with numHands 1: the
        // output limit cuts one. Then a tracked frame: the detector is skipped.
        let frames = [fixture.black, two, two]
        for (name, delegate, cache) in availableDelegates(fixture) {
            let landmarker = try makeLandmarker(fixture, numHands: 1, delegate: delegate,
                                                coreMLCache: cache, palmCandidates: limit)
            var records: [HandLandmarkerPalmCandidates] = []
            var stages: [HandLandmarkerStages] = []
            for (index, frame) in frames.enumerated() {
                let timestampMs = index * frameIntervalMs
                let result = try landmarker.detectForVideo(
                    cgImage: frame, timestampInMilliseconds: timestampMs)
                let record = try XCTUnwrap(result.palmCandidates, "\(name) frame \(index)")
                XCTAssertEqual(record.timestampMicroseconds, Int64(timestampMs) * 1000,
                               "\(name) frame \(index)")
                let stage = try XCTUnwrap(result.stages)
                XCTAssertEqual(record.palmDetectorRan, stage.palmDetectorRan,
                               "\(name) frame \(index)")
                XCTAssertEqual(record.afterSuppressionCount, stage.palmDetections.count,
                               "\(name) frame \(index)")
                XCTAssertNil(result.tensorCapture)
                records.append(record)
                stages.append(stage)
            }

            // Scores fell below the threshold.
            XCTAssertTrue(records[0].palmDetectorRan, name)
            XCTAssertEqual(records[0].candidates.count, limit, name)
            XCTAssertEqual(records[0].candidates.map(\.score),
                           records[0].candidates.map(\.score).sorted(by: >), "\(name): highest first")
            XCTAssertLessThan(try XCTUnwrap(records[0].candidates.first).score, 0.5, name)
            XCTAssertEqual(records[0].aboveThresholdCount, 0, name)
            XCTAssertEqual(records[0].afterSuppressionCount, 0, name)

            // Candidates merged by suppression, then cut by the output limit.
            let found = records[1]
            XCTAssertEqual(found.candidates.count, limit, name)
            XCTAssertEqual(found.candidates.map(\.score),
                           found.candidates.map(\.score).sorted(by: >), "\(name): highest first")
            XCTAssertGreaterThan(try XCTUnwrap(found.candidates.first).score, 0.5, name)
            XCTAssertEqual(try XCTUnwrap(found.candidates.first).keypoints.count, 7, name)
            XCTAssertGreaterThanOrEqual(found.afterSuppressionCount, 2, "\(name): two hands")
            XCTAssertGreaterThanOrEqual(found.aboveThresholdCount, found.afterSuppressionCount, name)
            if found.aboveThresholdCount < limit {
                XCTAssertEqual(found.candidates.filter { $0.score >= 0.5 }.count,
                               found.aboveThresholdCount, name)
            }
            XCTAssertEqual(stages[1].detectorHandRects.count, 1, "\(name): clipped to numHands")
            // Same coordinates as the palm detections: the best candidate lies
            // on one of them.
            let best = try XCTUnwrap(found.candidates.first)
            let centerX = best.xMin + best.width / 2, centerY = best.yMin + best.height / 2
            XCTAssertTrue(stages[1].palmDetections.contains {
                centerX > $0.xMin && centerX < $0.xMin + $0.width
                    && centerY > $0.yMin && centerY < $0.yMin + $0.height
            }, "\(name): best candidate is not on a palm detection")
            print("[\(name)] candidates: " + found.candidates.prefix(10).map {
                String(format: "%.2f@(%.3f,%.3f %.3fx%.3f)", $0.score, $0.xMin, $0.yMin,
                       $0.width, $0.height)
            }.joined(separator: " "))
            print("[\(name)] palm detections: " + stages[1].palmDetections.map {
                String(format: "%.2f@(%.3f,%.3f %.3fx%.3f)", $0.score, $0.xMin, $0.yMin,
                       $0.width, $0.height)
            }.joined(separator: " "))
            print("[\(name)] two hands, numHands 1: above threshold \(found.aboveThresholdCount), "
                  + "after suppression \(found.afterSuppressionCount), "
                  + "hand rects \(stages[1].detectorHandRects.count); "
                  + "black frame best score \(records[0].candidates.first?.score ?? -1)")

            // Detector skipped.
            XCTAssertFalse(records[2].palmDetectorRan, name)
            XCTAssertTrue(records[2].candidates.isEmpty, name)
            XCTAssertEqual(records[2].aboveThresholdCount, 0, name)
            XCTAssertEqual(records[2].afterSuppressionCount, 0, name)
        }
    }

    /// Palm candidates without the stage record, and in image mode.
    func testPalmCandidatesAlone() throws {
        let fixture = try fixture()
        let video = try makeLandmarker(fixture, stageOutputs: false, palmCandidates: 4)
        for (index, frame) in [fixture.hand, fixture.hand, fixture.black].enumerated() {
            let result = try video.detectForVideo(
                cgImage: frame, timestampInMilliseconds: index * frameIntervalMs)
            XCTAssertNil(result.stages)
            let record = try XCTUnwrap(result.palmCandidates, "frame \(index)")
            XCTAssertEqual(record.timestampMicroseconds, Int64(index * frameIntervalMs) * 1000)
            XCTAssertEqual(record.palmDetectorRan, index == 0, "frame \(index)")
            XCTAssertEqual(record.candidates.count, index == 0 ? 4 : 0, "frame \(index)")
            XCTAssertEqual(record.afterSuppressionCount >= 1, index == 0, "frame \(index)")
        }
        let image = try makeLandmarker(fixture, mode: .image, stageOutputs: false, palmCandidates: 4)
        let record = try XCTUnwrap(try image.detect(cgImage: fixture.hand).palmCandidates)
        XCTAssertTrue(record.palmDetectorRan)
        XCTAssertEqual(record.candidates.count, 4)
        XCTAssertGreaterThanOrEqual(record.aboveThresholdCount, 1)
    }
}
