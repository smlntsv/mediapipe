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
import ImageIO
import XCTest

@testable import MediaPipeTasks

// Tests for `HandLandmarkerOptions.palmSearch`.
//
// The grid arithmetic and the option bridging need no model. The pipeline
// tests are skipped unless MP_HAND_MODEL and MP_HAND_IMAGE are set:
//   MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=test_image.jpg swift test \
//     --filter PalmSearchTests
final class PalmSearchTests: XCTestCase {

    // MARK: - Grid arithmetic

    func testRecommendedGridUsesFewerTilesOnTheShortAxis() {
        let hd = CGSize(width: 1280, height: 720)
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: hd, targetMagnification: 2),
                       PalmSearchGrid(rows: 2, columns: 2))
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: hd, targetMagnification: 3),
                       PalmSearchGrid(rows: 2, columns: 3))
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: hd, targetMagnification: 4),
                       PalmSearchGrid(rows: 3, columns: 4))
        // Portrait: the long axis is the height.
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: CGSize(width: 720, height: 1280),
                                                  targetMagnification: 4),
                       PalmSearchGrid(rows: 4, columns: 3))
        // Square frame: both axes need the same count.
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: CGSize(width: 512, height: 512),
                                                  targetMagnification: 3),
                       PalmSearchGrid(rows: 3, columns: 3))
        XCTAssertEqual(PalmSearchGrid.recommended(frameSize: hd, targetMagnification: 1),
                       PalmSearchGrid(rows: 1, columns: 1))
    }

    func testRecommendedGridReachesTheTargetWithOverlap() {
        let hd = CGSize(width: 1280, height: 720)
        for target: Float in [2, 3, 4] {
            for overlap: Float in [0, 0.1, 0.2, 0.3] {
                let grid = PalmSearchGrid.recommended(frameSize: hd, targetMagnification: target,
                                                      overlap: overlap)
                XCTAssertGreaterThanOrEqual(grid.magnification(frameSize: hd), target - 1e-3,
                                            "\(target)x overlap \(overlap): \(grid)")
                // One column fewer would not reach it.
                let fewer = PalmSearchGrid(rows: grid.rows, columns: grid.columns - 1,
                                           overlap: overlap)
                XCTAssertLessThan(fewer.magnification(frameSize: hd), target - 1e-3)
            }
        }
    }

    func testMagnificationAccountsForOverlapAndAspect() {
        let hd = CGSize(width: 1280, height: 720)
        XCTAssertEqual(PalmSearchGrid(rows: 4, columns: 4).magnification(frameSize: hd), 4,
                       accuracy: 1e-4)
        XCTAssertEqual(PalmSearchGrid(rows: 3, columns: 4).magnification(frameSize: hd), 4,
                       accuracy: 1e-4)
        XCTAssertEqual(PalmSearchGrid(rows: 4, columns: 4, overlap: 0.2)
                        .magnification(frameSize: hd), 3.4, accuracy: 1e-4)
        XCTAssertEqual(PalmSearchGrid(rows: 2, columns: 2, overlap: 0.1)
                        .magnification(frameSize: hd), 1.9, accuracy: 1e-4)
        // One row of two tiles: the tile's long side is the frame height.
        XCTAssertEqual(PalmSearchGrid(rows: 1, columns: 2).magnification(frameSize: hd),
                       1280.0 / 720.0, accuracy: 1e-4)
    }

    func testOptionsBridge() {
        let plain = PalmSearchOptions().bridged
        XCTAssertTrue(plain.includeFullFrame)
        XCTAssertEqual(plain.gridRows, 0)
        XCTAssertEqual(plain.roundRobinTilesPerFrame, 0)
        XCTAssertFalse(plain.clipBeforeAssociation)

        let options = PalmSearchOptions(
            includeFullFrame: false,
            grid: PalmSearchGrid(rows: 3, columns: 4, overlap: 0.2),
            schedule: .roundRobin(perFrame: 2),
            acceptsExternalRegions: true,
            rejectTruncatedDetections: true,
            maxPalmFraction: 0.25,
            merge: .bestScore,
            clipBeforeAssociation: true).bridged
        XCTAssertFalse(options.includeFullFrame)
        XCTAssertEqual(options.gridRows, 3)
        XCTAssertEqual(options.gridColumns, 4)
        XCTAssertEqual(options.gridOverlap, 0.2)
        XCTAssertEqual(options.roundRobinTilesPerFrame, 2)
        XCTAssertTrue(options.acceptsExternalRegions)
        XCTAssertTrue(options.rejectTruncatedDetections)
        XCTAssertEqual(options.maxPalmFraction, 0.25)
        XCTAssertEqual(options.merge, 1)
        XCTAssertTrue(options.clipBeforeAssociation)

        // A schedule without a grid schedules nothing.
        XCTAssertEqual(PalmSearchOptions(schedule: .roundRobin(perFrame: 1)).bridged
                        .roundRobinTilesPerFrame, 0)
    }

    // MARK: - Pipeline

    private func env(_ key: String) -> String? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return value
    }

    private struct Fixture {
        let model: String
        /// The test image at 0.45 scale in the middle of a black canvas.
        let hand: CGImage
        /// The test image at 0.3 scale in the top-left quarter.
        let smallHand: CGImage
        let black: CGImage
    }

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

    private func fixture() throws -> Fixture {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let url = URL(fileURLWithPath: imagePath) as CFURL
        guard let imageSource = CGImageSourceCreateWithURL(url, nil),
              let source = CGImageSourceCreateImageAtIndex(imageSource, 0, nil),
              let hand = canvas(source, placements: [(0.45, 0, 0)]),
              // CoreGraphics' origin is bottom-left: +dy is up.
              let smallHand = canvas(source, placements: [(0.3, -0.25, 0.25)]),
              let black = canvas(source, placements: []) else {
            throw XCTSkip("Could not load or compose the test image at \(imagePath)")
        }
        return Fixture(model: model, hand: hand, smallHand: smallHand, black: black)
    }

    private func landmarker(_ fixture: Fixture, mode: RunningMode,
                            palmSearch: PalmSearchOptions?,
                            numHands: Int = 2) throws -> HandLandmarker {
        try HandLandmarker(options: HandLandmarkerOptions(
            modelPath: fixture.model, numHands: numHands, runningMode: mode,
            stageOutputs: true, palmCandidateLimit: 8, palmSearch: palmSearch))
    }

    private func coordinates(_ result: HandLandmarkerResult) -> [[Float]] {
        result.landmarks.map { $0.flatMap { [$0.x, $0.y, $0.z] } }
    }

    /// The whole frame through the search-region path gives what the pipeline
    /// gives without it: detections, rects, landmarks, value for value.
    func testFullFrameOnlyMatchesThePipelineWithoutPalmSearch() throws {
        let fx = try fixture()
        let frames = [fx.black, fx.hand, fx.hand, fx.black, fx.smallHand, fx.hand]
        for clip in [true, false] {
            let off = try landmarker(fx, mode: .video, palmSearch: nil)
            let on = try landmarker(fx, mode: .video,
                                    palmSearch: PalmSearchOptions(clipBeforeAssociation: clip))
            for (i, frame) in frames.enumerated() {
                let a = try off.detectForVideo(cgImage: frame, timestampInMilliseconds: i * 33)
                let b = try on.detectForVideo(cgImage: frame, timestampInMilliseconds: i * 33)
                XCTAssertEqual(coordinates(a), coordinates(b), "frame \(i) clip \(clip)")
                XCTAssertEqual(a.stages, b.stages, "frame \(i) clip \(clip)")
                XCTAssertEqual(a.palmCandidates, b.palmCandidates, "frame \(i) clip \(clip)")
                XCTAssertNil(a.palmSearch)
                let search = try XCTUnwrap(b.palmSearch)
                XCTAssertEqual(search.timestampMicroseconds, Int64(i * 33) * 1000)
                XCTAssertEqual(search.frameSequence, Int64(i))
                XCTAssertEqual(search.palmDetectorRan, b.stages?.palmDetectorRan)
                if search.palmDetectorRan {
                    XCTAssertEqual(search.regions.map(\.id), [0])
                    XCTAssertEqual(search.merged.map(\.detection),
                                   b.stages?.palmDetections)
                } else {
                    XCTAssertTrue(search.regions.isEmpty)
                }
            }
        }
    }

    /// A hand too small for the whole frame is found through a grid tile, and
    /// the record says which tile.
    func testGridFindsASmallHandAndRecordsItsRegion() throws {
        let fx = try fixture()
        let grid = PalmSearchGrid(rows: 2, columns: 2)
        let on = try landmarker(fx, mode: .image, palmSearch: PalmSearchOptions(grid: grid))
        let result = try on.detect(cgImage: fx.smallHand)
        let search = try XCTUnwrap(result.palmSearch)
        XCTAssertEqual(search.regions.map(\.id), [0, 1, 2, 3, 4])
        XCTAssertEqual(search.regions.map(\.kind), [.fullFrame, .grid, .grid, .grid, .grid])
        XCTAssertEqual(search.gridTileCount, 4)
        XCTAssertEqual(search.tilesPerFrame, 0)
        let tile = search.regions[1]
        XCTAssertEqual(tile.gridRow, 0)
        XCTAssertEqual(tile.gridColumn, 0)
        XCTAssertEqual(tile.requested.minX, 0, accuracy: 1e-6)
        XCTAssertEqual(tile.requested.width, 0.5, accuracy: 1e-6)
        // Each region carries its own candidates and clock stamps.
        for region in search.regions {
            XCTAssertEqual(region.candidates.count, 8)
            XCTAssertGreaterThan(region.cropStartNanoseconds, 0)
            XCTAssertGreaterThanOrEqual(region.detectionsReadyNanoseconds,
                                        region.cropStartNanoseconds)
        }
        XCTAssertGreaterThan(search.callEndNanoseconds, search.callStartNanoseconds)
        XCTAssertGreaterThanOrEqual(search.mergeDoneNanoseconds, search.regionsReadyNanoseconds)

        // The hand is in the top-left tile (region 1).
        // The test image shows two hands.
        XCTAssertGreaterThanOrEqual(result.landmarks.count, 1)
        let fromTopLeft = search.merged.filter { merged in
            merged.sources.contains { $0.regionID == 1 }
        }
        XCTAssertFalse(fromTopLeft.isEmpty)
        XCTAssertEqual(fromTopLeft.first?.outcome, .toLandmarks)
        XCTAssertEqual(fromTopLeft.first?.landmarkRoiIndex, 0)
        for merged in search.merged {
            for source in merged.sources {
                let region = try XCTUnwrap(search.regions.first { $0.id == source.regionID })
                XCTAssertEqual(region.detections[source.detectionIndex].outcome, .merged)
            }
        }
    }

    /// The round-robin cursor advances on detector frames only.
    func testRoundRobinCursorAdvancesPerDetectorFrame() throws {
        let fx = try fixture()
        let options = PalmSearchOptions(includeFullFrame: false,
                                        grid: PalmSearchGrid(rows: 1, columns: 3),
                                        schedule: .roundRobin(perFrame: 1))
        let on = try landmarker(fx, mode: .video, palmSearch: options, numHands: 1)
        var detectorFrames = 0
        // Black frames: the detector runs every frame, one tile each.
        for i in 0..<5 {
            let search = try XCTUnwrap(
                on.detectForVideo(cgImage: fx.black, timestampInMilliseconds: i * 33).palmSearch)
            XCTAssertTrue(search.palmDetectorRan)
            XCTAssertEqual(search.detectorFrameIndex, Int64(detectorFrames))
            XCTAssertEqual(search.cursorBefore, detectorFrames % 3)
            XCTAssertEqual(search.cursorAfter, (detectorFrames + 1) % 3)
            XCTAssertEqual(search.regions.map(\.id), [1 + detectorFrames % 3])
            detectorFrames += 1
        }
        // A large hand in the middle: found by the middle tile sooner or
        // later, then tracked, and the detector and the cursor stop.
        var skipped = 0
        for i in 5..<16 {
            let result = try on.detectForVideo(cgImage: fx.hand, timestampInMilliseconds: i * 33)
            let search = try XCTUnwrap(result.palmSearch)
            if search.palmDetectorRan {
                XCTAssertEqual(search.detectorFrameIndex, Int64(detectorFrames))
                XCTAssertEqual(search.cursorBefore, detectorFrames % 3)
                detectorFrames += 1
            } else {
                skipped += 1
                XCTAssertTrue(search.regions.isEmpty)
            }
        }
        XCTAssertGreaterThan(skipped, 0, "the hand was never tracked")
    }

    func testExternalRegions() throws {
        let fx = try fixture()
        let on = try landmarker(
            fx, mode: .image,
            palmSearch: PalmSearchOptions(includeFullFrame: false, acceptsExternalRegions: true))
        // No region: the detector has nothing to run on.
        let none = try on.detect(cgImage: fx.smallHand)
        XCTAssertTrue(none.landmarks.isEmpty)
        XCTAssertEqual(none.palmSearch?.regions.count, 0)

        // The top-left quarter, where the small hand is (origin top-left).
        let region = CGRect(x: 0, y: 0, width: 0.5, height: 0.5)
        let found = try on.detect(cgImage: fx.smallHand, searchRegions: [region])
        let search = try XCTUnwrap(found.palmSearch)
        XCTAssertEqual(search.regions.map(\.id), [1000])
        XCTAssertEqual(search.regions.first?.kind, .external)
        XCTAssertEqual(search.regions.first?.requested.width ?? 0, 0.5, accuracy: 1e-6)
        // The test image shows two hands; which of them the model finds at
        // this size is not what this test is about.
        XCTAssertGreaterThanOrEqual(found.landmarks.count, 1)

        // The bottom-right quarter holds nothing.
        let empty = try on.detect(cgImage: fx.smallHand,
                                  searchRegions: [CGRect(x: 0.5, y: 0.5, width: 0.5, height: 0.5)])
        XCTAssertTrue(empty.landmarks.isEmpty)
    }

    func testSearchRegionsRequireTheOption() throws {
        let fx = try fixture()
        let region = [CGRect(x: 0, y: 0, width: 0.5, height: 0.5)]
        let off = try landmarker(fx, mode: .image, palmSearch: nil)
        XCTAssertThrowsError(try off.detect(cgImage: fx.hand, searchRegions: region))
        let noExternal = try landmarker(fx, mode: .image, palmSearch: PalmSearchOptions())
        XCTAssertThrowsError(try noExternal.detect(cgImage: fx.hand, searchRegions: region))
        let on = try landmarker(fx, mode: .image,
                                palmSearch: PalmSearchOptions(acceptsExternalRegions: true))
        XCTAssertThrowsError(try on.detect(
            cgImage: fx.hand, searchRegions: [CGRect(x: 0.2, y: 0.2, width: 0, height: 0.5)]))
        XCTAssertNoThrow(try on.detect(cgImage: fx.hand, searchRegions: region))
    }
}
