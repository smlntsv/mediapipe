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
import ImageIO
import XCTest

@testable import MediaPipeTasksMac

// These tests run a real model on a real image and assert MediaPipe's result
// shapes and landmark counts. Because the `.task` models and test images are
// not committed, each case reads its paths from environment variables and is
// skipped when they are absent. Example:
//
//   MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=hand.jpg \
//   MP_POSE_MODEL=pose_landmarker.task MP_POSE_IMAGE=person.jpg \
//   MP_FACE_MODEL=face_landmarker.task MP_FACE_IMAGE=person.jpg \
//   swift test
final class LandmarkerResultShapeTests: XCTestCase {

    private func env(_ key: String) -> String? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return value
    }

    private func loadCGImage(_ path: String) throws -> CGImage {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let source = CGImageSourceCreateWithURL(url, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw XCTSkip("Could not load test image at \(path)")
        }
        return image
    }

    func testHandLandmarkerResultShape() throws {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = HandLandmarkerOptions()
        options.modelPath = model
        options.numHands = 2
        let result = try HandLandmarker(options: options).detect(cgImage: image)

        // Nested structure must be preserved (per-hand arrays, not flattened).
        XCTAssertFalse(result.landmarks.isEmpty, "expected at least one hand")
        XCTAssertEqual(result.landmarks[0].count, 21)
        XCTAssertEqual(result.worldLandmarks[0].count, 21)
        XCTAssertFalse(result.handedness.isEmpty)
        XCTAssertFalse(result.handedness[0].isEmpty)

        // Deprecated alias must mirror `handedness`.
        XCTAssertEqual(result.handednesses.count, result.handedness.count)
        XCTAssertEqual(result.handednesses[0].count, result.handedness[0].count)

        // Counts should line up across the per-hand arrays.
        XCTAssertEqual(result.landmarks.count, result.worldLandmarks.count)
        XCTAssertEqual(result.landmarks.count, result.handedness.count)
    }

    func testPoseLandmarkerResultShape() throws {
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = PoseLandmarkerOptions()
        options.modelPath = model
        let result = try PoseLandmarker(options: options).detect(cgImage: image)

        XCTAssertFalse(result.landmarks.isEmpty, "expected at least one pose")
        XCTAssertEqual(result.landmarks[0].count, 33)
        XCTAssertEqual(result.worldLandmarks[0].count, 33)
        // Masks are not produced in milestone 1.
        XCTAssertNil(result.segmentationMasks)
    }

    func testFaceLandmarkerResultShape() throws {
        guard let model = env("MP_FACE_MODEL"), let imagePath = env("MP_FACE_IMAGE") else {
            throw XCTSkip("Set MP_FACE_MODEL and MP_FACE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = FaceLandmarkerOptions()
        options.modelPath = model
        let result = try FaceLandmarker(options: options).detect(cgImage: image)

        XCTAssertFalse(result.faceLandmarks.isEmpty, "expected at least one face")
        XCTAssertEqual(result.faceLandmarks[0].count, 478)
        // Not requested by default.
        XCTAssertTrue(result.faceBlendshapes.isEmpty)
        XCTAssertTrue(result.facialTransformationMatrixes.isEmpty)
    }

    func testFaceLandmarkerBlendshapesAndMatrices() throws {
        guard let model = env("MP_FACE_MODEL"), let imagePath = env("MP_FACE_IMAGE") else {
            throw XCTSkip("Set MP_FACE_MODEL and MP_FACE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = FaceLandmarkerOptions()
        options.modelPath = model
        options.outputFaceBlendshapes = true
        options.outputFacialTransformationMatrixes = true
        let result = try FaceLandmarker(options: options).detect(cgImage: image)

        XCTAssertFalse(result.faceLandmarks.isEmpty, "expected at least one face")
        // One blendshape head per face, with categories populated.
        XCTAssertEqual(result.faceBlendshapes.count, result.faceLandmarks.count)
        XCTAssertFalse(result.faceBlendshapes[0].categories.isEmpty)
        // One 4x4 transformation matrix per face.
        XCTAssertEqual(result.facialTransformationMatrixes.count, result.faceLandmarks.count)
        let m = result.facialTransformationMatrixes[0]
        XCTAssertEqual(m.rows, 4)
        XCTAssertEqual(m.columns, 4)
        XCTAssertEqual(m.data.count, m.rows * m.columns)
    }
}
