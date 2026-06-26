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

// Verifies that `ImageProcessingOptions.rotationDegrees` is actually plumbed
// through to the native pipeline.
//
// The guarantee that matters: before this change the ObjC bridge passed
// `options = nullptr`, so rotation was *silently ignored*. The regression test
// below proves rotation now reaches the model by asserting that changing the
// rotation changes the detection result, while the default path (no rotation)
// is byte-for-byte unchanged.
//
// Why not a coordinate "round-trip"? MediaPipe returns landmarks in the supplied
// image's own coordinate frame regardless of rotation, and the bundled palm
// detector is highly rotation-robust — so on a normal image, rotating the pixels
// and applying the matching correction yields (correctly) the *same* landmarks
// whether or not rotation is set. That makes a coordinate round-trip unable to
// distinguish "rotation works" from "rotation ignored". Asserting that a
// non-zero rotation perturbs the result is the reliable signal.
//
// Skipped unless MP_HAND_MODEL and MP_HAND_IMAGE are set (model and image are
// not committed). Example:
//   MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=hand.jpg swift test
final class RotationOptionsTests: XCTestCase {

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

    /// Max per-landmark x/y difference between two hands (∞ if shapes differ).
    private func maxXY(_ a: [NormalizedLandmark], _ b: [NormalizedLandmark]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .greatestFiniteMagnitude }
        var m: Float = 0
        for i in 0..<a.count {
            m = max(m, abs(a[i].x - b[i].x))
            m = max(m, abs(a[i].y - b[i].y))
        }
        return m
    }

    // Pure validation — runs without a model or image.
    func testRotationDegreesMustBeMultipleOf90() {
        XCTAssertThrowsError(try ImageProcessingOptions(rotationDegrees: 45).validated()) { error in
            guard case MediaPipeError.invalidArgument = error else {
                return XCTFail("expected invalidArgument, got \(error)")
            }
        }
        for deg in [0, 90, -90, 180, 270, -270] {
            XCTAssertNoThrow(try ImageProcessingOptions(rotationDegrees: deg).validated(),
                             "rotationDegrees \(deg) should be valid")
        }
    }

    // Proves rotationDegrees reaches the model: the default path equals an
    // explicit 0°, and a 90° rotation changes what the model sees.
    func testRotationDegreesIsHonored() throws {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = HandLandmarkerOptions()
        options.modelPath = model
        options.numHands = 1  // single most-confident hand keeps the comparison stable
        let landmarker = try HandLandmarker(options: options)

        let defaultResult = try landmarker.detect(cgImage: image)
        try XCTSkipIf(defaultResult.landmarks.isEmpty, "No hand found in the test image.")
        let zero = try landmarker.detect(
            cgImage: image, imageProcessingOptions: ImageProcessingOptions(rotationDegrees: 0))
        let ninety = try landmarker.detect(
            cgImage: image, imageProcessingOptions: ImageProcessingOptions(rotationDegrees: 90))

        // The default (no options) must match an explicit 0° exactly — the
        // rotation plumbing must not perturb the existing behaviour.
        XCTAssertLessThan(maxXY(defaultResult.landmarks[0], zero.landmarks[0]), 0.001,
                          "Default detect() should equal rotationDegrees: 0.")

        // A 90° rotation feeds the model a different (sideways) image, so the
        // result must change. If rotation were ignored, this would be ~0.
        try XCTSkipIf(ninety.landmarks.isEmpty, "No hand at 90°; cannot compare.")
        let delta = maxXY(zero.landmarks[0], ninety.landmarks[0])
        print("[Rotation] max landmark delta between 0° and 90° = \(String(format: "%.3f", delta))")
        XCTAssertGreaterThan(delta, 0.1,
                             "rotationDegrees had no effect — is it reaching the model?")
    }
}
