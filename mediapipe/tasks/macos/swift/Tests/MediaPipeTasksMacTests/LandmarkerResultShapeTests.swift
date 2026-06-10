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

    /// Renders a CGImage into a `kCVPixelFormatType_32BGRA` CVPixelBuffer.
    private func makeBGRAPixelBuffer(from cgImage: CGImage) throws -> CVPixelBuffer {
        let w = cgImage.width, h = cgImage.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw XCTSkip("CVPixelBufferCreate failed (\(status))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        // 32BGRA == little-endian + premultipliedFirst.
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

    // MARK: - VIDEO mode (CPU)

    func testHandLandmarkerVideoMode() throws {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = HandLandmarkerOptions()
        options.modelPath = model
        options.numHands = 2
        options.delegate = .cpu
        options.runningMode = .video
        let landmarker = try HandLandmarker(options: options)
        for ts in [0, 33, 66] {  // increasing timestamps (~30fps)
            let result = try landmarker.detectForVideo(cgImage: image, timestampInMilliseconds: ts)
            XCTAssertFalse(result.landmarks.isEmpty, "expected a hand at t=\(ts)ms")
            XCTAssertEqual(result.landmarks[0].count, 21)
            XCTAssertEqual(result.worldLandmarks[0].count, 21)
        }
        // IMAGE-mode method must reject a VIDEO landmarker.
        XCTAssertThrowsError(try landmarker.detect(cgImage: image)) { error in
            guard case MediaPipeError.invalidRunningMode = error else {
                return XCTFail("expected invalidRunningMode, got \(error)")
            }
        }
    }

    func testPoseLandmarkerVideoMode() throws {
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = PoseLandmarkerOptions()
        options.modelPath = model
        options.runningMode = .video
        let landmarker = try PoseLandmarker(options: options)
        for ts in [0, 33, 66] {
            let result = try landmarker.detectForVideo(cgImage: image, timestampInMilliseconds: ts)
            XCTAssertFalse(result.landmarks.isEmpty, "expected a pose at t=\(ts)ms")
            XCTAssertEqual(result.landmarks[0].count, 33)
            XCTAssertEqual(result.worldLandmarks[0].count, 33)
        }
    }

    func testFaceLandmarkerVideoMode() throws {
        guard let model = env("MP_FACE_MODEL"), let imagePath = env("MP_FACE_IMAGE") else {
            throw XCTSkip("Set MP_FACE_MODEL and MP_FACE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = FaceLandmarkerOptions()
        options.modelPath = model
        options.runningMode = .video
        let landmarker = try FaceLandmarker(options: options)
        for ts in [0, 33, 66] {
            let result = try landmarker.detectForVideo(cgImage: image, timestampInMilliseconds: ts)
            XCTAssertFalse(result.faceLandmarks.isEmpty, "expected a face at t=\(ts)ms")
            XCTAssertEqual(result.faceLandmarks[0].count, 478)
        }
    }

    func testImageModeRejectsVideoCall() throws {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let options = HandLandmarkerOptions()
        options.modelPath = model  // default runningMode == .image
        let landmarker = try HandLandmarker(options: options)
        XCTAssertThrowsError(
            try landmarker.detectForVideo(cgImage: image, timestampInMilliseconds: 0)
        ) { error in
            guard case MediaPipeError.invalidRunningMode = error else {
                return XCTFail("expected invalidRunningMode, got \(error)")
            }
        }
    }

    // MARK: - Delegate

    func testGPUDelegateUnsupportedOnCPUArtifact() throws {
        try XCTSkipIf(mediaPipeGPUArtifactAvailable,
                      "Artifact is GPU-capable; .gpu is accepted (see GPU parity tests).")
        // The delegate check runs before native creation, so no model is needed.
        let options = HandLandmarkerOptions()
        options.modelPath = "/nonexistent.task"
        options.delegate = .gpu
        XCTAssertThrowsError(try HandLandmarker(options: options)) { error in
            guard case MediaPipeError.unsupportedDelegate = error else {
                return XCTFail("expected unsupportedDelegate, got \(error)")
            }
        }
    }

    // MARK: - GPU (only when a GPU-capable artifact is linked)

    func testHandLandmarkerGPUParity() throws {
        try XCTSkipUnless(mediaPipeGPUArtifactAvailable, "CPU-only artifact.")
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)

        func run(_ delegate: MediaPipeDelegate) throws -> HandLandmarkerResult {
            let o = HandLandmarkerOptions()
            o.modelPath = model; o.numHands = 2; o.delegate = delegate
            return try HandLandmarker(options: o).detect(cgImage: image)
        }
        let cpu = try run(.cpu)
        let gpu = try run(.gpu)

        // Same detection + landmark counts on both delegates.
        XCTAssertEqual(gpu.landmarks.count, cpu.landmarks.count)
        XCTAssertFalse(gpu.landmarks.isEmpty)
        XCTAssertEqual(gpu.landmarks[0].count, 21)
        XCTAssertEqual(gpu.worldLandmarks[0].count, 21)

        // CPU vs GPU normalized x/y should agree closely.
        var maxXY: Float = 0
        for h in 0..<min(cpu.landmarks.count, gpu.landmarks.count) {
            for i in 0..<min(cpu.landmarks[h].count, gpu.landmarks[h].count) {
                maxXY = max(maxXY, abs(cpu.landmarks[h][i].x - gpu.landmarks[h][i].x))
                maxXY = max(maxXY, abs(cpu.landmarks[h][i].y - gpu.landmarks[h][i].y))
            }
        }
        XCTAssertLessThan(maxXY, 0.02, "CPU vs GPU max x/y diff \(maxXY)")
    }

    func testPoseLandmarkerGPUSmoke() throws {
        try XCTSkipUnless(mediaPipeGPUArtifactAvailable, "CPU-only artifact.")
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let o = PoseLandmarkerOptions()
        o.modelPath = model; o.delegate = .gpu
        let result = try PoseLandmarker(options: o).detect(cgImage: image)
        XCTAssertFalse(result.landmarks.isEmpty)
        XCTAssertEqual(result.landmarks[0].count, 33)
    }

    func testFaceLandmarkerGPUSmoke() throws {
        try XCTSkipUnless(mediaPipeGPUArtifactAvailable, "CPU-only artifact.")
        guard let model = env("MP_FACE_MODEL"), let imagePath = env("MP_FACE_IMAGE") else {
            throw XCTSkip("Set MP_FACE_MODEL and MP_FACE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let o = FaceLandmarkerOptions()
        o.modelPath = model; o.delegate = .gpu
        let result = try FaceLandmarker(options: o).detect(cgImage: image)
        XCTAssertFalse(result.faceLandmarks.isEmpty)
        XCTAssertEqual(result.faceLandmarks[0].count, 478)
    }

    func testGPUVideoBenchmark() throws {
        try XCTSkipUnless(mediaPipeGPUArtifactAvailable, "CPU-only artifact.")
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let image = try loadCGImage(imagePath)
        let frames = 20

        func bench(_ delegate: MediaPipeDelegate) throws -> Double {
            let o = PoseLandmarkerOptions()
            o.modelPath = model; o.delegate = delegate; o.runningMode = .video
            let lm = try PoseLandmarker(options: o)
            // Warm up.
            _ = try lm.detectForVideo(cgImage: image, timestampInMilliseconds: 0)
            let start = ProcessInfo.processInfo.systemUptime
            for i in 1...frames {
                _ = try lm.detectForVideo(cgImage: image, timestampInMilliseconds: i * 33)
            }
            return (ProcessInfo.processInfo.systemUptime - start) / Double(frames) * 1000.0
        }
        let cpuMs = try bench(.cpu)
        let gpuMs = try bench(.gpu)
        print("[GPUVideoBenchmark] pose VIDEO ms/frame — CPU: \(String(format: "%.2f", cpuMs)), "
              + "GPU: \(String(format: "%.2f", gpuMs)) (\(frames) frames, report-only)")
    }

    // MARK: - CVPixelBuffer input

    func testHandLandmarkerPixelBufferImageMode() throws {
        guard let model = env("MP_HAND_MODEL"), let imagePath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_HAND_MODEL and MP_HAND_IMAGE to run this test.")
        }
        let cgImage = try loadCGImage(imagePath)
        let pixelBuffer = try makeBGRAPixelBuffer(from: cgImage)
        let options = HandLandmarkerOptions()
        options.modelPath = model
        options.numHands = 2
        let landmarker = try HandLandmarker(options: options)

        let pb = try landmarker.detect(pixelBuffer: pixelBuffer)
        XCTAssertFalse(pb.landmarks.isEmpty, "expected a hand from the pixel buffer")
        XCTAssertEqual(pb.landmarks[0].count, 21)
        XCTAssertEqual(pb.worldLandmarks[0].count, 21)

        // CVPixelBuffer (BGRA→RGBA) should match the CGImage path closely.
        let cg = try landmarker.detect(cgImage: cgImage)
        XCTAssertEqual(pb.landmarks.count, cg.landmarks.count)
        var maxXY: Float = 0
        for h in 0..<min(pb.landmarks.count, cg.landmarks.count) {
            for i in 0..<min(pb.landmarks[h].count, cg.landmarks[h].count) {
                maxXY = max(maxXY, abs(pb.landmarks[h][i].x - cg.landmarks[h][i].x))
                maxXY = max(maxXY, abs(pb.landmarks[h][i].y - cg.landmarks[h][i].y))
            }
        }
        XCTAssertLessThan(maxXY, 0.01, "pixelBuffer vs cgImage max x/y diff \(maxXY)")
    }

    func testPoseLandmarkerPixelBufferVideoMode() throws {
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let pixelBuffer = try makeBGRAPixelBuffer(from: try loadCGImage(imagePath))
        let options = PoseLandmarkerOptions()
        options.modelPath = model
        options.runningMode = .video
        let landmarker = try PoseLandmarker(options: options)
        for ts in [0, 33, 66] {
            let result = try landmarker.detectForVideo(
                pixelBuffer: pixelBuffer, timestampInMilliseconds: ts)
            XCTAssertFalse(result.landmarks.isEmpty, "expected a pose at t=\(ts)ms")
            XCTAssertEqual(result.landmarks[0].count, 33)
        }
    }

    func testPixelBufferVideoBenchmark() throws {
        guard let model = env("MP_POSE_MODEL"), let imagePath = env("MP_POSE_IMAGE") else {
            throw XCTSkip("Set MP_POSE_MODEL and MP_POSE_IMAGE to run this test.")
        }
        let pixelBuffer = try makeBGRAPixelBuffer(from: try loadCGImage(imagePath))
        let frames = 20

        func bench(_ delegate: MediaPipeDelegate) throws -> Double {
            let o = PoseLandmarkerOptions()
            o.modelPath = model; o.delegate = delegate; o.runningMode = .video
            let lm = try PoseLandmarker(options: o)
            _ = try lm.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: 0)
            let start = ProcessInfo.processInfo.systemUptime
            for i in 1...frames {
                _ = try lm.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: i * 33)
            }
            return (ProcessInfo.processInfo.systemUptime - start) / Double(frames) * 1000.0
        }
        let cpuMs = try bench(.cpu)
        let gpuMs = mediaPipeGPUArtifactAvailable ? try bench(.gpu) : -1
        print("[PixelBufferVideoBenchmark] pose VIDEO (CVPixelBuffer) ms/frame — "
              + "CPU: \(String(format: "%.2f", cpuMs)), "
              + "GPU: \(gpuMs < 0 ? "n/a" : String(format: "%.2f", gpuMs)) "
              + "(\(frames) frames, report-only)")
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
