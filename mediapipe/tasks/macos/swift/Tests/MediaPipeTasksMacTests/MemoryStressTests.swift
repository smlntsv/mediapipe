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
import Darwin
import ImageIO
import XCTest

@testable import MediaPipeTasksMac

/// Non-camera stress tests that run `detectForVideo(pixelBuffer:)` many times on
/// the same buffer and assert that resident memory (phys_footprint) stays
/// bounded. This isolates *library* leaks from AVCapture/SwiftUI demo leaks.
///
/// Env-gated like the other tests:
///   MP_HAND_MODEL / MP_POSE_MODEL / MP_FACE_MODEL  + a *_IMAGE for each.
/// Optional: MP_STRESS_FRAMES (default 1000), MP_STRESS_NO_POOL=1 to reproduce
/// the unbounded-autorelease behavior (omit the per-iteration autoreleasepool).
final class MemoryStressTests: XCTestCase {

    /// Resident memory (phys_footprint) in MB — the metric macOS uses for jetsam.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576.0
    }

    private var frames: Int { Int(ProcessInfo.processInfo.environment["MP_STRESS_FRAMES"] ?? "") ?? 1000 }
    private var usePool: Bool { ProcessInfo.processInfo.environment["MP_STRESS_NO_POOL"] != "1" }

    private func env(_ k: String) -> String? {
        guard let v = ProcessInfo.processInfo.environment[k], !v.isEmpty else { return nil }
        return v
    }

    private func bgraPixelBuffer(fromImageAt path: String) throws -> CVPixelBuffer {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let src = CGImageSourceCreateWithURL(url, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw XCTSkip("could not load \(path)")
        }
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, cg.width, cg.height,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { throw XCTSkip("CVPixelBufferCreate failed") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let info = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: cg.width, height: cg.height,
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info) else {
            throw XCTSkip("CGContext failed")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        return buffer
    }

    /// Runs `body` `measureFrames` times (each wrapped in autoreleasepool unless
    /// MP_STRESS_NO_POOL=1), after `warmup` un-measured iterations, and returns
    /// the footprint growth in MB across the measured window.
    ///
    /// `warmup` matters for the GPU path: the Metal/OpenGL texture caches keep a
    /// bounded (≈1 second) working set of in-flight IOSurfaces, which fills up
    /// once at the start. A large warmup absorbs that one-time ramp so the
    /// measured window reflects *steady-state* growth — i.e. a true leak — rather
    /// than the working-set fill.
    private func measureGrowth(label: String, warmup: Int, measureFrames: Int,
                               _ body: (Int) throws -> Void) rethrows -> Double {
        for i in 0..<warmup {
            if usePool { try autoreleasepool { try body(i) } } else { try body(i) }
        }
        let before = Self.footprintMB()
        for i in 0..<measureFrames {
            if usePool {
                try autoreleasepool { try body(warmup + i) }
            } else {
                try body(warmup + i)
            }
        }
        let after = Self.footprintMB()
        let growth = after - before
        print(String(format: "[MemoryStress] %@: %d warmup + %d frames, footprint %.1f → %.1f MB "
                     + "(Δ %+.1f MB = %+.3f MB/frame, pool=%@)",
                     label, warmup, measureFrames, before, after, growth,
                     growth / Double(measureFrames), usePool ? "yes" : "no"))
        return growth
    }

    // CPU is leak-free, so a small warmup + modest window is enough.
    private let cpuWarmup = 5
    private var cpuFrames: Int { frames }
    private let maxCPUGrowthMB = 150.0

    // GPU: the per-frame Metal/GL texture-cache flush (MPPMetalHelper.cc,
    // gpu_buffer_storage_cv_pixel_buffer.cc) makes the IOSurface working set
    // bounded. A long warmup saturates that working set, then we assert the
    // steady-state growth *rate* is near zero. The pre-fix leak was
    // ~0.77–1.54 MB/frame, so this threshold catches a regression with wide
    // margin while tolerating the bounded sawtooth (±~0.05 MB/frame at stress
    // speed).
    private let gpuWarmup = 1500
    private let gpuFrames = 4000
    private let maxGPURateMBPerFrame = 0.20

    // CPU stress (enforced: must stay bounded).
    func testHandStressCPU() throws { try assertCPUStable("hand", "MP_HAND_MODEL", "MP_HAND_IMAGE") }
    func testPoseStressCPU() throws { try assertCPUStable("pose", "MP_POSE_MODEL", "MP_POSE_IMAGE") }
    func testFaceStressCPU() throws { try assertCPUStable("face", "MP_FACE_MODEL", "MP_FACE_IMAGE") }

    // GPU stress (enforced: steady-state must be bounded after the working-set ramp).
    func testHandStressGPU() throws { try assertGPUBounded("hand", "MP_HAND_MODEL", "MP_HAND_IMAGE") }
    func testPoseStressGPU() throws { try assertGPUBounded("pose", "MP_POSE_MODEL", "MP_POSE_IMAGE") }
    func testFaceStressGPU() throws { try assertGPUBounded("face", "MP_FACE_MODEL", "MP_FACE_IMAGE") }

    private func assertCPUStable(_ kind: String, _ modelEnv: String, _ imageEnv: String) throws {
        guard let model = env(modelEnv), let img = env(imageEnv) else {
            throw XCTSkip("Set \(modelEnv) and \(imageEnv).")
        }
        let growth = try measure(kind: kind, model: model, image: img, delegate: .cpu,
                                 warmup: cpuWarmup, measureFrames: cpuFrames)
        XCTAssertLessThan(growth, maxCPUGrowthMB, "\(kind)/CPU grew \(growth) MB over \(cpuFrames) frames")
    }

    private func assertGPUBounded(_ kind: String, _ modelEnv: String, _ imageEnv: String) throws {
        try XCTSkipUnless(mediaPipeGPUArtifactAvailable, "CPU-only artifact.")
        guard let model = env(modelEnv), let img = env(imageEnv) else {
            throw XCTSkip("Set \(modelEnv) and \(imageEnv).")
        }
        // MP_STRESS_NO_POOL/MP_STRESS_FRAMES still apply to the CPU tests; the GPU
        // test uses fixed warmup/window so the steady-state rate is meaningful.
        let growth = try measure(kind: kind, model: model, image: img, delegate: .gpu,
                                 warmup: gpuWarmup, measureFrames: gpuFrames)
        let rate = growth / Double(gpuFrames)
        XCTAssertLessThan(
            rate, maxGPURateMBPerFrame,
            "\(kind)/GPU leaked \(rate) MB/frame (\(growth) MB over \(gpuFrames) steady-state frames "
            + "after \(gpuWarmup) warmup) — the macOS Metal texture-cache flush may have regressed.")
    }

    private func measure(kind: String, model: String, image: String,
                         delegate: MediaPipeDelegate, warmup: Int, measureFrames: Int) throws -> Double {
        let pb = try bgraPixelBuffer(fromImageAt: image)
        switch kind {
        case "hand":
            let o = HandLandmarkerOptions(); o.modelPath = model; o.numHands = 2
            o.delegate = delegate; o.runningMode = .video
            let lm = try HandLandmarker(options: o)
            return try measureGrowth(label: "hand/\(delegate.rawValue)", warmup: warmup, measureFrames: measureFrames) { ts in
                _ = try lm.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts + 1)
            }
        case "pose":
            let o = PoseLandmarkerOptions(); o.modelPath = model; o.delegate = delegate; o.runningMode = .video
            let lm = try PoseLandmarker(options: o)
            return try measureGrowth(label: "pose/\(delegate.rawValue)", warmup: warmup, measureFrames: measureFrames) { ts in
                _ = try lm.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts + 1)
            }
        default:
            let o = FaceLandmarkerOptions(); o.modelPath = model
            o.outputFaceBlendshapes = true; o.outputFacialTransformationMatrixes = true
            o.delegate = delegate; o.runningMode = .video
            let lm = try FaceLandmarker(options: o)
            return try measureGrowth(label: "face/\(delegate.rawValue)", warmup: warmup, measureFrames: measureFrames) { ts in
                _ = try lm.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts + 1)
            }
        }
    }
}

