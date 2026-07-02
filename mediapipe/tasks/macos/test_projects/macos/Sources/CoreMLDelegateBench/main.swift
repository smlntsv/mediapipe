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
//
// A/B benchmark for the Core ML (ANE) delegate across all three vision tasks.
//
// Hand runs VIDEO mode over a real clip (exercising the tracking loop and the
// detector-skip path); pose and face run VIDEO mode over a repeated still
// (first frame exercises the detector, steady state exercises tracking).
// For every task, .cpu / .gpu / .coreML are compared on latency and on
// landmark agreement against the .cpu baseline.
//
// Usage:
//   swift run -c release CoreMLDelegateBench <shared-dir>
// where <shared-dir> is mediapipe/tasks/macos/test_projects/shared
// (models/ + two-hands-only.mov + test_image*.jpg).

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MediaPipeTasksMac

let sharedDir = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : FileManager.default.currentDirectoryPath + "/../shared"
let modelsDir = sharedDir + "/models"

// MARK: - Helpers

func stats(_ values: [Double]) -> String {
    guard !values.isEmpty else { return "n/a" }
    let sorted = values.sorted()
    let mean = values.reduce(0, +) / Double(values.count)
    let p50 = sorted[sorted.count / 2]
    let p90 = sorted[min(Int(Double(sorted.count) * 0.9), sorted.count - 1)]
    return String(format: "%6.2f / %6.2f / %6.2f", mean, p50, p90)
}

func loadCGImage(_ path: String) -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        fatalError("cannot load image at \(path)")
    }
    return image
}

func forEachFrame(_ videoPath: String, _ body: (CVPixelBuffer, Int) throws -> Void) throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    guard let track = asset.tracks(withMediaType: .video).first else {
        fatalError("no video track in \(videoPath)")
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    reader.startReading()
    var lastTs = -1
    while reader.status == .reading, let sb = output.copyNextSampleBuffer() {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
        var ts = Int((CMSampleBufferGetPresentationTimeStamp(sb).seconds * 1000).rounded())
        if ts <= lastTs { ts = lastTs + 1 }
        lastTs = ts
        try body(pb, ts)
    }
}

struct PassResult {
    var ms: [Double] = []
    // Per frame: landmark sets (normalized), one [x,y] list per detection.
    var frames: [[[SIMD2<Float>]]] = []
}

func agreement(_ a: PassResult, _ b: PassResult, width: Float, height: Float) -> String {
    var pxErrors: [Double] = []
    var countMismatch = 0
    for i in 0..<min(a.frames.count, b.frames.count) {
        let fa = a.frames[i], fb = b.frames[i]
        if fa.count != fb.count { countMismatch += 1 }
        for da in fa {
            // Match by first-landmark proximity.
            guard !da.isEmpty, let db = fb.min(by: {
                simd_distance($0[0], da[0]) < simd_distance($1[0], da[0])
            }), db.count == da.count else { continue }
            var err: Float = 0
            for k in 0..<da.count {
                let dx = (da[k].x - db[k].x) * width
                let dy = (da[k].y - db[k].y) * height
                err += (dx * dx + dy * dy).squareRoot()
            }
            pxErrors.append(Double(err) / Double(da.count))
        }
    }
    return "px err mean/p50/p90: \(stats(pxErrors)) | count-mismatch frames: \(countMismatch)"
}

func detectionRate(_ r: PassResult, minCount: Int) -> Double {
    guard !r.frames.isEmpty else { return 0 }
    return Double(r.frames.filter { $0.count >= minCount }.count) / Double(r.frames.count) * 100
}

// MARK: - Hand: VIDEO over the real clip

func runHand(delegate: MediaPipeDelegate) throws -> PassResult {
    let options = HandLandmarkerOptions(
        modelPath: modelsDir + "/hand_landmarker.task",
        numHands: 2,
        delegate: delegate,
        coreMLModelCacheDirectory: modelsDir + "/coreml_models",
        runningMode: .video)
    let landmarker = try HandLandmarker(options: options)
    var result = PassResult()
    try forEachFrame(sharedDir + "/two-hands-only.mov") { pb, ts in
        let t0 = CFAbsoluteTimeGetCurrent()
        let res = try landmarker.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts)
        result.ms.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        result.frames.append(res.landmarks.map { $0.map { SIMD2($0.x, $0.y) } })
    }
    return result
}

// MARK: - Pose / Face: VIDEO over a repeated still

func runStill(_ makeDetect: () throws -> (CVPixelBuffer, Int) throws -> [[SIMD2<Float>]],
              image: CGImage, frames: Int) throws -> PassResult {
    // Render the CGImage into a BGRA pixel buffer once.
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height,
                        kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
                        &pb)
    guard let buffer = pb else { fatalError("CVPixelBufferCreate failed") }
    CVPixelBufferLockBaseAddress(buffer, [])
    let ctx = CGContext(
        data: CVPixelBufferGetBaseAddress(buffer),
        width: image.width, height: image.height, bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    CVPixelBufferUnlockBaseAddress(buffer, [])

    let detect = try makeDetect()
    var result = PassResult()
    for i in 0..<frames {
        let t0 = CFAbsoluteTimeGetCurrent()
        let landmarks = try detect(buffer, i * 33)
        result.ms.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        result.frames.append(landmarks)
    }
    return result
}

func runPose(delegate: MediaPipeDelegate, image: CGImage, frames: Int) throws -> PassResult {
    try runStill({
        let options = PoseLandmarkerOptions(
            modelPath: modelsDir + "/pose_landmarker.task",
            numPoses: 1,
            delegate: delegate,
            coreMLModelCacheDirectory: modelsDir + "/coreml_models",
            runningMode: .video)
        let landmarker = try PoseLandmarker(options: options)
        return { pb, ts in
            let res = try landmarker.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts)
            return res.landmarks.map { $0.map { SIMD2($0.x, $0.y) } }
        }
    }, image: image, frames: frames)
}

func runFace(delegate: MediaPipeDelegate, image: CGImage, frames: Int) throws -> PassResult {
    try runStill({
        let options = FaceLandmarkerOptions(
            modelPath: modelsDir + "/face_landmarker.task",
            numFaces: 1,
            outputFaceBlendshapes: true,  // exercises the (unconverted) blendshapes fallback
            delegate: delegate,
            coreMLModelCacheDirectory: modelsDir + "/coreml_models",
            runningMode: .video)
        let landmarker = try FaceLandmarker(options: options)
        return { pb, ts in
            let res = try landmarker.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts)
            return res.faceLandmarks.map { $0.map { SIMD2($0.x, $0.y) } }
        }
    }, image: image, frames: frames)
}

// MARK: - Main

let delegates: [MediaPipeDelegate] = [.cpu, .gpu, .coreML]
let stillFrames = 300

print("=== HAND (VIDEO, two-hands-only.mov, numHands=2) ===")
var handResults: [MediaPipeDelegate: PassResult] = [:]
for delegate in delegates {
    do {
        let r = try runHand(delegate: delegate)
        handResults[delegate] = r
        print(String(format: "%-7@ ms(mean/p50/p90): %@ | >=1 hand %5.1f%% | both %5.1f%%",
                     delegate.rawValue as NSString, stats(r.ms) as NSString,
                     detectionRate(r, minCount: 1), detectionRate(r, minCount: 2)))
    } catch {
        print("\(delegate.rawValue): FAILED — \(error)")
    }
}
if let cpu = handResults[.cpu], let coreml = handResults[.coreML] {
    print("coreML vs cpu agreement: \(agreement(coreml, cpu, width: 1280, height: 720))")
}

let fullBody = loadCGImage(sharedDir + "/test_image_full_body.jpg")
print("\n=== POSE (VIDEO, repeated still test_image_full_body.jpg, \(stillFrames) frames) ===")
var poseResults: [MediaPipeDelegate: PassResult] = [:]
for delegate in delegates {
    do {
        let r = try runPose(delegate: delegate, image: fullBody, frames: stillFrames)
        poseResults[delegate] = r
        // Skip the first (detector) frame in latency stats: steady-state is the
        // tracked path.
        print(String(format: "%-7@ ms(mean/p50/p90): %@ | first %6.2f | detected %5.1f%%",
                     delegate.rawValue as NSString,
                     stats(Array(r.ms.dropFirst(10))) as NSString,
                     r.ms.first ?? -1, detectionRate(r, minCount: 1)))
    } catch {
        print("\(delegate.rawValue): FAILED — \(error)")
    }
}
if let cpu = poseResults[.cpu], let coreml = poseResults[.coreML] {
    print("coreML vs cpu agreement: \(agreement(coreml, cpu, width: Float(fullBody.width), height: Float(fullBody.height)))")
}

let faceImage = loadCGImage(sharedDir + "/test_image.jpg")
print("\n=== FACE (VIDEO, repeated still test_image.jpg, \(stillFrames) frames, blendshapes on) ===")
var faceResults: [MediaPipeDelegate: PassResult] = [:]
for delegate in delegates {
    do {
        let r = try runFace(delegate: delegate, image: faceImage, frames: stillFrames)
        faceResults[delegate] = r
        print(String(format: "%-7@ ms(mean/p50/p90): %@ | first %6.2f | detected %5.1f%%",
                     delegate.rawValue as NSString,
                     stats(Array(r.ms.dropFirst(10))) as NSString,
                     r.ms.first ?? -1, detectionRate(r, minCount: 1)))
    } catch {
        print("\(delegate.rawValue): FAILED — \(error)")
    }
}
if let cpu = faceResults[.cpu], let coreml = faceResults[.coreML] {
    print("coreML vs cpu agreement: \(agreement(coreml, cpu, width: Float(faceImage.width), height: Float(faceImage.height)))")
}
