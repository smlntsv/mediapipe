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
//
// Native macOS smoke test: runs HandLandmarker, PoseLandmarker, and
// FaceLandmarker on the configured image and writes the results to a JSON file
// in the same envelope produced by the web parity project (including the model
// SHA256s and the options used, for parity verification).
//
// Reads its config from argv[1] or the PARITY_CONFIG environment variable.

import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import MediaPipeTasksMac

// MARK: - Config

struct Config: Codable {
    var handModel: String
    var poseModel: String
    var faceModel: String
    var image: String
    var output: String
    // Optional overrides (default to the values below); kept in the envelope so
    // both engines can be configured identically.
    var numHands: Int?
    var numPoses: Int?
    var numFaces: Int?
    var minDetectionConfidence: Float?
    var minPresenceConfidence: Float?
    var minTrackingConfidence: Float?
    var delegate: String?      // "CPU" | "GPU" (default CPU)
    var runningMode: String?   // "IMAGE" | "VIDEO" (default IMAGE)
}

// MARK: - JSON envelope (matches the web parity shape)

struct LandmarkJSON: Codable {
    let x: Float
    let y: Float
    let z: Float
    // Emitted as numbers (0 when the model does not provide them) for
    // Web-compatible comparison; the Swift API keeps them optional.
    let visibility: Float
    let presence: Float
    init(_ l: NormalizedLandmark) {
        x = l.x; y = l.y; z = l.z
        visibility = l.visibility ?? 0; presence = l.presence ?? 0
    }
    init(_ l: Landmark) {
        x = l.x; y = l.y; z = l.z
        visibility = l.visibility ?? 0; presence = l.presence ?? 0
    }
}

struct CategoryJSON: Codable {
    let index: Int
    let score: Float
    let categoryName: String?
    let displayName: String?
    init(_ c: MediaPipeTasksMac.Category) {
        index = c.index; score = c.score
        categoryName = c.categoryName; displayName = c.displayName
    }
}

struct ClassificationsJSON: Codable {
    let categories: [CategoryJSON]
    let headIndex: Int
    let headName: String?
}

struct MatrixJSON: Codable {
    let rows: Int
    let columns: Int
    let data: [Float]
}

struct HandJSON: Codable {
    let landmarks: [[LandmarkJSON]]
    let worldLandmarks: [[LandmarkJSON]]
    let handedness: [[CategoryJSON]]
    let handednesses: [[CategoryJSON]]
}

struct PoseJSON: Codable {
    let landmarks: [[LandmarkJSON]]
    let worldLandmarks: [[LandmarkJSON]]
    let segmentationMasks: [String]?
}

struct FaceJSON: Codable {
    let faceLandmarks: [[LandmarkJSON]]
    let faceBlendshapes: [ClassificationsJSON]
    let facialTransformationMatrixes: [MatrixJSON]
}

struct HandOptionsJSON: Codable {
    let numHands: Int
    let minHandDetectionConfidence: Float
    let minHandPresenceConfidence: Float
    let minTrackingConfidence: Float
    let runningMode: String
    let delegate: String
}
struct PoseOptionsJSON: Codable {
    let numPoses: Int
    let minPoseDetectionConfidence: Float
    let minPosePresenceConfidence: Float
    let minTrackingConfidence: Float
    let outputSegmentationMasks: Bool
    let runningMode: String
    let delegate: String
}
struct FaceOptionsJSON: Codable {
    let numFaces: Int
    let minFaceDetectionConfidence: Float
    let minFacePresenceConfidence: Float
    let minTrackingConfidence: Float
    let outputFaceBlendshapes: Bool
    let outputFacialTransformationMatrixes: Bool
    let runningMode: String
    let delegate: String
}
struct OptionsJSON: Codable {
    let hand: HandOptionsJSON
    let pose: PoseOptionsJSON
    let face: FaceOptionsJSON
}
struct ModelsJSON: Codable {
    let hand: String
    let pose: String
    let face: String
}

struct Envelope: Codable {
    let source: String
    let image: String
    let models: ModelsJSON
    let options: OptionsJSON
    let hand: HandJSON
    let pose: PoseJSON
    let face: FaceJSON
}

// MARK: - Helpers

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

func loadCGImage(_ path: String) -> CGImage {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        die("could not load image at \(path)")
    }
    return image
}

func sha256(_ path: String) -> String {
    guard let data = FileManager.default.contents(atPath: path) else {
        die("could not read model for hashing: \(path)")
    }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func loadConfig() -> Config {
    let path = CommandLine.arguments.count > 1
        ? CommandLine.arguments[1]
        : ProcessInfo.processInfo.environment["PARITY_CONFIG"]
    guard let path else {
        die("provide a config path as argv[1] or via PARITY_CONFIG. See config.example.json.")
    }
    guard let data = FileManager.default.contents(atPath: path) else {
        die("could not read config at \(path)")
    }
    do {
        return try JSONDecoder().decode(Config.self, from: data)
    } catch {
        die("invalid config JSON at \(path): \(error)")
    }
}

// MARK: - Run

let config = loadConfig()
let cgImage = loadCGImage(config.image)
let imageName = URL(fileURLWithPath: config.image).lastPathComponent

let numHands = config.numHands ?? 2
let numPoses = config.numPoses ?? 1
let numFaces = config.numFaces ?? 1
let minDetect = config.minDetectionConfidence ?? 0.5
let minPresence = config.minPresenceConfidence ?? 0.5
let minTrack = config.minTrackingConfidence ?? 0.5
let delegate = MediaPipeDelegate(rawValue: config.delegate ?? "CPU") ?? .cpu
let runningMode = RunningMode(rawValue: config.runningMode ?? "IMAGE") ?? .image
// For VIDEO mode on a single still image, use one frame at timestamp 0.
let videoTimestampMs = 0

let envelope: Envelope
do {
    let handOptions = HandLandmarkerOptions(
        modelPath: config.handModel, numHands: numHands,
        minHandDetectionConfidence: minDetect, minHandPresenceConfidence: minPresence,
        minTrackingConfidence: minTrack, delegate: delegate, runningMode: runningMode)
    let handLandmarker = try HandLandmarker(options: handOptions)
    let handResult = runningMode == .image
        ? try handLandmarker.detect(cgImage: cgImage)
        : try handLandmarker.detectForVideo(cgImage: cgImage, timestampInMilliseconds: videoTimestampMs)

    let poseOptions = PoseLandmarkerOptions(
        modelPath: config.poseModel, numPoses: numPoses,
        minPoseDetectionConfidence: minDetect, minPosePresenceConfidence: minPresence,
        minTrackingConfidence: minTrack, delegate: delegate, runningMode: runningMode)
    let poseLandmarker = try PoseLandmarker(options: poseOptions)
    let poseResult = runningMode == .image
        ? try poseLandmarker.detect(cgImage: cgImage)
        : try poseLandmarker.detectForVideo(cgImage: cgImage, timestampInMilliseconds: videoTimestampMs)

    let faceOptions = FaceLandmarkerOptions(
        modelPath: config.faceModel, numFaces: numFaces,
        minFaceDetectionConfidence: minDetect, minFacePresenceConfidence: minPresence,
        minTrackingConfidence: minTrack,
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
        delegate: delegate, runningMode: runningMode)
    let faceLandmarker = try FaceLandmarker(options: faceOptions)
    let faceResult = runningMode == .image
        ? try faceLandmarker.detect(cgImage: cgImage)
        : try faceLandmarker.detectForVideo(cgImage: cgImage, timestampInMilliseconds: videoTimestampMs)

    let hand = HandJSON(
        landmarks: handResult.landmarks.map { $0.map(LandmarkJSON.init) },
        worldLandmarks: handResult.worldLandmarks.map { $0.map(LandmarkJSON.init) },
        handedness: handResult.handedness.map { $0.map(CategoryJSON.init) },
        handednesses: handResult.handedness.map { $0.map(CategoryJSON.init) })

    let pose = PoseJSON(
        landmarks: poseResult.landmarks.map { $0.map(LandmarkJSON.init) },
        worldLandmarks: poseResult.worldLandmarks.map { $0.map(LandmarkJSON.init) },
        segmentationMasks: nil)

    let face = FaceJSON(
        faceLandmarks: faceResult.faceLandmarks.map { $0.map(LandmarkJSON.init) },
        faceBlendshapes: faceResult.faceBlendshapes.map {
            ClassificationsJSON(categories: $0.categories.map(CategoryJSON.init),
                                headIndex: $0.headIndex, headName: $0.headName)
        },
        facialTransformationMatrixes: faceResult.facialTransformationMatrixes.map {
            MatrixJSON(rows: $0.rows, columns: $0.columns, data: $0.data)
        })

    let runningModeStr = runningMode.rawValue
    let delegateStr = delegate.rawValue
    let options = OptionsJSON(
        hand: HandOptionsJSON(
            numHands: numHands, minHandDetectionConfidence: minDetect,
            minHandPresenceConfidence: minPresence, minTrackingConfidence: minTrack,
            runningMode: runningModeStr, delegate: delegateStr),
        pose: PoseOptionsJSON(
            numPoses: numPoses, minPoseDetectionConfidence: minDetect,
            minPosePresenceConfidence: minPresence, minTrackingConfidence: minTrack,
            outputSegmentationMasks: false, runningMode: runningModeStr, delegate: delegateStr),
        face: FaceOptionsJSON(
            numFaces: numFaces, minFaceDetectionConfidence: minDetect,
            minFacePresenceConfidence: minPresence, minTrackingConfidence: minTrack,
            outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
            runningMode: runningModeStr, delegate: delegateStr))

    let models = ModelsJSON(
        hand: sha256(config.handModel),
        pose: sha256(config.poseModel),
        face: sha256(config.faceModel))

    envelope = Envelope(source: "macos", image: imageName, models: models,
                        options: options, hand: hand, pose: pose, face: face)
} catch {
    die("detection failed: \(error.localizedDescription)")
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
do {
    let data = try encoder.encode(envelope)
    let outURL = URL(fileURLWithPath: config.output)
    try FileManager.default.createDirectory(
        at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: outURL)
    let summary = """
        macOS parity smoke test (\(imageName)):
          hand: \(envelope.hand.landmarks.count) × \(envelope.hand.landmarks.first?.count ?? 0)
          pose: \(envelope.pose.landmarks.count) × \(envelope.pose.landmarks.first?.count ?? 0)
          face: \(envelope.face.faceLandmarks.count) × \(envelope.face.faceLandmarks.first?.count ?? 0) \
        (blendshapes: \(envelope.face.faceBlendshapes.first?.categories.count ?? 0), \
        matrixes: \(envelope.face.facialTransformationMatrixes.count))
        wrote \(config.output)
        """
    FileHandle.standardError.write(Data((summary + "\n").utf8))
} catch {
    die("failed to write output: \(error)")
}
