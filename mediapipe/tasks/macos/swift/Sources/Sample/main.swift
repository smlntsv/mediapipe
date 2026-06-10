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
import Foundation
import ImageIO
import MediaPipeTasksMac

// A minimal CLI demonstrating MediaPipe Vision Tasks on macOS. It accepts any
// subset of model flags so each landmarker can be tested independently:
//
//   swift run mediapipe-macos-sample --hand hand_landmarker.task --image test.jpg
//   swift run mediapipe-macos-sample --hand h.task --pose p.task --face f.task --image test.jpg

struct Arguments {
    var handModel: String?
    var poseModel: String?
    var faceModel: String?
    var imagePath: String?
}

func parseArguments() -> Arguments {
    var args = Arguments()
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = iterator.next() {
        switch flag {
        case "--hand": args.handModel = iterator.next()
        case "--pose": args.poseModel = iterator.next()
        case "--face": args.faceModel = iterator.next()
        case "--image": args.imagePath = iterator.next()
        default:
            FileHandle.standardError.write(Data("Unknown argument: \(flag)\n".utf8))
        }
    }
    return args
}

func loadCGImage(path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return image
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

let arguments = parseArguments()

guard let imagePath = arguments.imagePath else {
    fail("missing --image <path>. "
        + "Usage: mediapipe-macos-sample --hand <model> [--pose <model>] [--face <model>] --image <image>")
}
guard let cgImage = loadCGImage(path: imagePath) else {
    fail("could not load image at \(imagePath)")
}

if arguments.handModel == nil && arguments.poseModel == nil && arguments.faceModel == nil {
    fail("provide at least one of --hand / --pose / --face <model.task>")
}

if let handModel = arguments.handModel {
    do {
        let options = HandLandmarkerOptions()
        options.modelPath = handModel
        options.numHands = 2
        let landmarker = try HandLandmarker(options: options)
        let result = try landmarker.detect(cgImage: cgImage)
        let count = result.landmarks.count
        let perHand = result.landmarks.first?.count ?? 0
        print("HandLandmarker: detected \(count) hand\(count == 1 ? "" : "s"), \(perHand) landmarks")
    } catch {
        fail("HandLandmarker failed: \(error.localizedDescription)")
    }
}

if let poseModel = arguments.poseModel {
    do {
        let options = PoseLandmarkerOptions()
        options.modelPath = poseModel
        options.numPoses = 1
        let landmarker = try PoseLandmarker(options: options)
        let result = try landmarker.detect(cgImage: cgImage)
        let count = result.landmarks.count
        let perPose = result.landmarks.first?.count ?? 0
        print("PoseLandmarker: detected \(count) pose\(count == 1 ? "" : "s"), \(perPose) landmarks")
    } catch {
        fail("PoseLandmarker failed: \(error.localizedDescription)")
    }
}

if let faceModel = arguments.faceModel {
    do {
        let options = FaceLandmarkerOptions()
        options.modelPath = faceModel
        options.numFaces = 1
        let landmarker = try FaceLandmarker(options: options)
        let result = try landmarker.detect(cgImage: cgImage)
        let count = result.landmarks.count
        let perFace = result.landmarks.first?.count ?? 0
        print("FaceLandmarker: detected \(count) face\(count == 1 ? "" : "s"), \(perFace) landmarks")
    } catch {
        fail("FaceLandmarker failed: \(error.localizedDescription)")
    }
}
