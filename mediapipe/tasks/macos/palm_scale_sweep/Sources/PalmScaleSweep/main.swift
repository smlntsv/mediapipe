// Usage:
//   ffmpeg ... -f rawvideo -pix_fmt bgra - | palm-scale-sweep \
//       --model hand_landmarker.task --width 1280 --height 720 \
//       --mode image|video --delegate cpu|gpu --fps 30.0003 \
//       --candidates 50 --out frames.jsonl
//
// One JSON object per frame:
//   i            frame index
//   ran          palm detector ran (stage record)
//   det          palm detections after suppression: [score, xMin, yMin, w, h]
//   cand         top-N palm candidates (no threshold, before suppression), same layout
//   above, kept  aboveThresholdCount, afterSuppressionCount
//   hands        [{h: handedness, hs: handedness score, lm: [x0, y0, x1, y1, ...]}]
// Coordinates are normalized to the frame as passed in.
import CoreVideo
import Foundation
import MediaPipeTasks

func die(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(2)
}

var model = ""
var width = 0
var height = 0
var mode = RunningMode.image
var delegate = MediaPipeDelegate.cpu
var fps = 30.0
var candidateLimit = 50
var numHands = 2
var outPath = ""

var argIterator = CommandLine.arguments.dropFirst().makeIterator()
func value(_ flag: String) -> String {
    guard let v = argIterator.next() else { die("\(flag) needs a value") }
    return v
}
while let flag = argIterator.next() {
    switch flag {
    case "--model": model = value(flag)
    case "--width": width = Int(value(flag)) ?? 0
    case "--height": height = Int(value(flag)) ?? 0
    case "--mode": mode = value(flag) == "video" ? .video : .image
    case "--delegate":
        switch value(flag) {
        case "cpu": delegate = .cpu
        case "gpu": delegate = .gpu
        default: die("--delegate must be cpu or gpu")
        }
    case "--fps": fps = Double(value(flag)) ?? fps
    case "--candidates": candidateLimit = Int(value(flag)) ?? candidateLimit
    case "--num-hands": numHands = Int(value(flag)) ?? numHands
    case "--out": outPath = value(flag)
    default: die("unknown argument \(flag)")
    }
}
guard !model.isEmpty, width > 0, height > 0, !outPath.isEmpty else {
    die("--model, --width, --height and --out are required")
}

let landmarker: HandLandmarker
do {
    landmarker = try HandLandmarker(options: HandLandmarkerOptions(
        modelPath: model,
        numHands: numHands,
        delegate: delegate,
        runningMode: mode,
        stageOutputs: true,
        palmCandidateLimit: candidateLimit))
} catch {
    die("HandLandmarker creation failed: \(error)")
}

FileManager.default.createFile(atPath: outPath, contents: nil)
guard let out = FileHandle(forWritingAtPath: outPath) else { die("cannot write \(outPath)") }


let frameBytes = width * height * 4
let stdin = FileHandle.standardInput
var buffer = Data()
var index = 0

func readFrame() -> Data? {
    while buffer.count < frameBytes {
        let chunk = stdin.readData(ofLength: max(frameBytes - buffer.count, 1 << 20))
        if chunk.isEmpty { return nil }
        buffer.append(chunk)
    }
    let frame = buffer.prefix(frameBytes)
    buffer.removeFirst(frameBytes)
    return Data(frame)
}

while let frame = readFrame() {
    autoreleasepool {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &pixelBuffer)
        guard let pb = pixelBuffer else { die("CVPixelBufferCreate failed") }
        CVPixelBufferLockBaseAddress(pb, [])
        let dst = CVPixelBufferGetBaseAddress(pb)!
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        frame.withUnsafeBytes { raw in
            let src = raw.baseAddress!
            for y in 0..<height {
                memcpy(dst + y * rowBytes, src + y * width * 4, width * 4)
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])

        let result: HandLandmarkerResult
        do {
            if mode == .video {
                let ts = Int((Double(index) * 1000.0 / fps).rounded())
                result = try landmarker.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts)
            } else {
                result = try landmarker.detect(pixelBuffer: pb)
            }
        } catch {
            die("detect failed on frame \(index): \(error)")
        }

        func f(_ v: Float) -> String { v.isFinite ? String(format: "%.4f", v) : "null" }
        func boxes(_ ds: [StageDetection]) -> String {
            "[" + ds.map { "[\(f($0.score)),\(f($0.xMin)),\(f($0.yMin)),\(f($0.width)),\(f($0.height))]" }
                .joined(separator: ",") + "]"
        }
        var line = "{\"i\":\(index)"
        if let stages = result.stages {
            line += ",\"ran\":\(stages.palmDetectorRan),\"det\":\(boxes(stages.palmDetections))"
        }
        if let pc = result.palmCandidates {
            line += ",\"cand\":\(boxes(pc.candidates)),\"above\":\(pc.aboveThresholdCount),\"kept\":\(pc.afterSuppressionCount)"
        }
        var hands: [String] = []
        for (h, landmarks) in result.landmarks.enumerated() {
            let category = h < result.handedness.count ? result.handedness[h].first : nil
            let lm = landmarks.map { "\(f($0.x)),\(f($0.y))" }.joined(separator: ",")
            hands.append("{\"h\":\"\(category?.categoryName ?? "")\",\"hs\":\(f(category?.score ?? 0)),\"lm\":[\(lm)]}")
        }
        line += ",\"hands\":[" + hands.joined(separator: ",") + "]}"
        out.write(line.data(using: .utf8)!)
        out.write("\n".data(using: .utf8)!)
        index += 1
    }
}
try? out.close()
FileHandle.standardError.write("frames \(index)\n".data(using: .utf8)!)
