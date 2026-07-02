// End-to-end A/B benchmark:
//   baseline : MediaPipeTasksMac HandLandmarker (TFLite, GPU/Metal, VIDEO mode)
//   coreml   : CoreMLHandPipeline (ANE / GPU), all marshaling on the clock
//
// Usage: pipeline-bench <video.mov> <hand_landmarker.task>

import AVFoundation
import CoreML
import CoreVideo
import Foundation
import MediaPipeTasksMac

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: pipeline-bench <video.mov> <hand_landmarker.task>")
    exit(1)
}
let videoPath = args[1]
let taskModelPath = args[2]

let scriptDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()  // Sources -> pipeline
let expDir = scriptDir.deletingLastPathComponent()            // -> coreml_ane
let palmPackage = expDir.appendingPathComponent("out_palm/palm_img_fp16.mlpackage")
let landPackage = expDir.appendingPathComponent("out/hand_landmarks_img_fp16.mlpackage")

// MARK: - Frame iteration

func forEachFrame(_ body: (CVPixelBuffer, Int) throws -> Void) throws -> Int {
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
    var n = 0
    while reader.status == .reading, let sb = output.copyNextSampleBuffer() {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
        var ts = Int((CMSampleBufferGetPresentationTimeStamp(sb).seconds * 1000).rounded())
        if ts <= lastTs { ts = lastTs + 1 }
        lastTs = ts
        try body(pb, ts)
        n += 1
    }
    return n
}

// MARK: - Stats helpers

func stats(_ values: [Double]) -> String {
    guard !values.isEmpty else { return "n/a" }
    let sorted = values.sorted()
    let mean = values.reduce(0, +) / Double(values.count)
    let p50 = sorted[sorted.count / 2]
    let p90 = sorted[min(Int(Double(sorted.count) * 0.9), sorted.count - 1)]
    return String(format: "%6.2f / %6.2f / %6.2f", mean, p50, p90)
}

struct FrameRecord {
    var ms: Double
    var hands: [[SIMD2<Float>]]  // frame-normalized 21 landmarks per hand
}

// MARK: - Pass 1: MediaPipe baseline (GPU)

print("=== baseline: MediaPipeTasksMac (TFLite), VIDEO mode, numHands=2 ===")
var baseline: [FrameRecord] = []
var baselineDelegate = "gpu"
do {
    let o = HandLandmarkerOptions()
    o.modelPath = taskModelPath
    o.numHands = 2
    o.runningMode = .video
    o.delegate = .gpu
    var lm: HandLandmarker
    do { lm = try HandLandmarker(options: o) } catch {
        baselineDelegate = "cpu (gpu init failed)"
        o.delegate = .cpu
        lm = try HandLandmarker(options: o)
    }
    let n = try forEachFrame { pb, ts in
        let t0 = CFAbsoluteTimeGetCurrent()
        let res = try lm.detectForVideo(pixelBuffer: pb, timestampInMilliseconds: ts)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let hands = res.landmarks.map { $0.map { SIMD2($0.x, $0.y) } }
        baseline.append(FrameRecord(ms: ms, hands: hands))
        if baseline.count % 100 == 0 {
            let recent = baseline.suffix(100).map(\.ms)
            print(String(format: "  frame %4d | last100 mean %6.2f ms",
                         baseline.count, recent.reduce(0, +) / Double(recent.count)))
            fflush(stdout)
        }
    }
    print("frames: \(n), delegate: \(baselineDelegate)")
    print("per-frame ms (mean/p50/p90): \(stats(baseline.map(\.ms)))")
    let bothPct = Double(baseline.filter { $0.hands.count >= 2 }.count) / Double(n) * 100
    print(String(format: "both-hands present: %.1f%%", bothPct))
}

// MARK: - Pass 2/3: CoreML pipeline (ANE, then GPU for context)

let palmCompiled = try await MLModel.compileModel(at: palmPackage)
let landCompiled = try await MLModel.compileModel(at: landPackage)

// MARK: - Debug: landmark leg seeded from baseline ground truth

if ProcessInfo.processInfo.environment["PIPE_DEBUG"] != nil {
    // Middle-most frame where the baseline found 2 hands.
    let twoHandFrames = baseline.enumerated().filter { $0.element.hands.count == 2 }.map(\.offset)
    if let target = twoHandFrames.count > 0 ? twoHandFrames[twoHandFrames.count / 2] : nil {
        FileHandle.standardError.write("DEBUG leg: seeding from baseline frame \(target)\n".data(using: .utf8)!)
        let pipeline = try CoreMLHandPipeline(
            palmURL: palmCompiled, landURL: landCompiled, computeUnits: .all)
        var idx = 0
        _ = try forEachFrame { pb, _ in
            if idx == target {
                let hands = baseline[target].hands
                let rois = hands.map { h in
                    landmarksToROI(h.map { SIMD3($0.x, $0.y, 0) },
                                   imageW: Float(CVPixelBufferGetWidth(pb)),
                                   imageH: Float(CVPixelBufferGetHeight(pb)),
                                   roiScale: 2.0)
                }
                try pipeline.debugLandmarkLeg(frame: pb, rois: rois,
                                              referenceWrists: hands.map { $0[0] })
                // Dump reference landmarks for offline math cross-check.
                let json = hands.map { h in h.map { ["x": $0.x, "y": $0.y] } }
                let data = try JSONSerialization.data(withJSONObject: json)
                try data.write(to: URL(fileURLWithPath: "/tmp/mp_build/ref_hands.json"))
                for (i, r) in rois.enumerated() {
                    FileHandle.standardError.write(
                        "DEBUG roiSwift[\(i)]: cx=\(r.cx) cy=\(r.cy) w=\(r.w) h=\(r.h) rot=\(r.rotation)\n".data(using: .utf8)!)
                }
            }
            idx += 1
        }
    }
}

func runCoreML(_ units: MLComputeUnits, label: String) throws -> [FrameRecord] {
    print("\n=== coreml pipeline (\(label)) ===")
    let pipeline = try CoreMLHandPipeline(
        palmURL: palmCompiled, landURL: landCompiled, computeUnits: units)
    var records: [FrameRecord] = []
    var times: [StageTimes] = []
    var palmRuns = 0
    let n = try forEachFrame { pb, _ in
        let (hands, t) = try pipeline.process(frame: pb)
        records.append(FrameRecord(ms: t.total * 1000,
                                   hands: hands.map { $0.landmarks.map { SIMD2($0.x, $0.y) } }))
        times.append(t)
        if t.palmRan { palmRuns += 1 }
        // Progress heartbeat: proves the run is alive and surfaces a broken
        // palm decode immediately (candidate count should be O(1..20)).
        if records.count % 50 == 0 {
            let recent = records.suffix(50).map(\.ms)
            let meanMs = recent.reduce(0, +) / Double(recent.count)
            print(String(format: "  frame %4d | last50 mean %6.2f ms | hands %d | palmRuns %d | palmCandidates %d",
                         records.count, meanMs, t.handCount, palmRuns, t.palmCandidates))
            fflush(stdout)
        }
    }
    print("frames: \(n)")
    print("per-frame ms (mean/p50/p90): \(stats(records.map(\.ms)))")
    let bothPct = Double(records.filter { $0.hands.count >= 2 }.count) / Double(n) * 100
    let anyPct = Double(records.filter { $0.hands.count >= 1 }.count) / Double(n) * 100
    print(String(format: "hands present: >=1 %.1f%% | both %.1f%%", anyPct, bothPct))

    let palmFrames = times.filter(\.palmRan)
    print("palm ran on \(palmFrames.count)/\(n) frames")
    func ms(_ f: (StageTimes) -> Double, _ subset: [StageTimes]) -> String {
        stats(subset.map { f($0) * 1000 })
    }
    print("stage ms (mean/p50/p90):")
    print("  palm  pre : \(ms(\.palmPre, palmFrames))")
    print("  palm  infer: \(ms(\.palmInfer, palmFrames))")
    print("  palm  post: \(ms(\.palmPost, palmFrames))")
    let landFrames = times.filter { $0.handCount > 0 }
    print("  land  pre : \(ms(\.landPre, landFrames))   (sum over hands)")
    print("  land  infer: \(ms(\.landInfer, landFrames)) (sum over hands)")
    print("  land  post: \(ms(\.landPost, landFrames))")
    return records
}

let ane = try runCoreML(.all, label: "ANE, .all")
let gpu = try runCoreML(.cpuAndGPU, label: "GPU, .cpuAndGPU")

// MARK: - Fidelity: CoreML(ANE) vs MediaPipe baseline landmarks

print("\n=== fidelity: coreml(ANE) vs mediapipe baseline ===")
var pxErrors: [Double] = []
var countMismatch = 0
let frameW: Float = 1280, frameH: Float = 720
for i in 0..<min(baseline.count, ane.count) {
    let b = baseline[i], c = ane[i]
    if b.hands.count != c.hands.count { countMismatch += 1 }
    guard !b.hands.isEmpty, !c.hands.isEmpty else { continue }
    // Match hands by wrist distance.
    for bh in b.hands {
        guard let ch = c.hands.min(by: {
            simd_distance($0[0], bh[0]) < simd_distance($1[0], bh[0])
        }) else { continue }
        var err: Float = 0
        for k in 0..<21 {
            let dx = (bh[k].x - ch[k].x) * frameW
            let dy = (bh[k].y - ch[k].y) * frameH
            err += (dx * dx + dy * dy).squareRoot()
        }
        pxErrors.append(Double(err) / 21)
    }
}
print("hand-count mismatch frames: \(countMismatch)/\(min(baseline.count, ane.count))")
print("mean per-landmark px error (matched hands, mean/p50/p90): \(stats(pxErrors))")
