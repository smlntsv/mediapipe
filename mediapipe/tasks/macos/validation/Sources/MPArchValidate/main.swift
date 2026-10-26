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

// mp-arch-validate — headless Hand Landmarker validation for one architecture.
//
//   mp-arch-validate env
//   mp-arch-validate image --model hand_landmarker.task --image a.jpg [--image b.jpg]
//                          --delegate cpu|gpu|coreml [--repeat 20] [--warmup 3]
//                          [--num-hands 2] [--coreml-cache DIR] --out report.json
//   mp-arch-validate video --model hand_landmarker.task --video clip.mov
//                          --delegate cpu|gpu|coreml [--loops 3] [--max-frames N]
//                          [--num-hands 2] [--coreml-cache DIR] [--min-hand-rate 0.25]
//                          --out report.json
//   mp-arch-validate frames --video clip.mov --at-ms 1000,4000 --out-dir DIR
//                          (extracts lossless PNG stills, run ONCE on the build
//                          machine so every target sees bit-identical pixels —
//                          H.264 decoders differ slightly across GPUs/OSes)
//
// Every report embeds the execution environment (process architecture,
// native vs Rosetta, CPU, Metal devices) and the SHA-256 of every input, so
// reports from different machines can be compared by compare_reports.py.
//
// Exit codes: 0 ok · 1 plausibility check failed · 2 usage · 3 landmarker
// creation failed (e.g. unsupported delegate) · 4 inference failed.
// A crash shows up as a signal exit in the runner script.

import AVFoundation
import CoreGraphics
import CoreVideo
import CryptoKit
import Darwin
import Foundation
import ImageIO
import MediaPipeTasks
import Metal

// MARK: - Small utilities

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func die(_ message: String, code: Int32 = 2) -> Never {
    log("error: \(message)")
    exit(code)
}

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
    return String(cString: buffer)
}

func sysctlInt(_ name: String) -> Int64? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0 else { return nil }
    if size == 4 {
        var value: Int32 = 0
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return Int64(value)
    }
    var value: Int64 = 0
    size = 8
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    return value
}

func sha256(of path: String) -> String {
    guard let data = FileManager.default.contents(atPath: path) else { return "unreadable" }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// Resident memory as accounted by the kernel for jetsam / Activity Monitor.
func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}

func nowMs() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000
}

func num(_ value: Float) -> Any { value.isFinite ? Double(value) : NSNull() }
func num(_ value: Double) -> Any { value.isFinite ? value : NSNull() }
func r5(_ value: Float) -> Any { value.isFinite ? (Double(value) * 1e5).rounded() / 1e5 : NSNull() }

func stats(_ values: [Double]) -> [String: Any] {
    guard !values.isEmpty else { return ["n": 0] }
    let sorted = values.sorted()
    func pct(_ p: Double) -> Double {
        sorted[min(sorted.count - 1, Int((p / 100) * Double(sorted.count - 1) + 0.5))]
    }
    let mean = values.reduce(0, +) / Double(values.count)
    return ["n": values.count, "mean": num(mean), "min": num(sorted.first!), "p50": num(pct(50)),
            "p90": num(pct(90)), "p99": num(pct(99)), "max": num(sorted.last!)]
}

func writeJSON(_ object: [String: Any], to path: String) {
    do {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        die("could not write \(path): \(error)")
    }
}

// MARK: - Environment

func processArch() -> String {
    #if arch(arm64)
    return "arm64"
    #elseif arch(x86_64)
    return "x86_64"
    #else
    return "unknown"
    #endif
}

func environment() -> [String: Any] {
    let arch = processArch()
    let translated = sysctlInt("sysctl.proc_translated") == 1
    let execution: String
    switch (arch, translated) {
    case ("arm64", _): execution = "native-arm64"
    case ("x86_64", true): execution = "rosetta-x86_64"
    case ("x86_64", false): execution = "native-x86_64"
    default: execution = "unknown"
    }
    var metal: [[String: Any]] = []
    for device in MTLCopyAllDevices() {
        metal.append([
            "name": device.name,
            "low_power": device.isLowPower,
            "removable": device.isRemovable,
            "headless": device.isHeadless,
            "unified_memory": device.hasUnifiedMemory,
            "family_apple7": device.supportsFamily(.apple7),
            "family_mac2": device.supportsFamily(.mac2),
            "recommended_max_working_set_mb": Double(device.recommendedMaxWorkingSetSize) / 1_048_576,
        ])
    }
    let os = ProcessInfo.processInfo.operatingSystemVersion
    return [
        "process_arch": arch,
        "rosetta_translated": translated,
        "execution": execution,
        "hw_model": sysctlString("hw.model") ?? "?",
        "cpu_brand": sysctlString("machdep.cpu.brand_string") ?? "?",
        "cpu_features": sysctlString("machdep.cpu.features") ?? "",
        "cpu_leaf7_features": sysctlString("machdep.cpu.leaf7_features") ?? "",
        "cpu_count": ProcessInfo.processInfo.activeProcessorCount,
        "memory_gb": Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824,
        "os_version": "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)",
        "os_build": sysctlString("kern.osversion") ?? "?",
        "metal_default_device": MTLCreateSystemDefaultDevice()?.name ?? NSNull(),
        "metal_devices": metal,
        "hostname_hash": String(sha256Hex(ProcessInfo.processInfo.hostName).prefix(12)),
    ]
}

func sha256Hex(_ string: String) -> String {
    SHA256.hash(data: Data(string.utf8)).map { String(format: "%02x", $0) }.joined()
}

// MARK: - Arguments

struct Options {
    var command = ""
    var model = ""
    var images: [String] = []
    var video = ""
    var delegate = "cpu"
    var repeatCount = 20
    var warmup = 3
    var loops = 3
    var maxFrames = Int.max
    var numHands = 2
    var coreMLCache: String?
    var out = ""
    var atMs: [Int] = []
    /// VIDEO: minimum fraction of frames with >= 1 hand for the run to count
    /// as plausible. Catches a delegate that runs without errors but returns
    /// nothing (the reference clip has hands in ~78% of frames).
    var minHandRate = 0.25
    var outDir = ""
}

func parse() -> Options {
    var options = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    guard !args.isEmpty else { die("usage: mp-arch-validate env|image|video [options]") }
    options.command = args.removeFirst()
    var it = args.makeIterator()
    func value(_ flag: String) -> String {
        guard let v = it.next() else { die("\(flag) needs a value") }
        return v
    }
    while let flag = it.next() {
        switch flag {
        case "--model": options.model = value(flag)
        case "--image": options.images.append(value(flag))
        case "--video": options.video = value(flag)
        case "--delegate": options.delegate = value(flag).lowercased()
        case "--repeat": options.repeatCount = Int(value(flag)) ?? options.repeatCount
        case "--warmup": options.warmup = Int(value(flag)) ?? options.warmup
        case "--loops": options.loops = Int(value(flag)) ?? options.loops
        case "--max-frames": options.maxFrames = Int(value(flag)) ?? options.maxFrames
        case "--num-hands": options.numHands = Int(value(flag)) ?? options.numHands
        case "--coreml-cache": options.coreMLCache = value(flag)
        case "--out": options.out = value(flag)
        case "--at-ms": options.atMs = value(flag).split(separator: ",").compactMap { Int($0) }
        case "--out-dir": options.outDir = value(flag)
        case "--min-hand-rate": options.minHandRate = Double(value(flag)) ?? options.minHandRate
        default: die("unknown argument \(flag)")
        }
    }
    return options
}

func delegate(named name: String) -> MediaPipeDelegate {
    switch name {
    case "cpu": return .cpu
    case "gpu": return .gpu
    case "coreml": return .coreML
    default: die("--delegate must be cpu, gpu or coreml")
    }
}

func makeLandmarker(_ options: Options, mode: RunningMode) -> (HandLandmarker, Double) {
    let start = nowMs()
    do {
        let landmarker = try HandLandmarker(options: HandLandmarkerOptions(
            modelPath: options.model,
            numHands: options.numHands,
            delegate: delegate(named: options.delegate),
            coreMLModelCacheDirectory: options.coreMLCache,
            runningMode: mode))
        return (landmarker, nowMs() - start)
    } catch {
        var report = baseReport(options)
        report["status"] = "init_failed"
        report["error"] = String(describing: error)
        if !options.out.isEmpty { writeJSON(report, to: options.out) }
        die("HandLandmarker creation failed (\(options.delegate)): \(error)", code: 3)
    }
}

func baseReport(_ options: Options) -> [String: Any] {
    [
        "tool": "mp-arch-validate",
        "schema": 1,
        "command": options.command,
        "delegate": options.delegate,
        "num_hands": options.numHands,
        "model": ["path": (options.model as NSString).lastPathComponent,
                  "sha256": sha256(of: options.model)],
        "environment": environment(),
        "started_at": ISO8601DateFormatter().string(from: Date()),
    ]
}

// MARK: - Result encoding + plausibility

let fingerChains: [[Int]] = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                             [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]

func encodeHands(_ result: HandLandmarkerResult) -> [[String: Any]] {
    result.landmarks.indices.map { i in
        let category = i < result.handedness.count ? result.handedness[i].first : nil
        let world = i < result.worldLandmarks.count ? result.worldLandmarks[i] : []
        return [
            "handedness": category?.categoryName ?? NSNull(),
            "handedness_score": category.map { num($0.score) } ?? NSNull(),
            "landmarks": result.landmarks[i].map { [num($0.x), num($0.y), num($0.z)] },
            "world_landmarks": world.map { [num($0.x), num($0.y), num($0.z)] },
        ]
    }
}

/// Anatomical sanity checks independent of any reference output.
func plausibility(_ result: HandLandmarkerResult, minHands: Int) -> [String] {
    var failures: [String] = []
    if result.landmarks.count < minHands {
        failures.append("expected >= \(minHands) hand(s), got \(result.landmarks.count)")
    }
    for (i, hand) in result.landmarks.enumerated() {
        let tag = "hand[\(i)]"
        guard hand.count == 21 else { failures.append("\(tag): \(hand.count) landmarks"); continue }
        if hand.contains(where: { !$0.x.isFinite || !$0.y.isFinite || !$0.z.isFinite }) {
            failures.append("\(tag): non-finite landmark"); continue
        }
        if hand.contains(where: { $0.x < -0.1 || $0.x > 1.1 || $0.y < -0.1 || $0.y > 1.1 }) {
            failures.append("\(tag): landmark outside [-0.1, 1.1]")
        }
        let xs = hand.map(\.x), ys = hand.map(\.y)
        let extent = max(xs.max()! - xs.min()!, ys.max()! - ys.min()!)
        if extent < 0.02 || extent > 1.2 { failures.append("\(tag): implausible extent \(extent)") }
        for chain in fingerChains {
            for (a, b) in zip(chain, chain.dropFirst()) {
                let d = hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)
                if d < 1e-4 { failures.append("\(tag): degenerate bone \(a)-\(b)") }
            }
        }
        if i < result.handedness.count, let c = result.handedness[i].first {
            if !["Left", "Right"].contains(c.categoryName ?? "") {
                failures.append("\(tag): handedness label \(c.categoryName ?? "nil")")
            }
            if !(0.5...1.0).contains(c.score) { failures.append("\(tag): handedness score \(c.score)") }
        } else {
            failures.append("\(tag): missing handedness")
        }
        if i < result.worldLandmarks.count, result.worldLandmarks[i].count == 21 {
            let w = result.worldLandmarks[i]
            let palm = sqrt(pow(w[0].x - w[9].x, 2) + pow(w[0].y - w[9].y, 2) + pow(w[0].z - w[9].z, 2))
            // Wrist → middle-finger MCP of a real hand is ~6-12 cm.
            if !(0.03...0.2).contains(palm) { failures.append("\(tag): world palm length \(palm) m") }
        } else {
            failures.append("\(tag): missing world landmarks")
        }
    }
    return failures
}

func maxLandmarkDelta(_ a: HandLandmarkerResult, _ b: HandLandmarkerResult) -> Double? {
    guard a.landmarks.count == b.landmarks.count else { return nil }
    var worst: Float = 0
    for (ha, hb) in zip(a.landmarks, b.landmarks) {
        guard ha.count == hb.count else { return nil }
        for (la, lb) in zip(ha, hb) { worst = max(worst, abs(la.x - lb.x), abs(la.y - lb.y)) }
    }
    return Double(worst)
}

// MARK: - IMAGE mode

func loadCGImage(_ path: String) -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        die("could not load image \(path)")
    }
    return image
}

func runImage(_ options: Options) -> Int32 {
    guard !options.model.isEmpty, !options.images.isEmpty, !options.out.isEmpty else {
        die("image needs --model, --image and --out")
    }
    var report = baseReport(options)
    let (landmarker, initMs) = makeLandmarker(options, mode: .image)
    report["init_ms"] = initMs
    var anyFailure = false
    var perImage: [[String: Any]] = []
    for path in options.images {
        let image = loadCGImage(path)
        var entry: [String: Any] = [
            "image": (path as NSString).lastPathComponent, "sha256": sha256(of: path),
            "width": image.width, "height": image.height,
        ]
        var timings: [Double] = []
        var first: HandLandmarkerResult?
        var worstRepeatDelta = 0.0
        var repeatCountMismatch = false
        do {
            for run in 0..<(options.warmup + options.repeatCount) {
                let t0 = nowMs()
                let result = try landmarker.detect(cgImage: image)
                let dt = nowMs() - t0
                if run == 0 { entry["first_inference_ms"] = dt }
                if run >= options.warmup { timings.append(dt) }
                if let f = first {
                    if let d = maxLandmarkDelta(f, result) { worstRepeatDelta = max(worstRepeatDelta, d) }
                    else { repeatCountMismatch = true }
                } else {
                    first = result
                }
            }
        } catch {
            entry["status"] = "inference_failed"
            entry["error"] = String(describing: error)
            perImage.append(entry)
            report["images"] = perImage
            report["status"] = "inference_failed"
            writeJSON(report, to: options.out)
            log("inference failed on \(path): \(error)")
            return 4
        }
        let result = first!
        let failures = plausibility(result, minHands: 1)
        anyFailure = anyFailure || !failures.isEmpty || repeatCountMismatch
        entry["hands"] = encodeHands(result)
        entry["latency_ms"] = stats(timings)
        entry["repeat_max_delta"] = worstRepeatDelta
        entry["repeat_hand_count_stable"] = !repeatCountMismatch
        entry["plausibility"] = ["pass": failures.isEmpty, "failures": failures]
        perImage.append(entry)
        let labels = result.handedness.map { $0.first?.categoryName ?? "?" }.joined(separator: ",")
        log(String(format: "%@ [%@] %@: %d hand(s) [%@] mean %.2f ms, repeat Δ %.2g, plausible=%@",
                   environment()["execution"] as! String, options.delegate,
                   (path as NSString).lastPathComponent, result.landmarks.count, labels,
                   (stats(timings)["mean"] as? Double) ?? -1, worstRepeatDelta,
                   failures.isEmpty ? "yes" : "NO \(failures)"))
    }
    report["images"] = perImage
    report["status"] = anyFailure ? "implausible" : "ok"
    writeJSON(report, to: options.out)
    return anyFailure ? 1 : 0
}

// MARK: - VIDEO mode (sustained)

final class VideoSource {
    let url: URL
    let durationMs: Double
    let width: Int
    let height: Int
    let nominalFPS: Float

    init(path: String) {
        url = URL(fileURLWithPath: path)
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { die("no video track in \(path)") }
        durationMs = asset.duration.seconds * 1000
        width = Int(track.naturalSize.width)
        height = Int(track.naturalSize.height)
        nominalFPS = track.nominalFrameRate
    }

    /// Decodes one pass as 32BGRA (the format the Swift API accepts). Returns
    /// false from `body` to stop early.
    func forEachFrame(maxFrames: Int, _ body: (CVPixelBuffer, Double) throws -> Bool) rethrows {
        let asset = AVURLAsset(url: url)
        let track = asset.tracks(withMediaType: .video).first!
        let reader: AVAssetReader
        do { reader = try AVAssetReader(asset: asset) } catch { die("AVAssetReader: \(error)") }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { die("cannot read \(url.path): \(String(describing: reader.error))") }
        defer { reader.cancelReading() }
        var count = 0
        while count < maxFrames, let sample = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let ptsMs = CMSampleBufferGetPresentationTimeStamp(sample).seconds * 1000
            count += 1
            let keepGoing = try autoreleasepool { try body(pixelBuffer, ptsMs) }
            if !keepGoing { break }
        }
    }
}

func linearSlope(_ points: [(Double, Double)]) -> Double? {
    guard points.count >= 3 else { return nil }
    let n = Double(points.count)
    let mx = points.map(\.0).reduce(0, +) / n
    let my = points.map(\.1).reduce(0, +) / n
    let sxx = points.map { pow($0.0 - mx, 2) }.reduce(0, +)
    guard sxx > 0 else { return nil }
    return points.map { ($0.0 - mx) * ($0.1 - my) }.reduce(0, +) / sxx
}

func runVideo(_ options: Options) -> Int32 {
    guard !options.model.isEmpty, !options.video.isEmpty, !options.out.isEmpty else {
        die("video needs --model, --video and --out")
    }
    let source = VideoSource(path: options.video)
    var report = baseReport(options)
    report["video"] = [
        "file": (options.video as NSString).lastPathComponent, "sha256": sha256(of: options.video),
        "width": source.width, "height": source.height, "duration_ms": source.durationMs,
        "nominal_fps": Double(source.nominalFPS),
    ]
    report["loops"] = options.loops
    let memBefore = physFootprintMB()
    let (landmarker, initMs) = makeLandmarker(options, mode: .video)
    report["init_ms"] = initMs

    var latencies: [Double] = []
    var memSamples: [(frame: Double, mb: Double)] = []
    var perLoop: [[String: Any]] = []
    var firstLoopFrames: [[String: Any]] = []
    var lastTimestamp = -1
    var globalFrame = 0
    var firstFrameMs: Double?
    var totalWithHand = 0
    var memAfterLoop0 = 0.0
    var peak = memBefore
    let wallStart = nowMs()
    for loop in 0..<options.loops {
        var frames = 0, withHand = 0, withTwo = 0
        var loopLatency: [Double] = []
        let loopOffset = Double(loop) * (source.durationMs + 1000)
        do {
            try source.forEachFrame(maxFrames: options.maxFrames) { pixelBuffer, ptsMs in
                var ts = Int((loopOffset + ptsMs).rounded())
                if ts <= lastTimestamp { ts = lastTimestamp + 1 }
                lastTimestamp = ts
                let t0 = nowMs()
                let result = try landmarker.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: ts)
                let dt = nowMs() - t0
                if firstFrameMs == nil { firstFrameMs = dt } else { latencies.append(dt) }
                loopLatency.append(dt)
                frames += 1
                globalFrame += 1
                if !result.landmarks.isEmpty { withHand += 1; totalWithHand += 1 }
                if result.landmarks.count >= 2 { withTwo += 1 }
                if loop == 0 {
                    firstLoopFrames.append([
                        "t": ts,
                        "hands": result.landmarks.indices.map { i -> [String: Any] in
                            [
                                "h": result.handedness[i].first?.categoryName ?? "?",
                                "xy": result.landmarks[i].flatMap { [r5($0.x), r5($0.y)] },
                            ]
                        },
                    ])
                }
                if globalFrame % 25 == 0 {
                    let mb = physFootprintMB()
                    peak = max(peak, mb)
                    memSamples.append((Double(globalFrame), mb))
                }
                return true
            }
        } catch {
            report["status"] = "inference_failed"
            report["error"] = String(describing: error)
            report["failed_at_frame"] = globalFrame
            writeJSON(report, to: options.out)
            log("inference failed at frame \(globalFrame): \(error)")
            return 4
        }
        let mbNow = physFootprintMB()
        peak = max(peak, mbNow)
        if loop == 0 { memAfterLoop0 = mbNow }
        perLoop.append([
            "loop": loop, "frames": frames, "frames_with_hand": withHand, "frames_with_two_hands": withTwo,
            "latency_ms": stats(loopLatency), "footprint_mb_end": mbNow,
        ])
        log(String(format: "loop %d: %d frames, hand %.1f%%, two %.1f%%, mean %.2f ms, footprint %.0f MB",
                   loop, frames, 100 * Double(withHand) / Double(max(frames, 1)),
                   100 * Double(withTwo) / Double(max(frames, 1)),
                   (stats(loopLatency)["mean"] as? Double) ?? -1, mbNow))
    }
    let wallMs = nowMs() - wallStart
    let detectMs = latencies.reduce(0, +) + (firstFrameMs ?? 0)
    let framesAfterLoop0 = Double(globalFrame) / Double(max(options.loops, 1))
    let steady = memSamples.filter { $0.frame > framesAfterLoop0 }
    report["frames_total"] = globalFrame
    report["first_frame_ms"] = firstFrameMs.map { num($0) } ?? NSNull()
    report["latency_ms"] = stats(latencies)
    report["throughput_fps_inference"] = Double(globalFrame) / (detectMs / 1000)
    report["throughput_fps_wall"] = Double(globalFrame) / (wallMs / 1000)
    report["per_loop"] = perLoop
    report["first_loop_frames"] = firstLoopFrames
    report["memory"] = [
        "footprint_mb_before_init": memBefore,
        "footprint_mb_after_loop0": memAfterLoop0,
        "footprint_mb_end": physFootprintMB(),
        "footprint_mb_peak": peak,
        // MB per 1000 frames after the first loop (warm-up/working-set ramp
        // excluded). The GPU path's texture caches give a bounded ramp; an
        // unbounded leak shows as a steady positive slope here.
        "slope_mb_per_1000_frames_after_loop0": linearSlope(steady.map { ($0.frame, $0.mb) })
            .map { num($0 * 1000) } ?? NSNull(),
        "samples": memSamples.map { [$0.frame, ($0.mb * 10).rounded() / 10] },
    ]
    let handRate = Double(totalWithHand) / Double(max(globalFrame, 1))
    let plausible = handRate >= options.minHandRate
    report["hand_rate"] = handRate
    report["plausibility"] = [
        "pass": plausible,
        "failures": plausible ? [] : [String(format: "hand rate %.3f < %.3f", handRate, options.minHandRate)],
    ]
    report["status"] = plausible ? "ok" : "implausible"
    writeJSON(report, to: options.out)
    if !plausible {
        log(String(format: "IMPLAUSIBLE: hands in %.1f%% of frames (minimum %.1f%%)",
                   100 * handRate, 100 * options.minHandRate))
    }
    log(String(format: "total %d frames: mean %.2f ms p90 %.2f, %.1f fps inference / %.1f fps wall, peak %.0f MB",
                   globalFrame, (stats(latencies)["mean"] as? Double) ?? -1,
                   (stats(latencies)["p90"] as? Double) ?? -1,
                   Double(globalFrame) / (detectMs / 1000), Double(globalFrame) / (wallMs / 1000), peak))
    return plausible ? 0 : 1
}

// MARK: - Still extraction

func runFrames(_ options: Options) -> Int32 {
    guard !options.video.isEmpty, !options.atMs.isEmpty, !options.outDir.isEmpty else {
        die("frames needs --video, --at-ms and --out-dir")
    }
    try? FileManager.default.createDirectory(atPath: options.outDir, withIntermediateDirectories: true)
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: options.video)))
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    let stem = ((options.video as NSString).lastPathComponent as NSString).deletingPathExtension
    for ms in options.atMs {
        let time = CMTime(value: CMTimeValue(ms), timescale: 1000)
        guard let image = try? generator.copyCGImage(at: time, actualTime: nil) else {
            die("could not extract frame at \(ms) ms")
        }
        let path = "\(options.outDir)/\(stem)_\(ms)ms.png"
        guard let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil) else {
            die("cannot write \(path)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { die("cannot write \(path)") }
        print(path)
    }
    return 0
}

// MARK: - Main

let options = parse()
switch options.command {
case "env":
    let data = try! JSONSerialization.data(withJSONObject: environment(), options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
case "image":
    exit(runImage(options))
case "video":
    exit(runVideo(options))
case "frames":
    exit(runFrames(options))
default:
    die("unknown command \(options.command) (env|image|video|frames)")
}
