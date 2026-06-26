import AVFoundation
import CoreGraphics
import ImageIO
import XCTest

@testable import MediaPipeTasksMac

// Measurement harness (not a pass/fail unit test): quantifies what the rotation
// fix buys on a *rotated* feed, by comparing detection rate with rotationDegrees
// = 0 (the old, ignored-rotation behaviour) vs the correct value.
//
// Methodology: the UPRIGHT detection is the pseudo-ground-truth. Each source
// frame is panned/scaled to simulate a moving hand (including hard near-edge
// positions), then rotated 90°. We report the detection rate of the rotated
// frames at each rotationDegrees. If the correct rotation beats 0°, rotation
// buys detections.
//
// Run with:
//   MP_VERIFY=1 MP_HAND_MODEL=hand_landmarker.task MP_HAND_IMAGE=hand.jpg \
//     swift test --filter VerifyRotationBenefit
final class VerifyRotationBenefit: XCTestCase {

    private func env(_ k: String) -> String? {
        let v = ProcessInfo.processInfo.environment[k]
        return (v?.isEmpty == false) ? v : nil
    }

    private func loadCGImage(_ path: String) throws -> CGImage {
        guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(s, 0, nil) else {
            throw XCTSkip("Could not load image at \(path)")
        }
        return img
    }

    /// Draws `src` into a same-size black canvas, scaled by `scale` and shifted
    /// by (`dx`,`dy`) in pixels — a cheap stand-in for a hand moving in frame.
    private func frame(_ src: CGImage, scale: CGFloat, dx: CGFloat, dy: CGFloat) -> CGImage? {
        let w = src.width, h = src.height
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bi)
        else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let sw = CGFloat(w) * scale, sh = CGFloat(h) * scale
        ctx.draw(src, in: CGRect(x: dx + (CGFloat(w) - sw) / 2, y: dy + (CGFloat(h) - sh) / 2,
                                 width: sw, height: sh))
        return ctx.makeImage()
    }

    private func rotate90(_ image: CGImage) -> CGImage? {
        let w = image.width, h = image.height
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: nil, width: h, height: w, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bi)
        else { return nil }
        ctx.translateBy(x: CGFloat(h), y: 0)
        ctx.rotate(by: .pi / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    func testMeasureRotationBenefit() throws {
        guard env("MP_VERIFY") != nil,
              let model = env("MP_HAND_MODEL"), let imgPath = env("MP_HAND_IMAGE") else {
            throw XCTSkip("Set MP_VERIFY=1, MP_HAND_MODEL, MP_HAND_IMAGE to run.")
        }
        let src = try loadCGImage(imgPath)

        // Build a sweep of frames: a few scales × a pan path that pushes the hand
        // toward the edges (harder cases where rotation tolerance breaks down).
        let scales: [CGFloat] = [1.0, 0.6, 0.4]
        let pans = stride(from: -0.35, through: 0.35, by: 0.1).map { CGFloat($0) }
        var frames: [CGImage] = []
        for s in scales {
            for p in pans {
                if let f = frame(src, scale: s, dx: p * CGFloat(src.width), dy: p * CGFloat(src.height)) {
                    frames.append(f)
                }
            }
        }

        let o = HandLandmarkerOptions(); o.modelPath = model; o.numHands = 1
        let lm = try HandLandmarker(options: o)

        func rate(_ imgs: [CGImage], _ r: Int) throws -> Double {
            var hits = 0
            for im in imgs {
                let res = try lm.detect(cgImage: im,
                                        imageProcessingOptions: ImageProcessingOptions(rotationDegrees: r))
                if !res.landmarks.isEmpty { hits += 1 }
            }
            return Double(hits) / Double(imgs.count)
        }

        // Upright (reference) detection rate.
        let uprightRate = try rate(frames, 0)

        // Rotated frames.
        let rotated = frames.compactMap(rotate90)
        var rateByR: [Int: Double] = [:]
        for r in [0, 90, -90, 180, 270] {
            rateByR[r] = try rate(rotated, r)
        }
        let bestNonZero = rateByR.filter { $0.key != 0 }.max { $0.value < $1.value }!

        // Accuracy probe: even when both detect, are the landmarks a *better*
        // hand with the correct rotation? Compare a rotation/scale/translation-
        // invariant shape signature (radial distances from the hand centroid)
        // against the upright reference for the same frame. Transform-free.
        func signature(_ hand: [NormalizedLandmark]) -> [Float] {
            var cx: Float = 0, cy: Float = 0
            for p in hand { cx += p.x; cy += p.y }
            let n = Float(hand.count); cx /= n; cy /= n
            let d = hand.map { hypotf($0.x - cx, $0.y - cy) }
            let mean = max(d.reduce(0, +) / n, 1e-6)
            return d.map { $0 / mean }
        }
        func shapeErr(_ a: [NormalizedLandmark], _ b: [NormalizedLandmark]) -> Float? {
            guard a.count == b.count, a.count == 21 else { return nil }
            let sa = signature(a), sb = signature(b)
            var m: Float = 0
            for i in 0..<sa.count { m = max(m, abs(sa[i] - sb[i])) }
            return m
        }
        var err0: [Float] = [], errBest: [Float] = []
        let bestR = bestNonZero.key
        for (i, up) in frames.enumerated() {
            let ref = try lm.detect(cgImage: up)
            guard let refHand = ref.landmarks.first else { continue }
            let r0 = try lm.detect(cgImage: rotated[i],
                                   imageProcessingOptions: ImageProcessingOptions(rotationDegrees: 0))
            let rb = try lm.detect(cgImage: rotated[i],
                                   imageProcessingOptions: ImageProcessingOptions(rotationDegrees: bestR))
            if let h = r0.landmarks.first, let e = shapeErr(h, refHand) { err0.append(e) }
            if let h = rb.landmarks.first, let e = shapeErr(h, refHand) { errBest.append(e) }
        }
        func mean(_ xs: [Float]) -> Float { xs.isEmpty ? -1 : xs.reduce(0, +) / Float(xs.count) }

        print("====== ROTATION BENEFIT (n=\(frames.count) frames) ======")
        print(String(format: "upright (reference) detection rate: %.0f%%", uprightRate * 100))
        for r in [0, 90, -90, 180, 270] {
            print(String(format: "rotated 90°, rotationDegrees=%4d : %.0f%%", r, (rateByR[r] ?? 0) * 100))
        }
        print(String(format: "==> detection buys: %+.0f pp (best %d°: %.0f%% vs 0°: %.0f%%)",
                     (bestNonZero.value - (rateByR[0] ?? 0)) * 100, bestNonZero.key,
                     bestNonZero.value * 100, (rateByR[0] ?? 0) * 100))
        print(String(format: "shape error vs upright ref — 0°: %.3f, best(%d°): %.3f (lower=more upright-like)",
                     mean(err0), bestR, mean(errBest)))
        print("========================================================")

        // Always passes — this is a report. The numbers are the deliverable.
        XCTAssertGreaterThan(frames.count, 0)
    }

    // Real-footage version. Point MP_VIDEO at a clip shot with the camera in its
    // production orientation (e.g. physically rotated 90°). Each frame is run in
    // IMAGE mode (fresh palm detection — no tracking to mask failures) at every
    // rotationDegrees. The rotation with the highest detection rate is the
    // correct mount value, and its lead over 0° is exactly what the fix buys on
    // YOUR footage.
    //
    //   MP_VERIFY=1 MP_VIDEO=clip.mov MP_HAND_MODEL=hand_landmarker.task \
    //     swift test --filter VerifyRotationBenefit/testMeasureRotationBenefitOnVideo
    func testMeasureRotationBenefitOnVideo() throws {
        guard env("MP_VERIFY") != nil, let model = env("MP_HAND_MODEL"),
              let videoPath = env("MP_VIDEO") else {
            throw XCTSkip("Set MP_VERIFY=1, MP_HAND_MODEL, MP_VIDEO to run.")
        }
        let maxFrames = Int(env("MP_VIDEO_MAX_FRAMES") ?? "600") ?? 600
        let expectedHands = Int(env("MP_VIDEO_NUM_HANDS") ?? "2") ?? 2

        let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw XCTSkip("No video track in \(videoPath)")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        reader.startReading()

        let o = HandLandmarkerOptions(); o.modelPath = model; o.numHands = expectedHands
        let lm = try HandLandmarker(options: o)

        let candidates = [0, 90, -90, 180, 270]
        var anyHand: [Int: Int] = [:]   // frames with >=1 hand, per rotation
        var allHands: [Int: Int] = [:]  // frames with == expectedHands, per rotation
        var scoreSum: [Int: Float] = [:]  // sum of top-hand handedness confidence
        var n = 0
        while n < maxFrames, reader.status == .reading, let sb = output.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            for r in candidates {
                let res = try lm.detect(
                    pixelBuffer: pb, imageProcessingOptions: ImageProcessingOptions(rotationDegrees: r))
                if !res.landmarks.isEmpty { anyHand[r, default: 0] += 1 }
                if res.landmarks.count >= expectedHands { allHands[r, default: 0] += 1 }
                if let cat = res.handedness.first?.first { scoreSum[r, default: 0] += cat.score }
            }
            n += 1
        }
        try XCTSkipIf(n == 0, "Decoded 0 frames from \(videoPath)")

        func pct(_ c: Int) -> Double { Double(c) / Double(n) * 100 }
        // Mean handedness confidence over frames where THAT rotation found a hand.
        func meanScore(_ r: Int) -> Float {
            let hits = anyHand[r] ?? 0
            return hits == 0 ? -1 : (scoreSum[r] ?? 0) / Float(hits)
        }
        let bestAny = candidates.filter { $0 != 0 }.max { (anyHand[$0] ?? 0) < (anyHand[$1] ?? 0) }!

        print("====== ROTATION BENEFIT ON VIDEO (\(n) frames, numHands=\(expectedHands)) ======")
        print("rotationDegrees |  >=1 hand  |  all \(expectedHands) hands | mean handedness conf")
        for r in candidates {
            print(String(format: "  %4d°        |   %5.1f%%   |   %5.1f%%   |   %.3f",
                         r, pct(anyHand[r] ?? 0), pct(allHands[r] ?? 0), meanScore(r)))
        }
        print(String(format: "==> fix buys (>=1 hand): %+.1f pp  (best %d°: %.1f%% vs 0°: %.1f%%)",
                     pct(anyHand[bestAny] ?? 0) - pct(anyHand[0] ?? 0), bestAny,
                     pct(anyHand[bestAny] ?? 0), pct(anyHand[0] ?? 0)))
        print(String(format: "==> confidence: best %d°: %.3f vs 0°: %.3f (%+.1f%%)",
                     bestAny, meanScore(bestAny), meanScore(0),
                     (meanScore(bestAny) / max(meanScore(0), 1e-6) - 1) * 100))
        print("=====================================================================")
        XCTAssertGreaterThan(n, 0)
    }

    // VIDEO-mode tracking continuity: the metric that actually matters for a
    // gesture app. Runs the clip in VIDEO mode (real PTS timestamps, tracking
    // enabled) and reports how often the hand is present, how many times tracking
    // drops out, and how long the gaps last. This is the baseline to improve
    // against once roiScale / velocity extrapolation lands in the graph.
    //
    //   MP_VERIFY=1 MP_VIDEO=clip.mov MP_HAND_MODEL=hand_landmarker.task \
    //     swift test --filter VerifyRotationBenefit/testVideoModeContinuity
    func testVideoModeContinuity() throws {
        guard env("MP_VERIFY") != nil, let model = env("MP_HAND_MODEL"),
              let videoPath = env("MP_VIDEO") else {
            throw XCTSkip("Set MP_VERIFY=1, MP_HAND_MODEL, MP_VIDEO to run.")
        }
        let maxFrames = Int(env("MP_VIDEO_MAX_FRAMES") ?? "100000") ?? 100000
        let numHands = Int(env("MP_VIDEO_NUM_HANDS") ?? "1") ?? 1
        let rotation = Int(env("MP_VIDEO_ROTATION") ?? "0") ?? 0

        let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw XCTSkip("No video track in \(videoPath)")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        reader.startReading()

        let o = HandLandmarkerOptions()
        o.modelPath = model; o.numHands = numHands; o.runningMode = .video
        let lm = try HandLandmarker(options: o)

        var present: [Bool] = []
        var lastTs = -1
        while present.count < maxFrames, reader.status == .reading,
              let sb = output.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            var ts = Int((CMSampleBufferGetPresentationTimeStamp(sb).seconds * 1000).rounded())
            if ts <= lastTs { ts = lastTs + 1 }  // VIDEO mode needs strictly increasing ts
            lastTs = ts
            let res = try lm.detectForVideo(
                pixelBuffer: pb, timestampInMilliseconds: ts,
                imageProcessingOptions: ImageProcessingOptions(rotationDegrees: rotation))
            present.append(!res.landmarks.isEmpty)
        }
        let n = present.count
        try XCTSkipIf(n == 0, "Decoded 0 frames from \(videoPath)")

        let hits = present.filter { $0 }.count
        // Gaps = runs of consecutive absent frames.
        var gaps: [Int] = []
        var i = 0
        while i < n {
            if !present[i] {
                var len = 0
                while i < n, !present[i] { len += 1; i += 1 }
                gaps.append(len)
            } else { i += 1 }
        }
        // Dropouts = present→absent transitions mid-stream (tracking losses).
        var dropouts = 0
        for k in 1..<max(n, 1) where present[k - 1] && !present[k] { dropouts += 1 }
        var longestStreak = 0, run = 0
        for p in present { if p { run += 1; longestStreak = max(longestStreak, run) } else { run = 0 } }
        let durSec = Double(lastTs) / 1000.0
        let meanGap = gaps.isEmpty ? 0 : Double(gaps.reduce(0, +)) / Double(gaps.count)
        let fps = durSec > 0 ? Double(n) / durSec : 0

        print("====== VIDEO-MODE CONTINUITY (\(n) frames, numHands=\(numHands), rot=\(rotation)°) ======")
        print(String(format: "duration ~%.1fs @ ~%.0f fps", durSec, fps))
        print(String(format: "hand present:      %.1f%% of frames (%d/%d)",
                     Double(hits) / Double(n) * 100, hits, n))
        print(String(format: "tracking dropouts: %d  (%.2f per second)",
                     dropouts, durSec > 0 ? Double(dropouts) / durSec : 0))
        print(String(format: "absence gaps:      %d  (mean %.1f frames / %.0f ms, max %d frames / %.0f ms)",
                     gaps.count, meanGap, meanGap / max(fps, 1e-6) * 1000,
                     gaps.max() ?? 0, Double(gaps.max() ?? 0) / max(fps, 1e-6) * 1000))
        print(String(format: "longest unbroken track: %d frames (%.1fs)",
                     longestStreak, Double(longestStreak) / max(fps, 1e-6)))
        print("====================================================================")
        XCTAssertGreaterThan(n, 0)
    }
}
