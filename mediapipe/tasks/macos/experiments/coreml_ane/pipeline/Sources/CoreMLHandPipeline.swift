// The Core ML/ANE hand landmarker pipeline: a faithful reimplementation of
// MediaPipe's HandLandmarkerGraph VIDEO-mode dataflow with Core ML inference.
//
//   frame -> [tracked < numHands?] -> letterbox 192 -> palm model -> decode
//              -> weighted NMS -> detection->ROI (x2.6) -> associate
//         -> per ROI: rotated crop 224 -> landmark model -> presence gate
//              -> project landmarks -> landmarks->ROI (x2.0) = next frame ROIs
//
// All marshaling costs (vImage warps, CVPixelBuffer feature values, output
// array conversion, anchor decode, NMS) are inside the timed region.

import CoreML
import CoreVideo
import Foundation

struct StageTimes {
    var palmPre = 0.0, palmInfer = 0.0, palmPost = 0.0
    var landPre = 0.0, landInfer = 0.0, landPost = 0.0
    var total = 0.0
    var palmRan = false
    var handCount = 0
    /// Detections above threshold BEFORE NMS on this frame's palm run —
    /// a healthy frame has O(1..20); thousands means the decode is broken.
    var palmCandidates = 0
}

final class CoreMLHandPipeline {
    private let palmModel: MLModel
    private let landModel: MLModel
    private let buffers = WarpBuffers()
    private var debugFirstFrame = ProcessInfo.processInfo.environment["PIPE_DEBUG"] != nil
    private var debugLandDumps = ProcessInfo.processInfo.environment["PIPE_DEBUG"] != nil ? 3 : 0
    private var frameIndex = 0
    private let anchors = generatePalmAnchors()
    private var rois: [ROI] = []

    private let numHands = 2
    private let minDetectionConfidence: Float = 0.5
    private let minPresenceConfidence: Float = 0.5
    private let associationIoU: Float = 0.5  // min_tracking_confidence

    // CoreML output feature names (see convert_image_input.py output mapping).
    private let palmRegressors = "var_876"   // [1,2016,18]
    private let palmLogits = "var_874"       // [1,2016,1]
    private let landLandmarks = "var_486"    // [1,63] x,y,z in 224 px
    private let landPresence = "var_491"     // [1,1]

    init(palmURL: URL, landURL: URL, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        palmModel = try MLModel(contentsOf: palmURL, configuration: config)
        landModel = try MLModel(contentsOf: landURL, configuration: config)
    }

    func reset() { rois = [] }

    func process(frame: CVPixelBuffer) throws -> (hands: [Hand], times: StageTimes) {
        var t = StageTimes()
        let frameW = Float(CVPixelBufferGetWidth(frame))
        let frameH = Float(CVPixelBufferGetHeight(frame))
        let tStart = now()

        // --- Palm detection (only when short of hands — DisallowIf semantics)
        if rois.count < numHands {
            t.palmRan = true
            var t0 = now()
            let lb = letterbox(src: frame, dst: buffers.palm, side: 192)
            t.palmPre = now() - t0

            if debugFirstFrame {
                debugFirstFrame = false
                dumpPNG(buffers.palm, to: "/tmp/mp_build/palm_letterbox_debug.png")
                let out = try palmModel.prediction(from: try provider(buffers.palm))
                let arrL = out.featureValue(for: palmLogits)!.multiArrayValue!
                let logits = floats(arrL)
                let top = logits.enumerated().max { $0.element < $1.element }!
                let msg = "DEBUG palm: logits[0..4]=\(logits.prefix(5).map { String(format: "%.2f", $0) })"
                    + " max=\(String(format: "%.2f", top.element))@\(top.offset)"
                    + " count>0=\(logits.filter { $0 > 0 }.count)"
                    + " dtype=\(arrL.dataType.rawValue) shape=\(arrL.shape) strides=\(arrL.strides)\n"
                FileHandle.standardError.write(msg.data(using: .utf8)!)
            }

            t0 = now()
            let out = try palmModel.prediction(from: try provider(buffers.palm))
            t.palmInfer = now() - t0

            t0 = now()
            let raw = floats(out.featureValue(for: palmRegressors)!.multiArrayValue!)
            let logits = floats(out.featureValue(for: palmLogits)!.multiArrayValue!)
            let candidates = decodePalms(
                raw: raw, logits: logits, anchors: anchors,
                scoreThreshold: minDetectionConfidence,
                padX: lb.padX, padY: lb.padY, scaleX: lb.scaleX, scaleY: lb.scaleY)
            t.palmCandidates = candidates.count
            let detections = weightedNMS(candidates)
            // HandAssociation: existing (tracked) ROIs have priority; add fresh
            // non-overlapping detections up to numHands.
            for d in detections.sorted(by: { $0.score > $1.score }) {
                guard rois.count < numHands else { break }
                let candidate = palmDetectionToROI(d, imageW: frameW, imageH: frameH)
                // Guard degenerate/NaN ROIs — a bad rect would make the crop
                // transform singular (vImage kvImageInvalidParameter).
                guard candidate.w.isFinite, candidate.h.isFinite,
                      candidate.cx.isFinite, candidate.cy.isFinite,
                      candidate.w > 1e-4, candidate.h > 1e-4 else { continue }
                if !rois.contains(where: { iou($0, candidate) > associationIoU }) {
                    rois.append(candidate)
                }
            }
            t.palmPost = now() - t0
        }

        // --- Landmark model per ROI
        var hands: [Hand] = []
        var nextROIs: [ROI] = []
        for roi in rois {
            var t0 = now()
            cropROI(src: frame, dst: buffers.land, roi: roi, side: 224)
            t.landPre += now() - t0

            t0 = now()
            let out = try landModel.prediction(from: try provider(buffers.land))
            t.landInfer += now() - t0

            t0 = now()
            let presence = floats(out.featureValue(for: landPresence)!.multiArrayValue!)[0]

            if debugLandDumps > 0 && frameIndex > 250 {
                debugLandDumps -= 1
                dumpPNG(buffers.land, to: "/tmp/mp_build/land_crop_\(debugLandDumps).png")
                let msg = String(format: "DEBUG land[%d]: roi cx=%.3f cy=%.3f w=%.3f h=%.3f rot=%.2f presence=%.3f\n",
                                 debugLandDumps, roi.cx, roi.cy, roi.w, roi.h, roi.rotation, presence)
                FileHandle.standardError.write(msg.data(using: .utf8)!)
            }
            if presence >= minPresenceConfidence {
                let raw = floats(out.featureValue(for: landLandmarks)!.multiArrayValue!)
                let landmarks = projectLandmarks(raw, roi: roi)
                let next = landmarksToROI(landmarks, imageW: frameW, imageH: frameH,
                                          roiScale: 2.0)
                hands.append(Hand(landmarks: landmarks, presence: presence, roiNext: next))
                nextROIs.append(next)
            }
            t.landPost += now() - t0
        }
        rois = nextROIs
        frameIndex += 1
        t.total = now() - tStart
        t.handCount = hands.count
        return (hands, t)
    }

    // MARK: - Debug: landmark leg in isolation

    /// Runs crop -> landmark model -> projection on caller-supplied ROIs and
    /// reports presence + projected wrist, dumping each crop. Used to isolate
    /// the tracking leg from the palm-detection leg.
    func debugLandmarkLeg(frame: CVPixelBuffer, rois: [ROI],
                          referenceWrists: [SIMD2<Float>]) throws {
        let frameW = Float(CVPixelBufferGetWidth(frame))
        let frameH = Float(CVPixelBufferGetHeight(frame))
        for (i, roi) in rois.enumerated() {
            // Variant sweep: which crop convention does the model accept?
            var variants: [(String, ROI)] = [
                ("as-is   ", roi),
                ("negRot  ", ROI(cx: roi.cx, cy: roi.cy, w: roi.w, h: roi.h, rotation: -roi.rotation)),
                ("zeroRot ", ROI(cx: roi.cx, cy: roi.cy, w: roi.w, h: roi.h, rotation: 0)),
                ("halfSize", ROI(cx: roi.cx, cy: roi.cy, w: roi.w / 2, h: roi.h / 2, rotation: roi.rotation)),
            ]
            // plus: center shifted DOWN by 0.1*h (i.e. shift sign flipped)
            variants.append(("shiftDn ", ROI(cx: roi.cx, cy: roi.cy + 0.2 * roi.h, w: roi.w, h: roi.h, rotation: roi.rotation)))
            for (name, v) in variants {
                cropROI(src: frame, dst: buffers.land, roi: v, side: 224)
                if name == "as-is   " {
                    dumpPNG(buffers.land, to: "/tmp/mp_build/leg_crop_\(i).png")
                }
                let out = try landModel.prediction(from: try provider(buffers.land))
                let presence = floats(out.featureValue(for: landPresence)!.multiArrayValue!)[0]
                let raw = floats(out.featureValue(for: landLandmarks)!.multiArrayValue!)
                let lm = projectLandmarks(raw, roi: v)
                let refW = i < referenceWrists.count ? referenceWrists[i] : SIMD2<Float>(-1, -1)
                let dx = (lm[0].x - refW.x) * frameW, dy = (lm[0].y - refW.y) * frameH
                let msg = String(format: "DEBUG leg[%d] %@: rot=%.2f | presence=%.3f | wristErrPx=%.1f\n",
                                 i, name as NSString, v.rotation, presence,
                                 (dx * dx + dy * dy).squareRoot())
                FileHandle.standardError.write(msg.data(using: .utf8)!)
            }
        }
    }

    // MARK: - Helpers

    private func provider(_ pb: CVPixelBuffer) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)])
    }

    @inline(__always) private func now() -> Double { CFAbsoluteTimeGetCurrent() }
}

/// MLMultiArray -> [Float] in logical (row-major) order, honoring STRIDES —
/// ANE-produced outputs are frequently padded/non-contiguous, so a raw linear
/// copy silently reads garbage/zeros. float32/float16 via raw memory (never
/// the NSNumber subscript, which would dominate the timings).
func floats(_ array: MLMultiArray) -> [Float] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    let n = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: n)

    // Fast path: strides describe a dense row-major layout.
    var dense = true
    var expect = 1
    for d in stride(from: shape.count - 1, through: 0, by: -1) {
        if strides[d] != expect { dense = false; break }
        expect *= shape[d]
    }

    func copyDense() {
        array.withUnsafeBytes { raw in
            switch array.dataType {
            case .float32:
                out.withUnsafeMutableBufferPointer {
                    $0.baseAddress!.update(from: raw.bindMemory(to: Float.self).baseAddress!, count: n)
                }
            case .float16:
                let src = raw.bindMemory(to: Float16.self)
                for i in 0..<n { out[i] = Float(src[i]) }
            default:
                for i in 0..<n { out[i] = array[i].floatValue }
            }
        }
    }

    func copyStrided() {
        array.withUnsafeBytes { raw in
            func element(_ offset: Int) -> Float {
                switch array.dataType {
                case .float32: return raw.bindMemory(to: Float.self)[offset]
                case .float16: return Float(raw.bindMemory(to: Float16.self)[offset])
                default: return array[offset].floatValue
                }
            }
            var idx = [Int](repeating: 0, count: shape.count)
            for i in 0..<n {
                var offset = 0
                for d in 0..<shape.count { offset += idx[d] * strides[d] }
                out[i] = element(offset)
                var d = shape.count - 1
                while d >= 0 {
                    idx[d] += 1
                    if idx[d] < shape[d] { break }
                    idx[d] = 0
                    d -= 1
                }
            }
        }
    }

    dense ? copyDense() : copyStrided()
    return out
}
