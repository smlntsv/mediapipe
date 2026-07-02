// Faithful Swift ports of the MediaPipe pre/post-processing math used by the
// hand landmarker graph. Constants and formulas taken from:
//   hand_detector_graph.cc (SSD anchors, TensorsToDetections, NMS,
//                           DetectionsToRects, RectTransformation x2.6)
//   hand_landmarks_to_rect_calculator.cc (rotation + landmark bbox)
//   rect_transformation_calculator.cc (shift / square_long)
//   landmark_projection_calculator.cc (crop -> frame projection)

import Foundation

// MARK: - Types

/// NormalizedRect: center/size in frame-normalized coords + rotation (rad).
struct ROI {
    var cx: Float
    var cy: Float
    var w: Float
    var h: Float
    var rotation: Float
}

struct Detection {
    var score: Float
    // Frame-normalized axis-aligned box.
    var cx: Float, cy: Float, w: Float, h: Float
    // 7 palm keypoints, frame-normalized.
    var keypoints: [SIMD2<Float>]
}

struct Hand {
    var landmarks: [SIMD3<Float>]  // frame-normalized x,y + z
    var presence: Float
    var roiNext: ROI
}

func normalizeRadians(_ angle: Float) -> Float {
    angle - 2 * .pi * floorf((angle + .pi) / (2 * .pi))
}

@inline(__always) func sigmoid(_ x: Float) -> Float {
    1 / (1 + expf(-min(max(x, -100), 100)))
}

/// Axis-aligned IoU ignoring rotation — matches mediapipe rectangle_util.
func iou(_ a: ROI, _ b: ROI) -> Float {
    let ax0 = a.cx - a.w / 2, ax1 = a.cx + a.w / 2
    let ay0 = a.cy - a.h / 2, ay1 = a.cy + a.h / 2
    let bx0 = b.cx - b.w / 2, bx1 = b.cx + b.w / 2
    let by0 = b.cy - b.h / 2, by1 = b.cy + b.h / 2
    let ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    let iy = max(0, min(ay1, by1) - max(ay0, by0))
    let inter = ix * iy
    let uni = a.w * a.h + b.w * b.h - inter
    return uni > 0 ? inter / uni : 0
}

// MARK: - SSD anchors (SsdAnchorsCalculator with the palm config)

struct Anchor { var x: Float; var y: Float }

/// strides [8,16,16,16], minScale .1484375, maxScale .75, aspect [1.0],
/// interpolatedScaleAspectRatio 1.0, fixedAnchorSize, offset 0.5 -> 2016.
func generatePalmAnchors() -> [Anchor] {
    let strides = [8, 16, 16, 16]
    let minScale: Float = 0.1484375, maxScale: Float = 0.75
    let inputSize = 192
    func calcScale(_ idx: Int) -> Float {
        minScale + (maxScale - minScale) * Float(idx) / Float(strides.count - 1)
    }
    var anchors: [Anchor] = []
    var layer = 0
    while layer < strides.count {
        var anchorsPerLoc = 0
        var last = layer
        while last < strides.count && strides[last] == strides[layer] {
            let scale = calcScale(last)
            _ = scale
            anchorsPerLoc += 1  // aspect 1.0
            // interpolated_scale_aspect_ratio == 1.0 adds one more per layer
            anchorsPerLoc += 1
            last += 1
        }
        let stride = strides[layer]
        let fm = Int(ceil(Float(inputSize) / Float(stride)))
        for y in 0..<fm {
            for x in 0..<fm {
                for _ in 0..<anchorsPerLoc {
                    anchors.append(Anchor(x: (Float(x) + 0.5) / Float(fm),
                                          y: (Float(y) + 0.5) / Float(fm)))
                }
            }
        }
        layer = last
    }
    precondition(anchors.count == 2016, "expected 2016 anchors, got \(anchors.count)")
    return anchors
}

// MARK: - TensorsToDetections (palm decode)

/// raw: [2016 x 18] (reverse_output_order: x,y,w,h then 7 keypoints x,y —
/// all offsets/sizes divided by 192; fixed anchor size). logits: [2016].
/// Letterbox undo maps the 192-square coords back to frame-normalized.
func decodePalms(
    raw: [Float], logits: [Float], anchors: [Anchor],
    scoreThreshold: Float,
    padX: Float, padY: Float, scaleX: Float, scaleY: Float
) -> [Detection] {
    var out: [Detection] = []
    let n = anchors.count
    out.reserveCapacity(8)
    for i in 0..<n {
        let score = sigmoid(logits[i])
        guard score >= scoreThreshold else { continue }
        let b = i * 18
        let a = anchors[i]
        // 192-square normalized coords.
        let cx = raw[b] / 192 + a.x
        let cy = raw[b + 1] / 192 + a.y
        let w = raw[b + 2] / 192
        let h = raw[b + 3] / 192
        func unletterbox(_ p: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2((p.x - padX) / scaleX, (p.y - padY) / scaleY)
        }
        let c = unletterbox(SIMD2(cx, cy))
        var kps: [SIMD2<Float>] = []
        kps.reserveCapacity(7)
        for k in 0..<7 {
            let kx = raw[b + 4 + 2 * k] / 192 + a.x
            let ky = raw[b + 5 + 2 * k] / 192 + a.y
            kps.append(unletterbox(SIMD2(kx, ky)))
        }
        out.append(Detection(score: score, cx: c.x, cy: c.y,
                             w: w / scaleX, h: h / scaleY, keypoints: kps))
    }
    return out
}

// MARK: - Weighted NMS (NonMaxSuppressionCalculator WEIGHTED, IoU 0.3)

func weightedNMS(_ detections: [Detection], threshold: Float = 0.3) -> [Detection] {
    // Index-based with a consumed mask — no per-iteration array rebuilds.
    let sorted = detections.sorted { $0.score > $1.score }
    var consumed = [Bool](repeating: false, count: sorted.count)
    var out: [Detection] = []
    for i in 0..<sorted.count {
        guard !consumed[i] else { continue }
        let top = sorted[i]
        let topROI = ROI(cx: top.cx, cy: top.cy, w: top.w, h: top.h, rotation: 0)
        // Weighted average of location data by score over all overlaps.
        var acc = Detection(score: top.score, cx: 0, cy: 0, w: 0, h: 0,
                            keypoints: Array(repeating: .zero, count: 7))
        var totalScore: Float = 0
        for j in i..<sorted.count {
            guard !consumed[j] else { continue }
            let d = sorted[j]
            let r = ROI(cx: d.cx, cy: d.cy, w: d.w, h: d.h, rotation: 0)
            guard iou(topROI, r) > threshold else { continue }
            consumed[j] = true
            totalScore += d.score
            acc.cx += d.cx * d.score; acc.cy += d.cy * d.score
            acc.w += d.w * d.score; acc.h += d.h * d.score
            for k in 0..<7 { acc.keypoints[k] += d.keypoints[k] * d.score }
        }
        acc.cx /= totalScore; acc.cy /= totalScore
        acc.w /= totalScore; acc.h /= totalScore
        for k in 0..<7 { acc.keypoints[k] /= totalScore }
        out.append(acc)
    }
    return out
}

// MARK: - RectTransformation (exact port)

func transformRect(_ rect: ROI, imageW: Float, imageH: Float,
                   scale: Float, shiftX: Float, shiftY: Float,
                   squareLong: Bool) -> ROI {
    var r = rect
    let width = r.w, height = r.h, rotation = r.rotation
    if rotation == 0 {
        r.cx += width * shiftX
        r.cy += height * shiftY
    } else {
        let xShift = (imageW * width * shiftX * cosf(rotation)
                      - imageH * height * shiftY * sinf(rotation)) / imageW
        let yShift = (imageW * width * shiftX * sinf(rotation)
                      + imageH * height * shiftY * cosf(rotation)) / imageH
        r.cx += xShift
        r.cy += yShift
    }
    if squareLong {
        let longSide = max(width * imageW, height * imageH)
        r.w = longSide / imageW
        r.h = longSide / imageH
    }
    r.w *= scale
    r.h *= scale
    return r
}

// MARK: - DetectionsToRects (palm detection -> ROI; kp0 wrist, kp2 middle MCP)

func palmDetectionToROI(_ d: Detection, imageW: Float, imageH: Float) -> ROI {
    let x0 = d.keypoints[0].x * imageW, y0 = d.keypoints[0].y * imageH
    let x1 = d.keypoints[2].x * imageW, y1 = d.keypoints[2].y * imageH
    let rotation = normalizeRadians(.pi / 2 - atan2f(-(y1 - y0), x1 - x0))
    let rect = ROI(cx: d.cx, cy: d.cy, w: d.w, h: d.h, rotation: rotation)
    return transformRect(rect, imageW: imageW, imageH: imageH,
                         scale: 2.6, shiftX: 0, shiftY: -0.5, squareLong: true)
}

// MARK: - HandLandmarksToRect (landmarks -> next-frame ROI)

/// Rotation from landmark indices 0/4/6/8 of the FULL 21-landmark list —
/// exactly as the graph executes (the calculator was written for a partial
/// list but the tasks graph feeds all 21).
func landmarksRotation(_ lm: [SIMD3<Float>], imageW: Float, imageH: Float) -> Float {
    let x0 = lm[0].x * imageW, y0 = lm[0].y * imageH
    var x1 = (lm[4].x + lm[8].x) / 2, y1 = (lm[4].y + lm[8].y) / 2
    x1 = (x1 + lm[6].x) / 2 * imageW
    y1 = (y1 + lm[6].y) / 2 * imageH
    return normalizeRadians(.pi / 2 - atan2f(-(y1 - y0), x1 - x0))
}

func landmarksToROI(_ lm: [SIMD3<Float>], imageW: Float, imageH: Float,
                    roiScale: Float) -> ROI {
    let rotation = landmarksRotation(lm, imageW: imageW, imageH: imageH)
    let reverse = normalizeRadians(-rotation)

    var minX = Float.greatestFiniteMagnitude, maxX = -Float.greatestFiniteMagnitude
    var minY = Float.greatestFiniteMagnitude, maxY = -Float.greatestFiniteMagnitude
    for p in lm {
        minX = min(minX, p.x); maxX = max(maxX, p.x)
        minY = min(minY, p.y); maxY = max(maxY, p.y)
    }
    let axisCX = (maxX + minX) / 2, axisCY = (maxY + minY) / 2

    var pMinX = Float.greatestFiniteMagnitude, pMaxX = -Float.greatestFiniteMagnitude
    var pMinY = Float.greatestFiniteMagnitude, pMaxY = -Float.greatestFiniteMagnitude
    for p in lm {
        let ox = (p.x - axisCX) * imageW
        let oy = (p.y - axisCY) * imageH
        let px = ox * cosf(reverse) - oy * sinf(reverse)
        let py = ox * sinf(reverse) + oy * cosf(reverse)
        pMinX = min(pMinX, px); pMaxX = max(pMaxX, px)
        pMinY = min(pMinY, py); pMaxY = max(pMaxY, py)
    }
    let projCX = (pMaxX + pMinX) / 2, projCY = (pMaxY + pMinY) / 2
    let cx = projCX * cosf(rotation) - projCY * sinf(rotation) + imageW * axisCX
    let cy = projCX * sinf(rotation) + projCY * cosf(rotation) + imageH * axisCY

    let rect = ROI(cx: cx / imageW, cy: cy / imageH,
                   w: (pMaxX - pMinX) / imageW, h: (pMaxY - pMinY) / imageH,
                   rotation: rotation)
    return transformRect(rect, imageW: imageW, imageH: imageH,
                         scale: roiScale, shiftX: 0, shiftY: -0.1, squareLong: true)
}

// MARK: - LandmarkProjection (224-crop coords -> frame-normalized)

func projectLandmarks(_ raw: [Float], roi: ROI) -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    out.reserveCapacity(21)
    let cosR = cosf(roi.rotation), sinR = sinf(roi.rotation)
    for i in 0..<21 {
        let nx = raw[3 * i] / 224 - 0.5
        let ny = raw[3 * i + 1] / 224 - 0.5
        let x = roi.cx + (nx * cosR - ny * sinR) * roi.w
        let y = roi.cy + (nx * sinR + ny * cosR) * roi.h
        let z = raw[3 * i + 2] / 224 * roi.w
        out.append(SIMD3(x, y, z))
    }
    return out
}
