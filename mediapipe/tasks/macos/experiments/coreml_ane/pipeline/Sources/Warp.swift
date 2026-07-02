// vImage-based crops replicating ImageToTensorCalculator's geometry:
//  - palm: whole frame letterboxed (keep-aspect, zero border) into 192x192
//  - landmark: rotated-square ROI crop into 224x224
// CPU path on purpose: leaves the GPU untouched (the point of an ANE pipeline)
// and its cost is charged to the measurement.

import Accelerate
import CoreVideo
import Foundation

final class WarpBuffers {
    let palm: CVPixelBuffer    // 192x192 BGRA
    let land: CVPixelBuffer    // 224x224 BGRA

    init() {
        func make(_ side: Int) -> CVPixelBuffer {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
            ]
            CVPixelBufferCreate(kCFAllocatorDefault, side, side,
                                kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            return pb!
        }
        palm = make(192)
        land = make(224)
    }
}

/// Applies `transform` (source px -> dest px, TOP-DOWN image coordinates)
/// from `src` into `dst`, zero-filling outside. Both buffers are BGRA.
///
/// vImageAffineWarp operates in CoreGraphics' BOTTOM-UP coordinate space
/// (empirically verified: a pure scale with zero translation renders into the
/// bottom of the destination). The given top-down transform M is therefore
/// conjugated with y-flips on both sides: M' = F_dst . M . F_src, i.e.
///   A=a  B=-b  C=-c  D=d  TX=tx + c*srcH  TY=dstH - ty - d*srcH
func warp(src: CVPixelBuffer, dst: CVPixelBuffer, transform: vImage_AffineTransform) {
    CVPixelBufferLockBaseAddress(src, .readOnly)
    CVPixelBufferLockBaseAddress(dst, [])
    defer {
        CVPixelBufferUnlockBaseAddress(dst, [])
        CVPixelBufferUnlockBaseAddress(src, .readOnly)
    }
    var srcBuf = vImage_Buffer(
        data: CVPixelBufferGetBaseAddress(src),
        height: vImagePixelCount(CVPixelBufferGetHeight(src)),
        width: vImagePixelCount(CVPixelBufferGetWidth(src)),
        rowBytes: CVPixelBufferGetBytesPerRow(src))
    var dstBuf = vImage_Buffer(
        data: CVPixelBufferGetBaseAddress(dst),
        height: vImagePixelCount(CVPixelBufferGetHeight(dst)),
        width: vImagePixelCount(CVPixelBufferGetWidth(dst)),
        rowBytes: CVPixelBufferGetBytesPerRow(dst))
    let srcH = Float(CVPixelBufferGetHeight(src))
    let dstH = Float(CVPixelBufferGetHeight(dst))
    var t = vImage_AffineTransform(
        a: transform.a, b: -transform.b,
        c: -transform.c, d: transform.d,
        tx: transform.tx + transform.c * srcH,
        ty: dstH - transform.ty - transform.d * srcH)
    var back: Pixel_8888 = (0, 0, 0, 255)
    let err = vImageAffineWarp_ARGB8888(&srcBuf, &dstBuf, nil, &t, &back,
                                        vImage_Flags(kvImageBackgroundColorFill))
    if err != kvImageNoError {
        fatalError("vImageAffineWarp failed: \(err)")
    }
}

/// Debug: write a BGRA pixel buffer as PNG.
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

func dumpPNG(_ pb: CVPixelBuffer, to path: String) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let ctx = CGContext(
        data: CVPixelBufferGetBaseAddress(pb), width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue)!
    let img = ctx.makeImage()!
    let dest = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

/// Letterbox the whole frame into a `side`-square. Returns the pad/scale info
/// (in square-normalized units) needed to undo the letterbox on decode.
func letterbox(src: CVPixelBuffer, dst: CVPixelBuffer, side: Float)
    -> (padX: Float, padY: Float, scaleX: Float, scaleY: Float) {
    let w = Float(CVPixelBufferGetWidth(src))
    let h = Float(CVPixelBufferGetHeight(src))
    let s = min(side / w, side / h)
    let padXpx = (side - s * w) / 2
    let padYpx = (side - s * h) / 2
    let t = vImage_AffineTransform(a: s, b: 0, c: 0, d: s, tx: padXpx, ty: padYpx)
    warp(src: src, dst: dst, transform: t)
    return (padX: padXpx / side, padY: padYpx / side,
            scaleX: s * w / side, scaleY: s * h / side)
}

/// Crop the (rotated, square) ROI into a `side`-square buffer.
func cropROI(src: CVPixelBuffer, dst: CVPixelBuffer, roi: ROI, side: Float) {
    let w = Float(CVPixelBufferGetWidth(src))
    let h = Float(CVPixelBufferGetHeight(src))
    let cx = roi.cx * w, cy = roi.cy * h
    let sizePx = max(roi.w * w, 1)  // square_long: w*W == h*H
    let k = side / sizePx
    let cosR = cosf(roi.rotation), sinR = sinf(roi.rotation)
    // dest = k * R(-rot) * (p - c) + side/2
    let a = k * cosR, b = -k * sinR
    let c = k * sinR, d = k * cosR
    let tx = side / 2 - (a * cx + c * cy)
    let ty = side / 2 - (b * cx + d * cy)
    warp(src: src, dst: dst,
         transform: vImage_AffineTransform(a: a, b: b, c: c, d: d, tx: tx, ty: ty))
}
