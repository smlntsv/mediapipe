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

import Accelerate
import CoreVideo
import Foundation
import MediaPipeTasksObjC

extension MPCImage {
    /// Creates a native MediaPipe image from a `CVPixelBuffer`.
    ///
    /// Milestone: only `kCVPixelFormatType_32BGRA` is supported. The buffer is
    /// converted to a tightly-packed RGBA8 buffer (the same representation used
    /// by the `CGImage` path) via a vImage channel permute, which also strips
    /// any per-row padding.
    public convenience init(pixelBuffer: CVPixelBuffer) throws {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw MediaPipeError.invalidImage(
                "Unsupported CVPixelBuffer format \(format); only kCVPixelFormatType_32BGRA "
                + "is supported.")
        }

        let lockFlags = CVPixelBufferLockFlags.readOnly
        guard CVPixelBufferLockBaseAddress(pixelBuffer, lockFlags) == kCVReturnSuccess else {
            throw MediaPipeError.invalidImage("Failed to lock CVPixelBuffer base address.")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, lockFlags) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard width > 0, height > 0, let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw MediaPipeError.invalidImage("CVPixelBuffer has no base address or zero size.")
        }

        let dstRowBytes = width * 4
        var rgba = [UInt8](repeating: 0, count: dstRowBytes * height)

        var src = vImage_Buffer(
            data: base,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: srcRowBytes)

        rgba.withUnsafeMutableBytes { dst in
            var dstBuffer = vImage_Buffer(
                data: dst.baseAddress,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: dstRowBytes)
            // Input order is B,G,R,A (channels 0,1,2,3). Output RGBA takes
            // channels [2,1,0,3] → R,G,B,A.
            var permuteMap: [UInt8] = [2, 1, 0, 3]
            _ = vImagePermuteChannels_ARGB8888(
                &src, &dstBuffer, &permuteMap, vImage_Flags(kvImageNoFlags))
        }

        try self.init(rgbaData: Data(rgba), width: width, height: height)
    }
}
