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

import AVFoundation
import SwiftUI

/// Hosts an `AVCaptureVideoPreviewLayer` (aspect-fit) for the live camera feed.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    let mirrored: Bool

    func makeNSView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateNSView(_ nsView: PreviewView, context: Context) {
        guard let connection = nsView.previewLayer.connection else { return }
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = mirrored
        }
    }

    final class PreviewView: NSView {
        override init(frame frameRect: NSRect) {
            super.init(frame: frameRect)
            layer = AVCaptureVideoPreviewLayer()
            wantsLayer = true
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

/// Maps the displayed (aspect-fit, letterboxed) video rect within a view of the
/// given size, matching `AVLayerVideoGravity.resizeAspect`.
func displayedVideoRect(viewSize: CGSize, bufferWidth: Int, bufferHeight: Int) -> CGRect {
    guard bufferWidth > 0, bufferHeight > 0, viewSize.width > 0, viewSize.height > 0 else {
        return CGRect(origin: .zero, size: viewSize)
    }
    let videoAspect = CGFloat(bufferWidth) / CGFloat(bufferHeight)
    let viewAspect = viewSize.width / viewSize.height
    if videoAspect > viewAspect {
        // Width-limited (letterbox top/bottom).
        let h = viewSize.width / videoAspect
        return CGRect(x: 0, y: (viewSize.height - h) / 2, width: viewSize.width, height: h)
    } else {
        // Height-limited (pillarbox left/right).
        let w = viewSize.height * videoAspect
        return CGRect(x: (viewSize.width - w) / 2, y: 0, width: w, height: viewSize.height)
    }
}
