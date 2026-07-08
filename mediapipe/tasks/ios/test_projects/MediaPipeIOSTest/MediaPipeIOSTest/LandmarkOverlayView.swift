//
//  LandmarkOverlayView.swift
//  MediaPipeIOSTest
//
//  Camera preview + landmark overlay. The preview layer uses .resizeAspect
//  and the overlay Canvas maps normalized landmark coordinates into the same
//  aspect-fit rect, so points land exactly on the video.
//

import AVFoundation
import SwiftUI

/// UIKit wrapper hosting the AVCaptureVideoPreviewLayer.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

struct LandmarkOverlay: View {
    @ObservedObject var hub: LandmarkerHub
    /// Pixel-buffer size, for aspect-fit mapping.
    let bufferSize: CGSize

    var body: some View {
        GeometryReader { geometry in
            let videoRect = aspectFitRect(content: bufferSize, container: geometry.size)
            Canvas { context, _ in
                if let hand = hub.lanes[.hand], hand.enabled {
                    draw(sets: hand.points, connections: hub.handConnections,
                         color: .green, pointRadius: 3, in: videoRect, context: &context)
                }
                if let pose = hub.lanes[.pose], pose.enabled {
                    draw(sets: pose.points, connections: hub.poseConnections,
                         color: .orange, pointRadius: 3, in: videoRect, context: &context)
                }
                if let face = hub.lanes[.face], face.enabled {
                    draw(sets: face.points, connections: [],
                         color: .cyan, pointRadius: 1, in: videoRect, context: &context)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func aspectFitRect(content: CGSize, container: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0 else { return .zero }
        let scale = min(container.width / content.width, container.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(
            x: (container.width - size.width) / 2,
            y: (container.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }

    private func draw(
        sets: [[CGPoint]],
        connections: [(Int, Int)],
        color: Color,
        pointRadius: CGFloat,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        for set in sets {
            let mapped = set.map { point in
                CGPoint(x: rect.minX + point.x * rect.width, y: rect.minY + point.y * rect.height)
            }
            if !connections.isEmpty {
                var path = Path()
                for (start, end) in connections where start < mapped.count && end < mapped.count {
                    path.move(to: mapped[start])
                    path.addLine(to: mapped[end])
                }
                context.stroke(path, with: .color(color.opacity(0.8)), lineWidth: 2)
            }
            var dots = Path()
            for point in mapped {
                dots.addEllipse(in: CGRect(
                    x: point.x - pointRadius, y: point.y - pointRadius,
                    width: pointRadius * 2, height: pointRadius * 2
                ))
            }
            context.fill(dots, with: .color(color))
        }
    }
}
