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

import MediaPipeTasksMac
import SwiftUI

/// Draws normalized landmarks (dots + connection lines) over the camera preview,
/// mapped into the aspect-fit video rect and mirrored to match the preview.
struct LandmarkOverlay: View {
    let bufferWidth: Int
    let bufferHeight: Int
    let mirrored: Bool

    let handResult: HandLandmarkerResult?
    let poseResult: PoseLandmarkerResult?
    let faceResult: FaceLandmarkerResult?

    var body: some View {
        Canvas { context, size in
            let rect = displayedVideoRect(viewSize: size, bufferWidth: bufferWidth, bufferHeight: bufferHeight)

            // Normalized (x, y) → view point, mirrored within the video rect.
            func point(_ x: Float, _ y: Float) -> CGPoint {
                let mx = mirrored ? (1 - CGFloat(x)) : CGFloat(x)
                return CGPoint(x: rect.minX + mx * rect.width,
                               y: rect.minY + CGFloat(y) * rect.height)
            }

            func drawConnections<L>(_ instances: [[L]], _ edges: [(Int, Int)],
                                    _ pos: (L) -> (Float, Float), color: Color, width: CGFloat) {
                for inst in instances {
                    var path = Path()
                    for (a, b) in edges where a < inst.count && b < inst.count {
                        let (ax, ay) = pos(inst[a]); let (bx, by) = pos(inst[b])
                        path.move(to: point(ax, ay)); path.addLine(to: point(bx, by))
                    }
                    context.stroke(path, with: .color(color), lineWidth: width)
                }
            }

            func drawDots<L>(_ instances: [[L]], _ pos: (L) -> (Float, Float), color: Color, radius: CGFloat) {
                for inst in instances {
                    for lm in inst {
                        let (x, y) = pos(lm); let p = point(x, y)
                        context.fill(Path(ellipseIn: CGRect(x: p.x - radius, y: p.y - radius,
                                                            width: radius * 2, height: radius * 2)),
                                     with: .color(color))
                    }
                }
            }

            // Face first (densest, drawn faint), then pose, then hand on top.
            if let face = faceResult {
                drawDots(face.faceLandmarks, { ($0.x, $0.y) }, color: .green.opacity(0.6), radius: 1.0)
            }
            if let pose = poseResult {
                drawConnections(pose.landmarks, Connections.pose, { ($0.x, $0.y) },
                                color: .cyan, width: 3)
                drawDots(pose.landmarks, { ($0.x, $0.y) }, color: .white, radius: 3)
            }
            if let hand = handResult {
                drawConnections(hand.landmarks, Connections.hand, { ($0.x, $0.y) },
                                color: .yellow, width: 2.5)
                drawDots(hand.landmarks, { ($0.x, $0.y) }, color: .red, radius: 3)
            }
        }
        .allowsHitTesting(false)
    }
}
