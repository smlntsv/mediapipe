//
//  TimingSparkline.swift
//  MediaPipeIOSTest
//
//  Tiny inference-time chart (port of the web client's sparkline): ring
//  buffer lives outside SwiftUI state, one polyline per redraw, y-axis
//  anchored at 0 and autoscaled to the window max (printed top-right) so
//  charts of different tasks are visually comparable.
//

import SwiftUI

struct TimingSparkline: View {
    let series: TimingSeries
    let color: Color
    /// Any changing value that signals "new sample" (the hub's tick). A new
    /// value makes this view struct differ from the previous one, which is
    /// what tells SwiftUI to re-render the Canvas.
    let tick: Int

    var body: some View {
        Canvas { [tick] context, size in
            _ = tick
            let values = series.snapshot()
            guard values.count > 1 else { return }
            let maxValue = values.max() ?? 1
            guard maxValue > 0 else { return }
            let scale = maxValue * 1.15
            let stepX = size.width / CGFloat(TimingSeries.capacity - 1)

            var path = Path()
            for (index, value) in values.enumerated() {
                let point = CGPoint(
                    x: CGFloat(index) * stepX,
                    y: size.height - CGFloat(value / scale) * size.height
                )
                if index == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            context.stroke(path, with: .color(color), lineWidth: 1)

            context.draw(
                Text(String(format: "%.0f", maxValue))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary),
                at: CGPoint(x: size.width - 8, y: 6)
            )
        }
        .id(tick)  // redraw per sample batch
    }
}
