//
//  ContentView.swift
//  MediaPipeIOSTest
//
//  Test bench for the MediaPipe fork's iOS landmarkers: live camera with
//  hand / face / pose overlays, a CPU / GPU / Core ML delegate switch, and
//  per-task inference-time sparklines.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraEngine()
    @StateObject private var hub = LandmarkerHub()

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                CameraPreviewView(session: camera.session)
                LandmarkOverlay(hub: hub, bufferSize: camera.bufferSize)
                if !camera.authorized {
                    Text("Camera access denied — enable it in Settings.")
                        .padding()
                        .background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 8))
                        .foregroundStyle(.white)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(.black)

            controls
        }
        .onAppear {
            camera.relay.onFrame = { [hub] pixelBuffer, timestampMs in
                hub.offerFrame(pixelBuffer, timestampMs: timestampMs)
            }
            camera.start()
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            HStack {
                Picker("Delegate", selection: $hub.delegateChoice) {
                    ForEach(DelegateChoice.allCases) { choice in
                        Text(choice.rawValue).tag(choice)
                    }
                }
                .pickerStyle(.segmented)

                Button {
                    camera.flipCamera()
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                }
                .buttonStyle(.bordered)
            }

            ForEach(LandmarkTaskKind.allCases) { kind in
                taskRow(kind)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
    }

    private func taskRow(_ kind: LandmarkTaskKind) -> some View {
        let state = hub.lanes[kind] ?? LandmarkerHub.LaneState()
        return HStack(spacing: 10) {
            Toggle(isOn: Binding(
                get: { hub.lanes[kind]?.enabled ?? false },
                set: { hub.setEnabled($0, for: kind) }
            )) {
                EmptyView()
            }
            .labelsHidden()
            .tint(color(for: kind))

            VStack(alignment: .leading, spacing: 0) {
                Text(kind.rawValue)
                    .font(.caption.bold())
                    .foregroundStyle(color(for: kind))
                Text(state.resolved)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
            .frame(width: 56, alignment: .leading)

            Text(String(format: "%5.1f ms", state.lastInferenceMs))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 64, alignment: .trailing)

            TimingSparkline(series: hub.series(for: kind), color: color(for: kind), tick: hub.tick)
                .frame(height: 26)
                .frame(maxWidth: .infinity)
                .background(.black.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                .opacity(state.enabled ? 1 : 0.3)
        }
    }

    private func color(for kind: LandmarkTaskKind) -> Color {
        switch kind {
        case .hand: return .green
        case .face: return .cyan
        case .pose: return .orange
        }
    }
}

#Preview {
    ContentView()
}
