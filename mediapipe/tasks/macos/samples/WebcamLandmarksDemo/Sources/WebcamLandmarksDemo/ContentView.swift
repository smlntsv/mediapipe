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

struct ContentView: View {
    @StateObject private var camera: CameraController
    @State private var task: DemoTask
    @State private var delegate: MediaPipeDelegate
    @State private var stage: PipelineStage
    @State private var lowPower: Bool

    init() {
        // Seed UI state from the controller (which honors WEBCAM_STAGE/DELEGATE/
        // LOWPOWER env overrides) so a launch-time mode shows correctly.
        let cam = CameraController()
        _camera = StateObject(wrappedValue: cam)
        _task = State(initialValue: .all)
        _delegate = State(initialValue: cam.initialDelegate)
        _stage = State(initialValue: cam.initialStage)
        _lowPower = State(initialValue: cam.initialLowPower)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: camera.session, mirrored: camera.mirrored)
                .overlay {
                    // Only mount the Canvas overlay in Full; in Preview/Convert it
                    // would redraw on every state change despite having no results.
                    if stage == .full {
                        LandmarkOverlay(
                            bufferWidth: camera.bufferWidth,
                            bufferHeight: camera.bufferHeight,
                            mirrored: camera.mirrored,
                            handResult: task.runsHand ? camera.handResult : nil,
                            poseResult: task.runsPose ? camera.poseResult : nil,
                            faceResult: task.runsFace ? camera.faceResult : nil)
                    }
                }

            VStack {
                controls
                Spacer()
                statsBar
            }
            .padding()

            if camera.permissionDenied {
                overlayMessage("Camera access denied.\nGrant it in System Settings ▸ Privacy & Security ▸ Camera, then relaunch.")
            } else if let status = camera.statusMessage {
                overlayMessage(status)
            }
        }
        .frame(minWidth: 760, minHeight: 500)
        .onAppear { camera.apply(task: task, delegate: delegate, stage: stage); camera.start() }
        .onDisappear { camera.stop() }
        .onChange(of: task) { camera.apply(task: $0, delegate: delegate, stage: stage) }
        .onChange(of: delegate) { camera.apply(task: task, delegate: $0, stage: stage) }
        .onChange(of: stage) { camera.apply(task: task, delegate: delegate, stage: $0) }
        .onChange(of: lowPower) { camera.setLowPower($0) }
    }

    private var controls: some View {
        HStack(spacing: 14) {
            Picker("Task", selection: $task) {
                ForEach(DemoTask.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).frame(maxWidth: 220)

            Picker("Delegate", selection: $delegate) {
                Text("CPU").tag(MediaPipeDelegate.cpu)
                Text("GPU").tag(MediaPipeDelegate.gpu)
            }.pickerStyle(.segmented).frame(maxWidth: 120)

            Picker("Stage", selection: $stage) {
                ForEach(PipelineStage.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).frame(maxWidth: 200)

            Toggle("Mirror", isOn: $camera.mirrored).toggleStyle(.switch)
            Toggle("Low power", isOn: $lowPower).toggleStyle(.switch)
                .help("640×480 instead of 1280×720")
            Spacer()
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var statsBar: some View {
        HStack(spacing: 18) {
            stat("FPS", String(format: "%.0f", camera.fps))
            stat("Inference", String(format: "%.1f ms", camera.inferenceMs))
            stat("Detections", "\(camera.detectionCount)")
            stat("Memory", String(format: "%.0f MB", camera.memoryMB))
            stat("Frames", "\(camera.frameCount)")
            stat("Delegate", delegate == .gpu ? "GPU" : "CPU")
            Spacer()
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.system(.body, design: .monospaced)).bold()
        }
    }

    private func overlayMessage(_ text: String) -> some View {
        Text(text).multilineTextAlignment(.center).padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10)).padding()
    }
}
