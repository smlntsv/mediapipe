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
import Combine
import CoreMedia
import CoreVideo
import Darwin
import Foundation
import MediaPipeTasksMac
import MediaPipeTasksObjC
import QuartzCore

/// Which task(s) to run.
enum DemoTask: String, CaseIterable, Identifiable {
    case hand = "Hand", pose = "Pose", face = "Face", all = "All"
    var id: String { rawValue }
    var runsHand: Bool { self == .hand || self == .all }
    var runsPose: Bool { self == .pose || self == .all }
    var runsFace: Bool { self == .face || self == .all }
}

/// Performance / debug stage. These exist to make CPU attribution obvious:
///   * Preview — preview layer ONLY. The `AVCaptureVideoDataOutput` is detached,
///     so no per-frame delegate callback, conversion, inference, or overlay runs.
///     CPU here is just the camera + `AVCaptureVideoPreviewLayer` compositing.
///   * Convert — data output + CVPixelBuffer→MPCImage conversion, no inference,
///     no overlay. CPU above Preview is attributable to capture + conversion.
///   * Full    — data output + inference + landmark overlay.
enum PipelineStage: String, CaseIterable, Identifiable {
    case previewOnly = "Preview"
    case conversionOnly = "Convert"
    case full = "Full"
    var id: String { rawValue }
    var needsVideoDataOutput: Bool { self != .previewOnly }
}

/// Drives the AVCaptureSession, runs the MediaPipe landmarkers in VIDEO mode on
/// each frame's CVPixelBuffer, and publishes results + stats to the UI.
final class CameraController: NSObject, ObservableObject,
                             AVCaptureVideoDataOutputSampleBufferDelegate {

    // Per-frame results (drive the overlay). Only published in `.full`.
    @Published var handResult: HandLandmarkerResult?
    @Published var poseResult: PoseLandmarkerResult?
    @Published var faceResult: FaceLandmarkerResult?
    @Published var bufferWidth: Int = 0
    @Published var bufferHeight: Int = 0

    // Stats (published at ~`statsHz`, NOT per frame, to avoid UI churn).
    @Published var fps: Double = 0
    @Published var inferenceMs: Double = 0
    @Published var detectionCount: Int = 0
    @Published var memoryMB: Double = 0
    @Published var frameCount: Int = 0

    @Published var permissionDenied = false
    @Published var statusMessage: String?

    @Published var mirrored = true

    let session = AVCaptureSession()

    private let videoQueue = DispatchQueue(label: "com.mediapipe.webcamdemo.video")
    private let output = AVCaptureVideoDataOutput()
    private var outputAttached = false
    private weak var device: AVCaptureDevice?

    private var hand: HandLandmarker?
    private var pose: PoseLandmarker?
    private var face: FaceLandmarker?

    private var task: DemoTask = .all
    private var delegate: MediaPipeDelegate = .cpu
    private var stage: PipelineStage = .full
    private var lowPower = false

    private var lastTimestampMs = 0
    private var lastFrameHostTime: CFTimeInterval = 0
    private var processed = 0

    // Coalesce per-frame results publishing (overlay) so a slow main thread never
    // queues unbounded work.
    private let publishLock = NSLock()
    private var resultsPublishScheduled = false

    // Throttle stats publishing to a few Hz (the overlay needs frame-rate updates,
    // but FPS/RSS/inference/frame-count text does not).
    private let statsHz: CFTimeInterval = 4
    private var lastStatsPublish: CFTimeInterval = 0
    private var statsPublishScheduled = false

    // Diagnostic stderr logging is OFF by default; enable with WEBCAM_DEBUG_LOG=1.
    private let debugLog = ProcessInfo.processInfo.environment["WEBCAM_DEBUG_LOG"] == "1"
    private let logEvery = 60
    private let targetFps = 30.0

    // MARK: - Lifecycle

    override init() {
        super.init()
        // Optional launch-time overrides, handy for benchmarking a single mode
        // without touching the UI (e.g. WEBCAM_STAGE=Preview WEBCAM_LOWPOWER=1).
        let env = ProcessInfo.processInfo.environment
        if let s = env["WEBCAM_STAGE"],
           let parsed = PipelineStage.allCases.first(where: { $0.rawValue.lowercased() == s.lowercased() }) {
            stage = parsed
        }
        if env["WEBCAM_DELEGATE"]?.uppercased() == "CPU" { delegate = .cpu }
        if env["WEBCAM_DELEGATE"]?.uppercased() == "GPU" { delegate = .gpu }
        if env["WEBCAM_LOWPOWER"] == "1" { lowPower = true }
    }

    /// The stage the UI should start in (reflects any WEBCAM_STAGE override).
    var initialStage: PipelineStage { stage }
    var initialDelegate: MediaPipeDelegate { delegate }
    var initialLowPower: Bool { lowPower }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: configureAndRun()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                if granted { self.configureAndRun() }
                else { DispatchQueue.main.async { self.permissionDenied = true } }
            }
        default:
            DispatchQueue.main.async { self.permissionDenied = true }
        }
    }

    func stop() { videoQueue.async { [weak self] in self?.session.stopRunning() } }

    /// Reconfigure task/delegate/stage (from the UI). Rebuilds landmarkers and
    /// attaches/detaches the data output on the video queue so it never races
    /// with frame processing.
    func apply(task: DemoTask, delegate: MediaPipeDelegate, stage: PipelineStage) {
        videoQueue.async { [weak self] in
            guard let self else { return }
            let stageChanged = stage != self.stage
            let rebuild = (task != self.task) || (delegate != self.delegate) || stageChanged
            self.task = task; self.delegate = delegate; self.stage = stage
            if stageChanged { self.syncOutputAttachment() }
            if rebuild { self.rebuildLandmarkers() }
        }
    }

    func setLowPower(_ on: Bool) {
        videoQueue.async { [weak self] in
            guard let self, on != self.lowPower else { return }
            self.lowPower = on
            self.applyCaptureFormat()
        }
    }

    // MARK: - Diagnostics

    /// Resident memory (phys_footprint) in MB — what macOS uses for jetsam.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576.0 : 0
    }

    // MARK: - Session setup

    private func configureAndRun() {
        videoQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            guard let device = AVCaptureDevice.default(for: .video),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async { self.statusMessage = "No camera available." }
                return
            }
            self.device = device
            self.session.addInput(input)
            // Fixed, practical capture format (see applyCaptureFormat): 1280x720,
            // or 640x480 in low-power mode, capped at 30 fps. Avoids needlessly
            // high webcam resolutions that inflate baseline CPU.
            self.session.sessionPreset = self.lowPower ? .vga640x480 : .hd1280x720
            self.output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            // Never queue frames: drop while the previous one is still processing.
            self.output.alwaysDiscardsLateVideoFrames = true
            self.output.setSampleBufferDelegate(self, queue: self.videoQueue)
            // Attach the data output only if the current stage needs it.
            self.syncOutputAttachment()
            self.session.commitConfiguration()
            self.applyCaptureFormat()
            self.rebuildLandmarkers()
            self.session.startRunning()
            DispatchQueue.main.async { self.statusMessage = nil }
        }
    }

    /// Adds/removes the `AVCaptureVideoDataOutput` to match the current stage.
    /// Preview uses only the preview layer, so detaching the output stops all
    /// per-frame delegate work — the key to a low-CPU Preview.
    private func syncOutputAttachment() {
        let want = stage.needsVideoDataOutput
        guard want != outputAttached else { return }
        session.beginConfiguration()
        if want {
            if session.canAddOutput(output) { session.addOutput(output); outputAttached = true }
        } else {
            session.removeOutput(output); outputAttached = false
        }
        session.commitConfiguration()
    }

    /// Sets the capture resolution (via preset) and caps the frame rate.
    private func applyCaptureFormat() {
        let preset: AVCaptureSession.Preset = lowPower ? .vga640x480 : .hd1280x720
        session.beginConfiguration()
        if session.canSetSessionPreset(preset) { session.sessionPreset = preset }
        session.commitConfiguration()
        if let device, (try? device.lockForConfiguration()) != nil {
            // Cap the frame rate to targetFps (a future UI could offer 15/30/60).
            let duration = CMTime(value: 1, timescale: Int32(targetFps))
            if device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
                $0.minFrameRate <= targetFps && targetFps <= $0.maxFrameRate }) {
                device.activeVideoMinFrameDuration = duration
                device.activeVideoMaxFrameDuration = duration
            }
            device.unlockForConfiguration()
        }
    }

    private func modelURL(_ name: String) -> String? {
        if let dir = ProcessInfo.processInfo.environment["WEBCAM_MODELS_DIR"] {
            let p = (dir as NSString).appendingPathComponent("\(name).task")
            if FileManager.default.fileExists(atPath: p) { return p }
        }
        return Bundle.main.url(forResource: name, withExtension: "task")?.path
    }

    private func rebuildLandmarkers() {
        hand = nil; pose = nil; face = nil
        lastTimestampMs = 0
        guard stage == .full else { return }  // no models needed for preview/convert
        var missing: [String] = []
        func make<T>(_ name: String, _ build: (String) throws -> T) -> T? {
            guard let path = modelURL(name) else { missing.append("\(name).task"); return nil }
            do { return try build(path) }
            catch { DispatchQueue.main.async { self.statusMessage = "Failed to load \(name): \(error.localizedDescription)" }; return nil }
        }
        if task.runsHand {
            hand = make("hand_landmarker") { p in
                let o = HandLandmarkerOptions(); o.modelPath = p; o.numHands = 2
                o.delegate = delegate; o.runningMode = .video; return try HandLandmarker(options: o)
            }
        }
        if task.runsPose {
            pose = make("pose_landmarker_full") { p in
                let o = PoseLandmarkerOptions(); o.modelPath = p; o.numPoses = 1
                o.delegate = delegate; o.runningMode = .video; return try PoseLandmarker(options: o)
            }
        }
        if task.runsFace {
            face = make("face_landmarker") { p in
                let o = FaceLandmarkerOptions(); o.modelPath = p; o.numFaces = 1
                o.delegate = delegate; o.runningMode = .video; return try FaceLandmarker(options: o)
            }
        }
        if !missing.isEmpty {
            DispatchQueue.main.async {
                self.statusMessage = "Missing models: \(missing.joined(separator: ", ")). Run download_models.sh + rebuild the app."
            }
        }
    }

    // MARK: - Frame processing (videoQueue)

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Drain every per-frame allocation (notably the bridged RGBA NSData and
        // the result objects) so nothing accumulates on this never-idle queue.
        autoreleasepool {
            processFrame(sampleBuffer)
        }
    }

    private func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        var ts = Int((CMTimeGetSeconds(pts) * 1000).rounded())
        if ts <= lastTimestampMs { ts = lastTimestampMs + 1 }
        lastTimestampMs = ts

        var newHand: HandLandmarkerResult?
        var newPose: PoseLandmarkerResult?
        var newFace: FaceLandmarkerResult?
        var inferMs = 0.0

        switch stage {
        case .previewOnly:
            // The output is detached in Preview, so this is normally unreachable;
            // guard anyway in case a stray buffered frame arrives mid-reconfigure.
            return
        case .conversionOnly:
            // Convert to MPCImage and immediately drop it (no inference).
            _ = try? MPCImage(pixelBuffer: pixelBuffer)
        case .full:
            let t0 = CACurrentMediaTime()
            do {
                if let hand { newHand = try hand.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: ts) }
                if let pose { newPose = try pose.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: ts) }
                if let face { newFace = try face.detectForVideo(pixelBuffer: pixelBuffer, timestampInMilliseconds: ts) }
            } catch {
                DispatchQueue.main.async { self.statusMessage = "Detection error: \(error.localizedDescription)" }
            }
            inferMs = (CACurrentMediaTime() - t0) * 1000
        }

        let now = CACurrentMediaTime()
        let instFps = lastFrameHostTime > 0 ? 1.0 / max(now - lastFrameHostTime, 1e-3) : 0
        lastFrameHostTime = now
        processed += 1

        let count = (newHand?.landmarks.count ?? 0)
            + (newPose?.landmarks.count ?? 0)
            + (newFace?.faceLandmarks.count ?? 0)

        if debugLog && processed % logEvery == 0 {
            let mem = Self.footprintMB()
            FileHandle.standardError.write(Data(String(
                format: "[mem] frame %d  task=%@ delegate=%@ stage=%@  %.1f fps  %.1f ms  det=%d  rss=%.1f MB  MPCImage.live=%ld\n",
                processed, task.rawValue, delegate.rawValue, stage.rawValue,
                instFps, inferMs, count, mem, MPCImage.liveInstanceCount()).utf8))
        }

        // Per-frame results publish (overlay) — only in Full, where results exist.
        // Coalesced: at most one pending main-thread update.
        if stage == .full {
            publishLock.lock()
            let already = resultsPublishScheduled
            resultsPublishScheduled = true
            publishLock.unlock()
            if !already {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.publishLock.lock(); self.resultsPublishScheduled = false; self.publishLock.unlock()
                    self.bufferWidth = w; self.bufferHeight = h
                    self.handResult = newHand; self.poseResult = newPose; self.faceResult = newFace
                }
            }
        }

        // Stats publish — throttled to `statsHz`, decoupled from the overlay.
        let statsDue = (now - lastStatsPublish) >= (1.0 / statsHz)
        guard statsDue else { return }
        lastStatsPublish = now
        publishLock.lock()
        let statsAlready = statsPublishScheduled
        statsPublishScheduled = true
        publishLock.unlock()
        if statsAlready { return }

        let mem = Self.footprintMB()
        let frames = processed
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.publishLock.lock(); self.statsPublishScheduled = false; self.publishLock.unlock()
            self.inferenceMs = self.inferenceMs * 0.7 + inferMs * 0.3
            self.fps = self.fps * 0.7 + instFps * 0.3
            self.detectionCount = count
            self.frameCount = frames
            self.memoryMB = mem
        }
    }
}
