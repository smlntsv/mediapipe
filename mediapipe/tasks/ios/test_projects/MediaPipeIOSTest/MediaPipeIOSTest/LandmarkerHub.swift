//
//  LandmarkerHub.swift
//  MediaPipeIOSTest
//
//  Runs hand / face / pose landmarkers over camera frames, one serial "lane"
//  per task (single-in-flight: a busy lane skips frames instead of queueing).
//  Mirrors the architecture of the macOS ScreenBar pipeline so results are
//  comparable: video running mode, monotonic timestamps, per-detect timing.
//

import Combine
import CoreVideo
import Foundation
import MediaPipeTasks
import QuartzCore

enum LandmarkTaskKind: String, CaseIterable, Identifiable, Sendable {
    case hand = "Hand"
    case face = "Face"
    case pose = "Pose"
    var id: String { rawValue }
}

enum DelegateChoice: String, CaseIterable, Identifiable, Sendable {
    case cpu = "CPU"
    case gpu = "GPU"
    case coreML = "Core ML"
    var id: String { rawValue }

    /// Candidate accelerators in fallback order (mirrors ScreenBar).
    var candidates: [MediaPipeDelegate] {
        switch self {
        case .cpu: return [.cpu]
        case .gpu: return [.gpu, .cpu]
        case .coreML: return [.coreML, .gpu, .cpu]
        }
    }
}

extension MediaPipeDelegate {
    var displayName: String {
        switch self {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .coreML: return "Core ML"
        @unknown default: return rawValue
        }
    }
}

/// Standard MediaPipe hand-skeleton edges (21 landmarks).
let handSkeletonEdges: [(Int, Int)] = [
    (0, 1), (1, 2), (2, 3), (3, 4),           // thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           // index
    (5, 9), (9, 10), (10, 11), (11, 12),      // middle
    (9, 13), (13, 14), (14, 15), (15, 16),    // ring
    (13, 17), (17, 18), (18, 19), (19, 20),   // pinky
    (0, 17),                                  // palm base
]

/// Standard MediaPipe (BlazePose) skeleton edges (33 landmarks).
let poseSkeletonEdges: [(Int, Int)] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  // face
    (11, 12), (11, 23), (12, 24), (23, 24),                                  // torso
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),              // left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),              // right arm
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),                        // left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),                        // right leg
]

/// Fixed-capacity ring buffer of inference timings, safe to append from a
/// lane queue and snapshot from the main thread.
nonisolated final class TimingSeries: @unchecked Sendable {
    static let capacity = 240

    private var values = [Float](repeating: 0, count: TimingSeries.capacity)
    private var writeIndex = 0
    private var count = 0
    private let lock = NSLock()

    func append(_ ms: Double) {
        lock.lock()
        values[writeIndex] = Float(ms)
        writeIndex = (writeIndex + 1) % Self.capacity
        count = min(count + 1, Self.capacity)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        writeIndex = 0
        count = 0
        lock.unlock()
    }

    /// Samples oldest → newest.
    func snapshot() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        guard count > 0 else { return [] }
        let start = (writeIndex - count + Self.capacity) % Self.capacity
        return (0..<count).map { values[(start + $0) % Self.capacity] }
    }
}

/// One landmark task's serial worker. Lives outside the MainActor: frames
/// arrive on the capture queue and inference runs on the lane's own queue.
nonisolated final class DetectorLane: @unchecked Sendable {
    let kind: LandmarkTaskKind
    let series = TimingSeries()

    /// Set on the lane queue; nil while (re)building or after a failure.
    /// Takes the camera pixel buffer + monotonic timestamp, returns normalized
    /// landmark sets.
    private var detect: ((CVPixelBuffer, Int) throws -> [[CGPoint]])?
    private let queue: DispatchQueue
    private let lock = NSLock()
    private var busy = false
    private var enabled = true
    private var lastTimestampMs = 0
    /// Called on the lane queue after each successful detection.
    var onResult: (@Sendable (LandmarkTaskKind, [[CGPoint]], Double) -> Void)?
    /// Called on the lane queue when a rebuild finishes (resolved name or error).
    var onStatus: (@Sendable (LandmarkTaskKind, String) -> Void)?

    init(kind: LandmarkTaskKind) {
        self.kind = kind
        queue = DispatchQueue(
            label: "com.dima.MediaPipeIOSTest.lane.\(kind.rawValue)", qos: .userInteractive
        )
    }

    func setEnabled(_ isEnabled: Bool) {
        lock.lock()
        enabled = isEnabled
        lock.unlock()
        if !isEnabled {
            series.clear()
            onResult?(kind, [], 0)
        }
    }

    /// Rebuilds the landmarker on the lane queue, walking the fallback chain.
    func rebuild(choice: DelegateChoice) {
        queue.async { [self] in
            detect = nil
            series.clear()
            for candidate in choice.candidates {
                do {
                    detect = try Self.makeDetector(kind: kind, delegate: candidate)
                    lastTimestampMs = 0
                    onStatus?(kind, candidate.displayName)
                    return
                } catch {
                    continue
                }
            }
            onStatus?(kind, "failed")
        }
    }

    /// Called on the capture queue. Skips the frame when disabled or busy.
    /// (A lane whose landmarker is still building admits the frame and drops
    /// it on the lane queue — one wasted hop, no shared state to race on.)
    func offer(_ pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        lock.lock()
        let admitted = enabled && !busy
        if admitted { busy = true }
        lock.unlock()
        guard admitted else { return }

        queue.async { [self] in
            defer {
                lock.lock()
                busy = false
                lock.unlock()
            }
            guard let detect else { return }
            var ts = timestampMs
            if ts <= lastTimestampMs { ts = lastTimestampMs + 1 }
            lastTimestampMs = ts
            do {
                let started = CACurrentMediaTime()
                let points = try detect(pixelBuffer, ts)
                let elapsedMs = (CACurrentMediaTime() - started) * 1000
                series.append(elapsedMs)
                onResult?(kind, points, elapsedMs)
            } catch {
                // Timestamp or detection error: drop the frame.
            }
        }
    }

    // MARK: - Landmarker construction

    private static func modelPath(_ name: String) throws -> String {
        guard let path = Bundle.main.path(forResource: name, ofType: "task") else {
            throw NSError(
                domain: "MediaPipeIOSTest", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "\(name).task missing from app bundle"]
            )
        }
        return path
    }

    private static func makeDetector(
        kind: LandmarkTaskKind, delegate: MediaPipeDelegate
    ) throws -> (CVPixelBuffer, Int) throws -> [[CGPoint]] {
        func toPoints(_ landmarks: [[NormalizedLandmark]]) -> [[CGPoint]] {
            landmarks.map { $0.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) } }
        }
        // coreMLModelCacheDirectory is left nil: the wrapper defaults it to the
        // .task file's directory, and the converted <sha>.mlmodelc models are
        // bundled flat next to the .task files.
        switch kind {
        case .hand:
            let landmarker = try HandLandmarker(options: HandLandmarkerOptions(
                modelPath: try modelPath("hand_landmarker"),
                numHands: 2, delegate: delegate, runningMode: .video))
            return { pb, ts in
                toPoints(try landmarker.detectForVideo(
                    pixelBuffer: pb, timestampInMilliseconds: ts).landmarks)
            }
        case .face:
            let landmarker = try FaceLandmarker(options: FaceLandmarkerOptions(
                modelPath: try modelPath("face_landmarker"),
                numFaces: 1, delegate: delegate, runningMode: .video))
            return { pb, ts in
                toPoints(try landmarker.detectForVideo(
                    pixelBuffer: pb, timestampInMilliseconds: ts).faceLandmarks)
            }
        case .pose:
            let landmarker = try PoseLandmarker(options: PoseLandmarkerOptions(
                modelPath: try modelPath("pose_landmarker"),
                numPoses: 1, delegate: delegate, runningMode: .video))
            return { pb, ts in
                toPoints(try landmarker.detectForVideo(
                    pixelBuffer: pb, timestampInMilliseconds: ts).landmarks)
            }
        }
    }
}

// MARK: - Hub

@MainActor
final class LandmarkerHub: ObservableObject {
    struct LaneState {
        var enabled = true
        /// Accelerator actually in use after the fallback chain ("Core ML",
        /// "GPU", "CPU", "failed", or "…" while building).
        var resolved = "…"
        var lastInferenceMs: Double = 0
        /// Normalized landmark sets from the latest successful detection.
        var points: [[CGPoint]] = []
    }

    @Published private(set) var lanes: [LandmarkTaskKind: LaneState] = [
        .hand: LaneState(), .face: LaneState(), .pose: LaneState(),
    ]
    /// Bumped per result; drives sparkline redraws.
    @Published private(set) var tick = 0
    @Published var delegateChoice: DelegateChoice = .gpu {
        didSet { rebuildAll() }
    }

    let handLane = DetectorLane(kind: .hand)
    let faceLane = DetectorLane(kind: .face)
    let poseLane = DetectorLane(kind: .pose)

    /// Skeleton edges for the overlay (the Swift package doesn't expose the
    /// connection topology, so these are the standard MediaPipe sets).
    let handConnections = handSkeletonEdges
    let poseConnections = poseSkeletonEdges

    private var allLanes: [DetectorLane] { [handLane, faceLane, poseLane] }

    init() {
        for lane in allLanes {
            lane.onResult = { [weak self] kind, points, ms in
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.lanes[kind]?.points = points
                    if ms > 0 { self.lanes[kind]?.lastInferenceMs = ms }
                    self.tick += 1
                }
            }
            lane.onStatus = { [weak self] kind, resolved in
                DispatchQueue.main.async {
                    self?.lanes[kind]?.resolved = resolved
                }
            }
        }
        rebuildAll()
    }

    func series(for kind: LandmarkTaskKind) -> TimingSeries {
        lane(for: kind).series
    }

    func setEnabled(_ enabled: Bool, for kind: LandmarkTaskKind) {
        lanes[kind]?.enabled = enabled
        lane(for: kind).setEnabled(enabled)
    }

    /// Entry point from the capture queue.
    nonisolated func offerFrame(_ pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        handLane.offer(pixelBuffer, timestampMs: timestampMs)
        faceLane.offer(pixelBuffer, timestampMs: timestampMs)
        poseLane.offer(pixelBuffer, timestampMs: timestampMs)
    }

    private func lane(for kind: LandmarkTaskKind) -> DetectorLane {
        switch kind {
        case .hand: return handLane
        case .face: return faceLane
        case .pose: return poseLane
        }
    }

    private func rebuildAll() {
        for kind in LandmarkTaskKind.allCases {
            lanes[kind]?.resolved = "…"
        }
        let choice = delegateChoice
        for lane in allLanes {
            lane.rebuild(choice: choice)
        }
    }
}
