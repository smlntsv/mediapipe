//
//  CameraEngine.swift
//  MediaPipeIOSTest
//
//  Front-camera capture for the landmarker test bench. Delivers upright
//  (portrait-rotated), mirrored BGRA pixel buffers on a capture queue —
//  BGRA because that's the one format MediaPipe's CVPixelBuffer path accepts,
//  mirrored so overlay coordinates match what the user sees.
//

import AVFoundation
import Combine
import CoreVideo
import Foundation

/// Receives frames on the capture queue. Kept outside the MainActor world:
/// AVFoundation calls it on its own queue.
nonisolated final class CameraFrameRelay: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
    @unchecked Sendable {
    /// Called on the capture queue with each frame and its presentation time (ms).
    var onFrame: (@Sendable (CVPixelBuffer, Int) -> Void)?

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ptsMs = Int(CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds * 1000)
        onFrame?(pixelBuffer, ptsMs)
    }
}

@MainActor
final class CameraEngine: NSObject, ObservableObject {
    let session = AVCaptureSession()
    let relay = CameraFrameRelay()

    /// Size of delivered pixel buffers (after rotation), for overlay mapping.
    @Published private(set) var bufferSize = CGSize(width: 720, height: 1280)
    @Published private(set) var authorized = true
    @Published private(set) var usingFrontCamera = true

    private let captureQueue = DispatchQueue(
        label: "com.dima.MediaPipeIOSTest.capture", qos: .userInteractive
    )
    private var configured = false

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndRun()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor [weak self] in
                    self?.authorized = granted
                    if granted { self?.configureAndRun() }
                }
            }
        default:
            authorized = false
        }
    }

    func flipCamera() {
        usingFrontCamera.toggle()
        configured = false
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        session.commitConfiguration()
        configureAndRun()
    }

    private func configureAndRun() {
        guard !configured else { return }
        let position: AVCaptureDevice.Position = usingFrontCamera ? .front : .back
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: position
        ) else { return }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(relay, queue: captureQueue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        // Deliver upright portrait buffers so MediaPipe needs no rotation
        // hint, mirrored (front camera) so overlays match the preview.
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if usingFrontCamera, connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }
        session.commitConfiguration()

        bufferSize = CGSize(width: 720, height: 1280)
        configured = true

        let session = self.session
        captureQueue.async {
            if !session.isRunning { session.startRunning() }
        }
    }
}
