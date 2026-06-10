// swift-tools-version:5.9
// Native macOS SwiftUI webcam demo that validates the live-video path:
// AVCaptureSession → CVPixelBuffer → MediaPipeTasksMac VIDEO mode → GPU delegate.
import PackageDescription

let package = Package(
    name: "WebcamLandmarksDemo",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "WebcamLandmarksDemo", targets: ["WebcamLandmarksDemo"]),
    ],
    dependencies: [
        // Repo root (where the MediaPipeTasksMac Package.swift lives).
        .package(path: "../../../../..")
    ],
    targets: [
        .executableTarget(
            name: "WebcamLandmarksDemo",
            dependencies: [
                .product(name: "MediaPipeTasksMac", package: "mediapipe")
            ],
            path: "Sources/WebcamLandmarksDemo"
        ),
    ]
)
