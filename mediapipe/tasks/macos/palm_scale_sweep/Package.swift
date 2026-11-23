// swift-tools-version:5.9
// palm-scale-sweep: reads raw BGRA frames from stdin (produced by ffmpeg, see
// sweep.py), runs the Hand Landmarker on each one through the public Swift API
// and writes one JSON line per frame with the raw debug outputs (palm
// detections, palm candidates, landmarks). No analysis here; sweep.py does it.
//
// Consumes the root package by path. SwiftPM names a path dependency after its
// directory, so the product's package is the worktree folder name.
import PackageDescription

let package = Package(
    name: "PalmScaleSweep",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "palm-scale-sweep", targets: ["PalmScaleSweep"]),
    ],
    dependencies: [
        .package(path: "../../../.."),
    ],
    targets: [
        .executableTarget(
            name: "PalmScaleSweep",
            dependencies: [.product(name: "MediaPipeTasks", package: "mediapipe-hand-stage-outputs")],
            path: "Sources/PalmScaleSweep"
        ),
    ]
)
