// swift-tools-version:5.9
// End-to-end A/B: MediaPipeTasksMac (TFLite/Metal) vs a faithful Core ML/ANE
// reimplementation of the hand landmarker pipeline. See ../README.md.
import PackageDescription

let package = Package(
    name: "pipeline-bench",
    platforms: [.macOS("14.4")],
    dependencies: [
        // The mediapipe repo root package (MediaPipeTasksMac baseline).
        .package(path: "../../../../../..")
    ],
    targets: [
        .executableTarget(
            name: "pipeline-bench",
            dependencies: [.product(name: "MediaPipeTasksMac", package: "mediapipe")],
            path: "Sources"
        )
    ]
)
