// swift-tools-version:5.9
// ANE-vs-GPU-vs-CPU benchmark for the Core ML-converted MediaPipe hand
// landmark model. See ../README.md.
import PackageDescription

let package = Package(
    name: "ane-bench",
    platforms: [.macOS("14.4")],
    targets: [
        .executableTarget(name: "ane-bench", path: "Sources")
    ]
)
