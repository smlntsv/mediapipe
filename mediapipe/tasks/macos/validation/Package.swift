// swift-tools-version:5.9
// Headless architecture-validation harness for the MediaPipeTasks package.
//
// Builds `mp-arch-validate`, a CLI that runs Hand Landmarker inference on fixed
// images (IMAGE mode) and on video files (sustained VIDEO mode) through the
// PUBLIC Swift API, and writes machine-comparable JSON reports. It needs no
// camera and no Accessibility permission (video frames are decoded from files
// with AVAssetReader), so it runs over SSH.
//
// Consumes the root package by path, so it links whatever
// mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework holds
// (e.g. the universal macos-arm64_x86_64 slice). Build universal with:
//   swift build -c release --arch arm64 --arch x86_64
// See README.md in this directory.
import PackageDescription

let package = Package(
    name: "MPArchValidate",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "mp-arch-validate", targets: ["MPArchValidate"]),
    ],
    dependencies: [
        .package(path: "../../../.."),
    ],
    targets: [
        .executableTarget(
            name: "MPArchValidate",
            dependencies: [.product(name: "MediaPipeTasks", package: "mediapipe")],
            path: "Sources/MPArchValidate"
        ),
    ]
)
