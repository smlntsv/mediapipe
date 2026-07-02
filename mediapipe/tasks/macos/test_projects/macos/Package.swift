// swift-tools-version:5.9
// Minimal native macOS smoke-test app that depends on the local root
// MediaPipeTasksMac package (by relative path) and exports parity JSON.
import PackageDescription

let package = Package(
    name: "ParitySmokeTest",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "ParitySmokeTest", targets: ["ParitySmokeTest"]),
        .executable(name: "CoreMLDelegateBench", targets: ["CoreMLDelegateBench"]),
    ],
    dependencies: [
        // Repo root (where the MediaPipeTasksMac Package.swift lives).
        .package(path: "../../../../..")
    ],
    targets: [
        .executableTarget(
            name: "ParitySmokeTest",
            dependencies: [
                .product(name: "MediaPipeTasksMac", package: "mediapipe")
            ],
            path: "Sources/ParitySmokeTest"
        ),
        .executableTarget(
            name: "CoreMLDelegateBench",
            dependencies: [
                .product(name: "MediaPipeTasksMac", package: "mediapipe")
            ],
            path: "Sources/CoreMLDelegateBench"
        ),
    ]
)
