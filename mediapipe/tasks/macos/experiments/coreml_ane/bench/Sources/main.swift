// ANE benchmark for the Core ML-converted MediaPipe hand landmark model.
//
// For each MLComputeUnits configuration:
//   1. loads the compiled model and reports per-op device placement via
//      MLComputePlan (macOS 14.4+) — the proof of whether layers actually land
//      on the Neural Engine, not just a request for them to;
//   2. runs warmup + timed predictions and reports mean/p50/p90 latency.
//
// Usage: swift run -c release ane-bench <model.mlpackage> [iterations]

import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: ane-bench <model.mlpackage> [iterations]")
    exit(1)
}
let modelURL = URL(fileURLWithPath: args[1])
let iterations = args.count >= 3 ? Int(args[2]) ?? 200 : 200

// Compile once (no-op cost repeated per config otherwise).
let compiledURL = try await MLModel.compileModel(at: modelURL)

let configs: [(String, MLComputeUnits)] = [
    ("cpuOnly", .cpuOnly),
    ("cpuAndGPU", .cpuAndGPU),
    ("cpuAndNeuralEngine", .cpuAndNeuralEngine),
    ("all", .all),
]

func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "?"
    }
}

/// Per-op preferred-device counts from the compute plan.
func placement(for config: MLModelConfiguration) async -> [String: Int] {
    guard let plan = try? await MLComputePlan.load(contentsOf: compiledURL, configuration: config)
    else { return [:] }
    guard case .program(let program) = plan.modelStructure else { return [:] }
    var counts: [String: Int] = [:]
    for (_, function) in program.functions {
        for op in function.block.operations {
            guard let usage = plan.deviceUsage(for: op) else { continue }
            counts[deviceName(usage.preferred), default: 0] += 1
        }
    }
    return counts
}

/// Random input matching the model's (single multi-array) input description.
func makeInput(_ model: MLModel) throws -> MLDictionaryFeatureProvider {
    var features: [String: MLFeatureValue] = [:]
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        guard let constraint = desc.multiArrayConstraint else {
            fatalError("unsupported input type for \(name)")
        }
        let array = try MLMultiArray(shape: constraint.shape, dataType: .float32)
        let count = array.count
        array.withUnsafeMutableBytes { ptr, _ in
            let f = ptr.bindMemory(to: Float.self)
            for i in 0..<count { f[i] = Float.random(in: 0..<1) }
        }
        features[name] = MLFeatureValue(multiArray: array)
    }
    return try MLDictionaryFeatureProvider(dictionary: features)
}

print("model: \(modelURL.lastPathComponent), \(iterations) timed iterations per config\n")
let h1 = "computeUnits".padding(toLength: 20, withPad: " ", startingAt: 0)
let h2 = "op placement (preferred)".padding(toLength: 24, withPad: " ", startingAt: 0)
print(h1 + " | " + h2 + " | latency ms mean/p50/p90")

for (name, units) in configs {
    let config = MLModelConfiguration()
    config.computeUnits = units

    let counts = await placement(for: config)
    let placementStr = counts.isEmpty
        ? "n/a"
        : counts.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: " ")

    do {
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        let input = try makeInput(model)
        for _ in 0..<30 { _ = try model.prediction(from: input) }  // warmup

        var lat: [Double] = []
        lat.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: input)
            lat.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        lat.sort()
        let mean = lat.reduce(0, +) / Double(lat.count)
        let p50 = lat[lat.count / 2]
        let p90 = lat[min(Int(Double(lat.count) * 0.9), lat.count - 1)]
        let pad = name.padding(toLength: 20, withPad: " ", startingAt: 0)
        let padPlace = placementStr.padding(toLength: 24, withPad: " ", startingAt: 0)
        print(pad + " | " + padPlace + " | " + String(format: "%5.2f / %5.2f / %5.2f", mean, p50, p90))
    } catch {
        print("\(name): FAILED — \(error.localizedDescription)")
    }
}
