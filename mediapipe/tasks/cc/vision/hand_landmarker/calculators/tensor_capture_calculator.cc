/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/tensor_capture.h"

namespace mediapipe {

namespace {
using ::mediapipe::tasks::vision::hand_landmarker::TensorCapture;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kEnableTag[] = "ENABLE";
constexpr char kTickTag[] = "TICK";
constexpr char kCaptureTag[] = "CAPTURE";
}  // namespace

// Debugging aid: copies the input tensor of a model to the CPU on the
// timestamps where the caller asked for it.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The packet that is also connected to the inference calculator. It has no
//     packet at a timestamp where inference is skipped.
//   ENABLE - bool
//     Whether to capture at this timestamp. Absent or false: the calculator
//     does not touch the tensor (no view is requested, so a GPU tensor is not
//     read back) and emits nothing.
//   TICK - any @Optional
//     A stream that has a packet at every timestamp the model could run at.
//     Only makes the calculator run at timestamps where TENSORS is absent.
//
// Outputs:
//   CAPTURE - TensorCapture
//     Emitted exactly when ENABLE is true. `has_tensor` is false when TENSORS
//     had no packet at this timestamp; otherwise the capture holds a copy of
//     the first tensor.
class TensorCaptureCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
    cc->Inputs().Tag(kEnableTag).Set<bool>();
    if (cc->Inputs().HasTag(kTickTag)) {
      cc->Inputs().Tag(kTickTag).SetAny();
    }
    cc->Outputs().Tag(kCaptureTag).Set<TensorCapture>();
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& enable = cc->Inputs().Tag(kEnableTag);
    if (enable.IsEmpty() || !enable.Get<bool>()) {
      return absl::OkStatus();
    }
    auto capture = std::make_unique<TensorCapture>();
    const auto& tensors_in = cc->Inputs().Tag(kTensorsTag);
    if (!tensors_in.IsEmpty()) {
      const auto& tensors = tensors_in.Get<std::vector<Tensor>>();
      if (!tensors.empty()) {
        const Tensor& tensor = tensors[0];
        capture->has_tensor = true;
        capture->element_type = tensor.element_type();
        capture->shape = tensor.shape().dims;
        auto data = std::make_shared<std::vector<uint8_t>>(tensor.bytes());
        // Reads a GPU tensor back to the CPU.
        auto view = tensor.GetCpuReadView();
        std::memcpy(data->data(), view.buffer<uint8_t>(), data->size());
        capture->data = std::move(data);
      }
    }
    cc->Outputs().Tag(kCaptureTag).Add(capture.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TensorCaptureCalculator);

// Collects the per-ROI captures of the landmark model loop.
typedef EndLoopCalculator<std::vector<TensorCapture>>
    EndLoopTensorCaptureCalculator;
REGISTER_CALCULATOR(EndLoopTensorCaptureCalculator);

}  // namespace mediapipe
