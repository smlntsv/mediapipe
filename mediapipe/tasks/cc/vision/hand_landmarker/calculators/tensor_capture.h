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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_TENSOR_CAPTURE_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_TENSOR_CAPTURE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// A CPU copy of the first input tensor of one model invocation, taken by
// TensorCaptureCalculator from the packet the inference calculator receives.
struct TensorCapture {
  // False when no tensor reached the model at this timestamp, i.e. inference
  // was skipped. The remaining fields are then unset.
  bool has_tensor = false;
  Tensor::ElementType element_type = Tensor::ElementType::kNone;
  std::vector<int> shape;
  // The tensor's bytes, `shape` elements of `element_type` in row-major order.
  // Shared so that copying a capture does not copy the data.
  std::shared_ptr<const std::vector<uint8_t>> data;
};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_TENSOR_CAPTURE_H_
