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

#include <cstring>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/tensor_capture.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::vision::hand_landmarker::TensorCapture;

constexpr char kNode[] = R"pb(
  calculator: "TensorCaptureCalculator"
  input_stream: "TENSORS:tensors"
  input_stream: "ENABLE:enable"
  input_stream: "TICK:tick"
  output_stream: "CAPTURE:capture"
)pb";

void AddTensor(CalculatorRunner* runner, int64_t ts) {
  auto tensors = std::make_unique<std::vector<Tensor>>();
  tensors->emplace_back(Tensor::ElementType::kFloat32,
                        Tensor::Shape{1, 2, 2, 3});
  {
    auto view = tensors->back().GetCpuWriteView();
    float* values = view.buffer<float>();
    for (int i = 0; i < 12; ++i) values[i] = ts + i * 0.5f;
  }
  runner->MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(tensors.release()).At(Timestamp(ts)));
}

void AddFlag(CalculatorRunner* runner, int64_t ts, bool enable) {
  runner->MutableInputs()->Tag("ENABLE").packets.push_back(
      MakePacket<bool>(enable).At(Timestamp(ts)));
  runner->MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<int>(0).At(Timestamp(ts)));
}

TEST(TensorCaptureCalculatorTest, CapturesOnlyEnabledTimestamps) {
  CalculatorRunner runner(kNode);
  // ts 1: tensor, not enabled. ts 2: tensor, enabled. ts 3: no tensor,
  // enabled. ts 4: no tensor, not enabled.
  AddTensor(&runner, 1);
  AddFlag(&runner, 1, false);
  AddTensor(&runner, 2);
  AddFlag(&runner, 2, true);
  AddFlag(&runner, 3, true);
  AddFlag(&runner, 4, false);
  MP_ASSERT_OK(runner.Run());

  const auto& packets = runner.Outputs().Tag("CAPTURE").packets;
  ASSERT_EQ(packets.size(), 2);

  EXPECT_EQ(packets[0].Timestamp(), Timestamp(2));
  const auto& captured = packets[0].Get<TensorCapture>();
  EXPECT_TRUE(captured.has_tensor);
  EXPECT_EQ(captured.element_type, Tensor::ElementType::kFloat32);
  EXPECT_THAT(captured.shape, testing::ElementsAre(1, 2, 2, 3));
  ASSERT_NE(captured.data, nullptr);
  ASSERT_EQ(captured.data->size(), 12 * sizeof(float));
  std::vector<float> values(12);
  std::memcpy(values.data(), captured.data->data(), captured.data->size());
  for (int i = 0; i < 12; ++i) EXPECT_FLOAT_EQ(values[i], 2 + i * 0.5f);

  EXPECT_EQ(packets[1].Timestamp(), Timestamp(3));
  const auto& skipped = packets[1].Get<TensorCapture>();
  EXPECT_FALSE(skipped.has_tensor);
  EXPECT_EQ(skipped.data, nullptr);
  EXPECT_TRUE(skipped.shape.empty());
}

// A tensor that was never written cannot be read: requesting a read view of it
// is a fatal error. Passing one through with capture not requested therefore
// proves that the calculator requests no view of the tensor, and so triggers
// no GPU-to-CPU readback, on such a frame.
TEST(TensorCaptureCalculatorTest, DoesNotReadTheTensorWhenNotEnabled) {
  CalculatorRunner runner(kNode);
  for (int64_t ts : {1, 2}) {
    auto tensors = std::make_unique<std::vector<Tensor>>();
    tensors->emplace_back(Tensor::ElementType::kFloat32,
                          Tensor::Shape{1, 2, 2, 3});
    runner.MutableInputs()->Tag("TENSORS").packets.push_back(
        Adopt(tensors.release()).At(Timestamp(ts)));
  }
  // ts 1: capture explicitly off. ts 2: no request packet at all.
  AddFlag(&runner, 1, false);
  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<int>(0).At(Timestamp(2)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("CAPTURE").packets.empty());
}

TEST(TensorCaptureCalculatorDeathTest, ReadingAnUnwrittenTensorIsFatal) {
  // The premise of the test above.
  EXPECT_DEATH(
      {
        CalculatorRunner runner(kNode);
        auto tensors = std::make_unique<std::vector<Tensor>>();
        tensors->emplace_back(Tensor::ElementType::kFloat32,
                              Tensor::Shape{1, 2, 2, 3});
        runner.MutableInputs()->Tag("TENSORS").packets.push_back(
            Adopt(tensors.release()).At(Timestamp(1)));
        AddFlag(&runner, 1, true);
        runner.Run().IgnoreError();
      },
      "Tensor must be written prior to read from");
}

}  // namespace
}  // namespace mediapipe
