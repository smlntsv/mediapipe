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


#include <chrono>
#include <cstdint>
#include <vector>

#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

// Fork addition, debugging aid. Emits the monotonic clock, in nanoseconds,
// at the moment a packet of its input reaches it: when the calculator that
// produced the packet has finished and this one is scheduled.
//
// Inputs:
//   (no tag) - any
// Outputs:
//   (no tag) - int64_t
//     One packet per input packet, at the same timestamp.
class MonotonicClockCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).Set<int64_t>();
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Index(0).IsEmpty()) return absl::OkStatus();
    const int64_t now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    cc->Outputs().Index(0).AddPacket(
        MakePacket<int64_t>(now).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(MonotonicClockCalculator);

// Collects the clock stamps of a loop, one per iteration.
typedef EndLoopCalculator<std::vector<int64_t>> EndLoopInt64Calculator;
REGISTER_CALCULATOR(EndLoopInt64Calculator);

}  // namespace mediapipe
