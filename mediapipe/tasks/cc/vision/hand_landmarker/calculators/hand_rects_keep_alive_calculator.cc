/* Copyright 2025 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_rects_keep_alive_calculator.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_rects_keep_alive_calculator.pb.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe {
namespace tasks {
namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api3::Calculator;
using ::mediapipe::api3::CalculatorContext;

}  // namespace

class HandRectsKeepAliveNodeImpl
    : public Calculator<HandRectsKeepAliveNode, HandRectsKeepAliveNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<HandRectsKeepAliveNode>& cc) override {
    options_ = cc.options.Get();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<HandRectsKeepAliveNode>& cc) override {
    std::vector<NormalizedRect> incoming;
    if (cc.rects_in) {
      incoming = cc.rects_in.GetOrDie();
    }
    const float threshold = options_.min_similarity_threshold();

    // Age currently-held rects: a held rect overlapped by an incoming rect has
    // been recovered (or is covered by another hand) and is released; the rest
    // stay held until they exceed the miss budget.
    std::vector<HeldRect> next_held;
    for (HeldRect& held : held_) {
      MP_ASSIGN_OR_RETURN(
          bool recovered,
          mediapipe::DoesRectOverlap(held.rect, incoming, threshold));
      if (!recovered) {
        held.misses += 1;
        if (held.misses <= options_.max_miss_frames()) {
          next_held.push_back(std::move(held));
        }
      }
    }
    // Rects live on the previous frame that vanished this frame start holding
    // (miss 1), unless an existing held rect already covers that region.
    std::vector<NormalizedRect> held_rects;
    held_rects.reserve(next_held.size());
    for (const HeldRect& held : next_held) {
      held_rects.push_back(held.rect);
    }
    for (NormalizedRect& rect : prev_live_) {
      MP_ASSIGN_OR_RETURN(
          bool still_present,
          mediapipe::DoesRectOverlap(rect, incoming, threshold));
      if (still_present) continue;
      MP_ASSIGN_OR_RETURN(
          bool already_held,
          mediapipe::DoesRectOverlap(rect, held_rects, threshold));
      if (already_held) continue;
      if (options_.max_miss_frames() >= 1) {
        held_rects.push_back(rect);
        next_held.push_back({std::move(rect), /*misses=*/1});
      }
    }
    held_ = std::move(next_held);
    prev_live_ = incoming;

    // Output: real rects first (they win downstream clipping), held appended.
    auto output = std::make_unique<std::vector<NormalizedRect>>(
        std::move(incoming));
    for (const HeldRect& held : held_) {
      output->push_back(held.rect);
    }
    cc.rects_out.Send(std::move(output));
    return absl::OkStatus();
  }

 private:
  struct HeldRect {
    NormalizedRect rect;
    int misses = 0;
  };

  HandRectsKeepAliveCalculatorOptions options_;
  // Rects emitted as live (actually detected) on the previous frame.
  std::vector<NormalizedRect> prev_live_;
  // Rects currently being held, with their consecutive-miss counts.
  std::vector<HeldRect> held_;
};

}  // namespace tasks
}  // namespace mediapipe
