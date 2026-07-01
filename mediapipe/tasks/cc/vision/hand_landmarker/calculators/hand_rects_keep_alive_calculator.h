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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_RECTS_KEEP_ALIVE_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_RECTS_KEEP_ALIVE_CALCULATOR_H_

#include <vector>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_rects_keep_alive_calculator.pb.h"

namespace mediapipe {
namespace tasks {

// Stateful keep-alive for the hand tracking loop ("frames without a hand").
//
// Sits on the HandLandmarkerGraph back edge, between the per-frame
// HAND_RECT_NEXT_FRAME vector and the PreviousLoopbackCalculator LOOP input.
// The upstream presence gate drops a hand's next-frame ROI the instant its
// presence score dips below min_hand_presence_confidence (e.g. one motion-blur
// frame). Without intervention that (a) shrinks the tracked-rect vector, so
// the palm detector re-runs, and (b) discards the last-known ROI, so recovery
// must wait for a fresh palm detection.
//
// This calculator remembers rects that were present on the previous frame and,
// when one goes missing, keeps re-injecting it into the loop for up to
// max_miss_frames frames. While held: the tracked-rect count stays up (palm
// detector stays skipped) and the landmark model keeps retrying the last-known
// region, so a momentarily-blurred hand is re-acquired by the cheap landmark
// model instead of a full re-detection. A held rect is released as soon as an
// incoming rect overlaps it (recovered), or dropped once it has been missing
// for more than max_miss_frames frames.
//
// Multi-hand safe: holding is per-rect, matched to incoming rects by IoU, so
// one flickering hand does not disturb the other tracks.
//
// Input:
//  No tag - Vector of NormalizedRect (this frame's tracked hand rects).
//
// Output:
//  No tag - Vector of NormalizedRect (input rects first, held rects appended).
struct HandRectsKeepAliveNode : public api3::Node<"HandRectsKeepAliveCalculator"> {
  template <typename S>
  struct Contract {
    // This frame's hand rects (after presence gating and deduplication).
    api3::Input<S, std::vector<NormalizedRect>> rects_in{""};

    // Same rects with held (recently vanished) rects appended.
    api3::Output<S, std::vector<NormalizedRect>> rects_out{""};

    api3::Options<S, mediapipe::HandRectsKeepAliveCalculatorOptions> options;
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_RECTS_KEEP_ALIVE_CALCULATOR_H_
