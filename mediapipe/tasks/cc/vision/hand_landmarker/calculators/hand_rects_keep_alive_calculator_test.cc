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

#include <initializer_list>
#include <utility>
#include <vector>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;
using ::testing::ElementsAre;
using ::mediapipe::EqualsProto;

NormalizedRect Rect(float cx, float cy) {
  NormalizedRect rect;
  rect.set_x_center(cx);
  rect.set_y_center(cy);
  rect.set_width(0.2);
  rect.set_height(0.2);
  return rect;
}

api3::Packet<std::vector<NormalizedRect>> Frame(
    std::initializer_list<NormalizedRect> rects) {
  return api3::MakePacket<std::vector<NormalizedRect>>(
      std::vector<NormalizedRect>(rects));
}

// Builds the graph function for a single keep-alive node with the given miss
// budget. State persists across runner.Run() calls (each call is the next
// frame on the same live graph).
auto KeepAliveGraphFn(int max_miss_frames) {
  return [max_miss_frames](GenericGraph& graph,
                           Stream<std::vector<NormalizedRect>> rects)
             -> Stream<std::vector<NormalizedRect>> {
    auto& node = graph.AddNode<tasks::HandRectsKeepAliveNode>();
    {
      mediapipe::HandRectsKeepAliveCalculatorOptions& options =
          *node.options.Mutable();
      options.set_max_miss_frames(max_miss_frames);
      options.set_min_similarity_threshold(0.5);
    }
    node.rects_in.Set(rects);
    return node.rects_out.Get();
  };
}

TEST(HandRectsKeepAliveCalculatorTest, HoldsVanishedRectThenDrops) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(KeepAliveGraphFn(/*max_miss_frames=*/2)).Create());
  const NormalizedRect a = Rect(0.3, 0.3);

  // Frame 1: rect present -> passthrough.
  MP_ASSERT_OK_AND_ASSIGN(auto out1, runner.Run(Frame({a})));
  EXPECT_THAT(out1.GetOrDie(), ElementsAre(EqualsProto(a)));

  // Frames 2 and 3: rect vanished -> held (miss 1 and 2).
  MP_ASSERT_OK_AND_ASSIGN(auto out2, runner.Run(Frame({})));
  EXPECT_THAT(out2.GetOrDie(), ElementsAre(EqualsProto(a)));
  MP_ASSERT_OK_AND_ASSIGN(auto out3, runner.Run(Frame({})));
  EXPECT_THAT(out3.GetOrDie(), ElementsAre(EqualsProto(a)));

  // Frame 4: miss budget exceeded -> dropped.
  MP_ASSERT_OK_AND_ASSIGN(auto out4, runner.Run(Frame({})));
  EXPECT_TRUE(out4.GetOrDie().empty());
}

TEST(HandRectsKeepAliveCalculatorTest, ReleasesHeldRectOnRecovery) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(KeepAliveGraphFn(/*max_miss_frames=*/3)).Create());
  const NormalizedRect a = Rect(0.3, 0.3);
  const NormalizedRect a_recovered = Rect(0.32, 0.31);  // overlaps `a`

  MP_ASSERT_OK_AND_ASSIGN(auto out1, runner.Run(Frame({a})));
  MP_ASSERT_OK_AND_ASSIGN(auto out2, runner.Run(Frame({})));
  EXPECT_THAT(out2.GetOrDie(), ElementsAre(EqualsProto(a)));

  // Recovery: incoming rect overlaps the held one -> held copy released, no
  // duplicate.
  MP_ASSERT_OK_AND_ASSIGN(auto out3, runner.Run(Frame({a_recovered})));
  EXPECT_THAT(out3.GetOrDie(), ElementsAre(EqualsProto(a_recovered)));
}

TEST(HandRectsKeepAliveCalculatorTest, HoldsOneHandWhileOtherStaysLive) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(KeepAliveGraphFn(/*max_miss_frames=*/2)).Create());
  const NormalizedRect a = Rect(0.2, 0.2);
  const NormalizedRect b = Rect(0.8, 0.8);
  const NormalizedRect a_recovered = Rect(0.22, 0.2);

  MP_ASSERT_OK_AND_ASSIGN(auto out1, runner.Run(Frame({a, b})));
  EXPECT_THAT(out1.GetOrDie(), ElementsAre(EqualsProto(a), EqualsProto(b)));

  // Hand A vanishes; B stays. Output: live B first, held A appended.
  MP_ASSERT_OK_AND_ASSIGN(auto out2, runner.Run(Frame({b})));
  EXPECT_THAT(out2.GetOrDie(), ElementsAre(EqualsProto(b), EqualsProto(a)));

  // Hand A recovers near its old spot; held copy released.
  MP_ASSERT_OK_AND_ASSIGN(auto out3, runner.Run(Frame({a_recovered, b})));
  EXPECT_THAT(out3.GetOrDie(),
              ElementsAre(EqualsProto(a_recovered), EqualsProto(b)));
}

TEST(HandRectsKeepAliveCalculatorTest, ZeroMissFramesIsPassthrough) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(KeepAliveGraphFn(/*max_miss_frames=*/0)).Create());
  const NormalizedRect a = Rect(0.3, 0.3);

  MP_ASSERT_OK_AND_ASSIGN(auto out1, runner.Run(Frame({a})));
  EXPECT_THAT(out1.GetOrDie(), ElementsAre(EqualsProto(a)));
  MP_ASSERT_OK_AND_ASSIGN(auto out2, runner.Run(Frame({})));
  EXPECT_TRUE(out2.GetOrDie().empty());
}

}  // namespace
}  // namespace mediapipe
