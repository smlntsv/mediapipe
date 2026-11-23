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

#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionKind;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegions;

CalculatorGraphConfig::Node Node(const std::string& options) {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "PalmSearchRegionsCalculator"
        input_stream: "IMAGE_SIZE:image_size"
        input_stream: "NORM_RECT:norm_rect"
        input_stream: "EXTERNAL_REGIONS:external"
        output_stream: "RECTS:rects"
        output_stream: "REGIONS:regions"
        options {
          [mediapipe.tasks.vision.hand_detector.proto.PalmSearchOptions.ext] {
            )pb" +
      options + "}}");
}

void AddFrame(CalculatorRunner* runner, int64_t ts, float rotation = 0,
              std::vector<NormalizedRect> external = {}) {
  runner->MutableInputs()->Tag("IMAGE_SIZE").packets.push_back(
      MakePacket<std::pair<int, int>>(1280, 720).At(Timestamp(ts)));
  NormalizedRect rect;
  rect.set_x_center(0.5f);
  rect.set_y_center(0.5f);
  rect.set_width(1.0f);
  rect.set_height(1.0f);
  rect.set_rotation(rotation);
  runner->MutableInputs()->Tag("NORM_RECT").packets.push_back(
      MakePacket<NormalizedRect>(rect).At(Timestamp(ts)));
  runner->MutableInputs()->Tag("EXTERNAL_REGIONS").packets.push_back(
      MakePacket<std::vector<NormalizedRect>>(std::move(external))
          .At(Timestamp(ts)));
}

std::vector<int> Ids(const PalmSearchRegions& regions) {
  std::vector<int> ids;
  for (const auto& region : regions.regions) ids.push_back(region.id);
  return ids;
}

using ::testing::ElementsAre;

TEST(PalmSearchRegionsCalculatorTest, FullFrameOnlyPassesTheTaskRect) {
  CalculatorRunner runner(Node(""));
  AddFrame(&runner, 1, -1.5707964f);
  MP_ASSERT_OK(runner.Run());
  const auto& rects = runner.Outputs().Tag("RECTS").packets;
  ASSERT_EQ(rects.size(), 1);
  const auto& rect = rects[0].Get<std::vector<NormalizedRect>>();
  ASSERT_EQ(rect.size(), 1);
  EXPECT_FLOAT_EQ(rect[0].width(), 1.0f);
  EXPECT_FLOAT_EQ(rect[0].rotation(), -1.5707964f);
  const auto& regions =
      runner.Outputs().Tag("REGIONS").packets[0].Get<PalmSearchRegions>();
  EXPECT_THAT(Ids(regions), ElementsAre(0));
  EXPECT_EQ(regions.regions[0].kind, PalmSearchRegionKind::kFullFrame);
  EXPECT_FLOAT_EQ(regions.regions[0].sampled_height * 720, 1280.0f);
}

TEST(PalmSearchRegionsCalculatorTest, GridAllTilesWithFullFrame) {
  CalculatorRunner runner(Node("grid_rows: 2 grid_columns: 2"));
  AddFrame(&runner, 1);
  AddFrame(&runner, 2);
  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(packets.size(), 2);
  for (int i = 0; i < 2; ++i) {
    const auto& regions = packets[i].Get<PalmSearchRegions>();
    EXPECT_THAT(Ids(regions), ElementsAre(0, 1, 2, 3, 4));
    EXPECT_EQ(regions.schedule.detector_frame_index, i);
    EXPECT_EQ(regions.schedule.grid_tile_count, 4);
    EXPECT_EQ(regions.schedule.tiles_per_frame, 0);
  }
  const auto& rects =
      runner.Outputs().Tag("RECTS").packets[0].Get<std::vector<NormalizedRect>>();
  ASSERT_EQ(rects.size(), 5);
  EXPECT_FLOAT_EQ(rects[1].x_center(), 0.25f);
  EXPECT_FLOAT_EQ(rects[1].y_center(), 0.25f);
  EXPECT_FLOAT_EQ(rects[1].width(), 0.5f);
  EXPECT_FLOAT_EQ(rects[4].x_center(), 0.75f);
  EXPECT_FLOAT_EQ(rects[4].y_center(), 0.75f);
}

// The cursor advances once per timestamp the calculator runs at.
TEST(PalmSearchRegionsCalculatorTest, RoundRobinWithoutFullFrame) {
  CalculatorRunner runner(Node("include_full_frame: false grid_rows: 1 "
                               "grid_columns: 3 round_robin_tiles_per_frame: 1"));
  // Timestamps 1, 2, 5, 9: gaps are frames on which the detector is skipped.
  for (int64_t ts : {1, 2, 5, 9}) AddFrame(&runner, ts);
  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(packets.size(), 4);
  const int expected[] = {1, 2, 3, 1};
  for (int i = 0; i < 4; ++i) {
    const auto& regions = packets[i].Get<PalmSearchRegions>();
    EXPECT_THAT(Ids(regions), ElementsAre(expected[i]));
    EXPECT_EQ(regions.schedule.cursor_before, i % 3);
    EXPECT_EQ(regions.schedule.cursor_after, (i + 1) % 3);
    EXPECT_EQ(regions.schedule.detector_frame_index, i);
  }
}

TEST(PalmSearchRegionsCalculatorTest, ExternalRegionsWithRotation) {
  CalculatorRunner runner(Node(""));
  NormalizedRect a;
  a.set_x_center(0.2f);
  a.set_y_center(0.3f);
  a.set_width(0.25f);
  a.set_height(0.5f);
  NormalizedRect empty;  // skipped, but keeps its index
  NormalizedRect b = a;
  b.set_x_center(0.8f);
  AddFrame(&runner, 1, -1.5707964f, {a, empty, b});
  MP_ASSERT_OK(runner.Run());
  const auto& regions =
      runner.Outputs().Tag("REGIONS").packets[0].Get<PalmSearchRegions>();
  EXPECT_THAT(Ids(regions), ElementsAre(0, 1000, 1002));
  EXPECT_EQ(regions.regions[1].kind, PalmSearchRegionKind::kExternal);
  const auto& rects =
      runner.Outputs().Tag("RECTS").packets[0].Get<std::vector<NormalizedRect>>();
  ASSERT_EQ(rects.size(), 3);
  // Quarter turn: width and height swap, in the other axis's pixels.
  EXPECT_FLOAT_EQ(rects[1].width(), 0.5f * 720 / 1280);
  EXPECT_FLOAT_EQ(rects[1].height(), 0.25f * 1280 / 720);
  EXPECT_FLOAT_EQ(rects[1].rotation(), -1.5707964f);
  EXPECT_FLOAT_EQ(rects[2].x_center(), 0.8f);
}

// Nothing to run: a regions packet, no rects packet.
TEST(PalmSearchRegionsCalculatorTest, NoRegion) {
  CalculatorRunner runner(Node("include_full_frame: false"));
  AddFrame(&runner, 1);
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("RECTS").packets.empty());
  ASSERT_EQ(runner.Outputs().Tag("REGIONS").packets.size(), 1);
  EXPECT_TRUE(runner.Outputs()
                  .Tag("REGIONS")
                  .packets[0]
                  .Get<PalmSearchRegions>()
                  .regions.empty());
}

}  // namespace
}  // namespace mediapipe
