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
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchDetectionOutcome;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRecord;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegion;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionKind;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegions;

using Detections = std::vector<Detection>;
using RegionDetections = std::vector<Detections>;

CalculatorGraphConfig::Node Node(const std::string& options) {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "PalmSearchMergeCalculator"
        input_stream: "REGIONS:regions"
        input_stream: "REGION_DETECTIONS:region_detections"
        output_stream: "DETECTIONS:detections"
        output_stream: "RECORD:record"
        options {
          [mediapipe.tasks.vision.hand_detector.proto.PalmSearchOptions.ext] {
            )pb" +
      options + "}}");
}

Detection Box(float score, float xmin, float ymin, float width, float height) {
  Detection detection;
  detection.add_score(score);
  detection.add_label("Palm");
  auto* location = detection.mutable_location_data();
  location->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* box = location->mutable_relative_bounding_box();
  box->set_xmin(xmin);
  box->set_ymin(ymin);
  box->set_width(width);
  box->set_height(height);
  auto* keypoint = location->add_relative_keypoints();
  keypoint->set_x(xmin + width / 2);
  keypoint->set_y(ymin + height / 2);
  return detection;
}

PalmSearchRegion FullFrame() {
  PalmSearchRegion region;
  region.id = 0;
  region.kind = PalmSearchRegionKind::kFullFrame;
  return region;
}

// A grid tile sampled as the square window with this center and size.
PalmSearchRegion Tile(int id, float x_center, float y_center, float width,
                      float height) {
  PalmSearchRegion region;
  region.id = id;
  region.kind = PalmSearchRegionKind::kGrid;
  region.requested_x_center = region.sampled_x_center = x_center;
  region.requested_y_center = region.sampled_y_center = y_center;
  region.requested_width = region.sampled_width = width;
  region.requested_height = region.sampled_height = height;
  return region;
}

struct Outputs {
  Detections detections;
  PalmSearchRecord record;
};

Outputs RunMerge(const std::string& options, std::vector<PalmSearchRegion> regions,
            RegionDetections region_detections) {
  CalculatorRunner runner(Node(options));
  auto regions_packet = std::make_unique<PalmSearchRegions>();
  regions_packet->regions = std::move(regions);
  runner.MutableInputs()->Tag("REGIONS").packets.push_back(
      Adopt(regions_packet.release()).At(Timestamp(1)));
  runner.MutableInputs()->Tag("REGION_DETECTIONS").packets.push_back(
      MakePacket<RegionDetections>(std::move(region_detections))
          .At(Timestamp(1)));
  MP_EXPECT_OK(runner.Run());
  Outputs outputs;
  const auto& detections = runner.Outputs().Tag("DETECTIONS").packets;
  const auto& records = runner.Outputs().Tag("RECORD").packets;
  EXPECT_EQ(detections.size(), 1);
  EXPECT_EQ(records.size(), 1);
  if (!detections.empty()) outputs.detections = detections[0].Get<Detections>();
  if (!records.empty()) outputs.record = records[0].Get<PalmSearchRecord>();
  return outputs;
}

// One region: its detections come out unchanged and in order, even when two
// of them overlap by more than the merge threshold.
TEST(PalmSearchMergeCalculatorTest, SingleRegionIsPassedThrough) {
  const Detections input = {Box(0.9f, 0.10f, 0.10f, 0.20f, 0.20f),
                            Box(0.8f, 0.12f, 0.12f, 0.20f, 0.20f),
                            Box(0.6f, 0.60f, 0.60f, 0.10f, 0.10f)};
  Outputs out = RunMerge("", {FullFrame()}, {input});
  ASSERT_EQ(out.detections.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(out.detections[i].SerializeAsString(),
              input[i].SerializeAsString());
    ASSERT_EQ(out.record.merged[i].sources.size(), 1);
    EXPECT_EQ(out.record.merged[i].sources[0].region_id, 0);
    EXPECT_EQ(out.record.merged[i].sources[0].detection_index, i);
    EXPECT_EQ(out.record.regions[0].detections[i].merged_index, i);
  }
}

TEST(PalmSearchMergeCalculatorTest, EmptyRegionGivesEmptyDetections) {
  Outputs out = RunMerge("", {FullFrame()}, {Detections{}});
  EXPECT_TRUE(out.detections.empty());
  EXPECT_TRUE(out.record.merged.empty());
  ASSERT_EQ(out.record.regions.size(), 1);
}

// The same palm seen by two regions becomes one detection, averaged by score.
TEST(PalmSearchMergeCalculatorTest, MergesAcrossRegionsWeighted) {
  Outputs out = RunMerge(
      "", {FullFrame(), Tile(1, 0.25f, 0.25f, 0.5f, 0.5f)},
      {{Box(0.6f, 0.10f, 0.10f, 0.20f, 0.20f)},
       {Box(0.9f, 0.12f, 0.10f, 0.20f, 0.20f),
        Box(0.7f, 0.30f, 0.30f, 0.10f, 0.10f)}});
  ASSERT_EQ(out.detections.size(), 2);
  // Seeded by the 0.9 detection of the tile, joined by the full frame's.
  EXPECT_FLOAT_EQ(out.detections[0].score(0), 0.9f);
  const auto& box = out.detections[0].location_data().relative_bounding_box();
  EXPECT_NEAR(box.xmin(), (0.12f * 0.9f + 0.10f * 0.6f) / 1.5f, 1e-6);
  EXPECT_NEAR(box.width(), 0.20f, 1e-6);
  ASSERT_EQ(out.record.merged[0].sources.size(), 2);
  EXPECT_EQ(out.record.merged[0].sources[0].region_id, 1);
  EXPECT_EQ(out.record.merged[0].sources[1].region_id, 0);
  EXPECT_EQ(out.record.regions[0].detections[0].merged_index, 0);
  EXPECT_EQ(out.record.regions[1].detections[0].merged_index, 0);
  EXPECT_EQ(out.record.regions[1].detections[1].merged_index, 1);
  EXPECT_FLOAT_EQ(out.detections[1].score(0), 0.7f);
}

TEST(PalmSearchMergeCalculatorTest, BestScoreKeepsTheSeedUnchanged) {
  const Detection best = Box(0.9f, 0.12f, 0.10f, 0.20f, 0.20f);
  Outputs out = RunMerge("merge: BEST_SCORE",
                    {FullFrame(), Tile(1, 0.25f, 0.25f, 0.5f, 0.5f)},
                    {{Box(0.6f, 0.10f, 0.10f, 0.20f, 0.20f)}, {best}});
  ASSERT_EQ(out.detections.size(), 1);
  EXPECT_EQ(out.detections[0].SerializeAsString(), best.SerializeAsString());
  EXPECT_EQ(out.record.merged[0].sources.size(), 2);
}

// Boxes of different regions that overlap less than the threshold stay apart.
TEST(PalmSearchMergeCalculatorTest, LowOverlapIsNotMerged) {
  Outputs out =
      RunMerge("", {FullFrame(), Tile(1, 0.25f, 0.25f, 0.5f, 0.5f)},
          {{Box(0.9f, 0.10f, 0.10f, 0.20f, 0.20f)},
           {Box(0.8f, 0.25f, 0.10f, 0.20f, 0.20f)}});
  EXPECT_EQ(out.detections.size(), 2);
}

TEST(PalmSearchMergeCalculatorTest, RejectsTruncatedOnInnerEdgesOnly) {
  // The tile's window is the top-left quarter: its left and top edges are
  // frame edges, its right and bottom edges are not.
  const PalmSearchRegion tile = Tile(1, 0.25f, 0.25f, 0.5f, 0.5f);
  Outputs out = RunMerge("reject_truncated_detections: true", {tile},
                    {{Box(0.9f, 0.00f, 0.00f, 0.10f, 0.10f),    // frame corner
                      Box(0.8f, 0.40f, 0.20f, 0.10f, 0.10f),    // right edge
                      Box(0.7f, 0.20f, 0.42f, 0.10f, 0.08f),    // bottom edge
                      Box(0.6f, 0.20f, 0.20f, 0.10f, 0.10f)}});  // inside
  ASSERT_EQ(out.record.regions[0].detections.size(), 4);
  const auto& detections = out.record.regions[0].detections;
  EXPECT_EQ(detections[0].outcome, PalmSearchDetectionOutcome::kMerged);
  EXPECT_EQ(detections[1].outcome,
            PalmSearchDetectionOutcome::kRejectedTruncated);
  EXPECT_EQ(detections[1].merged_index, -1);
  EXPECT_EQ(detections[2].outcome,
            PalmSearchDetectionOutcome::kRejectedTruncated);
  EXPECT_EQ(detections[3].outcome, PalmSearchDetectionOutcome::kMerged);
  EXPECT_EQ(out.detections.size(), 2);
}

TEST(PalmSearchMergeCalculatorTest, TruncationIsOffByDefault) {
  Outputs out = RunMerge("", {Tile(1, 0.25f, 0.25f, 0.5f, 0.5f)},
                    {{Box(0.8f, 0.40f, 0.20f, 0.10f, 0.10f)}});
  EXPECT_EQ(out.detections.size(), 1);
}

// The full frame is exempt from both rejection rules.
TEST(PalmSearchMergeCalculatorTest, SizeRejectionSparesTheFullFrame) {
  Outputs out = RunMerge("max_palm_fraction: 0.2",
                    {FullFrame(), Tile(1, 0.75f, 0.75f, 0.5f, 0.5f)},
                    {{Box(0.9f, 0.10f, 0.10f, 0.40f, 0.40f)},
                     {Box(0.8f, 0.55f, 0.55f, 0.30f, 0.15f),
                      Box(0.7f, 0.60f, 0.80f, 0.10f, 0.10f)}});
  EXPECT_EQ(out.record.regions[0].detections[0].outcome,
            PalmSearchDetectionOutcome::kMerged);
  EXPECT_EQ(out.record.regions[1].detections[0].outcome,
            PalmSearchDetectionOutcome::kRejectedSize);
  EXPECT_EQ(out.record.regions[1].detections[1].outcome,
            PalmSearchDetectionOutcome::kMerged);
  EXPECT_EQ(out.detections.size(), 2);
}

// No region ran: a record without detections, and no detections packet.
TEST(PalmSearchMergeCalculatorTest, NoRegionDetectionsPacket) {
  CalculatorRunner runner(Node(""));
  runner.MutableInputs()->Tag("REGIONS").packets.push_back(
      MakePacket<PalmSearchRegions>().At(Timestamp(1)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("DETECTIONS").packets.empty());
  ASSERT_EQ(runner.Outputs().Tag("RECORD").packets.size(), 1);
}

}  // namespace
}  // namespace mediapipe
