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

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/palm_search_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search_grid.h"

namespace mediapipe {

namespace {
using ::mediapipe::tasks::vision::hand_detector::proto::PalmSearchOptions;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegion;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionKind;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegions;

constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kExternalRegionsTag[] = "EXTERNAL_REGIONS";
constexpr char kRectsTag[] = "RECTS";
constexpr char kRegionsTag[] = "REGIONS";

int64_t MonotonicNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// Fork addition. Region generation of the palm search path: decides which
// regions of the frame the palm model runs on at this timestamp. It is the
// only place that knows about grids and schedules; the nodes after it get a
// list of rects.
//
// The calculator runs only at timestamps where IMAGE_SIZE has a packet. The
// hand landmarker graph gates the detector's image, so in stream mode that
// is exactly the frames on which the detector runs, and state kept here
// (the schedule) advances per detector frame.
//
// Inputs:
//   IMAGE_SIZE - std::pair<int, int>
//     Width and height of the frame.
//   NORM_RECT - NormalizedRect @Optional
//     The task's rect for the whole frame. Only its rotation differs from
//     the full-frame default; the hand landmarker accepts no region of
//     interest.
//   EXTERNAL_REGIONS - std::vector<NormalizedRect> @Optional
//     Regions chosen by the caller for this frame: axis-aligned boxes in
//     normalized coordinates of the frame as passed in (their rotation is
//     ignored). Entries without a positive width and height are skipped.
//
// Region order: the full frame (`include_full_frame`), the grid tiles due on
// this detector frame in grid order from the schedule cursor, the external
// regions in the order given.
//
// Outputs:
//   RECTS - std::vector<NormalizedRect>
//     One rect per region, for image-to-tensor, with the task's rotation.
//     No packet when there is no region to run at this timestamp.
//   REGIONS - PalmSearchRegions
//     The same regions described in frame coordinates, in the same order.
//     Emitted at every timestamp the calculator runs at.
class PalmSearchRegionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    if (cc->Inputs().HasTag(kNormRectTag)) {
      cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
    }
    if (cc->Inputs().HasTag(kExternalRegionsTag)) {
      cc->Inputs().Tag(kExternalRegionsTag).Set<std::vector<NormalizedRect>>();
    }
    cc->Outputs().Tag(kRectsTag).Set<std::vector<NormalizedRect>>();
    cc->Outputs().Tag(kRegionsTag).Set<PalmSearchRegions>();
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<PalmSearchOptions>();
    tiles_ = tasks::vision::hand_landmarker::PalmSearchGridTiles(
        options_.grid_rows(), options_.grid_columns(), options_.grid_overlap());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kImageSizeTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& [image_width, image_height] =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    RET_CHECK(image_width > 0 && image_height > 0);

    NormalizedRect frame_rect;
    frame_rect.set_x_center(0.5f);
    frame_rect.set_y_center(0.5f);
    frame_rect.set_width(1.0f);
    frame_rect.set_height(1.0f);
    frame_rect.set_rotation(0.0f);
    if (cc->Inputs().HasTag(kNormRectTag) &&
        !cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
      frame_rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
    }

    auto regions = std::make_unique<PalmSearchRegions>();
    auto rects = std::make_unique<std::vector<NormalizedRect>>();
    regions->schedule.detector_frame_index = detector_frame_index_++;

    if (options_.include_full_frame()) {
      PalmSearchRegion region;
      region.id = tasks::vision::hand_landmarker::kPalmSearchFullFrameRegionId;
      region.kind = PalmSearchRegionKind::kFullFrame;
      tasks::vision::hand_landmarker::SetPalmSearchSampledWindow(
          image_width, image_height, &region);
      regions->regions.push_back(region);
      // The task's own rect, untouched: the model sees what it sees without
      // search regions.
      rects->push_back(frame_rect);
    }

    auto add_region = [&](PalmSearchRegion region) {
      tasks::vision::hand_landmarker::SetPalmSearchSampledWindow(
          image_width, image_height, &region);
      rects->push_back(tasks::vision::hand_landmarker::PalmSearchRegionRect(
          region, frame_rect.rotation(), image_width, image_height));
      regions->regions.push_back(region);
    };

    regions->schedule.grid_tile_count = tiles_.size();
    regions->schedule.tiles_per_frame =
        std::max(0, options_.round_robin_tiles_per_frame());
    regions->schedule.cursor_before = cursor_;
    for (int index : tasks::vision::hand_landmarker::PalmSearchScheduledTiles(
             tiles_.size(), options_.round_robin_tiles_per_frame(), cursor_,
             &cursor_)) {
      add_region(tiles_[index]);
    }
    regions->schedule.cursor_after = cursor_;

    if (cc->Inputs().HasTag(kExternalRegionsTag) &&
        !cc->Inputs().Tag(kExternalRegionsTag).IsEmpty()) {
      const auto& external = cc->Inputs()
                                 .Tag(kExternalRegionsTag)
                                 .Get<std::vector<NormalizedRect>>();
      for (int i = 0; i < external.size(); ++i) {
        if (!(external[i].width() > 0 && external[i].height() > 0)) continue;
        PalmSearchRegion region;
        region.id =
            tasks::vision::hand_landmarker::kPalmSearchExternalRegionIdBase + i;
        region.kind = PalmSearchRegionKind::kExternal;
        region.requested_x_center = external[i].x_center();
        region.requested_y_center = external[i].y_center();
        region.requested_width = external[i].width();
        region.requested_height = external[i].height();
        add_region(region);
      }
    }

    regions->regions_ready_ns = MonotonicNowNs();
    if (!rects->empty()) {
      cc->Outputs().Tag(kRectsTag).Add(rects.release(), cc->InputTimestamp());
    }
    cc->Outputs().Tag(kRegionsTag).Add(regions.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  PalmSearchOptions options_;
  // The grid, fixed for the life of the graph.
  std::vector<PalmSearchRegion> tiles_;
  // Advance once per Process call, that is per detector frame.
  int64_t detector_frame_index_ = 0;
  int cursor_ = 0;
};
REGISTER_CALCULATOR(PalmSearchRegionsCalculator);

}  // namespace mediapipe
