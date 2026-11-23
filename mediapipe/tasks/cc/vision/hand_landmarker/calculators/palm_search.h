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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_H_

#include <cstdint>
#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// Records of the palm search regions path (PalmSearchOptions). Raw values of
// one frame; nothing here is derived from more than one frame.
//
// All rects and detections are in normalized coordinates of the input image
// as passed in, like every other hand landmarker output.

// Region ids. They do not change from frame to frame for a given grid.
inline constexpr int kPalmSearchFullFrameRegionId = 0;
// A grid tile is `kPalmSearchGridRegionIdBase + row * columns + column`.
inline constexpr int kPalmSearchGridRegionIdBase = 1;
// The i-th external region of a call is `kPalmSearchExternalRegionIdBase + i`.
inline constexpr int kPalmSearchExternalRegionIdBase = 1000;

enum class PalmSearchRegionKind {
  kFullFrame = 0,
  kGrid = 1,
  kExternal = 2,
};

// One region the palm model was run on.
struct PalmSearchRegion {
  int id = 0;
  PalmSearchRegionKind kind = PalmSearchRegionKind::kFullFrame;
  // Grid tiles only, otherwise -1.
  int grid_row = -1;
  int grid_column = -1;
  // The region as asked for: an axis-aligned box of the frame.
  float requested_x_center = 0.5f;
  float requested_y_center = 0.5f;
  float requested_width = 1.0f;
  float requested_height = 1.0f;
  // The window the model input was sampled from. Image-to-tensor grows the
  // region to the square aspect ratio of the model input around its center,
  // so the window is a square in pixels and can reach outside the frame
  // (those pixels are black). Not clipped to the frame here.
  float sampled_x_center = 0.5f;
  float sampled_y_center = 0.5f;
  float sampled_width = 1.0f;
  float sampled_height = 1.0f;
};

// State of the region schedule on one detector frame.
struct PalmSearchSchedule {
  // Number of detector frames before this one since the landmarker was
  // created. A frame on which the detector is skipped does not count.
  int64_t detector_frame_index = 0;
  // Number of grid tiles, and how many of them run per detector frame (0:
  // all of them).
  int grid_tile_count = 0;
  int tiles_per_frame = 0;
  // Index of the first grid tile run on this frame and of the first one of
  // the next detector frame. Both 0 without a round-robin schedule.
  int cursor_before = 0;
  int cursor_after = 0;
};

// Output of PalmSearchRegionsCalculator: the regions of one detector frame,
// in the order the model is run on them.
struct PalmSearchRegions {
  std::vector<PalmSearchRegion> regions;
  PalmSearchSchedule schedule;
  // Monotonic clock, nanoseconds, when the list was emitted.
  int64_t regions_ready_ns = 0;
};

// What became of one detection of one region.
enum class PalmSearchDetectionOutcome {
  // Went into the merge; `merged_index` says where it ended up.
  kMerged = 0,
  // Dropped by `reject_truncated_detections`.
  kRejectedTruncated = 1,
  // Dropped by `max_palm_fraction`.
  kRejectedSize = 2,
};

struct PalmSearchRegionDetection {
  // The detection as it left the region's own non-max suppression, projected
  // to the frame.
  mediapipe::Detection detection;
  PalmSearchDetectionOutcome outcome = PalmSearchDetectionOutcome::kMerged;
  // Index into `PalmSearchRecord::merged` of the detection it became or was
  // folded into; -1 when it was rejected.
  int merged_index = -1;
};

// Everything one region produced.
struct PalmSearchRegionResult {
  PalmSearchRegion region;
  std::vector<PalmSearchRegionDetection> detections;
  // Only with `debug_palm_candidates` > 0: the highest scored decoded boxes
  // of this region whatever their score, and the number of decoded boxes
  // that passed the score threshold (the input of the region's non-max
  // suppression).
  std::vector<mediapipe::Detection> candidates;
  int above_threshold_count = 0;
  // Monotonic clock, nanoseconds: when the region entered the loop and when
  // its projected detections left it. 0 when not measured. Loop iterations
  // can overlap in time, so these are not exclusive costs.
  int64_t crop_start_ns = 0;
  int64_t detections_ready_ns = 0;
};

// Where a merged detection came from.
struct PalmSearchSource {
  int region_id = 0;
  // Index into that region's `detections`.
  int detection_index = 0;
};

struct PalmSearchMergedDetection {
  mediapipe::Detection detection;
  // Highest scored first: the first source is the detection the merged one
  // took its score and label from.
  std::vector<PalmSearchSource> sources;
};

// Output of PalmSearchMergeCalculator for one detector frame.
struct PalmSearchRecord {
  PalmSearchSchedule schedule;
  int64_t regions_ready_ns = 0;
  // In the order the model was run on them.
  std::vector<PalmSearchRegionResult> regions;
  // Highest score first. These are the graph's PALM_DETECTIONS; the i-th one
  // becomes the i-th palm rect and the i-th detector hand rect.
  std::vector<PalmSearchMergedDetection> merged;
  // Monotonic clock, nanoseconds, when `merged` was emitted.
  int64_t merge_done_ns = 0;
};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_H_
