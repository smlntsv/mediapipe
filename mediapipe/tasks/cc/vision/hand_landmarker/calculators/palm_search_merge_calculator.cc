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

#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/palm_search_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search.h"

namespace mediapipe {

// Collects the detections of every region of the palm search loop: one
// std::vector<Detection> per iteration.
typedef EndLoopCalculator<std::vector<std::vector<Detection>>>
    EndLoopDetectionVectorCalculator;
REGISTER_CALCULATOR(EndLoopDetectionVectorCalculator);

namespace {
using ::mediapipe::tasks::vision::hand_detector::proto::PalmSearchOptions;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchDetectionOutcome;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchMergedDetection;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRecord;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegion;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionDetection;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionKind;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegionResult;
using ::mediapipe::tasks::vision::hand_landmarker::PalmSearchRegions;

using Detections = std::vector<Detection>;
using RegionDetections = std::vector<Detections>;

constexpr char kRegionsTag[] = "REGIONS";
constexpr char kRegionDetectionsTag[] = "REGION_DETECTIONS";
constexpr char kRegionCandidatesTag[] = "REGION_CANDIDATES";
constexpr char kRegionThresholdedTag[] = "REGION_THRESHOLDED";
constexpr char kRegionStartNsTag[] = "REGION_START_NS";
constexpr char kRegionReadyNsTag[] = "REGION_READY_NS";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kRecordTag[] = "RECORD";
constexpr char kCandidatesTag[] = "CANDIDATES";
constexpr char kThresholdedTag[] = "THRESHOLDED";

// A box edge closer than this fraction of the sampled window to a window
// edge counts as touching it: one pixel of the 192 pixel model input.
constexpr float kEdgeTolerance = 1.0f / 192.0f;
// A window edge this close to the frame edge, or beyond it, is a frame edge.
constexpr float kFrameEdgeTolerance = 1e-4f;

int64_t MonotonicNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

Rectangle_f BoxOf(const Detection& detection) {
  const auto& box = detection.location_data().relative_bounding_box();
  return Rectangle_f(box.xmin(), box.ymin(), box.width(), box.height());
}

// Intersection over union, as NonMaxSuppressionCalculator computes it.
float Iou(const Rectangle_f& rect1, const Rectangle_f& rect2) {
  if (!rect1.Intersects(rect2)) return 0.0f;
  const float intersection_area = Rectangle_f(rect1).Intersect(rect2).Area();
  const float normalization = rect1.Area() + rect2.Area() - intersection_area;
  return normalization > 0.0f ? intersection_area / normalization : 0.0f;
}

// Whether `detection`'s box touches an edge of the window `region` was
// sampled from, other than an edge on or outside the frame edge.
bool IsTruncated(const Detection& detection, const PalmSearchRegion& region) {
  const auto& box = detection.location_data().relative_bounding_box();
  const float left = region.sampled_x_center - region.sampled_width / 2;
  const float right = region.sampled_x_center + region.sampled_width / 2;
  const float top = region.sampled_y_center - region.sampled_height / 2;
  const float bottom = region.sampled_y_center + region.sampled_height / 2;
  const float tolerance_x = region.sampled_width * kEdgeTolerance;
  const float tolerance_y = region.sampled_height * kEdgeTolerance;
  if (left > kFrameEdgeTolerance && box.xmin() <= left + tolerance_x) {
    return true;
  }
  if (right < 1.0f - kFrameEdgeTolerance &&
      box.xmin() + box.width() >= right - tolerance_x) {
    return true;
  }
  if (top > kFrameEdgeTolerance && box.ymin() <= top + tolerance_y) {
    return true;
  }
  if (bottom < 1.0f - kFrameEdgeTolerance &&
      box.ymin() + box.height() >= bottom - tolerance_y) {
    return true;
  }
  return false;
}

// A detection that passed the filters, waiting to be merged.
struct Item {
  int region_index;
  int detection_index;
  float score;
};

// Score-weighted average of the boxes and keypoints of `cluster`, written
// into a copy of its first detection. Same arithmetic as the WEIGHTED
// algorithm of NonMaxSuppressionCalculator.
Detection WeightedDetection(const std::vector<Item>& cluster,
                            const PalmSearchRecord& record) {
  const Detection& first = record.regions[cluster[0].region_index]
                               .detections[cluster[0].detection_index]
                               .detection;
  Detection weighted = first;
  const int num_keypoints = first.location_data().relative_keypoints_size();
  std::vector<float> keypoints(num_keypoints * 2);
  float w_xmin = 0.0f;
  float w_ymin = 0.0f;
  float w_xmax = 0.0f;
  float w_ymax = 0.0f;
  float total_score = 0.0f;
  for (const Item& item : cluster) {
    total_score += item.score;
    const auto& location_data = record.regions[item.region_index]
                                    .detections[item.detection_index]
                                    .detection.location_data();
    const auto& bbox = location_data.relative_bounding_box();
    w_xmin += bbox.xmin() * item.score;
    w_ymin += bbox.ymin() * item.score;
    w_xmax += (bbox.xmin() + bbox.width()) * item.score;
    w_ymax += (bbox.ymin() + bbox.height()) * item.score;
    for (int i = 0; i < num_keypoints; ++i) {
      keypoints[i * 2] += location_data.relative_keypoints(i).x() * item.score;
      keypoints[i * 2 + 1] +=
          location_data.relative_keypoints(i).y() * item.score;
    }
  }
  auto* box = weighted.mutable_location_data()->mutable_relative_bounding_box();
  box->set_xmin(w_xmin / total_score);
  box->set_ymin(w_ymin / total_score);
  box->set_width((w_xmax / total_score) - box->xmin());
  box->set_height((w_ymax / total_score) - box->ymin());
  for (int i = 0; i < num_keypoints; ++i) {
    auto* keypoint =
        weighted.mutable_location_data()->mutable_relative_keypoints(i);
    keypoint->set_x(keypoints[i * 2] / total_score);
    keypoint->set_y(keypoints[i * 2 + 1] / total_score);
  }
  return weighted;
}
}  // namespace

// Fork addition. The merge step of the palm search path: takes the palm
// detections of every region of one frame, optionally drops some of them,
// and merges what different regions found of the same palm.
//
// It knows nothing about how the regions were chosen.
//
// Detections of one region have already been through that region's non-max
// suppression and are never merged with each other here, so with a single
// region the output is that region's detections, unchanged and in order.
// Across regions: detections are taken highest score first; each one
// collects the detections of other regions whose box overlaps its own by
// more than `merge_min_suppression_threshold` (intersection over union) and
// becomes one output detection. Output order is by descending score of the
// first detection of each cluster.
//
// Inputs:
//   REGIONS - PalmSearchRegions
//     The regions of this frame (PalmSearchRegionsCalculator).
//   REGION_DETECTIONS - std::vector<std::vector<Detection>>
//     Per region, in the same order: its detections in frame coordinates.
//     No packet when no region was run.
//   REGION_CANDIDATES - std::vector<std::vector<Detection>> @Optional
//   REGION_THRESHOLDED - std::vector<std::vector<Detection>> @Optional
//     Per region: the palm candidate streams of `debug_palm_candidates`.
//   REGION_START_NS - std::vector<int64_t> @Optional
//   REGION_READY_NS - std::vector<int64_t> @Optional
//     Per region: clock stamps for the record.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     The merged detections. Emitted (possibly empty) whenever
//     REGION_DETECTIONS has a packet.
//   RECORD - PalmSearchRecord @Optional
//     What every region produced and what became of it. Emitted whenever
//     REGIONS has a packet.
//   CANDIDATES - std::vector<Detection> @Optional
//   THRESHOLDED - std::vector<Detection> @Optional
//     The per-region candidate streams concatenated in region order.
class PalmSearchMergeCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kRegionsTag).Set<PalmSearchRegions>();
    cc->Inputs().Tag(kRegionDetectionsTag).Set<RegionDetections>();
    for (const char* tag : {kRegionCandidatesTag, kRegionThresholdedTag}) {
      if (cc->Inputs().HasTag(tag)) {
        cc->Inputs().Tag(tag).Set<RegionDetections>();
      }
    }
    for (const char* tag : {kRegionStartNsTag, kRegionReadyNsTag}) {
      if (cc->Inputs().HasTag(tag)) {
        cc->Inputs().Tag(tag).Set<std::vector<int64_t>>();
      }
    }
    cc->Outputs().Tag(kDetectionsTag).Set<Detections>();
    if (cc->Outputs().HasTag(kRecordTag)) {
      cc->Outputs().Tag(kRecordTag).Set<PalmSearchRecord>();
    }
    for (const char* tag : {kCandidatesTag, kThresholdedTag}) {
      if (cc->Outputs().HasTag(tag)) {
        cc->Outputs().Tag(tag).Set<Detections>();
      }
    }
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<PalmSearchOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kRegionsTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& regions = cc->Inputs().Tag(kRegionsTag).Get<PalmSearchRegions>();
    auto record = std::make_unique<PalmSearchRecord>();
    record->schedule = regions.schedule;
    record->regions_ready_ns = regions.regions_ready_ns;
    record->regions.resize(regions.regions.size());
    for (int r = 0; r < regions.regions.size(); ++r) {
      record->regions[r].region = regions.regions[r];
    }

    const bool ran = !cc->Inputs().Tag(kRegionDetectionsTag).IsEmpty();
    if (ran) {
      const auto& region_detections =
          cc->Inputs().Tag(kRegionDetectionsTag).Get<RegionDetections>();
      RET_CHECK_EQ(region_detections.size(), regions.regions.size())
          << "One detection vector per region is expected.";
      std::vector<Item> items = Filter(region_detections, record.get());
      Merge(std::move(items), record.get());

      auto detections = std::make_unique<Detections>();
      detections->reserve(record->merged.size());
      for (const auto& merged : record->merged) {
        detections->push_back(merged.detection);
      }
      record->merge_done_ns = MonotonicNowNs();
      cc->Outputs()
          .Tag(kDetectionsTag)
          .Add(detections.release(), cc->InputTimestamp());
    }

    for (const char* tag : {kRegionStartNsTag, kRegionReadyNsTag}) {
      if (!cc->Inputs().HasTag(tag) || cc->Inputs().Tag(tag).IsEmpty()) continue;
      const auto& stamps = cc->Inputs().Tag(tag).Get<std::vector<int64_t>>();
      for (int r = 0; r < stamps.size() && r < record->regions.size(); ++r) {
        (tag == kRegionStartNsTag ? record->regions[r].crop_start_ns
                                  : record->regions[r].detections_ready_ns) =
            stamps[r];
      }
    }
    AddCandidates(cc, kRegionCandidatesTag, kCandidatesTag, record.get(),
                  /*count_only=*/false);
    AddCandidates(cc, kRegionThresholdedTag, kThresholdedTag, record.get(),
                  /*count_only=*/true);

    if (cc->Outputs().HasTag(kRecordTag)) {
      cc->Outputs().Tag(kRecordTag).Add(record.release(), cc->InputTimestamp());
    }
    return absl::OkStatus();
  }

 private:
  // Copies the detections into the record, applies the optional rejection
  // rules and returns the detections that go on to the merge.
  std::vector<Item> Filter(const RegionDetections& region_detections,
                           PalmSearchRecord* record) const {
    std::vector<Item> items;
    for (int r = 0; r < region_detections.size(); ++r) {
      PalmSearchRegionResult& result = record->regions[r];
      const bool is_full_frame =
          result.region.kind == PalmSearchRegionKind::kFullFrame;
      result.detections.resize(region_detections[r].size());
      for (int d = 0; d < region_detections[r].size(); ++d) {
        PalmSearchRegionDetection& entry = result.detections[d];
        entry.detection = region_detections[r][d];
        const auto& box =
            entry.detection.location_data().relative_bounding_box();
        if (!is_full_frame && options_.reject_truncated_detections() &&
            IsTruncated(entry.detection, result.region)) {
          entry.outcome = PalmSearchDetectionOutcome::kRejectedTruncated;
        } else if (!is_full_frame && options_.max_palm_fraction() > 0 &&
                   (box.width() > options_.max_palm_fraction() ||
                    box.height() > options_.max_palm_fraction())) {
          entry.outcome = PalmSearchDetectionOutcome::kRejectedSize;
        } else {
          items.push_back({r, d,
                           entry.detection.score_size() > 0
                               ? entry.detection.score(0)
                               : 0.0f});
        }
      }
    }
    return items;
  }

  void Merge(std::vector<Item> items, PalmSearchRecord* record) const {
    // Stable: equal scores keep region order, then the region's own order.
    std::stable_sort(
        items.begin(), items.end(),
        [](const Item& a, const Item& b) { return a.score > b.score; });
    std::vector<Item> remaining;
    std::vector<Item> cluster;
    while (!items.empty()) {
      const Item seed = items[0];
      const Rectangle_f seed_box =
          BoxOf(record->regions[seed.region_index]
                    .detections[seed.detection_index]
                    .detection);
      remaining.clear();
      cluster.clear();
      cluster.push_back(seed);
      for (int i = 1; i < items.size(); ++i) {
        const Item& item = items[i];
        const bool joins =
            item.region_index != seed.region_index &&
            Iou(BoxOf(record->regions[item.region_index]
                          .detections[item.detection_index]
                          .detection),
                seed_box) > options_.merge_min_suppression_threshold();
        (joins ? cluster : remaining).push_back(item);
      }
      PalmSearchMergedDetection merged;
      if (cluster.size() > 1 && options_.merge() == PalmSearchOptions::WEIGHTED) {
        merged.detection = WeightedDetection(cluster, *record);
      } else {
        merged.detection = record->regions[seed.region_index]
                               .detections[seed.detection_index]
                               .detection;
      }
      const int merged_index = record->merged.size();
      for (const Item& item : cluster) {
        PalmSearchRegionResult& result = record->regions[item.region_index];
        result.detections[item.detection_index].merged_index = merged_index;
        merged.sources.push_back({result.region.id, item.detection_index});
      }
      record->merged.push_back(std::move(merged));
      items.swap(remaining);
    }
  }

  // Concatenates a per-region candidate stream into its legacy single-vector
  // output and copies it into the record.
  void AddCandidates(CalculatorContext* cc, const char* input_tag,
                     const char* output_tag, PalmSearchRecord* record,
                     bool count_only) const {
    if (!cc->Inputs().HasTag(input_tag) || cc->Inputs().Tag(input_tag).IsEmpty()) {
      return;
    }
    const auto& per_region = cc->Inputs().Tag(input_tag).Get<RegionDetections>();
    auto all = std::make_unique<Detections>();
    for (int r = 0; r < per_region.size(); ++r) {
      all->insert(all->end(), per_region[r].begin(), per_region[r].end());
      if (r >= record->regions.size()) continue;
      if (count_only) {
        record->regions[r].above_threshold_count = per_region[r].size();
      } else {
        record->regions[r].candidates = per_region[r];
      }
    }
    if (cc->Outputs().HasTag(output_tag)) {
      cc->Outputs().Tag(output_tag).Add(all.release(), cc->InputTimestamp());
    }
  }

  PalmSearchOptions options_;
};
REGISTER_CALCULATOR(PalmSearchMergeCalculator);

}  // namespace mediapipe
