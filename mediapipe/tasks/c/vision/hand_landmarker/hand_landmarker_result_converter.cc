/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result_converter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;
using CppHandLandmarkerStages =
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerStages;
using CppHandLandmarkerTensorCaptures =
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerTensorCaptures;
using CppHandLandmarkerPalmCandidates =
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerPalmCandidates;
using CppHandLandmarkerPalmSearch =
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerPalmSearch;
using CppTensorCapture =
    ::mediapipe::tasks::vision::hand_landmarker::TensorCapture;

namespace {

void ConvertRects(const std::vector<mediapipe::NormalizedRect>& in,
                  MpNormalizedRotatedRect** out, uint32_t* out_count) {
  *out_count = in.size();
  *out = in.empty() ? nullptr : new MpNormalizedRotatedRect[in.size()];
  for (uint32_t i = 0; i < in.size(); ++i) {
    MpNormalizedRotatedRect& rect = (*out)[i];
    rect.x_center = in[i].x_center();
    rect.y_center = in[i].y_center();
    rect.width = in[i].width();
    rect.height = in[i].height();
    rect.rotation = in[i].rotation();
    rect.has_rect_id = in[i].has_rect_id();
    rect.rect_id = in[i].rect_id();
  }
}

void CloseRects(MpNormalizedRotatedRect** rects, uint32_t* count) {
  delete[] *rects;
  *rects = nullptr;
  *count = 0;
}

void ConvertDetection(const mediapipe::Detection& detection,
                      MpNormalizedDetection* out) {
  const auto& location = detection.location_data();
  const auto& box = location.relative_bounding_box();
  MpNormalizedDetection& c_detection = *out;
  c_detection.score = detection.score_size() > 0 ? detection.score(0) : 0.0f;
  c_detection.x_min = box.xmin();
  c_detection.y_min = box.ymin();
  c_detection.width = box.width();
  c_detection.height = box.height();
  c_detection.keypoints_count = location.relative_keypoints_size();
  c_detection.keypoints =
      c_detection.keypoints_count == 0
          ? nullptr
          : new MpNormalizedPoint[c_detection.keypoints_count];
  for (uint32_t j = 0; j < c_detection.keypoints_count; ++j) {
    c_detection.keypoints[j].x = location.relative_keypoints(j).x();
    c_detection.keypoints[j].y = location.relative_keypoints(j).y();
  }
}

void ConvertDetections(const std::vector<mediapipe::Detection>& in,
                       MpNormalizedDetection** out, uint32_t* out_count) {
  *out_count = in.size();
  *out = in.empty() ? nullptr : new MpNormalizedDetection[in.size()];
  for (uint32_t i = 0; i < in.size(); ++i) {
    ConvertDetection(in[i], &(*out)[i]);
  }
}

void CloseDetections(MpNormalizedDetection** detections, uint32_t* count) {
  for (uint32_t i = 0; i < *count; ++i) {
    delete[] (*detections)[i].keypoints;
  }
  delete[] *detections;
  *detections = nullptr;
  *count = 0;
}

void ConvertTensorCapture(const CppTensorCapture& in, MpTensorCapture* out) {
  *out = MpTensorCapture{};
  if (!in.has_tensor) return;
  out->has_tensor = true;
  out->element_type = static_cast<int32_t>(in.element_type);
  out->shape_count = in.shape.size();
  if (!in.shape.empty()) {
    out->shape = new int32_t[in.shape.size()];
    std::copy(in.shape.begin(), in.shape.end(), out->shape);
  }
  if (in.data != nullptr && !in.data->empty()) {
    out->data_size = in.data->size();
    out->data = new uint8_t[in.data->size()];
    std::memcpy(out->data, in.data->data(), in.data->size());
  }
}

void CloseTensorCapture(MpTensorCapture* capture) {
  delete[] capture->shape;
  delete[] capture->data;
  *capture = MpTensorCapture{};
}

MpHandLandmarkerTensorCapture* ConvertTensorCaptures(
    const CppHandLandmarkerTensorCaptures& in) {
  auto* out = new MpHandLandmarkerTensorCapture{};
  out->timestamp_us = in.timestamp_us;
  ConvertTensorCapture(in.palm_detector_input, &out->palm_detector_input);
  out->landmark_inputs_count = in.landmark_inputs.size();
  if (!in.landmark_inputs.empty()) {
    out->landmark_inputs = new MpTensorCapture[in.landmark_inputs.size()];
  }
  for (uint32_t i = 0; i < out->landmark_inputs_count; ++i) {
    ConvertTensorCapture(in.landmark_inputs[i], &out->landmark_inputs[i]);
  }
  return out;
}

void CloseTensorCaptures(MpHandLandmarkerTensorCapture* capture) {
  if (capture == nullptr) return;
  CloseTensorCapture(&capture->palm_detector_input);
  for (uint32_t i = 0; i < capture->landmark_inputs_count; ++i) {
    CloseTensorCapture(&capture->landmark_inputs[i]);
  }
  delete[] capture->landmark_inputs;
  delete capture;
}

MpHandLandmarkerPalmCandidates* ConvertPalmCandidates(
    const CppHandLandmarkerPalmCandidates& in) {
  auto* out = new MpHandLandmarkerPalmCandidates{};
  out->timestamp_us = in.timestamp_us;
  out->palm_detector_ran = in.palm_detector_ran;
  ConvertDetections(in.candidates, &out->candidates, &out->candidates_count);
  out->above_threshold_count = in.above_threshold_count;
  out->after_suppression_count = in.after_suppression_count;
  return out;
}

MpHandLandmarkerPalmSearch* ConvertPalmSearch(
    const CppHandLandmarkerPalmSearch& in) {
  auto* out = new MpHandLandmarkerPalmSearch{};
  out->timestamp_us = in.timestamp_us;
  out->frame_sequence = in.frame_sequence;
  out->palm_detector_ran = in.palm_detector_ran;
  out->detector_frame_index = in.record.schedule.detector_frame_index;
  out->grid_tile_count = in.record.schedule.grid_tile_count;
  out->tiles_per_frame = in.record.schedule.tiles_per_frame;
  out->cursor_before = in.record.schedule.cursor_before;
  out->cursor_after = in.record.schedule.cursor_after;
  out->call_start_ns = in.call_start_ns;
  out->call_end_ns = in.call_end_ns;
  out->regions_ready_ns = in.record.regions_ready_ns;
  out->merge_done_ns = in.record.merge_done_ns;
  out->landmark_rois_ready_ns = in.landmark_rois_ready_ns;
  out->landmarks_done_ns = in.landmarks_done_ns;

  out->regions_count = in.record.regions.size();
  out->regions = in.record.regions.empty()
                     ? nullptr
                     : new MpPalmSearchRegion[in.record.regions.size()]();
  for (uint32_t r = 0; r < out->regions_count; ++r) {
    const auto& cpp_region = in.record.regions[r];
    MpPalmSearchRegion& region = out->regions[r];
    region.id = cpp_region.region.id;
    region.kind = static_cast<int>(cpp_region.region.kind);
    region.grid_row = cpp_region.region.grid_row;
    region.grid_column = cpp_region.region.grid_column;
    region.requested_x_center = cpp_region.region.requested_x_center;
    region.requested_y_center = cpp_region.region.requested_y_center;
    region.requested_width = cpp_region.region.requested_width;
    region.requested_height = cpp_region.region.requested_height;
    region.sampled_x_center = cpp_region.region.sampled_x_center;
    region.sampled_y_center = cpp_region.region.sampled_y_center;
    region.sampled_width = cpp_region.region.sampled_width;
    region.sampled_height = cpp_region.region.sampled_height;
    region.detections_count = cpp_region.detections.size();
    region.detections =
        cpp_region.detections.empty()
            ? nullptr
            : new MpPalmSearchRegionDetection[cpp_region.detections.size()]();
    for (uint32_t d = 0; d < region.detections_count; ++d) {
      ConvertDetection(cpp_region.detections[d].detection,
                       &region.detections[d].detection);
      region.detections[d].outcome =
          static_cast<int>(cpp_region.detections[d].outcome);
      region.detections[d].merged_index = cpp_region.detections[d].merged_index;
    }
    ConvertDetections(cpp_region.candidates, &region.candidates,
                      &region.candidates_count);
    region.above_threshold_count = cpp_region.above_threshold_count;
    region.crop_start_ns = cpp_region.crop_start_ns;
    region.detections_ready_ns = cpp_region.detections_ready_ns;
  }

  out->merged_count = in.record.merged.size();
  out->merged = in.record.merged.empty()
                    ? nullptr
                    : new MpPalmSearchMergedDetection[in.record.merged.size()]();
  for (uint32_t m = 0; m < out->merged_count; ++m) {
    const auto& cpp_merged = in.record.merged[m];
    MpPalmSearchMergedDetection& merged = out->merged[m];
    ConvertDetection(cpp_merged.detection, &merged.detection);
    merged.sources_count = cpp_merged.sources.size();
    merged.sources = cpp_merged.sources.empty()
                         ? nullptr
                         : new MpPalmSearchSource[cpp_merged.sources.size()];
    for (uint32_t s = 0; s < merged.sources_count; ++s) {
      merged.sources[s].region_id = cpp_merged.sources[s].region_id;
      merged.sources[s].detection_index = cpp_merged.sources[s].detection_index;
    }
    merged.outcome = m < in.merged_outcomes.size()
                         ? static_cast<int>(in.merged_outcomes[m])
                         : 0;
    merged.overlapped_rect_index = m < in.merged_overlapped_rect_index.size()
                                       ? in.merged_overlapped_rect_index[m]
                                       : -1;
    merged.landmark_roi_index = m < in.merged_landmark_roi_index.size()
                                    ? in.merged_landmark_roi_index[m]
                                    : -1;
  }
  return out;
}

void ClosePalmSearch(MpHandLandmarkerPalmSearch* search) {
  if (search == nullptr) return;
  for (uint32_t r = 0; r < search->regions_count; ++r) {
    MpPalmSearchRegion& region = search->regions[r];
    for (uint32_t d = 0; d < region.detections_count; ++d) {
      delete[] region.detections[d].detection.keypoints;
    }
    delete[] region.detections;
    CloseDetections(&region.candidates, &region.candidates_count);
  }
  delete[] search->regions;
  for (uint32_t m = 0; m < search->merged_count; ++m) {
    delete[] search->merged[m].detection.keypoints;
    delete[] search->merged[m].sources;
  }
  delete[] search->merged;
  delete search;
}

void ClosePalmCandidates(MpHandLandmarkerPalmCandidates* candidates) {
  if (candidates == nullptr) return;
  CloseDetections(&candidates->candidates, &candidates->candidates_count);
  delete candidates;
}

MpHandLandmarkerStages* ConvertStages(const CppHandLandmarkerStages& in) {
  auto* out = new MpHandLandmarkerStages{};
  out->timestamp_us = in.timestamp_us;
  out->palm_detector_ran = in.palm_detector_ran;
  ConvertRects(in.previous_rects, &out->previous_rects,
               &out->previous_rects_count);

  ConvertDetections(in.palm_detections, &out->palm_detections,
                    &out->palm_detections_count);

  ConvertRects(in.palm_rects, &out->palm_rects, &out->palm_rects_count);
  ConvertRects(in.detector_hand_rects, &out->detector_hand_rects,
               &out->detector_hand_rects_count);
  ConvertRects(in.landmark_rois, &out->landmark_rois,
               &out->landmark_rois_count);

  out->presence_scores_count = in.presence_scores.size();
  if (!in.presence_scores.empty()) {
    out->presence_scores = new float[in.presence_scores.size()];
    std::copy(in.presence_scores.begin(), in.presence_scores.end(),
              out->presence_scores);
  }
  out->presence_passed_count = in.presence_passed.size();
  if (!in.presence_passed.empty()) {
    out->presence_passed = new bool[in.presence_passed.size()];
    std::copy(in.presence_passed.begin(), in.presence_passed.end(),
              out->presence_passed);
  }

  out->landmarks_before_dedup_count = in.landmarks_before_dedup.size();
  if (!in.landmarks_before_dedup.empty()) {
    out->landmarks_before_dedup =
        new MpNormalizedLandmarks[out->landmarks_before_dedup_count];
  }
  for (uint32_t i = 0; i < out->landmarks_before_dedup_count; ++i) {
    CppConvertToNormalizedLandmarks(in.landmarks_before_dedup[i].landmarks,
                                    &out->landmarks_before_dedup[i]);
  }

  ConvertRects(in.hand_rects_before_dedup, &out->hand_rects_before_dedup,
               &out->hand_rects_before_dedup_count);
  ConvertRects(in.hand_rects_next_frame, &out->hand_rects_next_frame,
               &out->hand_rects_next_frame_count);
  ConvertRects(in.keep_alive_rects, &out->keep_alive_rects,
               &out->keep_alive_rects_count);

  out->keep_alive_miss_counts_count = in.keep_alive_miss_counts.size();
  if (!in.keep_alive_miss_counts.empty()) {
    out->keep_alive_miss_counts = new int32_t[in.keep_alive_miss_counts.size()];
    std::copy(in.keep_alive_miss_counts.begin(),
              in.keep_alive_miss_counts.end(), out->keep_alive_miss_counts);
  }
  return out;
}

void CloseStages(MpHandLandmarkerStages* stages) {
  if (stages == nullptr) return;
  CloseRects(&stages->previous_rects, &stages->previous_rects_count);
  CloseDetections(&stages->palm_detections, &stages->palm_detections_count);
  CloseRects(&stages->palm_rects, &stages->palm_rects_count);
  CloseRects(&stages->detector_hand_rects, &stages->detector_hand_rects_count);
  CloseRects(&stages->landmark_rois, &stages->landmark_rois_count);
  delete[] stages->presence_scores;
  delete[] stages->presence_passed;
  for (uint32_t i = 0; i < stages->landmarks_before_dedup_count; ++i) {
    CppCloseNormalizedLandmarks(&stages->landmarks_before_dedup[i]);
  }
  delete[] stages->landmarks_before_dedup;
  CloseRects(&stages->hand_rects_before_dedup,
             &stages->hand_rects_before_dedup_count);
  CloseRects(&stages->hand_rects_next_frame,
             &stages->hand_rects_next_frame_count);
  CloseRects(&stages->keep_alive_rects, &stages->keep_alive_rects_count);
  delete[] stages->keep_alive_miss_counts;
  delete stages;
}

}  // namespace

void CppConvertToHandLandmarkerResult(
    const mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult& in,
    MpHandLandmarkerResult* out) {
  out->handedness_count = in.handedness.size();
  out->handedness = new MpCategories[out->handedness_count];

  for (uint32_t i = 0; i < out->handedness_count; ++i) {
    uint32_t categories_count = in.handedness[i].categories.size();
    out->handedness[i].categories_count = categories_count;
    out->handedness[i].categories = new MpCategory[categories_count];

    for (uint32_t j = 0; j < categories_count; ++j) {
      const auto& cpp_category = in.handedness[i].categories[j];
      CppConvertToCategory(cpp_category, &out->handedness[i].categories[j]);
    }
  }

  out->hand_landmarks_count = in.hand_landmarks.size();
  out->hand_landmarks = new MpNormalizedLandmarks[out->hand_landmarks_count];
  for (uint32_t i = 0; i < out->hand_landmarks_count; ++i) {
    std::vector<CppNormalizedLandmark> cpp_normalized_landmarks;
    for (uint32_t j = 0; j < in.hand_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.hand_landmarks[i].landmarks[j];
      cpp_normalized_landmarks.push_back(cpp_landmark);
    }
    CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                    &out->hand_landmarks[i]);
  }

  out->hand_world_landmarks_count = in.hand_world_landmarks.size();
  out->hand_world_landmarks = new MpLandmarks[out->hand_world_landmarks_count];
  for (uint32_t i = 0; i < out->hand_world_landmarks_count; ++i) {
    std::vector<CppLandmark> cpp_landmarks;
    for (uint32_t j = 0; j < in.hand_world_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.hand_world_landmarks[i].landmarks[j];
      cpp_landmarks.push_back(cpp_landmark);
    }
    CppConvertToLandmarks(cpp_landmarks, &out->hand_world_landmarks[i]);
  }

  out->stages = in.stages.has_value() ? ConvertStages(*in.stages) : nullptr;
  out->tensor_capture = in.tensor_capture.has_value()
                            ? ConvertTensorCaptures(*in.tensor_capture)
                            : nullptr;
  out->palm_candidates = in.palm_candidates.has_value()
                             ? ConvertPalmCandidates(*in.palm_candidates)
                             : nullptr;
  out->palm_search = in.palm_search.has_value()
                         ? ConvertPalmSearch(*in.palm_search)
                         : nullptr;
}

void CppCloseHandLandmarkerResult(MpHandLandmarkerResult* result) {
  for (uint32_t i = 0; i < result->handedness_count; ++i) {
    CppCloseCategories(&result->handedness[i]);
  }
  delete[] result->handedness;

  for (uint32_t i = 0; i < result->hand_landmarks_count; ++i) {
    CppCloseNormalizedLandmarks(&result->hand_landmarks[i]);
  }
  delete[] result->hand_landmarks;

  for (uint32_t i = 0; i < result->hand_world_landmarks_count; ++i) {
    CppCloseLandmarks(&result->hand_world_landmarks[i]);
  }
  delete[] result->hand_world_landmarks;

  result->handedness = nullptr;
  result->hand_landmarks = nullptr;
  result->hand_world_landmarks = nullptr;

  result->handedness_count = 0;
  result->hand_landmarks_count = 0;
  result->hand_world_landmarks_count = 0;

  CloseStages(result->stages);
  result->stages = nullptr;
  CloseTensorCaptures(result->tensor_capture);
  result->tensor_capture = nullptr;
  ClosePalmCandidates(result->palm_candidates);
  result->palm_candidates = nullptr;
  ClosePalmSearch(result->palm_search);
  result->palm_search = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
