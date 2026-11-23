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

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

void InitHandLandmarkerResult(
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult*
        cpp_result) {
  // Initialize handedness
  mediapipe::tasks::components::containers::Category cpp_category = {
      /* index= */ 1,
      /* score= */ 0.8f,
      /* category_name= */ "handeness_label_1",
      /* display_name= */ "handeness_display_name_1"};
  mediapipe::tasks::components::containers::Classifications
      classifications_for_handedness;
  classifications_for_handedness.categories.push_back(cpp_category);
  cpp_result->handedness.push_back(classifications_for_handedness);

  // Initialize hand_landmarks
  mediapipe::tasks::components::containers::NormalizedLandmark
      cpp_normalized_landmark = {/* x= */ 0.1f,
                                 /* y= */ 0.2f,
                                 /* z= */ 0.3f};
  mediapipe::tasks::components::containers::NormalizedLandmarks
      cpp_normalized_landmarks;
  cpp_normalized_landmarks.landmarks.push_back(cpp_normalized_landmark);
  cpp_result->hand_landmarks.push_back(cpp_normalized_landmarks);

  // Initialize hand_world_landmarks
  mediapipe::tasks::components::containers::Landmark cpp_landmark = {
      /* x= */ 1.0f,
      /* y= */ 1.1f,
      /* z= */ 1.2f};
  mediapipe::tasks::components::containers::Landmarks cpp_landmarks;
  cpp_landmarks.landmarks.push_back(cpp_landmark);
  cpp_result->hand_world_landmarks.push_back(cpp_landmarks);
}

TEST(HandLandmarkerResultConverterTest, ConvertsCustomResult) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  InitHandLandmarkerResult(&cpp_result);

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  // Verify conversion of hand_landmarks
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_landmarks_count, cpp_result.hand_landmarks.size());

  for (uint32_t i = 0; i < c_result.hand_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_landmarks[i].landmarks_count,
              cpp_result.hand_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.hand_landmarks[i].landmarks_count; ++j) {
      const auto& landmark = cpp_result.hand_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].x, landmark.x);
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].y, landmark.y);
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].z, landmark.z);
    }
  }

  // Verify conversion of hand_world_landmarks
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks_count,
            cpp_result.hand_world_landmarks.size());

  for (uint32_t i = 0; i < c_result.hand_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_world_landmarks[i].landmarks_count,
              cpp_result.hand_world_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.hand_world_landmarks[i].landmarks_count;
         ++j) {
      const auto& landmark = cpp_result.hand_world_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].x,
                      landmark.x);
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].y,
                      landmark.y);
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].z,
                      landmark.z);
    }
  }

  CppCloseHandLandmarkerResult(&c_result);
}

TEST(HandLandmarkerResultConverterTest, FreesMemory) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  InitHandLandmarkerResult(&cpp_result);

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.handedness, nullptr);
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);

  CppCloseHandLandmarkerResult(&c_result);

  EXPECT_EQ(c_result.handedness, nullptr);
  EXPECT_EQ(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks, nullptr);
}

TEST(HandLandmarkerResultConverterTest, StagesAreNullWhenAbsent) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  InitHandLandmarkerResult(&cpp_result);

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);
  EXPECT_EQ(c_result.stages, nullptr);
  EXPECT_EQ(c_result.tensor_capture, nullptr);
  EXPECT_EQ(c_result.palm_candidates, nullptr);
  EXPECT_EQ(c_result.palm_search, nullptr);
  CppCloseHandLandmarkerResult(&c_result);
}

TEST(HandLandmarkerResultConverterTest, ConvertsAndFreesTensorCapture) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  auto& capture = cpp_result.tensor_capture.emplace();
  capture.timestamp_us = 66000;
  // The palm detector was skipped; the landmark model ran on the first of two
  // rects.
  auto& ran = capture.landmark_inputs.emplace_back();
  ran.has_tensor = true;
  ran.element_type = mediapipe::Tensor::ElementType::kFloat32;
  ran.shape = {1, 1, 2, 3};
  const std::vector<float> values = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, -1.0f};
  auto bytes = std::make_shared<std::vector<uint8_t>>(values.size() *
                                                      sizeof(float));
  std::memcpy(bytes->data(), values.data(), bytes->size());
  ran.data = bytes;
  capture.landmark_inputs.emplace_back();

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  ASSERT_NE(c_result.tensor_capture, nullptr);
  const MpHandLandmarkerTensorCapture& c_capture = *c_result.tensor_capture;
  EXPECT_EQ(c_capture.timestamp_us, 66000);
  EXPECT_FALSE(c_capture.palm_detector_input.has_tensor);
  EXPECT_EQ(c_capture.palm_detector_input.data, nullptr);
  EXPECT_EQ(c_capture.palm_detector_input.data_size, 0);
  ASSERT_EQ(c_capture.landmark_inputs_count, 2);
  const MpTensorCapture& c_ran = c_capture.landmark_inputs[0];
  EXPECT_TRUE(c_ran.has_tensor);
  EXPECT_EQ(c_ran.element_type, MP_TENSOR_ELEMENT_FLOAT32);
  ASSERT_EQ(c_ran.shape_count, 4);
  EXPECT_EQ(c_ran.shape[2], 2);
  EXPECT_EQ(c_ran.shape[3], 3);
  ASSERT_EQ(c_ran.data_size, values.size() * sizeof(float));
  EXPECT_EQ(std::memcmp(c_ran.data, values.data(), c_ran.data_size), 0);
  EXPECT_FALSE(c_capture.landmark_inputs[1].has_tensor);

  CppCloseHandLandmarkerResult(&c_result);
  EXPECT_EQ(c_result.tensor_capture, nullptr);
}

TEST(HandLandmarkerResultConverterTest, ConvertsAndFreesPalmCandidates) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  auto& candidates = cpp_result.palm_candidates.emplace();
  candidates.timestamp_us = 99000;
  candidates.palm_detector_ran = true;
  mediapipe::Detection detection;
  detection.add_score(0.31f);
  auto* box = detection.mutable_location_data()->mutable_relative_bounding_box();
  box->set_xmin(0.1f);
  box->set_width(0.2f);
  auto* keypoint = detection.mutable_location_data()->add_relative_keypoints();
  keypoint->set_x(0.4f);
  keypoint->set_y(0.6f);
  candidates.candidates = {detection, detection};
  candidates.above_threshold_count = 7;
  candidates.after_suppression_count = 3;

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  ASSERT_NE(c_result.palm_candidates, nullptr);
  const MpHandLandmarkerPalmCandidates& c_candidates = *c_result.palm_candidates;
  EXPECT_EQ(c_candidates.timestamp_us, 99000);
  EXPECT_TRUE(c_candidates.palm_detector_ran);
  ASSERT_EQ(c_candidates.candidates_count, 2);
  EXPECT_FLOAT_EQ(c_candidates.candidates[1].score, 0.31f);
  EXPECT_FLOAT_EQ(c_candidates.candidates[1].width, 0.2f);
  ASSERT_EQ(c_candidates.candidates[1].keypoints_count, 1);
  EXPECT_FLOAT_EQ(c_candidates.candidates[1].keypoints[0].y, 0.6f);
  EXPECT_EQ(c_candidates.above_threshold_count, 7);
  EXPECT_EQ(c_candidates.after_suppression_count, 3);

  CppCloseHandLandmarkerResult(&c_result);
  EXPECT_EQ(c_result.palm_candidates, nullptr);
}

TEST(HandLandmarkerResultConverterTest, ConvertsAndFreesPalmSearch) {
  using namespace ::mediapipe::tasks::vision::hand_landmarker;  // NOLINT
  HandLandmarkerResult cpp_result;
  auto& search = cpp_result.palm_search.emplace();
  search.timestamp_us = 33000;
  search.frame_sequence = 7;
  search.palm_detector_ran = true;
  search.call_start_ns = 100;
  search.call_end_ns = 900;
  search.landmark_rois_ready_ns = 600;
  search.landmarks_done_ns = 800;
  search.record.schedule.detector_frame_index = 4;
  search.record.schedule.grid_tile_count = 4;
  search.record.schedule.tiles_per_frame = 1;
  search.record.schedule.cursor_before = 2;
  search.record.schedule.cursor_after = 3;
  search.record.regions_ready_ns = 200;
  search.record.merge_done_ns = 500;

  mediapipe::Detection detection;
  detection.add_score(0.75f);
  auto* box = detection.mutable_location_data()->mutable_relative_bounding_box();
  box->set_xmin(0.1f);
  box->set_ymin(0.2f);
  box->set_width(0.3f);
  box->set_height(0.4f);
  auto* keypoint = detection.mutable_location_data()->add_relative_keypoints();
  keypoint->set_x(0.25f);
  keypoint->set_y(0.35f);

  search.record.regions.resize(2);
  search.record.regions[0].region.id = 0;
  search.record.regions[1].region.id = 3;
  search.record.regions[1].region.kind = PalmSearchRegionKind::kGrid;
  search.record.regions[1].region.grid_row = 1;
  search.record.regions[1].region.grid_column = 0;
  search.record.regions[1].region.requested_width = 0.5f;
  search.record.regions[1].region.sampled_height = 0.9f;
  search.record.regions[1].detections.resize(2);
  search.record.regions[1].detections[0].detection = detection;
  search.record.regions[1].detections[0].merged_index = 0;
  search.record.regions[1].detections[1].detection = detection;
  search.record.regions[1].detections[1].outcome =
      PalmSearchDetectionOutcome::kRejectedTruncated;
  search.record.regions[1].candidates = {detection};
  search.record.regions[1].above_threshold_count = 5;
  search.record.regions[1].crop_start_ns = 210;
  search.record.regions[1].detections_ready_ns = 400;

  search.record.merged.resize(1);
  search.record.merged[0].detection = detection;
  search.record.merged[0].sources = {{3, 0}};
  search.merged_outcomes = {PalmSearchHandOutcome::kOverlapsTrackedRect};
  search.merged_overlapped_rect_index = {0};
  search.merged_landmark_roi_index = {-1};

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);
  ASSERT_NE(c_result.palm_search, nullptr);
  const MpHandLandmarkerPalmSearch& c_search = *c_result.palm_search;
  EXPECT_EQ(c_search.timestamp_us, 33000);
  EXPECT_EQ(c_search.frame_sequence, 7);
  EXPECT_TRUE(c_search.palm_detector_ran);
  EXPECT_EQ(c_search.detector_frame_index, 4);
  EXPECT_EQ(c_search.grid_tile_count, 4);
  EXPECT_EQ(c_search.tiles_per_frame, 1);
  EXPECT_EQ(c_search.cursor_before, 2);
  EXPECT_EQ(c_search.cursor_after, 3);
  EXPECT_EQ(c_search.call_start_ns, 100);
  EXPECT_EQ(c_search.call_end_ns, 900);
  EXPECT_EQ(c_search.regions_ready_ns, 200);
  EXPECT_EQ(c_search.merge_done_ns, 500);
  EXPECT_EQ(c_search.landmark_rois_ready_ns, 600);
  EXPECT_EQ(c_search.landmarks_done_ns, 800);
  ASSERT_EQ(c_search.regions_count, 2);
  EXPECT_EQ(c_search.regions[0].detections, nullptr);
  EXPECT_EQ(c_search.regions[0].candidates, nullptr);
  const MpPalmSearchRegion& c_region = c_search.regions[1];
  EXPECT_EQ(c_region.id, 3);
  EXPECT_EQ(c_region.kind, 1);
  EXPECT_EQ(c_region.grid_row, 1);
  EXPECT_EQ(c_region.grid_column, 0);
  EXPECT_FLOAT_EQ(c_region.requested_width, 0.5f);
  EXPECT_FLOAT_EQ(c_region.sampled_height, 0.9f);
  ASSERT_EQ(c_region.detections_count, 2);
  EXPECT_FLOAT_EQ(c_region.detections[0].detection.score, 0.75f);
  EXPECT_FLOAT_EQ(c_region.detections[0].detection.x_min, 0.1f);
  ASSERT_EQ(c_region.detections[0].detection.keypoints_count, 1);
  EXPECT_FLOAT_EQ(c_region.detections[0].detection.keypoints[0].y, 0.35f);
  EXPECT_EQ(c_region.detections[0].outcome, 0);
  EXPECT_EQ(c_region.detections[0].merged_index, 0);
  EXPECT_EQ(c_region.detections[1].outcome, 1);
  EXPECT_EQ(c_region.detections[1].merged_index, -1);
  EXPECT_EQ(c_region.candidates_count, 1);
  EXPECT_EQ(c_region.above_threshold_count, 5);
  EXPECT_EQ(c_region.crop_start_ns, 210);
  EXPECT_EQ(c_region.detections_ready_ns, 400);
  ASSERT_EQ(c_search.merged_count, 1);
  EXPECT_FLOAT_EQ(c_search.merged[0].detection.height, 0.4f);
  ASSERT_EQ(c_search.merged[0].sources_count, 1);
  EXPECT_EQ(c_search.merged[0].sources[0].region_id, 3);
  EXPECT_EQ(c_search.merged[0].sources[0].detection_index, 0);
  EXPECT_EQ(c_search.merged[0].outcome, 1);
  EXPECT_EQ(c_search.merged[0].overlapped_rect_index, 0);
  EXPECT_EQ(c_search.merged[0].landmark_roi_index, -1);

  CppCloseHandLandmarkerResult(&c_result);
  EXPECT_EQ(c_result.palm_search, nullptr);
}

TEST(HandLandmarkerResultConverterTest, ConvertsAndFreesStages) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  auto& stages = cpp_result.stages.emplace();
  stages.timestamp_us = 33000;
  stages.palm_detector_ran = true;

  mediapipe::NormalizedRect rect;
  rect.set_x_center(0.25f);
  rect.set_y_center(0.5f);
  rect.set_width(0.2f);
  rect.set_height(0.3f);
  rect.set_rotation(1.5f);
  rect.set_rect_id(7);
  stages.landmark_rois = {rect, mediapipe::NormalizedRect()};

  mediapipe::Detection detection;
  detection.add_score(0.9f);
  auto* location = detection.mutable_location_data();
  location->mutable_relative_bounding_box()->set_xmin(0.1f);
  location->mutable_relative_bounding_box()->set_ymin(0.2f);
  location->mutable_relative_bounding_box()->set_width(0.3f);
  location->mutable_relative_bounding_box()->set_height(0.4f);
  auto* keypoint = location->add_relative_keypoints();
  keypoint->set_x(0.6f);
  keypoint->set_y(0.7f);
  stages.palm_detections = {detection};

  stages.presence_scores = {0.8f, 0.1f};
  stages.presence_passed = {true, false};
  ::mediapipe::tasks::components::containers::NormalizedLandmarks landmarks;
  landmarks.landmarks.push_back({0.1f, 0.2f, 0.3f});
  stages.landmarks_before_dedup = {landmarks};
  stages.keep_alive_rects = {rect};
  stages.keep_alive_miss_counts = {2};

  MpHandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  // An empty result still carries the stages.
  EXPECT_EQ(c_result.hand_landmarks_count, 0);
  ASSERT_NE(c_result.stages, nullptr);
  const MpHandLandmarkerStages& c_stages = *c_result.stages;
  EXPECT_EQ(c_stages.timestamp_us, 33000);
  EXPECT_TRUE(c_stages.palm_detector_ran);

  ASSERT_EQ(c_stages.landmark_rois_count, 2);
  EXPECT_FLOAT_EQ(c_stages.landmark_rois[0].x_center, 0.25f);
  EXPECT_FLOAT_EQ(c_stages.landmark_rois[0].y_center, 0.5f);
  EXPECT_FLOAT_EQ(c_stages.landmark_rois[0].width, 0.2f);
  EXPECT_FLOAT_EQ(c_stages.landmark_rois[0].height, 0.3f);
  EXPECT_FLOAT_EQ(c_stages.landmark_rois[0].rotation, 1.5f);
  EXPECT_TRUE(c_stages.landmark_rois[0].has_rect_id);
  EXPECT_EQ(c_stages.landmark_rois[0].rect_id, 7);
  EXPECT_FALSE(c_stages.landmark_rois[1].has_rect_id);

  ASSERT_EQ(c_stages.palm_detections_count, 1);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].score, 0.9f);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].x_min, 0.1f);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].y_min, 0.2f);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].width, 0.3f);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].height, 0.4f);
  ASSERT_EQ(c_stages.palm_detections[0].keypoints_count, 1);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].keypoints[0].x, 0.6f);
  EXPECT_FLOAT_EQ(c_stages.palm_detections[0].keypoints[0].y, 0.7f);

  ASSERT_EQ(c_stages.presence_scores_count, 2);
  EXPECT_FLOAT_EQ(c_stages.presence_scores[0], 0.8f);
  EXPECT_FLOAT_EQ(c_stages.presence_scores[1], 0.1f);
  ASSERT_EQ(c_stages.presence_passed_count, 2);
  EXPECT_TRUE(c_stages.presence_passed[0]);
  EXPECT_FALSE(c_stages.presence_passed[1]);

  ASSERT_EQ(c_stages.landmarks_before_dedup_count, 1);
  ASSERT_EQ(c_stages.landmarks_before_dedup[0].landmarks_count, 1);
  EXPECT_FLOAT_EQ(c_stages.landmarks_before_dedup[0].landmarks[0].y, 0.2f);

  ASSERT_EQ(c_stages.keep_alive_rects_count, 1);
  ASSERT_EQ(c_stages.keep_alive_miss_counts_count, 1);
  EXPECT_EQ(c_stages.keep_alive_miss_counts[0], 2);

  // Streams that produced nothing convert to empty arrays.
  EXPECT_EQ(c_stages.previous_rects_count, 0);
  EXPECT_EQ(c_stages.previous_rects, nullptr);
  EXPECT_EQ(c_stages.palm_rects_count, 0);

  CppCloseHandLandmarkerResult(&c_result);
  EXPECT_EQ(c_result.stages, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
