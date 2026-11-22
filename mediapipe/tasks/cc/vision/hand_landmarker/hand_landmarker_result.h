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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/tensor_capture.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// Per-stage outputs of the hand pipeline for one frame, filled only when
// `HandLandmarkerOptions::debug_stage_outputs` is set. Every field is the
// content of one internal graph stream, copied unchanged; a vector is empty
// when its stream carried no packet on this frame (or an empty vector).
//
// All rects, detections and landmarks are in normalized coordinates of the
// input image as passed in, whatever `rotation_degrees` the image processing
// options carry (the same frame as the result landmarks). A rect's `rotation`
// is in radians, clockwise, in that frame as well.
struct HandLandmarkerStages {
  // Timestamp (microseconds) of the graph output this record was read from.
  // In video and live stream mode it equals the input timestamp.
  int64_t timestamp_us = 0;

  // Stream mode only. Rects carried in from the previous frame: the previous
  // frame's `hand_rects_next_frame`, or its `keep_alive_rects` when
  // `tracking_grace_frames` > 0.
  std::vector<mediapipe::NormalizedRect> previous_rects;

  // Whether the palm detector ran on this frame. In stream mode it is skipped
  // when `previous_rects` holds at least `num_hands` rects; in image mode it
  // always runs.
  bool palm_detector_ran = false;

  // Palm detector outputs: detections after non-max suppression, the palm
  // rects derived from them, and those rects expanded to whole-hand rects
  // (clipped to `num_hands`). All empty when the detector did not run. When it
  // ran and found no palm, `palm_detections` is empty and the two rect vectors
  // each hold one all-zero rect (DetectionsToRectsCalculator's
  // `output_zero_rect_for_empty_detections`).
  std::vector<mediapipe::Detection> palm_detections;
  std::vector<mediapipe::NormalizedRect> palm_rects;
  std::vector<mediapipe::NormalizedRect> detector_hand_rects;

  // Rects handed to the landmark model, in order: `previous_rects` followed
  // by the non-overlapping `detector_hand_rects` (HandAssociationCalculator),
  // clipped to `num_hands`. In image mode these are the `detector_hand_rects`.
  // HandAssociationCalculator gives every rect a fresh `rect_id` on every
  // frame (the rects carried over from the previous frame have none), so the
  // id does not identify a hand across frames; image mode sets no id.
  std::vector<mediapipe::NormalizedRect> landmark_rois;

  // The landmark model's hand presence score, and whether it passed
  // `min_hand_presence_confidence`, per `landmark_rois` entry in the same
  // order. The model skips an all-zero rect, which has no entry here; such a
  // rect is always the last one.
  std::vector<float> presence_scores;
  std::vector<bool> presence_passed;

  // Per ROI that passed presence, in `landmark_rois` order: the landmarks and
  // the rect predicted for the next frame, before
  // HandLandmarksDeduplicationCalculator.
  std::vector<components::containers::NormalizedLandmarks>
      landmarks_before_dedup;
  std::vector<mediapipe::NormalizedRect> hand_rects_before_dedup;

  // Next-frame rects after deduplication, one per hand in the result (the
  // landmarks after deduplication are `HandLandmarkerResult::hand_landmarks`).
  std::vector<mediapipe::NormalizedRect> hand_rects_next_frame;

  // Stream mode with `tracking_grace_frames` > 0 only. Output of
  // HandRectsKeepAliveCalculator: the rects handed to the next frame
  // (`hand_rects_next_frame` first, held rects appended) and, per rect, 0 if
  // it is live or the number of consecutive frames it has been missing.
  std::vector<mediapipe::NormalizedRect> keep_alive_rects;
  std::vector<int> keep_alive_miss_counts;
};

// CPU copies of the tensors given to the two models on one frame. Set on a
// result only for a frame the caller asked to capture (and only when
// `HandLandmarkerOptions::debug_tensor_capture` is set).
//
// A tensor is what the inference calculator receives: the image region after
// crop, rotation, resize and value normalization.
struct HandLandmarkerTensorCaptures {
  // Timestamp (microseconds) of the graph output this record was read from.
  int64_t timestamp_us = 0;

  // Input of the palm detection model. `has_tensor` is false when the palm
  // detector did not run on this frame.
  TensorCapture palm_detector_input;

  // Input of the landmark model, one entry per rect it is given, in that
  // order (`HandLandmarkerStages::landmark_rois`). `has_tensor` is false for a
  // rect the model was not run on (an all-zero rect).
  std::vector<TensorCapture> landmark_inputs;
};

// What became of the palm detector's candidates on one frame. Set on every
// result when `HandLandmarkerOptions::debug_palm_candidates` is positive.
struct HandLandmarkerPalmCandidates {
  // Timestamp (microseconds) of the graph output this record was read from.
  int64_t timestamp_us = 0;

  // Whether the palm detector ran on this frame. The other fields are empty
  // or 0 when it did not.
  bool palm_detector_ran = false;

  // The `debug_palm_candidates` highest scored boxes decoded from the model
  // output, highest first, whatever their score and before non-max
  // suppression. Same coordinates as `HandLandmarkerStages::palm_detections`.
  std::vector<mediapipe::Detection> candidates;

  // Number of decoded boxes whose score passed
  // `min_hand_detection_confidence`: the input of non-max suppression.
  int above_threshold_count = 0;

  // Number of detections non-max suppression produced from them. The first
  // `num_hands` of these become hand rects.
  int after_suppression_count = 0;
};

// The hand landmarks detection result from HandLandmarker, where each vector
// element represents a single hand detected in the image.
struct HandLandmarkerResult {
  // Classification of handedness.
  // Note: Left and Right handedness correspond to the subject's hands, not the
  // viewer. Passing a horizontally mirrored image will invert these outputs.
  std::vector<components::containers::Classifications> handedness;
  // Detected hand landmarks in normalized image coordinates.
  std::vector<components::containers::NormalizedLandmarks> hand_landmarks;
  // Detected hand landmarks in world coordinates.
  std::vector<components::containers::Landmarks> hand_world_landmarks;
  // Per-stage debug outputs. Set only when
  // `HandLandmarkerOptions::debug_stage_outputs` is true.
  std::optional<HandLandmarkerStages> stages;
  // Model input tensors. Set only on a frame the caller asked to capture.
  std::optional<HandLandmarkerTensorCaptures> tensor_capture;
  // Palm candidate diagnostics. Set only when
  // `HandLandmarkerOptions::debug_palm_candidates` is positive.
  std::optional<HandLandmarkerPalmCandidates> palm_candidates;
};

HandLandmarkerResult ConvertToHandLandmarkerResult(
    const std::vector<mediapipe::ClassificationList>& handedness_proto,
    const std::vector<mediapipe::NormalizedLandmarkList>& hand_landmarks_proto,
    const std::vector<mediapipe::LandmarkList>& hand_world_landmarks_proto);

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_
