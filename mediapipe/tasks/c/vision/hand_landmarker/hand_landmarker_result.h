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

#ifndef MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_RESULT_HAND_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_RESULT_HAND_LANDMARKER_RESULT_H_

#include <stdbool.h>
#include <stdint.h>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"

#ifdef __cplusplus
extern "C" {
#endif

// A rotated rectangle in normalized image coordinates (the C form of
// mediapipe.NormalizedRect).
struct MpNormalizedRotatedRect {
  // Center of the rectangle, normalized by the image width / height.
  float x_center;
  float y_center;

  // Size of the rectangle, normalized by the image width / height.
  float width;
  float height;

  // Rotation around the center, in radians, clockwise.
  float rotation;

  // Whether `rect_id` is set.
  bool has_rect_id;

  // Id assigned to the rectangle by the tracking loop. Valid only if
  // `has_rect_id` is true. Reassigned on every frame: it does not identify a
  // hand across frames.
  int64_t rect_id;
};

// A point in normalized image coordinates.
struct MpNormalizedPoint {
  float x;
  float y;
};

// A detection with a float bounding box in normalized image coordinates (the
// C form of a mediapipe.Detection that carries a relative bounding box).
struct MpNormalizedDetection {
  // Score of the detection. 0 if the detection has no score.
  float score;

  // Bounding box, normalized by the image width / height.
  float x_min;
  float y_min;
  float width;
  float height;

  // Keypoints of the detection in normalized image coordinates.
  struct MpNormalizedPoint* keypoints;

  // The number of elements in the keypoints array.
  uint32_t keypoints_count;
};

// Per-stage outputs of the hand pipeline for one frame. Present in a result
// only when `MpHandLandmarkerOptions::debug_stage_outputs` is true.
//
// Every array is the content of one internal graph stream, copied unchanged;
// its count is 0 when that stream produced nothing on this frame. Rects,
// detections and landmarks are in normalized coordinates of the input image as
// passed in, whatever `rotation_degrees` the image processing options carry
// (the same frame as the result landmarks); so is a rect's `rotation`.
typedef struct MpHandLandmarkerStages {
  // Timestamp (microseconds) of the graph output this record was read from.
  // In video and live stream mode it equals the input timestamp.
  int64_t timestamp_us;

  // Whether the palm detector ran on this frame.
  bool palm_detector_ran;

  // Video / live stream mode only. Rects carried in from the previous frame,
  // including rects held by the tracking grace.
  struct MpNormalizedRotatedRect* previous_rects;
  uint32_t previous_rects_count;

  // Palm detections after non-max suppression.
  struct MpNormalizedDetection* palm_detections;
  uint32_t palm_detections_count;

  // Palm rects derived from the palm detections. When the detector ran and
  // found no palm, this and `detector_hand_rects` each hold one all-zero rect.
  struct MpNormalizedRotatedRect* palm_rects;
  uint32_t palm_rects_count;

  // Palm rects expanded to whole-hand rects, clipped to `num_hands`.
  struct MpNormalizedRotatedRect* detector_hand_rects;
  uint32_t detector_hand_rects_count;

  // Rects handed to the landmark model: previous rects followed by the
  // non-overlapping detector hand rects, clipped to `num_hands`.
  struct MpNormalizedRotatedRect* landmark_rois;
  uint32_t landmark_rois_count;

  // Hand presence score per `landmark_rois` entry, in the same order. The
  // landmark model skips an all-zero rect, which has no entry here or in
  // `presence_passed`; such a rect is always the last one.
  float* presence_scores;
  uint32_t presence_scores_count;

  // Whether the presence score passed `min_hand_presence_confidence`, per
  // `landmark_rois` entry, in the same order.
  bool* presence_passed;
  uint32_t presence_passed_count;

  // Landmarks per ROI that passed presence, before deduplication.
  struct MpNormalizedLandmarks* landmarks_before_dedup;
  uint32_t landmarks_before_dedup_count;

  // Next-frame rect per ROI that passed presence, before deduplication.
  struct MpNormalizedRotatedRect* hand_rects_before_dedup;
  uint32_t hand_rects_before_dedup_count;

  // Next-frame rects after deduplication, one per hand in the result.
  struct MpNormalizedRotatedRect* hand_rects_next_frame;
  uint32_t hand_rects_next_frame_count;

  // Video / live stream mode with `tracking_grace_frames` > 0 only. The rects
  // handed to the next frame: `hand_rects_next_frame` first, held rects
  // appended.
  struct MpNormalizedRotatedRect* keep_alive_rects;
  uint32_t keep_alive_rects_count;

  // Per `keep_alive_rects` entry: 0 if the rect is live, otherwise the number
  // of consecutive frames it has been missing.
  int32_t* keep_alive_miss_counts;
  uint32_t keep_alive_miss_counts_count;
} MpHandLandmarkerStages;

// Element type of a captured tensor (the values of mediapipe::Tensor's
// ElementType).
enum MpTensorElementType {
  MP_TENSOR_ELEMENT_NONE = 0,
  MP_TENSOR_ELEMENT_FLOAT16 = 1,
  MP_TENSOR_ELEMENT_FLOAT32 = 2,
  MP_TENSOR_ELEMENT_UINT8 = 3,
  MP_TENSOR_ELEMENT_INT8 = 4,
  MP_TENSOR_ELEMENT_INT32 = 5,
  MP_TENSOR_ELEMENT_INT64 = 6,
  MP_TENSOR_ELEMENT_CHAR = 7,
  MP_TENSOR_ELEMENT_BOOL = 8,
};

// A CPU copy of the input tensor of one model invocation.
struct MpTensorCapture {
  // False when no tensor reached the model, i.e. inference was skipped. The
  // remaining fields are then 0 / null.
  bool has_tensor;

  // One of MpTensorElementType.
  int32_t element_type;

  // Dimensions of the tensor.
  int32_t* shape;
  uint32_t shape_count;

  // The tensor's bytes: the product of `shape` elements of `element_type`, in
  // row-major order.
  uint8_t* data;
  uint64_t data_size;
};

// The tensors given to the two models on one frame. Present in a result only
// for a detect call that set `MpImageProcessingOptions::capture_tensors` on a
// hand landmarker created with `debug_tensor_capture`. A tensor is what the
// inference step receives: the image region after crop, rotation, resize and
// value normalization.
typedef struct MpHandLandmarkerTensorCapture {
  // Timestamp (microseconds) of the graph output this record was read from.
  int64_t timestamp_us;

  // Input of the palm detection model. `has_tensor` is false when the palm
  // detector did not run on this frame.
  struct MpTensorCapture palm_detector_input;

  // Input of the landmark model, one entry per rect it is given, in that order
  // (`MpHandLandmarkerStages::landmark_rois`). `has_tensor` is false for a
  // rect the model was not run on (an all-zero rect).
  struct MpTensorCapture* landmark_inputs;
  uint32_t landmark_inputs_count;
} MpHandLandmarkerTensorCapture;

// What became of the palm detector's candidates on one frame. Present in every
// result of a hand landmarker created with `debug_palm_candidates` > 0.
typedef struct MpHandLandmarkerPalmCandidates {
  // Timestamp (microseconds) of the graph output this record was read from.
  int64_t timestamp_us;

  // Whether the palm detector ran on this frame. The other fields are 0 /
  // null when it did not.
  bool palm_detector_ran;

  // The `debug_palm_candidates` highest scored boxes decoded from the model
  // output, highest first, whatever their score and before non-max
  // suppression. Same coordinates as `MpHandLandmarkerStages::palm_detections`.
  struct MpNormalizedDetection* candidates;
  uint32_t candidates_count;

  // Number of decoded boxes whose score passed
  // `min_hand_detection_confidence`: the input of non-max suppression.
  uint32_t above_threshold_count;

  // Number of detections non-max suppression produced from them. The first
  // `num_hands` of these become hand rects.
  uint32_t after_suppression_count;
} MpHandLandmarkerPalmCandidates;

// The hand landmarker result from HandLandmarker, where each vector
// element represents a single hand detected in the image.
typedef struct MpHandLandmarkerResult {
  // Classification of handedness.
  // Note: Left and Right handedness correspond to the subject's hands, not the
  // viewer. Passing a horizontally mirrored image will invert these outputs.
  struct MpCategories* handedness;

  // The number of elements in the handedness array.
  uint32_t handedness_count;

  // Detected hand landmarks in normalized image coordinates.
  struct MpNormalizedLandmarks* hand_landmarks;

  // The number of elements in the hand_landmarks array.
  uint32_t hand_landmarks_count;

  // Detected hand landmarks in world coordinates.
  struct MpLandmarks* hand_world_landmarks;

  // The number of elements in the hand_world_landmarks array.
  uint32_t hand_world_landmarks_count;

  // Per-stage debug outputs. Null unless the hand landmarker was created with
  // `debug_stage_outputs`. Owned by the result and freed with it.
  struct MpHandLandmarkerStages* stages;

  // Model input tensors. Null unless this detect call asked for a capture (see
  // MpHandLandmarkerTensorCapture). Owned by the result and freed with it.
  struct MpHandLandmarkerTensorCapture* tensor_capture;

  // Palm candidate diagnostics. Null unless the hand landmarker was created
  // with `debug_palm_candidates` > 0. Owned by the result and freed with it.
  struct MpHandLandmarkerPalmCandidates* palm_candidates;
} MpHandLandmarkerResult;

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_RESULT_HAND_LANDMARKER_RESULT_H_
