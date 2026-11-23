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

// One detection of one palm search region.
struct MpPalmSearchRegionDetection {
  // As it left the region's own non-max suppression, in frame coordinates.
  struct MpNormalizedDetection detection;
  // 0: went into the merge. 1: dropped by `reject_truncated_detections`.
  // 2: dropped by `max_palm_fraction`.
  int outcome;
  // Index into `MpHandLandmarkerPalmSearch::merged` of the detection it
  // became or was folded into; -1 when it was dropped.
  int merged_index;
};

// One region the palm model was run on, and what it produced.
struct MpPalmSearchRegion {
  // 0: the full frame. 1 + row * columns + column: a grid tile. 1000 + i: the
  // i-th external region of the call.
  int id;
  // 0: full frame, 1: grid tile, 2: external region.
  int kind;
  // Grid tiles only, otherwise -1.
  int grid_row;
  int grid_column;
  // The region as asked for: an axis-aligned box of the frame.
  float requested_x_center;
  float requested_y_center;
  float requested_width;
  float requested_height;
  // The window the model input was sampled from: the requested box grown
  // around its center to a square in pixels. Not clipped to the frame; the
  // part outside the frame is black.
  float sampled_x_center;
  float sampled_y_center;
  float sampled_width;
  float sampled_height;

  struct MpPalmSearchRegionDetection* detections;
  uint32_t detections_count;

  // Only with `debug_palm_candidates` > 0: the highest scored decoded boxes
  // of this region whatever their score, and the number of decoded boxes
  // that passed the score threshold.
  struct MpNormalizedDetection* candidates;
  uint32_t candidates_count;
  uint32_t above_threshold_count;

  // Monotonic clock, nanoseconds: when the region entered the loop and when
  // its detections left it. Iterations can overlap in time.
  int64_t crop_start_ns;
  int64_t detections_ready_ns;
};

// Where a merged detection came from.
struct MpPalmSearchSource {
  int region_id;
  // Index into that region's `detections`.
  int detection_index;
};

struct MpPalmSearchMergedDetection {
  struct MpNormalizedDetection detection;
  // Highest scored first.
  struct MpPalmSearchSource* sources;
  uint32_t sources_count;
  // What became of its hand rect. 0: given to the landmark model. 1: dropped
  // by the association, overlapped a rect carried over from the previous
  // frame. 2: dropped by the association, overlapped the hand rect of an
  // earlier detection of this frame. 3: cut off by the clip to `num_hands`
  // after the association. 4: cut off by the clip to `num_hands` before the
  // association (`clip_before_association`).
  int outcome;
  // For outcomes 1 and 2: index of the overlapped rect in the association's
  // output (the previous frame's rects first, then the kept detector rects).
  // -1 otherwise.
  int overlapped_rect_index;
  // For outcome 0: index into `MpHandLandmarkerStages::landmark_rois`. -1
  // otherwise.
  int landmark_roi_index;
};

// The palm search record of one frame. Present in every result of a hand
// landmarker created with `palm_search_enabled` and `debug_stage_outputs`.
// Raw values of one frame; all coordinates are normalized to the input image
// as passed in.
typedef struct MpHandLandmarkerPalmSearch {
  // Timestamp (microseconds) of the graph output this record was read from.
  int64_t timestamp_us;
  // Image and video mode: the number of detect calls on this landmarker
  // before this one. -1 in live stream mode.
  int64_t frame_sequence;

  // Whether the palm detector ran on this frame. In stream mode it is skipped
  // when the rects carried over from the previous frame already number
  // `num_hands`. The schedule, the detector's clock stamps, `regions` and
  // `merged` are 0 / null when it did not.
  bool palm_detector_ran;

  // Schedule state: the number of detector frames before this one, the
  // number of grid tiles, how many run per detector frame (0: all), and the
  // index of the first tile run on this detector frame and on the next.
  int64_t detector_frame_index;
  int grid_tile_count;
  int tiles_per_frame;
  int cursor_before;
  int cursor_after;

  // Monotonic clock, nanoseconds; 0 where not measured. When the detect call
  // was entered and when the graph had produced its outputs (image and video
  // mode), when the region list and the merged detections were emitted, when
  // the landmark model's rects were emitted and when its landmarks were
  // collected.
  int64_t call_start_ns;
  int64_t call_end_ns;
  int64_t regions_ready_ns;
  int64_t merge_done_ns;
  int64_t landmark_rois_ready_ns;
  int64_t landmarks_done_ns;

  // In the order the model was run on them.
  struct MpPalmSearchRegion* regions;
  uint32_t regions_count;

  // The merged detections, highest score first: the graph's palm detections.
  struct MpPalmSearchMergedDetection* merged;
  uint32_t merged_count;
} MpHandLandmarkerPalmSearch;

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

  // Palm search record. Null unless the hand landmarker was created with
  // `palm_search_enabled` and `debug_stage_outputs`. Owned by the result and
  // freed with it. Kept as the last field.
  struct MpHandLandmarkerPalmSearch* palm_search;
} MpHandLandmarkerResult;

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_RESULT_HAND_LANDMARKER_RESULT_H_
