/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// Fork addition. Configures the palm search regions path; see
// `HandLandmarkerOptions::palm_search`. The defaults are not a recommended
// configuration: they run the whole frame only.
struct PalmSearchOptions {
  // The whole frame is one of the regions on every detector frame.
  bool include_full_frame = true;

  // A grid of tiles over the frame, each of them a region. Off when rows or
  // columns is 0. `grid_overlap` is the fraction of a tile's width (height)
  // it shares with its horizontal (vertical) neighbour, in [0, 1).
  int grid_rows = 0;
  int grid_columns = 0;
  float grid_overlap = 0.0f;

  // Number of grid tiles run per detector frame, cycling through the grid. 0:
  // every tile on every detector frame. A detector frame is a frame on which
  // the palm detector runs at all; the cycle does not advance on others.
  int round_robin_tiles_per_frame = 0;

  // Allows the `search_regions` argument of the detect methods.
  bool accepts_external_regions = false;

  // Drops a detection whose box touches an edge of the window its region was
  // sampled from, unless that edge is a frame edge. Never applied to the
  // full-frame region.
  bool reject_truncated_detections = false;

  // Drops a detection of a grid tile or external region whose box is wider or
  // taller than this fraction of the frame. 0: off.
  float max_palm_fraction = 0.0f;

  // How detections of different regions that overlap are merged.
  enum class Merge {
    // Boxes and keypoints averaged by score, as in the detector's own
    // non-max suppression.
    kWeighted = 0,
    // The highest scored detection, unchanged.
    kBestScore = 1,
  };
  Merge merge = Merge::kWeighted;

  // Hands the detector's rects to the association with tracked hands already
  // clipped to `num_hands`, as the graph without search regions does. When
  // false the only clip to `num_hands` is the one after the association.
  bool clip_before_association = false;
};

struct HandLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks library, such as specifying
  // the TfLite model bundle file with metadata, accelerator options, op
  // resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // HandLandmarker has three running modes:
  // 1) The image mode for detecting hand landmarks on single image inputs.
  // 2) The video mode for detecting hand landmarks on the decoded frames of a
  //    video.
  // 3) The live stream mode for detecting hand landmarks on the live stream of
  //    input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The maximum number of hands can be detected by the HandLandmarker.
  int num_hands = 1;

  // The minimum confidence score for the hand detection to be considered
  // successful.
  float min_hand_detection_confidence = 0.5;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  float min_hand_presence_confidence = 0.5;

  // The minimum confidence score for the hand tracking to be considered
  // successful.
  float min_tracking_confidence = 0.5;

  // Scale factor for the frame-to-frame region of interest derived from the
  // previous frame's hand landmarks. Larger values expand the tracking crop so
  // a fast-moving hand stays inside it between frames. Default 2.0.
  float roi_scale = 2.0;

  // In VIDEO/LIVE_STREAM mode: number of consecutive frames a vanished hand's
  // ROI is kept in the tracking loop before its track is dropped. While held,
  // the palm detector stays skipped and the landmark model keeps retrying the
  // last-known region, so a momentarily-blurred hand is re-acquired quickly.
  // 0 (default) drops a hand immediately (original behavior).
  int tracking_grace_frames = 0;

  // Debugging aid. When true, every result carries `stages`: the per-stage
  // outputs of the hand pipeline for that frame (see HandLandmarkerStages).
  // When false (default) the graph and the results are unchanged.
  bool debug_stage_outputs = false;

  // Debugging aid. When true, the `capture_tensors` argument of the detect
  // methods may be set, and the result of such a call carries
  // `tensor_capture`: CPU copies of the tensors given to the palm detection
  // model and to the landmark model on that frame. A call without
  // `capture_tensors` does not read any tensor. When false (default) the graph
  // and the results are unchanged.
  bool debug_tensor_capture = false;

  // Debugging aid. When positive, every result carries `palm_candidates`,
  // with this many of the palm detector's highest scored candidates. 0
  // (default) leaves the graph and the results unchanged.
  int debug_palm_candidates = 0;

  // Palm search regions. When set, the palm detection model runs on a list of
  // regions of the frame instead of on the whole frame only, and the
  // detections of all regions are merged. With `debug_stage_outputs` every
  // result then also carries `palm_search`. When unset (default) the graph
  // and the results are unchanged.
  std::optional<PalmSearchOptions> palm_search;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<HandLandmarkerResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Performs hand landmarks detection on the given image.
//
// TODO  add the link to DevSite.
// This API expects a pre-trained hand landmarker model asset bundle.
//
// Inputs:
//   Image
//     - The image that hand landmarks detection runs on.
//   std::optional<NormalizedRect>
//     - If provided, can be used to specify the rotation to apply to the image
//       before performing hand landmarks detection, by setting its 'rotation'
//       field in radians (e.g. 'M_PI / 2' for a 90° anti-clockwise rotation).
//       Note that specifying a region-of-interest using the 'x_center',
//       'y_center', 'width' and 'height' fields is NOT supported and will
//       result in an invalid argument error being returned.
// Outputs:
//   HandLandmarkerResult
//     - The hand landmarks detection results.
class HandLandmarker : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a HandLandmarker from a HandLandmarkerOptions to process image data
  // or streaming data. Hand landmarker can be created with one of the following
  // three running modes:
  // 1) Image mode for detecting hand landmarks on single image inputs. Users
  //    provide mediapipe::Image to the `Detect` method, and will receive the
  //    detected hand landmarks results as the return value.
  // 2) Video mode for detecting hand landmarks on the decoded frames of a
  //    video. Users call `DetectForVideo` method, and will receive the detected
  //    hand landmarks results as the return value.
  // 3) Live stream mode for detecting hand landmarks on the live stream of the
  //    input data, such as from camera. Users call `DetectAsync` to push the
  //    image data into the HandLandmarker, the detected results along with the
  //    input timestamp and the image that hand landmarker runs on will be
  //    available in the result callback when the hand landmarker finishes the
  //    work.
  static absl::StatusOr<std::unique_ptr<HandLandmarker>> Create(
      std::unique_ptr<HandLandmarkerOptions> options);

  // Performs hand landmarks detection on the given image.
  // Only use this method when the HandLandmarker is created with the image
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: Describes how the input image will be preprocessed
  // after the yuv support is implemented.
  //
  // `capture_tensors` requires `debug_tensor_capture` and makes the result
  // carry `tensor_capture`; the same holds for the other detect methods.
  absl::StatusOr<HandLandmarkerResult> Detect(
      Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt,
      bool capture_tensors = false,
      const std::vector<components::containers::RectF>& search_regions = {});

  // Performs hand landmarks detection on the provided video frame.
  // Only use this method when the HandLandmarker is created with the video
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<HandLandmarkerResult> DetectForVideo(
      Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt,
      bool capture_tensors = false,
      const std::vector<components::containers::RectF>& search_regions = {});

  // Sends live image data to perform hand landmarks detection, and the results
  // will be available via the "result_callback" provided in the
  // HandLandmarkerOptions. Only use this method when the HandLandmarker
  // is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the hand landmarker. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A vector of HandLandmarkerResult, each is the detected results
  //     for a input frame.
  //   - The const reference to the corresponding input image that the hand
  //     landmarker runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status DetectAsync(Image image, int64_t timestamp_ms,
                           std::optional<core::ImageProcessingOptions>
                               image_processing_options = std::nullopt,
                           bool capture_tensors = false,
                           const std::vector<components::containers::RectF>&
                               search_regions = {});

  // Shuts down the HandLandmarker when all works are done.
  absl::Status Close() { return runner_->Close(); }

 private:
  // Adds the per-frame capture request to `inputs` when the graph was built
  // with tensor capture; fails when a capture is requested without it.
  absl::Status AddCaptureRequest(bool capture_tensors, Timestamp timestamp,
                                 tasks::core::PacketMap* inputs) const;

  // Whether the graph was built with `debug_tensor_capture`.
  bool tensor_capture_enabled_ = false;
  // Adds the per-frame search regions packet when the graph has that input;
  // an error if regions are given and it has not.
  absl::Status AddSearchRegions(
      const std::vector<components::containers::RectF>& search_regions,
      Timestamp timestamp, tasks::core::PacketMap* inputs) const;

  // `HandLandmarkerOptions::num_hands`, for the debug records.
  int num_hands_ = 1;
  // Whether the graph has the per-call search regions input.
  bool external_regions_enabled_ = false;
  // Number of Detect / DetectForVideo calls so far.
  int64_t frame_sequence_ = 0;
};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
