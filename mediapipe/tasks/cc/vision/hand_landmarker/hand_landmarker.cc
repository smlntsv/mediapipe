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

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

namespace {

using HandLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    hand_landmarker::proto::HandLandmarkerGraphOptions;

using ::mediapipe::NormalizedRect;

constexpr char kTaskName[] = "HandLandmarker";
constexpr char kHandLandmarkerGraphTypeName[] =
    "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessStreamName[] = "handedness";
constexpr char kHandLandmarksTag[] = "LANDMARKS";
constexpr char kHandLandmarksStreamName[] = "landmarks";
constexpr char kHandWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandWorldLandmarksStreamName[] = "world_landmarks";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Debug stage outputs: {subgraph output tag, stream name}. Wired only when
// `debug_stage_outputs` is set.
struct StageStream {
  const char* tag;
  const char* name;
};
constexpr StageStream kStagePalmDetections = {"PALM_DETECTIONS",
                                              "stage_palm_detections"};
constexpr StageStream kStagePalmRects = {"PALM_RECTS", "stage_palm_rects"};
constexpr StageStream kStageDetectorHandRects = {"STAGE_DETECTOR_HAND_RECTS",
                                                 "stage_detector_hand_rects"};
constexpr StageStream kStageLandmarkRois = {"STAGE_LANDMARK_ROIS",
                                            "stage_landmark_rois"};
constexpr StageStream kStagePresence = {"STAGE_PRESENCE", "stage_presence"};
constexpr StageStream kStagePresenceScore = {"STAGE_PRESENCE_SCORE",
                                             "stage_presence_score"};
constexpr StageStream kStageLandmarksBeforeDedup = {
    "STAGE_LANDMARKS_BEFORE_DEDUP", "stage_landmarks_before_dedup"};
constexpr StageStream kStageHandRectsBeforeDedup = {
    "STAGE_HAND_RECTS_BEFORE_DEDUP", "stage_hand_rects_before_dedup"};
constexpr StageStream kStageHandRectsNextFrame = {
    "HAND_RECT_NEXT_FRAME", "stage_hand_rects_next_frame"};
// Stream mode only.
constexpr StageStream kStagePrevHandRects = {"STAGE_PREV_HAND_RECTS",
                                             "stage_prev_hand_rects"};
constexpr StageStream kStageHasEnoughHands = {"STAGE_HAS_ENOUGH_HANDS",
                                              "stage_has_enough_hands"};
// Stream mode with a positive tracking grace only.
constexpr StageStream kStageKeepAliveRects = {"STAGE_KEEP_ALIVE_RECTS",
                                              "stage_keep_alive_rects"};
constexpr StageStream kStageKeepAliveMissCounts = {
    "STAGE_KEEP_ALIVE_MISS_COUNTS", "stage_keep_alive_miss_counts"};

// Tensor capture streams, wired only when `debug_tensor_capture` is set.
constexpr char kCaptureTensorsTag[] = "CAPTURE_TENSORS";
constexpr char kCaptureTensorsStreamName[] = "capture_tensors_in";
constexpr StageStream kPalmInputTensorCapture = {"PALM_INPUT_TENSOR_CAPTURE",
                                                 "palm_input_tensor_capture"};
constexpr StageStream kLandmarkInputTensorCaptures = {
    "LANDMARK_INPUT_TENSOR_CAPTURES", "landmark_input_tensor_captures"};
// Palm candidate streams, wired only when `debug_palm_candidates` is positive.
constexpr StageStream kPalmCandidates = {"PALM_CANDIDATES", "palm_candidates"};
constexpr StageStream kPalmThresholdedDetections = {
    "PALM_THRESHOLDED_DETECTIONS", "palm_thresholded_detections"};

// Palm search streams, wired only with `palm_search`.
constexpr StageStream kPalmInputTensorCaptures = {
    "PALM_INPUT_TENSOR_CAPTURES", "palm_input_tensor_captures"};
// With `debug_stage_outputs` as well.
constexpr StageStream kStagePalmSearchRecord = {"STAGE_PALM_SEARCH_RECORD",
                                                "stage_palm_search_record"};
constexpr StageStream kStageAssociationOverlaps = {
    "STAGE_ASSOCIATION_OVERLAPS", "stage_association_overlaps"};
constexpr StageStream kStageLandmarkRoisReadyNs = {
    "STAGE_LANDMARK_ROIS_READY_NS", "stage_landmark_rois_ready_ns"};
constexpr StageStream kStageLandmarksDoneNs = {"STAGE_LANDMARKS_DONE_NS",
                                               "stage_landmarks_done_ns"};
// Per-call regions, wired only with `accepts_external_regions`.
constexpr char kSearchRegionsTag[] = "SEARCH_REGIONS";
constexpr char kSearchRegionsStreamName[] = "search_regions_in";
// Caller-side clock stamps and call count of the palm search record.
struct CallInfo {
  int64_t frame_sequence = -1;
  int64_t call_start_ns = 0;
  int64_t call_end_ns = 0;
};

int64_t MonotonicNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Copies the payload of the named output packet into `out`. Leaves `out`
// untouched when the stream is not part of the graph or carried no packet at
// this timestamp.
template <typename T>
void CopyStage(const tasks::core::PacketMap& packets, const StageStream& stream,
               T* out) {
  auto it = packets.find(stream.name);
  if (it == packets.end() || it->second.IsEmpty()) return;
  *out = it->second.Get<T>();
}

// Fills `result->stages` from the debug output streams, if the graph was built
// with them. A plain map lookup when it was not.
void AddStagesIfRequested(const tasks::core::PacketMap& packets,
                          HandLandmarkerResult* result) {
  if (packets.find(kStageLandmarkRois.name) == packets.end()) return;
  HandLandmarkerStages& stages = result->stages.emplace();
  if (auto it = packets.find(kHandLandmarksStreamName); it != packets.end()) {
    stages.timestamp_us = it->second.Timestamp().Value();
  }
  CopyStage(packets, kStagePrevHandRects, &stages.previous_rects);
  // The detector's inputs are gated by DisallowIf(has_enough_hands), which
  // lets them through when the flag is false or absent. Image mode has no
  // gate.
  auto has_enough_hands = packets.find(kStageHasEnoughHands.name);
  stages.palm_detector_ran = has_enough_hands == packets.end() ||
                             has_enough_hands->second.IsEmpty() ||
                             !has_enough_hands->second.Get<bool>();
  CopyStage(packets, kStagePalmDetections, &stages.palm_detections);
  CopyStage(packets, kStagePalmRects, &stages.palm_rects);
  CopyStage(packets, kStageDetectorHandRects, &stages.detector_hand_rects);
  CopyStage(packets, kStageLandmarkRois, &stages.landmark_rois);
  CopyStage(packets, kStagePresenceScore, &stages.presence_scores);
  CopyStage(packets, kStagePresence, &stages.presence_passed);
  std::vector<NormalizedLandmarkList> landmarks_before_dedup;
  CopyStage(packets, kStageLandmarksBeforeDedup, &landmarks_before_dedup);
  stages.landmarks_before_dedup.reserve(landmarks_before_dedup.size());
  for (const auto& landmarks : landmarks_before_dedup) {
    stages.landmarks_before_dedup.push_back(
        components::containers::ConvertToNormalizedLandmarks(landmarks));
  }
  CopyStage(packets, kStageHandRectsBeforeDedup,
            &stages.hand_rects_before_dedup);
  CopyStage(packets, kStageHandRectsNextFrame, &stages.hand_rects_next_frame);
  CopyStage(packets, kStageKeepAliveRects, &stages.keep_alive_rects);
  CopyStage(packets, kStageKeepAliveMissCounts, &stages.keep_alive_miss_counts);
}

// Fills `result->tensor_capture` on a frame the caller asked to capture. The
// palm capture stream carries a packet exactly on those frames, whether or not
// the detector ran.
void AddTensorCaptureIfRequested(const tasks::core::PacketMap& packets,
                                 HandLandmarkerResult* result) {
  auto palm = packets.find(kPalmInputTensorCapture.name);
  if (auto palms = packets.find(kPalmInputTensorCaptures.name);
      palms != packets.end()) {
    // Palm search path: per-region captures, with no packet when no region
    // ran. A requested frame is one where either model's captures arrived.
    auto landmarks = packets.find(kLandmarkInputTensorCaptures.name);
    const bool has_landmarks =
        landmarks != packets.end() && !landmarks->second.IsEmpty();
    if (palms->second.IsEmpty() && !has_landmarks) return;
    HandLandmarkerTensorCaptures& capture = result->tensor_capture.emplace();
    if (auto it = packets.find(kHandLandmarksStreamName);
        it != packets.end()) {
      capture.timestamp_us = it->second.Timestamp().Value();
    }
    CopyStage(packets, kPalmInputTensorCaptures, &capture.palm_detector_inputs);
    if (!capture.palm_detector_inputs.empty()) {
      capture.palm_detector_input = capture.palm_detector_inputs[0];
    }
    CopyStage(packets, kLandmarkInputTensorCaptures, &capture.landmark_inputs);
    return;
  }
  if (palm == packets.end() || palm->second.IsEmpty()) return;
  HandLandmarkerTensorCaptures& capture = result->tensor_capture.emplace();
  if (auto it = packets.find(kHandLandmarksStreamName); it != packets.end()) {
    capture.timestamp_us = it->second.Timestamp().Value();
  }
  capture.palm_detector_input = palm->second.Get<TensorCapture>();
  CopyStage(packets, kLandmarkInputTensorCaptures, &capture.landmark_inputs);
}

// Fills `result->palm_search` if the graph was built with the palm search
// record stream. Needs `result->stages`, which the same option adds.
void AddPalmSearchIfRequested(const tasks::core::PacketMap& packets,
                              int num_hands, const CallInfo& call,
                              HandLandmarkerResult* result) {
  auto record = packets.find(kStagePalmSearchRecord.name);
  if (record == packets.end() || !result->stages.has_value()) return;
  const HandLandmarkerStages& stages = *result->stages;
  HandLandmarkerPalmSearch& out = result->palm_search.emplace();
  out.timestamp_us = stages.timestamp_us;
  out.palm_detector_ran = stages.palm_detector_ran;
  out.frame_sequence = call.frame_sequence;
  out.call_start_ns = call.call_start_ns;
  out.call_end_ns = call.call_end_ns;
  CopyStage(packets, kStageLandmarkRoisReadyNs, &out.landmark_rois_ready_ns);
  CopyStage(packets, kStageLandmarksDoneNs, &out.landmarks_done_ns);
  if (record->second.IsEmpty()) return;
  out.record = record->second.Get<PalmSearchRecord>();

  // A HandAssociationCalculator is not part of every palm search graph.
  const bool has_association =
      packets.find(kStageAssociationOverlaps.name) != packets.end();
  std::vector<int> overlaps;
  CopyStage(packets, kStageAssociationOverlaps, &overlaps);

  const int merged_count = out.record.merged.size();
  // Without a merged detection the detector's rect vector holds one all-zero
  // placeholder, which is not a detection.
  const int rect_count =
      merged_count == 0
          ? 0
          : std::min<int>(merged_count, stages.detector_hand_rects.size());
  const int base_count = stages.previous_rects.size();
  out.merged_outcomes.resize(merged_count);
  out.merged_overlapped_rect_index.assign(merged_count, -1);
  out.merged_landmark_roi_index.assign(merged_count, -1);
  int position = base_count;
  for (int i = 0; i < merged_count; ++i) {
    if (i >= rect_count) {
      out.merged_outcomes[i] = PalmSearchHandOutcome::kClippedBeforeAssociation;
      continue;
    }
    const int overlapped =
        has_association && i < overlaps.size() ? overlaps[i] : -1;
    if (overlapped >= 0) {
      out.merged_outcomes[i] = overlapped < base_count
                                   ? PalmSearchHandOutcome::kOverlapsTrackedRect
                                   : PalmSearchHandOutcome::kOverlapsEarlierRect;
      out.merged_overlapped_rect_index[i] = overlapped;
      continue;
    }
    if (position < num_hands) {
      out.merged_outcomes[i] = PalmSearchHandOutcome::kToLandmarks;
      out.merged_landmark_roi_index[i] = position;
    } else {
      out.merged_outcomes[i] = PalmSearchHandOutcome::kClippedByNumHands;
    }
    ++position;
  }
}

// Fills `result->palm_candidates` if the graph was built with the palm
// candidate streams.
void AddPalmCandidatesIfRequested(const tasks::core::PacketMap& packets,
                                  HandLandmarkerResult* result) {
  auto candidates = packets.find(kPalmCandidates.name);
  if (candidates == packets.end()) return;
  HandLandmarkerPalmCandidates& out = result->palm_candidates.emplace();
  if (auto it = packets.find(kHandLandmarksStreamName); it != packets.end()) {
    out.timestamp_us = it->second.Timestamp().Value();
  }
  out.palm_detector_ran = !candidates->second.IsEmpty();
  CopyStage(packets, kPalmCandidates, &out.candidates);
  std::vector<mediapipe::Detection> detections;
  CopyStage(packets, kPalmThresholdedDetections, &detections);
  out.above_threshold_count = detections.size();
  detections.clear();
  CopyStage(packets, kStagePalmDetections, &detections);
  out.after_suppression_count = detections.size();
}

// Fills the debug records of `result` that the graph was built for.
void AddDebugRecords(const tasks::core::PacketMap& packets, int num_hands,
                     const CallInfo& call, HandLandmarkerResult* result) {
  AddStagesIfRequested(packets, result);
  AddTensorCaptureIfRequested(packets, result);
  AddPalmCandidatesIfRequested(packets, result);
  AddPalmSearchIfRequested(packets, num_hands, call, result);
}

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.hand_ladnamrker.HandLandmarkerGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<HandLandmarkerGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHandLandmarkerGraphTypeName);
  std::vector<StageStream> stage_streams;
  if (options->debug_stage_outputs()) {
    stage_streams = {kStagePalmDetections,
                     kStagePalmRects,
                     kStageDetectorHandRects,
                     kStageLandmarkRois,
                     kStagePresence,
                     kStagePresenceScore,
                     kStageLandmarksBeforeDedup,
                     kStageHandRectsBeforeDedup,
                     kStageHandRectsNextFrame};
    if (options->base_options().use_stream_mode()) {
      stage_streams.push_back(kStagePrevHandRects);
      stage_streams.push_back(kStageHasEnoughHands);
      if (options->tracking_grace_frames() > 0) {
        stage_streams.push_back(kStageKeepAliveRects);
        stage_streams.push_back(kStageKeepAliveMissCounts);
      }
    }
  }
  const bool tensor_capture = options->debug_tensor_capture();
  const bool palm_search =
      options->hand_detector_graph_options().has_palm_search();
  if (tensor_capture) {
    stage_streams.push_back(palm_search ? kPalmInputTensorCaptures
                                        : kPalmInputTensorCapture);
    stage_streams.push_back(kLandmarkInputTensorCaptures);
  }
  if (palm_search && options->debug_stage_outputs()) {
    stage_streams.push_back(kStagePalmSearchRecord);
    stage_streams.push_back(kStageLandmarkRoisReadyNs);
    stage_streams.push_back(kStageLandmarksDoneNs);
    // Image mode has an association only without `clip_before_association`.
    if (options->base_options().use_stream_mode() ||
        !options->hand_detector_graph_options()
             .palm_search()
             .clip_before_association()) {
      stage_streams.push_back(kStageAssociationOverlaps);
    }
  }
  if (options->hand_detector_graph_options().debug_palm_candidates() > 0) {
    stage_streams.push_back(kPalmCandidates);
    stage_streams.push_back(kPalmThresholdedDetections);
    // Its packet count is the number of detections after suppression.
    if (!options->debug_stage_outputs()) {
      stage_streams.push_back(kStagePalmDetections);
    }
  }
  std::vector<std::string> input_tags = {kImageTag, kNormRectTag};
  if (tensor_capture) {
    input_tags.push_back(kCaptureTensorsTag);
    graph.In(kCaptureTensorsTag).SetName(kCaptureTensorsStreamName);
  }
  if (palm_search && options->hand_detector_graph_options()
                         .palm_search()
                         .accepts_external_regions()) {
    input_tags.push_back(kSearchRegionsTag);
    graph.In(kSearchRegionsTag).SetName(kSearchRegionsStreamName);
  }
  subgraph.GetOptions<HandLandmarkerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  subgraph.Out(kHandednessTag).SetName(kHandednessStreamName) >>
      graph.Out(kHandednessTag);
  subgraph.Out(kHandLandmarksTag).SetName(kHandLandmarksStreamName) >>
      graph.Out(kHandLandmarksTag);
  subgraph.Out(kHandWorldLandmarksTag).SetName(kHandWorldLandmarksStreamName) >>
      graph.Out(kHandWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  for (const StageStream& stream : stage_streams) {
    subgraph.Out(stream.tag).SetName(stream.name) >> graph.Out(stream.tag);
  }
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(graph, subgraph, input_tags,
                                                 kHandLandmarksTag);
  }
  for (const std::string& tag : input_tags) {
    graph.In(tag) >> subgraph.In(tag);
  }
  return graph.GetConfig();
}

// Fork addition: on Apple platforms keep the TFLite path (Metal delegate on
// GPU, XNNPACK on CPU). LiteRT's GPU accelerator is a closed-source prebuilt
// that the fork does not ship, so with LiteRT a GPU request fails with "No GPU
// accelerator is registered in the LiteRT environment".
#if defined(__APPLE__)
constexpr bool kUseLiteRt = false;
#else
constexpr bool kUseLiteRt = true;
#endif  // defined(__APPLE__)

// Converts the user-facing HandLandmarkerOptions struct to the internal
// HandLandmarkerGraphOptions proto.
std::unique_ptr<HandLandmarkerGraphOptionsProto>
ConvertHandLandmarkerGraphOptionsProto(HandLandmarkerOptions* options) {
  auto options_proto = std::make_unique<HandLandmarkerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options),
                                             /*use_litert=*/kUseLiteRt));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure hand detector options.
  auto* hand_detector_graph_options =
      options_proto->mutable_hand_detector_graph_options();
  hand_detector_graph_options->set_num_hands(options->num_hands);
  hand_detector_graph_options->set_min_detection_confidence(
      options->min_hand_detection_confidence);

  // Configure hand landmark detector options.
  options_proto->set_min_tracking_confidence(options->min_tracking_confidence);
  options_proto->set_tracking_grace_frames(options->tracking_grace_frames);
  auto* hand_landmarks_detector_graph_options =
      options_proto->mutable_hand_landmarks_detector_graph_options();
  hand_landmarks_detector_graph_options->set_min_detection_confidence(
      options->min_hand_presence_confidence);
  hand_landmarks_detector_graph_options->set_roi_scale(options->roi_scale);
  // Set only when requested, so the serialized options (and with them the
  // graph config) are unchanged otherwise.
  if (options->debug_stage_outputs) {
    options_proto->set_debug_stage_outputs(true);
  }
  if (options->debug_tensor_capture) {
    options_proto->set_debug_tensor_capture(true);
  }
  if (options->debug_palm_candidates > 0) {
    hand_detector_graph_options->set_debug_palm_candidates(
        options->debug_palm_candidates);
  }
  if (options->palm_search.has_value()) {
    auto* palm_search = hand_detector_graph_options->mutable_palm_search();
    palm_search->set_include_full_frame(
        options->palm_search->include_full_frame);
    palm_search->set_clip_before_association(
        options->palm_search->clip_before_association);
    palm_search->set_grid_rows(options->palm_search->grid_rows);
    palm_search->set_grid_columns(options->palm_search->grid_columns);
    palm_search->set_grid_overlap(options->palm_search->grid_overlap);
    palm_search->set_round_robin_tiles_per_frame(
        options->palm_search->round_robin_tiles_per_frame);
    palm_search->set_accepts_external_regions(
        options->palm_search->accepts_external_regions);
    palm_search->set_reject_truncated_detections(
        options->palm_search->reject_truncated_detections);
    palm_search->set_max_palm_fraction(options->palm_search->max_palm_fraction);
    palm_search->set_merge(
        options->palm_search->merge == PalmSearchOptions::Merge::kBestScore
            ? hand_detector::proto::PalmSearchOptions::BEST_SCORE
            : hand_detector::proto::PalmSearchOptions::WEIGHTED);
  }

  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HandLandmarker>> HandLandmarker::Create(
    std::unique_ptr<HandLandmarkerOptions> options) {
  auto options_proto = ConvertHandLandmarkerGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  const int num_hands = options->num_hands;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback = [=](absl::StatusOr<tasks::core::PacketMap>
                               status_or_packets) {
      if (!status_or_packets.ok()) {
        Image image;
        result_callback(status_or_packets.status(), image,
                        Timestamp::Unset().Value());
        return;
      }
      if (status_or_packets.value()[kImageOutStreamName].IsEmpty()) {
        return;
      }
      Packet image_packet = status_or_packets.value()[kImageOutStreamName];
      if (status_or_packets.value()[kHandLandmarksStreamName].IsEmpty()) {
        Packet empty_packet =
            status_or_packets.value()[kHandLandmarksStreamName];
        HandLandmarkerResult empty_result;
        AddDebugRecords(status_or_packets.value(), num_hands, CallInfo(),
                        &empty_result);
        result_callback(
            {std::move(empty_result)}, image_packet.Get<Image>(),
            empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        return;
      }
      Packet handedness_packet =
          status_or_packets.value()[kHandednessStreamName];
      Packet hand_landmarks_packet =
          status_or_packets.value()[kHandLandmarksStreamName];
      Packet hand_world_landmarks_packet =
          status_or_packets.value()[kHandWorldLandmarksStreamName];
      HandLandmarkerResult result = ConvertToHandLandmarkerResult(
          /* handedness= */ handedness_packet
              .Get<std::vector<ClassificationList>>(),
          /* hand_landmarks= */
          hand_landmarks_packet.Get<std::vector<NormalizedLandmarkList>>(),
          /* hand_world_landmarks= */
          hand_world_landmarks_packet.Get<std::vector<LandmarkList>>());
      AddDebugRecords(status_or_packets.value(), num_hands, CallInfo(), &result);
      result_callback(
          {std::move(result)}, image_packet.Get<Image>(),
          hand_landmarks_packet.Timestamp().Value() /
              kMicroSecondsPerMilliSecond);
    };
  }
  const bool tensor_capture = options->debug_tensor_capture;
  const bool external_regions = options->palm_search.has_value() &&
                                options->palm_search->accepts_external_regions;
  auto landmarker = core::VisionTaskApiFactory::Create<
      HandLandmarker, HandLandmarkerGraphOptionsProto>(
      {.config = CreateGraphConfig(
           std::move(options_proto),
           options->running_mode == core::RunningMode::LIVE_STREAM),
       .task_name = kTaskName,
       .task_running_mode = core::GetCoreRunningMode(options->running_mode),
       .op_resolver = std::move(options->base_options.op_resolver),
       .packets_callback = std::move(packets_callback),
       .disable_default_service = options->base_options.disable_default_service,
       .host_environment = options->base_options.host_environment,
       .host_system = options->base_options.host_system,
       .host_version = options->base_options.host_version,
       .app_id = options->base_options.app_id,
       .app_version = options->base_options.app_version,
       .ca_bundle_path = options->base_options.ca_bundle_path});
  if (landmarker.ok()) {
    (*landmarker)->tensor_capture_enabled_ = tensor_capture;
    (*landmarker)->num_hands_ = num_hands;
    (*landmarker)->external_regions_enabled_ = external_regions;
  }
  return landmarker;
}

absl::Status HandLandmarker::AddCaptureRequest(
    bool capture_tensors, Timestamp timestamp,
    tasks::core::PacketMap* inputs) const {
  if (!tensor_capture_enabled_) {
    if (capture_tensors) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "capture_tensors requires the debug_tensor_capture option.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    return absl::OkStatus();
  }
  // One packet per frame, so that the stream's timestamp bound follows the
  // image stream.
  Packet request = MakePacket<bool>(capture_tensors);
  (*inputs)[kCaptureTensorsStreamName] =
      timestamp == Timestamp::Unset() ? request : request.At(timestamp);
  return absl::OkStatus();
}

absl::Status HandLandmarker::AddSearchRegions(
    const std::vector<components::containers::RectF>& search_regions,
    Timestamp timestamp, tasks::core::PacketMap* inputs) const {
  if (!external_regions_enabled_) {
    if (!search_regions.empty()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "search_regions requires palm_search.accepts_external_regions.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    return absl::OkStatus();
  }
  auto rects = std::make_unique<std::vector<NormalizedRect>>();
  rects->reserve(search_regions.size());
  for (const auto& region : search_regions) {
    if (!(region.left < region.right && region.top < region.bottom)) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Expected search regions with left < right and top < bottom.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    NormalizedRect rect;
    rect.set_x_center((region.left + region.right) / 2);
    rect.set_y_center((region.top + region.bottom) / 2);
    rect.set_width(region.right - region.left);
    rect.set_height(region.bottom - region.top);
    rects->push_back(rect);
  }
  // One packet per frame, so that the stream's timestamp bound follows the
  // image stream.
  Packet packet = Adopt(rects.release());
  (*inputs)[kSearchRegionsStreamName] =
      timestamp == Timestamp::Unset() ? packet : packet.At(timestamp);
  return absl::OkStatus();
}

absl::StatusOr<HandLandmarkerResult> HandLandmarker::Detect(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options,
    bool capture_tensors,
    const std::vector<components::containers::RectF>& search_regions) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ABSL_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                        ConvertToNormalizedRect(image_processing_options, image,
                                                /*roi_allowed=*/false));
  tasks::core::PacketMap inputs = {
      {kImageInStreamName, MakePacket<Image>(std::move(image))},
      {kNormRectStreamName, MakePacket<NormalizedRect>(std::move(norm_rect))}};
  ABSL_RETURN_IF_ERROR(
      AddCaptureRequest(capture_tensors, Timestamp::Unset(), &inputs));
  ABSL_RETURN_IF_ERROR(
      AddSearchRegions(search_regions, Timestamp::Unset(), &inputs));
  CallInfo call;
  call.frame_sequence = frame_sequence_++;
  call.call_start_ns = MonotonicNowNs();
  ABSL_ASSIGN_OR_RETURN(auto output_packets,
                        ProcessImageData(std::move(inputs)));
  call.call_end_ns = MonotonicNowNs();
  if (output_packets[kHandLandmarksStreamName].IsEmpty()) {
    HandLandmarkerResult empty_result;
    AddDebugRecords(output_packets, num_hands_, call, &empty_result);
    return empty_result;
  }
  HandLandmarkerResult result = ConvertToHandLandmarkerResult(
      /* handedness= */
      output_packets[kHandednessStreamName]
          .Get<std::vector<mediapipe::ClassificationList>>(),
      /* hand_landmarks= */
      output_packets[kHandLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* hand_world_landmarks */
      output_packets[kHandWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
  AddDebugRecords(output_packets, num_hands_, call, &result);
  return result;
}

absl::StatusOr<HandLandmarkerResult> HandLandmarker::DetectForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options,
    bool capture_tensors,
    const std::vector<components::containers::RectF>& search_regions) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ABSL_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                        ConvertToNormalizedRect(image_processing_options, image,
                                                /*roi_allowed=*/false));
  const Timestamp timestamp(timestamp_ms * kMicroSecondsPerMilliSecond);
  tasks::core::PacketMap inputs = {
      {kImageInStreamName, MakePacket<Image>(std::move(image)).At(timestamp)},
      {kNormRectStreamName,
       MakePacket<NormalizedRect>(std::move(norm_rect)).At(timestamp)}};
  ABSL_RETURN_IF_ERROR(AddCaptureRequest(capture_tensors, timestamp, &inputs));
  ABSL_RETURN_IF_ERROR(AddSearchRegions(search_regions, timestamp, &inputs));
  CallInfo call;
  call.frame_sequence = frame_sequence_++;
  call.call_start_ns = MonotonicNowNs();
  ABSL_ASSIGN_OR_RETURN(auto output_packets,
                        ProcessVideoData(std::move(inputs)));
  call.call_end_ns = MonotonicNowNs();
  if (output_packets[kHandLandmarksStreamName].IsEmpty()) {
    HandLandmarkerResult empty_result;
    AddDebugRecords(output_packets, num_hands_, call, &empty_result);
    return empty_result;
  }
  HandLandmarkerResult result = ConvertToHandLandmarkerResult(
      /* handedness= */
      output_packets[kHandednessStreamName]
          .Get<std::vector<mediapipe::ClassificationList>>(),
      /* hand_landmarks= */
      output_packets[kHandLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* hand_world_landmarks */
      output_packets[kHandWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
  AddDebugRecords(output_packets, num_hands_, call, &result);
  return result;
}

absl::Status HandLandmarker::DetectAsync(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options,
    bool capture_tensors,
    const std::vector<components::containers::RectF>& search_regions) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ABSL_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                        ConvertToNormalizedRect(image_processing_options, image,
                                                /*roi_allowed=*/false));
  const Timestamp timestamp(timestamp_ms * kMicroSecondsPerMilliSecond);
  tasks::core::PacketMap inputs = {
      {kImageInStreamName, MakePacket<Image>(std::move(image)).At(timestamp)},
      {kNormRectStreamName,
       MakePacket<NormalizedRect>(std::move(norm_rect)).At(timestamp)}};
  ABSL_RETURN_IF_ERROR(AddCaptureRequest(capture_tensors, timestamp, &inputs));
  ABSL_RETURN_IF_ERROR(AddSearchRegions(search_regions, timestamp, &inputs));
  return SendLiveStreamData(std::move(inputs));
}

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
