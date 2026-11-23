// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#import "MPCHandLandmarker.h"

#import "MPCError.h"
#import "MPCImageInternal.h"
#import "MPCMarshal.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

// Swift passes the C enum values as plain integers (MediaPipeDelegate.cValue
// and RunningMode.cValue in Modes.swift). Upstream has renumbered these enums
// before, so fail the build instead of silently selecting the wrong delegate.
static_assert(MP_DELEGATE_CPU == 1, "Update MediaPipeDelegate.cValue");
static_assert(MP_DELEGATE_GPU == 2, "Update MediaPipeDelegate.cValue");
static_assert(MP_DELEGATE_COREML == 100, "Update MediaPipeDelegate.cValue");
static_assert(MP_RUNNING_MODE_IMAGE == 1, "Update RunningMode.cValue");
static_assert(MP_RUNNING_MODE_VIDEO == 2, "Update RunningMode.cValue");

namespace {

// Builds the native image-processing options. Only rotation is exposed; the
// landmarker tasks reject a region of interest. The returned struct is
// value-typed; pass its address to the detect call.
MpImageProcessingOptions MakeImageProcessingOptions(int rotationDegrees,
                                                    BOOL captureTensors) {
  MpImageProcessingOptions options{};
  options.rotation_degrees = rotationDegrees;
  options.has_region_of_interest = 0;
  options.capture_tensors = captureTensors ? 1 : 0;
  return options;
}

NSArray<MPCNormalizedDetection *> *CopyDetections(const MpNormalizedDetection *detections,
                                                  uint32_t count) {
  NSMutableArray<MPCNormalizedDetection *> *out = [NSMutableArray arrayWithCapacity:count];
  for (uint32_t i = 0; i < count; ++i) {
    const MpNormalizedDetection &d = detections[i];
    NSMutableArray<NSNumber *> *keypoints =
        [NSMutableArray arrayWithCapacity:d.keypoints_count * 2];
    for (uint32_t j = 0; j < d.keypoints_count; ++j) {
      [keypoints addObject:@(d.keypoints[j].x)];
      [keypoints addObject:@(d.keypoints[j].y)];
    }
    [out addObject:[[MPCNormalizedDetection alloc] initWithScore:d.score
                                                            xMin:d.x_min
                                                            yMin:d.y_min
                                                           width:d.width
                                                          height:d.height
                                             keypointCoordinates:keypoints]];
  }
  return out;
}

MPCTensorCapture *CopyTensorCapture(const MpTensorCapture &capture) {
  MPCTensorCapture *out = [[MPCTensorCapture alloc] init];
  out.hasTensor = capture.has_tensor ? YES : NO;
  if (!capture.has_tensor) {
    return out;
  }
  out.elementType = capture.element_type;
  NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:capture.shape_count];
  for (uint32_t i = 0; i < capture.shape_count; ++i) {
    [shape addObject:@(capture.shape[i])];
  }
  out.shape = shape;
  out.data = [NSData dataWithBytes:capture.data length:(NSUInteger)capture.data_size];
  return out;
}

// Deep-copies the captured tensors, or returns nil when the result carries
// none (no capture was requested for this frame).
MPCHandLandmarkerTensorCapture *_Nullable BuildTensorCapture(
    const MpHandLandmarkerTensorCapture *capture) {
  if (capture == NULL) {
    return nil;
  }
  MPCHandLandmarkerTensorCapture *out = [[MPCHandLandmarkerTensorCapture alloc] init];
  out.timestampUs = capture->timestamp_us;
  out.palmDetectorInput = CopyTensorCapture(capture->palm_detector_input);
  NSMutableArray<MPCTensorCapture *> *inputs =
      [NSMutableArray arrayWithCapacity:capture->landmark_inputs_count];
  for (uint32_t i = 0; i < capture->landmark_inputs_count; ++i) {
    [inputs addObject:CopyTensorCapture(capture->landmark_inputs[i])];
  }
  out.landmarkInputs = inputs;
  return out;
}

// Fills the search regions of `options` from flat {left, top, right, bottom}
// quadruples. `storage` owns the array and must outlive the detect call.
void SetSearchRegions(NSArray<NSNumber *> *_Nullable regions, std::vector<MpRectF> *storage,
                      MpImageProcessingOptions *options) {
  storage->clear();
  for (NSUInteger i = 0; i + 3 < regions.count; i += 4) {
    MpRectF rect{};
    rect.left = regions[i].floatValue;
    rect.top = regions[i + 1].floatValue;
    rect.right = regions[i + 2].floatValue;
    rect.bottom = regions[i + 3].floatValue;
    storage->push_back(rect);
  }
  options->search_regions = storage->empty() ? NULL : storage->data();
  options->search_regions_count = (int)storage->size();
}

// Deep-copies the palm search record, or returns nil when the result carries
// none.
MPCHandLandmarkerPalmSearch *_Nullable BuildPalmSearch(const MpHandLandmarkerPalmSearch *search) {
  if (search == NULL) {
    return nil;
  }
  MPCHandLandmarkerPalmSearch *out = [[MPCHandLandmarkerPalmSearch alloc] init];
  out.timestampUs = search->timestamp_us;
  out.frameSequence = search->frame_sequence;
  out.palmDetectorRan = search->palm_detector_ran ? YES : NO;
  out.detectorFrameIndex = search->detector_frame_index;
  out.gridTileCount = search->grid_tile_count;
  out.tilesPerFrame = search->tiles_per_frame;
  out.cursorBefore = search->cursor_before;
  out.cursorAfter = search->cursor_after;
  out.callStartNs = search->call_start_ns;
  out.callEndNs = search->call_end_ns;
  out.regionsReadyNs = search->regions_ready_ns;
  out.mergeDoneNs = search->merge_done_ns;
  out.landmarkRoisReadyNs = search->landmark_rois_ready_ns;
  out.landmarksDoneNs = search->landmarks_done_ns;

  NSMutableArray<MPCPalmSearchRegion *> *regions =
      [NSMutableArray arrayWithCapacity:search->regions_count];
  for (uint32_t r = 0; r < search->regions_count; ++r) {
    const MpPalmSearchRegion &in = search->regions[r];
    MPCPalmSearchRegion *region = [[MPCPalmSearchRegion alloc] init];
    region.regionId = in.id;
    region.kind = in.kind;
    region.gridRow = in.grid_row;
    region.gridColumn = in.grid_column;
    region.requestedXCenter = in.requested_x_center;
    region.requestedYCenter = in.requested_y_center;
    region.requestedWidth = in.requested_width;
    region.requestedHeight = in.requested_height;
    region.sampledXCenter = in.sampled_x_center;
    region.sampledYCenter = in.sampled_y_center;
    region.sampledWidth = in.sampled_width;
    region.sampledHeight = in.sampled_height;
    NSMutableArray<MPCPalmSearchRegionDetection *> *detections =
        [NSMutableArray arrayWithCapacity:in.detections_count];
    for (uint32_t d = 0; d < in.detections_count; ++d) {
      MPCPalmSearchRegionDetection *detection = [[MPCPalmSearchRegionDetection alloc] init];
      detection.detection = CopyDetections(&in.detections[d].detection, 1)[0];
      detection.outcome = in.detections[d].outcome;
      detection.mergedIndex = in.detections[d].merged_index;
      [detections addObject:detection];
    }
    region.detections = detections;
    region.candidates = CopyDetections(in.candidates, in.candidates_count);
    region.aboveThresholdCount = in.above_threshold_count;
    region.cropStartNs = in.crop_start_ns;
    region.detectionsReadyNs = in.detections_ready_ns;
    [regions addObject:region];
  }
  out.regions = regions;

  NSMutableArray<MPCPalmSearchMergedDetection *> *merged =
      [NSMutableArray arrayWithCapacity:search->merged_count];
  for (uint32_t m = 0; m < search->merged_count; ++m) {
    const MpPalmSearchMergedDetection &in = search->merged[m];
    MPCPalmSearchMergedDetection *detection = [[MPCPalmSearchMergedDetection alloc] init];
    detection.detection = CopyDetections(&in.detection, 1)[0];
    NSMutableArray<NSNumber *> *sources = [NSMutableArray arrayWithCapacity:in.sources_count * 2];
    for (uint32_t s = 0; s < in.sources_count; ++s) {
      [sources addObject:@(in.sources[s].region_id)];
      [sources addObject:@(in.sources[s].detection_index)];
    }
    detection.sources = sources;
    detection.outcome = in.outcome;
    detection.overlappedRectIndex = in.overlapped_rect_index;
    detection.landmarkRoiIndex = in.landmark_roi_index;
    [merged addObject:detection];
  }
  out.merged = merged;
  return out;
}

// Deep-copies the palm candidate diagnostics, or returns nil when the result
// carries none (the option is off).
MPCHandLandmarkerPalmCandidates *_Nullable BuildPalmCandidates(
    const MpHandLandmarkerPalmCandidates *candidates) {
  if (candidates == NULL) {
    return nil;
  }
  MPCHandLandmarkerPalmCandidates *out = [[MPCHandLandmarkerPalmCandidates alloc] init];
  out.timestampUs = candidates->timestamp_us;
  out.palmDetectorRan = candidates->palm_detector_ran ? YES : NO;
  out.candidates = CopyDetections(candidates->candidates, candidates->candidates_count);
  out.aboveThresholdCount = candidates->above_threshold_count;
  out.afterSuppressionCount = candidates->after_suppression_count;
  return out;
}

NSArray<MPCRotatedRect *> *CopyRects(const MpNormalizedRotatedRect *rects,
                                     uint32_t count) {
  NSMutableArray<MPCRotatedRect *> *out = [NSMutableArray arrayWithCapacity:count];
  for (uint32_t i = 0; i < count; ++i) {
    const MpNormalizedRotatedRect &r = rects[i];
    [out addObject:[[MPCRotatedRect alloc] initWithXCenter:r.x_center
                                                   yCenter:r.y_center
                                                     width:r.width
                                                    height:r.height
                                                  rotation:r.rotation
                                                 hasRectId:r.has_rect_id ? YES : NO
                                                    rectId:r.rect_id]];
  }
  return out;
}

// Deep-copies the native stage outputs, or returns nil when the result carries
// none (the option is off).
MPCHandLandmarkerStages *_Nullable BuildStages(const MpHandLandmarkerStages *stages) {
  if (stages == NULL) {
    return nil;
  }
  MPCHandLandmarkerStages *out = [[MPCHandLandmarkerStages alloc] init];
  out.timestampUs = stages->timestamp_us;
  out.palmDetectorRan = stages->palm_detector_ran ? YES : NO;
  out.previousRects = CopyRects(stages->previous_rects, stages->previous_rects_count);

  out.palmDetections = CopyDetections(stages->palm_detections, stages->palm_detections_count);

  out.palmRects = CopyRects(stages->palm_rects, stages->palm_rects_count);
  out.detectorHandRects =
      CopyRects(stages->detector_hand_rects, stages->detector_hand_rects_count);
  out.landmarkRois = CopyRects(stages->landmark_rois, stages->landmark_rois_count);

  NSMutableArray<NSNumber *> *scores =
      [NSMutableArray arrayWithCapacity:stages->presence_scores_count];
  for (uint32_t i = 0; i < stages->presence_scores_count; ++i) {
    [scores addObject:@(stages->presence_scores[i])];
  }
  out.presenceScores = scores;
  NSMutableArray<NSNumber *> *passed =
      [NSMutableArray arrayWithCapacity:stages->presence_passed_count];
  for (uint32_t i = 0; i < stages->presence_passed_count; ++i) {
    [passed addObject:stages->presence_passed[i] ? @YES : @NO];
  }
  out.presencePassed = passed;

  NSMutableArray<NSArray<MPCLandmark *> *> *landmarks =
      [NSMutableArray arrayWithCapacity:stages->landmarks_before_dedup_count];
  for (uint32_t i = 0; i < stages->landmarks_before_dedup_count; ++i) {
    [landmarks addObject:MPCCopyLandmarks(stages->landmarks_before_dedup[i])];
  }
  out.landmarksBeforeDedup = landmarks;

  out.handRectsBeforeDedup = CopyRects(stages->hand_rects_before_dedup,
                                       stages->hand_rects_before_dedup_count);
  out.handRectsNextFrame =
      CopyRects(stages->hand_rects_next_frame, stages->hand_rects_next_frame_count);
  out.keepAliveRects = CopyRects(stages->keep_alive_rects, stages->keep_alive_rects_count);
  NSMutableArray<NSNumber *> *missCounts =
      [NSMutableArray arrayWithCapacity:stages->keep_alive_miss_counts_count];
  for (uint32_t i = 0; i < stages->keep_alive_miss_counts_count; ++i) {
    [missCounts addObject:@(stages->keep_alive_miss_counts[i])];
  }
  out.keepAliveMissCounts = missCounts;
  return out;
}

// Deep-copies a native result into ObjC objects. Does NOT close the native
// result (the caller owns that).
MPCHandLandmarkerResult *BuildResult(const MpHandLandmarkerResult &result) {
  NSMutableArray<NSArray<MPCLandmark *> *> *landmarks =
      [NSMutableArray arrayWithCapacity:result.hand_landmarks_count];
  for (uint32_t i = 0; i < result.hand_landmarks_count; ++i) {
    [landmarks addObject:MPCCopyLandmarks(result.hand_landmarks[i])];
  }
  NSMutableArray<NSArray<MPCLandmark *> *> *worldLandmarks =
      [NSMutableArray arrayWithCapacity:result.hand_world_landmarks_count];
  for (uint32_t i = 0; i < result.hand_world_landmarks_count; ++i) {
    [worldLandmarks addObject:MPCCopyLandmarks(result.hand_world_landmarks[i])];
  }
  NSMutableArray<NSArray<MPCCategory *> *> *handedness =
      [NSMutableArray arrayWithCapacity:result.handedness_count];
  for (uint32_t i = 0; i < result.handedness_count; ++i) {
    [handedness addObject:MPCCopyCategories(result.handedness[i])];
  }
  MPCHandLandmarkerResult *out =
      [[MPCHandLandmarkerResult alloc] initWithLandmarks:landmarks
                                          worldLandmarks:worldLandmarks
                                              handedness:handedness
                                                  stages:BuildStages(result.stages)];
  out.tensorCapture = BuildTensorCapture(result.tensor_capture);
  out.palmCandidates = BuildPalmCandidates(result.palm_candidates);
  out.palmSearch = BuildPalmSearch(result.palm_search);
  return out;
}

}  // namespace

@implementation MPCHandLandmarker {
  // Exclusively owned native landmarker; closed exactly once in -dealloc.
  MpHandLandmarkerPtr _landmarker;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numHands:(NSInteger)numHands
      minHandDetectionConfidence:(float)minHandDetectionConfidence
       minHandPresenceConfidence:(float)minHandPresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                         roiScale:(float)roiScale
             trackingGraceFrames:(NSInteger)trackingGraceFrames
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                            error:(NSError **)error {
  return [self initWithModelPath:modelPath
                        numHands:numHands
      minHandDetectionConfidence:minHandDetectionConfidence
       minHandPresenceConfidence:minHandPresenceConfidence
           minTrackingConfidence:minTrackingConfidence
                        roiScale:roiScale
             trackingGraceFrames:trackingGraceFrames
                        delegate:delegate
             coreMLModelCacheDir:coreMLModelCacheDir
                     runningMode:runningMode
                    stageOutputs:NO
                           error:error];
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numHands:(NSInteger)numHands
      minHandDetectionConfidence:(float)minHandDetectionConfidence
       minHandPresenceConfidence:(float)minHandPresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                         roiScale:(float)roiScale
             trackingGraceFrames:(NSInteger)trackingGraceFrames
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                     stageOutputs:(BOOL)stageOutputs
                            error:(NSError **)error {
  return [self initWithModelPath:modelPath
                        numHands:numHands
      minHandDetectionConfidence:minHandDetectionConfidence
       minHandPresenceConfidence:minHandPresenceConfidence
           minTrackingConfidence:minTrackingConfidence
                        roiScale:roiScale
             trackingGraceFrames:trackingGraceFrames
                        delegate:delegate
             coreMLModelCacheDir:coreMLModelCacheDir
                     runningMode:runningMode
                    stageOutputs:stageOutputs
                   tensorCapture:NO
              palmCandidateLimit:0
                           error:error];
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numHands:(NSInteger)numHands
      minHandDetectionConfidence:(float)minHandDetectionConfidence
       minHandPresenceConfidence:(float)minHandPresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                         roiScale:(float)roiScale
             trackingGraceFrames:(NSInteger)trackingGraceFrames
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                     stageOutputs:(BOOL)stageOutputs
                    tensorCapture:(BOOL)tensorCapture
               palmCandidateLimit:(NSInteger)palmCandidateLimit
                            error:(NSError **)error {
  return [self initWithModelPath:modelPath
                        numHands:numHands
      minHandDetectionConfidence:minHandDetectionConfidence
       minHandPresenceConfidence:minHandPresenceConfidence
           minTrackingConfidence:minTrackingConfidence
                        roiScale:roiScale
             trackingGraceFrames:trackingGraceFrames
                        delegate:delegate
             coreMLModelCacheDir:coreMLModelCacheDir
                     runningMode:runningMode
                    stageOutputs:stageOutputs
                   tensorCapture:tensorCapture
              palmCandidateLimit:palmCandidateLimit
                      palmSearch:nil
                           error:error];
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numHands:(NSInteger)numHands
      minHandDetectionConfidence:(float)minHandDetectionConfidence
       minHandPresenceConfidence:(float)minHandPresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                         roiScale:(float)roiScale
             trackingGraceFrames:(NSInteger)trackingGraceFrames
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                     stageOutputs:(BOOL)stageOutputs
                    tensorCapture:(BOOL)tensorCapture
               palmCandidateLimit:(NSInteger)palmCandidateLimit
                       palmSearch:(nullable MPCPalmSearchOptions *)palmSearch
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpHandLandmarkerOptions options{};
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = (MpDelegate)delegate;
  options.base_options.coreml_model_cache_dir = coreMLModelCacheDir.UTF8String;
  options.running_mode = (MpRunningMode)runningMode;
  options.num_hands = (int)numHands;
  options.min_hand_detection_confidence = minHandDetectionConfidence;
  options.min_hand_presence_confidence = minHandPresenceConfidence;
  options.min_tracking_confidence = minTrackingConfidence;
  options.roi_scale = roiScale;
  options.tracking_grace_frames = (int)trackingGraceFrames;
  options.debug_stage_outputs = stageOutputs ? true : false;
  options.debug_tensor_capture = tensorCapture ? true : false;
  options.debug_palm_candidates = (int)palmCandidateLimit;
  if (palmSearch != nil) {
    options.palm_search_enabled = true;
    options.palm_search.include_full_frame = palmSearch.includeFullFrame ? true : false;
    options.palm_search.grid_rows = (int)palmSearch.gridRows;
    options.palm_search.grid_columns = (int)palmSearch.gridColumns;
    options.palm_search.grid_overlap = palmSearch.gridOverlap;
    options.palm_search.round_robin_tiles_per_frame = (int)palmSearch.roundRobinTilesPerFrame;
    options.palm_search.accepts_external_regions =
        palmSearch.acceptsExternalRegions ? true : false;
    options.palm_search.reject_truncated_detections =
        palmSearch.rejectTruncatedDetections ? true : false;
    options.palm_search.max_palm_fraction = palmSearch.maxPalmFraction;
    options.palm_search.merge = (int)palmSearch.merge;
    options.palm_search.clip_before_association =
        palmSearch.clipBeforeAssociation ? true : false;
  }
  options.result_callback = nullptr;

  char *errorMsg = NULL;
  MpHandLandmarkerPtr landmarker = NULL;
  MpStatus status = MpHandLandmarkerCreate(&options, &landmarker, &errorMsg);
  if (status != kMpOk) {
    if (error) {
      *error = MPCMakeError(status, errorMsg);
    }
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    return nil;
  }

  _landmarker = landmarker;
  return self;
}

- (MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                         rotationDegrees:(int)rotationDegrees
                                   error:(NSError **)error {
  return [self detectImage:image rotationDegrees:rotationDegrees captureTensors:NO error:error];
}

- (MPCHandLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                 rotationDegrees:(int)rotationDegrees
                                     timestampMs:(int64_t)timestampMs
                                           error:(NSError **)error {
  return [self detectForVideoImage:image
                   rotationDegrees:rotationDegrees
                       timestampMs:timestampMs
                    captureTensors:NO
                             error:error];
}

- (MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                         rotationDegrees:(int)rotationDegrees
                          captureTensors:(BOOL)captureTensors
                                   error:(NSError **)error {
  return [self detectImage:image
           rotationDegrees:rotationDegrees
            captureTensors:captureTensors
             searchRegions:nil
                     error:error];
}

- (MPCHandLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                 rotationDegrees:(int)rotationDegrees
                                     timestampMs:(int64_t)timestampMs
                                  captureTensors:(BOOL)captureTensors
                                           error:(NSError **)error {
  return [self detectForVideoImage:image
                   rotationDegrees:rotationDegrees
                       timestampMs:timestampMs
                    captureTensors:captureTensors
                     searchRegions:nil
                             error:error];
}

- (MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                         rotationDegrees:(int)rotationDegrees
                          captureTensors:(BOOL)captureTensors
                           searchRegions:(nullable NSArray<NSNumber *> *)searchRegions
                                   error:(NSError **)error {
  char *errorMsg = NULL;
  MpHandLandmarkerResult result{};
  MpImageProcessingOptions options =
      MakeImageProcessingOptions(rotationDegrees, captureTensors);
  std::vector<MpRectF> regionStorage;
  SetSearchRegions(searchRegions, &regionStorage, &options);
  MpStatus status = MpHandLandmarkerDetectImage(
      _landmarker, image.imagePtr, &options, &result, &errorMsg);
  if (status != kMpOk) {
    if (error) {
      *error = MPCMakeError(status, errorMsg);
    }
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    return nil;
  }
  MPCHandLandmarkerResult *out = BuildResult(result);
  MpHandLandmarkerCloseResult(&result);
  return out;
}

- (MPCHandLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                 rotationDegrees:(int)rotationDegrees
                                     timestampMs:(int64_t)timestampMs
                                  captureTensors:(BOOL)captureTensors
                                   searchRegions:(nullable NSArray<NSNumber *> *)searchRegions
                                           error:(NSError **)error {
  char *errorMsg = NULL;
  MpHandLandmarkerResult result{};
  MpImageProcessingOptions options =
      MakeImageProcessingOptions(rotationDegrees, captureTensors);
  std::vector<MpRectF> regionStorage;
  SetSearchRegions(searchRegions, &regionStorage, &options);
  MpStatus status = MpHandLandmarkerDetectForVideo(
      _landmarker, image.imagePtr, &options, timestampMs, &result,
      &errorMsg);
  if (status != kMpOk) {
    if (error) {
      *error = MPCMakeError(status, errorMsg);
    }
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    return nil;
  }
  MPCHandLandmarkerResult *out = BuildResult(result);
  MpHandLandmarkerCloseResult(&result);
  return out;
}

- (void)dealloc {
  if (_landmarker) {
    char *errorMsg = NULL;
    MpHandLandmarkerClose(_landmarker, &errorMsg);
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    _landmarker = NULL;
  }
}

@end
