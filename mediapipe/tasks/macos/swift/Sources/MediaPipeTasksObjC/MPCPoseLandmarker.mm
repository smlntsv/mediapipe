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

#import "MPCPoseLandmarker.h"

#import "MPCError.h"
#import "MPCImageInternal.h"
#import "MPCMarshal.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"

namespace {

// Builds the native image-processing options. Only rotation is exposed; the
// landmarker tasks reject a region of interest. The returned struct is
// value-typed; pass its address to the detect call.
MpImageProcessingOptions MakeImageProcessingOptions(int rotationDegrees) {
  MpImageProcessingOptions options{};
  options.rotation_degrees = rotationDegrees;
  options.has_region_of_interest = 0;
  return options;
}

MPCPoseLandmarkerResult *BuildResult(const MpPoseLandmarkerResult &result) {
  NSMutableArray<NSArray<MPCLandmark *> *> *landmarks =
      [NSMutableArray arrayWithCapacity:result.pose_landmarks_count];
  for (uint32_t i = 0; i < result.pose_landmarks_count; ++i) {
    [landmarks addObject:MPCCopyLandmarks(result.pose_landmarks[i])];
  }
  NSMutableArray<NSArray<MPCLandmark *> *> *worldLandmarks =
      [NSMutableArray arrayWithCapacity:result.pose_world_landmarks_count];
  for (uint32_t i = 0; i < result.pose_world_landmarks_count; ++i) {
    [worldLandmarks addObject:MPCCopyLandmarks(result.pose_world_landmarks[i])];
  }
  return [[MPCPoseLandmarkerResult alloc] initWithLandmarks:landmarks
                                            worldLandmarks:worldLandmarks];
}

}  // namespace

@implementation MPCPoseLandmarker {
  MpPoseLandmarkerPtr _landmarker;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numPoses:(NSInteger)numPoses
      minPoseDetectionConfidence:(float)minPoseDetectionConfidence
       minPosePresenceConfidence:(float)minPosePresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpPoseLandmarkerOptions options{};
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = (MpDelegate)delegate;
  options.base_options.coreml_model_cache_dir = coreMLModelCacheDir.UTF8String;
  options.running_mode = (MpRunningMode)runningMode;
  options.num_poses = (int)numPoses;
  options.min_pose_detection_confidence = minPoseDetectionConfidence;
  options.min_pose_presence_confidence = minPosePresenceConfidence;
  options.min_tracking_confidence = minTrackingConfidence;
  options.output_segmentation_masks = false;
  options.result_callback = nullptr;

  char *errorMsg = NULL;
  MpPoseLandmarkerPtr landmarker = NULL;
  MpStatus status = MpPoseLandmarkerCreate(&options, &landmarker, &errorMsg);
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

- (MPCPoseLandmarkerResult *)detectImage:(MPCImage *)image
                         rotationDegrees:(int)rotationDegrees
                                   error:(NSError **)error {
  char *errorMsg = NULL;
  MpPoseLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
  MpStatus status = MpPoseLandmarkerDetectImage(
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
  MPCPoseLandmarkerResult *out = BuildResult(result);
  MpPoseLandmarkerCloseResult(&result);
  return out;
}

- (MPCPoseLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                 rotationDegrees:(int)rotationDegrees
                                     timestampMs:(int64_t)timestampMs
                                           error:(NSError **)error {
  char *errorMsg = NULL;
  MpPoseLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
  MpStatus status = MpPoseLandmarkerDetectForVideo(
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
  MPCPoseLandmarkerResult *out = BuildResult(result);
  MpPoseLandmarkerCloseResult(&result);
  return out;
}

- (void)dealloc {
  if (_landmarker) {
    char *errorMsg = NULL;
    MpPoseLandmarkerClose(_landmarker, &errorMsg);
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    _landmarker = NULL;
  }
}

@end
