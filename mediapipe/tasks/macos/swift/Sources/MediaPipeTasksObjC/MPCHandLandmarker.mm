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

#import "MPCHandLandmarker.h"

#import "MPCError.h"
#import "MPCImageInternal.h"
#import "MPCMarshal.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

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
  return [[MPCHandLandmarkerResult alloc] initWithLandmarks:landmarks
                                            worldLandmarks:worldLandmarks
                                                handedness:handedness];
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
                         delegate:(int)delegate
                      runningMode:(int)runningMode
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpHandLandmarkerOptions options{};
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = (MpDelegate)delegate;
  options.running_mode = (MpRunningMode)runningMode;
  options.num_hands = (int)numHands;
  options.min_hand_detection_confidence = minHandDetectionConfidence;
  options.min_hand_presence_confidence = minHandPresenceConfidence;
  options.min_tracking_confidence = minTrackingConfidence;
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
  char *errorMsg = NULL;
  MpHandLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
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
                                           error:(NSError **)error {
  char *errorMsg = NULL;
  MpHandLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
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
