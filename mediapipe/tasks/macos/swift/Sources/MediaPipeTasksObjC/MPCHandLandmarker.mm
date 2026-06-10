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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

@implementation MPCHandLandmarker {
  // Exclusively owned native landmarker; closed exactly once in -dealloc.
  MpHandLandmarkerPtr _landmarker;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numHands:(NSInteger)numHands
      minHandDetectionConfidence:(float)minHandDetectionConfidence
       minHandPresenceConfidence:(float)minHandPresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpHandLandmarkerOptions options{};  // value-init: zeroes base_options, etc.
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = MP_DELEGATE_CPU;
  options.running_mode = MP_RUNNING_MODE_IMAGE;
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
                                   error:(NSError **)error {
  char *errorMsg = NULL;
  MpHandLandmarkerResult result{};
  MpStatus status = MpHandLandmarkerDetectImage(
      _landmarker, image.imagePtr, /*options=*/nullptr, &result, &errorMsg);
  if (status != kMpOk) {
    if (error) {
      *error = MPCMakeError(status, errorMsg);
    }
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    return nil;
  }

  // Deep-copy EVERYTHING into Objective-C objects before closing the native
  // result. After MpHandLandmarkerCloseResult, no native pointers remain live.
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

  MpHandLandmarkerCloseResult(&result);

  return [[MPCHandLandmarkerResult alloc] initWithLandmarks:landmarks
                                            worldLandmarks:worldLandmarks
                                                handedness:handedness];
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
