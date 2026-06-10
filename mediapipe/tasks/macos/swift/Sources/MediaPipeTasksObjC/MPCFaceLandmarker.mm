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

#import "MPCFaceLandmarker.h"

#import "MPCError.h"
#import "MPCImageInternal.h"
#import "MPCMarshal.h"

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

@implementation MPCFaceLandmarker {
  // Exclusively owned native landmarker; closed exactly once in -dealloc.
  MpFaceLandmarkerPtr _landmarker;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numFaces:(NSInteger)numFaces
      minFaceDetectionConfidence:(float)minFaceDetectionConfidence
       minFacePresenceConfidence:(float)minFacePresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpFaceLandmarkerOptions options{};
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = MP_DELEGATE_CPU;
  options.running_mode = MP_RUNNING_MODE_IMAGE;
  options.num_faces = (int)numFaces;
  options.min_face_detection_confidence = minFaceDetectionConfidence;
  options.min_face_presence_confidence = minFacePresenceConfidence;
  options.min_tracking_confidence = minTrackingConfidence;
  options.output_face_blendshapes = false;
  options.output_facial_transformation_matrixes = false;
  options.result_callback = nullptr;

  char *errorMsg = NULL;
  MpFaceLandmarkerPtr landmarker = NULL;
  MpStatus status = MpFaceLandmarkerCreate(&options, &landmarker, &errorMsg);
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

- (MPCFaceLandmarkerResult *)detectImage:(MPCImage *)image
                                   error:(NSError **)error {
  char *errorMsg = NULL;
  MpFaceLandmarkerResult result{};
  MpStatus status = MpFaceLandmarkerDetectImage(
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

  // Deep-copy before closing the native result. Blendshapes and the facial
  // transformation matrix are intentionally not surfaced in milestone 1.
  NSMutableArray<NSArray<MPCLandmark *> *> *landmarks =
      [NSMutableArray arrayWithCapacity:result.face_landmarks_count];
  for (uint32_t i = 0; i < result.face_landmarks_count; ++i) {
    [landmarks addObject:MPCCopyLandmarks(result.face_landmarks[i])];
  }

  MpFaceLandmarkerCloseResult(&result);

  return [[MPCFaceLandmarkerResult alloc] initWithLandmarks:landmarks];
}

- (void)dealloc {
  if (_landmarker) {
    char *errorMsg = NULL;
    MpFaceLandmarkerClose(_landmarker, &errorMsg);
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    _landmarker = NULL;
  }
}

@end
