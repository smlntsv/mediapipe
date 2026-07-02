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

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

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

MPCFaceLandmarkerResult *BuildResult(const MpFaceLandmarkerResult &result) {
  NSMutableArray<NSArray<MPCLandmark *> *> *landmarks =
      [NSMutableArray arrayWithCapacity:result.face_landmarks_count];
  for (uint32_t i = 0; i < result.face_landmarks_count; ++i) {
    [landmarks addObject:MPCCopyLandmarks(result.face_landmarks[i])];
  }

  NSMutableArray<MPCClassifications *> *blendshapes =
      [NSMutableArray arrayWithCapacity:result.face_blendshapes_count];
  for (uint32_t i = 0; i < result.face_blendshapes_count; ++i) {
    [blendshapes addObject:[[MPCClassifications alloc]
                               initWithCategories:MPCCopyCategories(result.face_blendshapes[i])
                                        headIndex:0
                                         headName:nil]];
  }

  NSMutableArray<MPCMatrix *> *matrixes =
      [NSMutableArray arrayWithCapacity:result.facial_transformation_matrixes_count];
  for (uint32_t i = 0; i < result.facial_transformation_matrixes_count; ++i) {
    const MpMatrix &m = result.facial_transformation_matrixes[i];
    const NSUInteger count = (NSUInteger)m.rows * (NSUInteger)m.cols;
    NSMutableArray<NSNumber *> *data = [NSMutableArray arrayWithCapacity:count];
    for (NSUInteger k = 0; k < count; ++k) {
      [data addObject:@(m.data[k])];
    }
    [matrixes addObject:[[MPCMatrix alloc] initWithRows:(NSInteger)m.rows
                                                columns:(NSInteger)m.cols
                                                   data:data]];
  }

  return [[MPCFaceLandmarkerResult alloc] initWithLandmarks:landmarks
                                                blendshapes:blendshapes
                                     transformationMatrixes:matrixes];
}

}  // namespace

@implementation MPCFaceLandmarker {
  MpFaceLandmarkerPtr _landmarker;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                         numFaces:(NSInteger)numFaces
      minFaceDetectionConfidence:(float)minFaceDetectionConfidence
       minFacePresenceConfidence:(float)minFacePresenceConfidence
           minTrackingConfidence:(float)minTrackingConfidence
             outputFaceBlendshapes:(BOOL)outputFaceBlendshapes
outputFacialTransformationMatrixes:(BOOL)outputFacialTransformationMatrixes
                         delegate:(int)delegate
             coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                      runningMode:(int)runningMode
                            error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  MpFaceLandmarkerOptions options{};
  options.base_options.model_asset_path = modelPath.UTF8String;
  options.base_options.delegate = (MpDelegate)delegate;
  options.base_options.coreml_model_cache_dir = coreMLModelCacheDir.UTF8String;
  options.running_mode = (MpRunningMode)runningMode;
  options.num_faces = (int)numFaces;
  options.min_face_detection_confidence = minFaceDetectionConfidence;
  options.min_face_presence_confidence = minFacePresenceConfidence;
  options.min_tracking_confidence = minTrackingConfidence;
  options.output_face_blendshapes = outputFaceBlendshapes ? true : false;
  options.output_facial_transformation_matrixes =
      outputFacialTransformationMatrixes ? true : false;
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
                         rotationDegrees:(int)rotationDegrees
                                   error:(NSError **)error {
  char *errorMsg = NULL;
  MpFaceLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
  MpStatus status = MpFaceLandmarkerDetectImage(
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
  MPCFaceLandmarkerResult *out = BuildResult(result);
  MpFaceLandmarkerCloseResult(&result);
  return out;
}

- (MPCFaceLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                 rotationDegrees:(int)rotationDegrees
                                     timestampMs:(int64_t)timestampMs
                                           error:(NSError **)error {
  char *errorMsg = NULL;
  MpFaceLandmarkerResult result{};
  MpImageProcessingOptions options = MakeImageProcessingOptions(rotationDegrees);
  MpStatus status = MpFaceLandmarkerDetectForVideo(
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
  MPCFaceLandmarkerResult *out = BuildResult(result);
  MpFaceLandmarkerCloseResult(&result);
  return out;
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
