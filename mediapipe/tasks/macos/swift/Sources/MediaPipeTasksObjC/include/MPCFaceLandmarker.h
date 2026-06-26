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

#import <Foundation/Foundation.h>

#import "MPCFaceLandmarkerResult.h"
#import "MPCImage.h"

NS_ASSUME_NONNULL_BEGIN

// Objective-C bridge to the MediaPipe C FaceLandmarker (image mode, CPU
// delegate). Exclusively owns the native landmarker and closes it exactly once
// on deallocation.
@interface MPCFaceLandmarker : NSObject

// `delegate` and `runningMode` use the MediaPipe C enum integer values.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                  numFaces:(NSInteger)numFaces
               minFaceDetectionConfidence:(float)minFaceDetectionConfidence
                minFacePresenceConfidence:(float)minFacePresenceConfidence
                    minTrackingConfidence:(float)minTrackingConfidence
                      outputFaceBlendshapes:(BOOL)outputFaceBlendshapes
        outputFacialTransformationMatrixes:(BOOL)outputFacialTransformationMatrixes
                                  delegate:(int)delegate
                               runningMode:(int)runningMode
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

// `rotationDegrees` is a clockwise multiple of 90 applied before inference.
- (nullable MPCFaceLandmarkerResult *)detectImage:(MPCImage *)image
                                  rotationDegrees:(int)rotationDegrees
                                            error:(NSError **)error;

- (nullable MPCFaceLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                          rotationDegrees:(int)rotationDegrees
                                              timestampMs:(int64_t)timestampMs
                                                    error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
