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

#import "MPCImage.h"
#import "MPCPoseLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

// Objective-C bridge to the MediaPipe C PoseLandmarker (image mode, CPU
// delegate). Exclusively owns the native landmarker and closes it exactly once
// on deallocation.
@interface MPCPoseLandmarker : NSObject

// `delegate` and `runningMode` use the MediaPipe C enum integer values.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                  numPoses:(NSInteger)numPoses
               minPoseDetectionConfidence:(float)minPoseDetectionConfidence
                minPosePresenceConfidence:(float)minPosePresenceConfidence
                    minTrackingConfidence:(float)minTrackingConfidence
                                  delegate:(int)delegate
                      coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                               runningMode:(int)runningMode
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

// `rotationDegrees` is a clockwise multiple of 90 applied before inference.
- (nullable MPCPoseLandmarkerResult *)detectImage:(MPCImage *)image
                                  rotationDegrees:(int)rotationDegrees
                                            error:(NSError **)error;

- (nullable MPCPoseLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                          rotationDegrees:(int)rotationDegrees
                                              timestampMs:(int64_t)timestampMs
                                                    error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
