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

#import "MPCHandLandmarkerResult.h"
#import "MPCImage.h"

NS_ASSUME_NONNULL_BEGIN

// Objective-C bridge to the MediaPipe C HandLandmarker (image mode, CPU
// delegate). Exclusively owns the native landmarker and closes it exactly once
// on deallocation.
@interface MPCHandLandmarker : NSObject

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                  numHands:(NSInteger)numHands
               minHandDetectionConfidence:(float)minHandDetectionConfidence
                minHandPresenceConfidence:(float)minHandPresenceConfidence
                    minTrackingConfidence:(float)minTrackingConfidence
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

// Runs hand landmark detection on `image`. Returns a fully-copied result, or
// nil with `error` set on failure.
- (nullable MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                                            error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
