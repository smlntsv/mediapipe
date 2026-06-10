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

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                  numFaces:(NSInteger)numFaces
               minFaceDetectionConfidence:(float)minFaceDetectionConfidence
                minFacePresenceConfidence:(float)minFacePresenceConfidence
                    minTrackingConfidence:(float)minTrackingConfidence
                      outputFaceBlendshapes:(BOOL)outputFaceBlendshapes
        outputFacialTransformationMatrixes:(BOOL)outputFacialTransformationMatrixes
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

- (nullable MPCFaceLandmarkerResult *)detectImage:(MPCImage *)image
                                            error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
