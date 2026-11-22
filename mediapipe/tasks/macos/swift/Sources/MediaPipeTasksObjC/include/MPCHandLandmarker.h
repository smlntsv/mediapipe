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

// `stageOutputs` makes every result carry `stages` (per-stage debug outputs).
// `tensorCapture` allows `captureTensors:YES` on the detect methods.
// `palmCandidateLimit` > 0 makes every result carry `palmCandidates`.
// `delegate` and `runningMode` use the MediaPipe C enum integer values
// (MpDelegate / MpRunningMode).
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
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
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

// Same as above with `tensorCapture` off and `palmCandidateLimit` 0.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
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
                                     error:(NSError **)error;

// Same as above with `stageOutputs` off.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                  numHands:(NSInteger)numHands
               minHandDetectionConfidence:(float)minHandDetectionConfidence
                minHandPresenceConfidence:(float)minHandPresenceConfidence
                    minTrackingConfidence:(float)minTrackingConfidence
                                  roiScale:(float)roiScale
                      trackingGraceFrames:(NSInteger)trackingGraceFrames
                                  delegate:(int)delegate
                      coreMLModelCacheDir:(nullable NSString *)coreMLModelCacheDir
                               runningMode:(int)runningMode
                                     error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

// Runs hand landmark detection on `image` (IMAGE running mode). Returns a
// fully-copied result, or nil with `error` set on failure. `rotationDegrees` is
// a clockwise multiple of 90 applied before inference.
- (nullable MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                                  rotationDegrees:(int)rotationDegrees
                                            error:(NSError **)error;

// Runs hand landmark detection on a video frame (VIDEO running mode).
- (nullable MPCHandLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                          rotationDegrees:(int)rotationDegrees
                                              timestampMs:(int64_t)timestampMs
                                                    error:(NSError **)error;

// The two methods above with a capture request. `captureTensors` makes the
// result carry `tensorCapture`; it needs a landmarker created with
// `tensorCapture` and is ignored otherwise.
- (nullable MPCHandLandmarkerResult *)detectImage:(MPCImage *)image
                                  rotationDegrees:(int)rotationDegrees
                                   captureTensors:(BOOL)captureTensors
                                            error:(NSError **)error;

- (nullable MPCHandLandmarkerResult *)detectForVideoImage:(MPCImage *)image
                                          rotationDegrees:(int)rotationDegrees
                                              timestampMs:(int64_t)timestampMs
                                           captureTensors:(BOOL)captureTensors
                                                    error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
