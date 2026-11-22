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

#import "MPCLandmark.h"

NS_ASSUME_NONNULL_BEGIN

// A rotated rectangle in normalized image coordinates.
@interface MPCRotatedRect : NSObject

@property(nonatomic, readonly) float xCenter;
@property(nonatomic, readonly) float yCenter;
@property(nonatomic, readonly) float width;
@property(nonatomic, readonly) float height;
// Radians, clockwise.
@property(nonatomic, readonly) float rotation;
@property(nonatomic, readonly) BOOL hasRectId;
@property(nonatomic, readonly) int64_t rectId;

- (instancetype)initWithXCenter:(float)xCenter
                        yCenter:(float)yCenter
                          width:(float)width
                         height:(float)height
                       rotation:(float)rotation
                      hasRectId:(BOOL)hasRectId
                         rectId:(int64_t)rectId NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

// A detection with a float bounding box in normalized image coordinates.
@interface MPCNormalizedDetection : NSObject

@property(nonatomic, readonly) float score;
@property(nonatomic, readonly) float xMin;
@property(nonatomic, readonly) float yMin;
@property(nonatomic, readonly) float width;
@property(nonatomic, readonly) float height;
// Keypoints as a flat list: x0, y0, x1, y1, ...
@property(nonatomic, readonly) NSArray<NSNumber *> *keypointCoordinates;

- (instancetype)initWithScore:(float)score
                         xMin:(float)xMin
                         yMin:(float)yMin
                        width:(float)width
                       height:(float)height
          keypointCoordinates:(NSArray<NSNumber *> *)keypointCoordinates
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

// Per-stage outputs of the hand pipeline for one frame (the ObjC copy of
// MpHandLandmarkerStages). All data is fully copied out of native memory.
@interface MPCHandLandmarkerStages : NSObject

@property(nonatomic) int64_t timestampUs;
@property(nonatomic) BOOL palmDetectorRan;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *previousRects;
@property(nonatomic, copy) NSArray<MPCNormalizedDetection *> *palmDetections;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *palmRects;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *detectorHandRects;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *landmarkRois;
@property(nonatomic, copy) NSArray<NSNumber *> *presenceScores;
@property(nonatomic, copy) NSArray<NSNumber *> *presencePassed;
@property(nonatomic, copy) NSArray<NSArray<MPCLandmark *> *> *landmarksBeforeDedup;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *handRectsBeforeDedup;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *handRectsNextFrame;
@property(nonatomic, copy) NSArray<MPCRotatedRect *> *keepAliveRects;
@property(nonatomic, copy) NSArray<NSNumber *> *keepAliveMissCounts;

@end

NS_ASSUME_NONNULL_END
