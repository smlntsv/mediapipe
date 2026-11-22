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

#import "MPCHandLandmarkerStages.h"

@implementation MPCRotatedRect

- (instancetype)initWithXCenter:(float)xCenter
                        yCenter:(float)yCenter
                          width:(float)width
                         height:(float)height
                       rotation:(float)rotation
                      hasRectId:(BOOL)hasRectId
                         rectId:(int64_t)rectId {
  self = [super init];
  if (self) {
    _xCenter = xCenter;
    _yCenter = yCenter;
    _width = width;
    _height = height;
    _rotation = rotation;
    _hasRectId = hasRectId;
    _rectId = rectId;
  }
  return self;
}

@end

@implementation MPCNormalizedDetection

- (instancetype)initWithScore:(float)score
                         xMin:(float)xMin
                         yMin:(float)yMin
                        width:(float)width
                       height:(float)height
          keypointCoordinates:(NSArray<NSNumber *> *)keypointCoordinates {
  self = [super init];
  if (self) {
    _score = score;
    _xMin = xMin;
    _yMin = yMin;
    _width = width;
    _height = height;
    _keypointCoordinates = [keypointCoordinates copy];
  }
  return self;
}

@end

@implementation MPCHandLandmarkerStages

- (instancetype)init {
  self = [super init];
  if (self) {
    _previousRects = @[];
    _palmDetections = @[];
    _palmRects = @[];
    _detectorHandRects = @[];
    _landmarkRois = @[];
    _presenceScores = @[];
    _presencePassed = @[];
    _landmarksBeforeDedup = @[];
    _handRectsBeforeDedup = @[];
    _handRectsNextFrame = @[];
    _keepAliveRects = @[];
    _keepAliveMissCounts = @[];
  }
  return self;
}

@end
