// Copyright 2026 The MediaPipe Authors.
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

#import "MPCHandLandmarkerStages.h"

NS_ASSUME_NONNULL_BEGIN

// Configuration of the palm search regions path (the ObjC copy of
// MpPalmSearchOptions). A new instance runs the whole frame only.
@interface MPCPalmSearchOptions : NSObject

@property(nonatomic) BOOL includeFullFrame;
@property(nonatomic) NSInteger gridRows;
@property(nonatomic) NSInteger gridColumns;
@property(nonatomic) float gridOverlap;
// 0: every grid tile on every detector frame.
@property(nonatomic) NSInteger roundRobinTilesPerFrame;
@property(nonatomic) BOOL acceptsExternalRegions;
@property(nonatomic) BOOL rejectTruncatedDetections;
// 0: off.
@property(nonatomic) float maxPalmFraction;
// 0: weighted, 1: best score.
@property(nonatomic) NSInteger merge;
@property(nonatomic) BOOL clipBeforeAssociation;

@end

// One detection of one region (the ObjC copy of MpPalmSearchRegionDetection).
@interface MPCPalmSearchRegionDetection : NSObject

@property(nonatomic, strong) MPCNormalizedDetection *detection;
@property(nonatomic) NSInteger outcome;
@property(nonatomic) NSInteger mergedIndex;

@end

// One region and what it produced (the ObjC copy of MpPalmSearchRegion). The
// two rects are {xCenter, yCenter, width, height}.
@interface MPCPalmSearchRegion : NSObject

@property(nonatomic) NSInteger regionId;
@property(nonatomic) NSInteger kind;
@property(nonatomic) NSInteger gridRow;
@property(nonatomic) NSInteger gridColumn;
@property(nonatomic) float requestedXCenter;
@property(nonatomic) float requestedYCenter;
@property(nonatomic) float requestedWidth;
@property(nonatomic) float requestedHeight;
@property(nonatomic) float sampledXCenter;
@property(nonatomic) float sampledYCenter;
@property(nonatomic) float sampledWidth;
@property(nonatomic) float sampledHeight;
@property(nonatomic, copy) NSArray<MPCPalmSearchRegionDetection *> *detections;
@property(nonatomic, copy) NSArray<MPCNormalizedDetection *> *candidates;
@property(nonatomic) NSInteger aboveThresholdCount;
@property(nonatomic) int64_t cropStartNs;
@property(nonatomic) int64_t detectionsReadyNs;

@end

// One merged detection (the ObjC copy of MpPalmSearchMergedDetection).
@interface MPCPalmSearchMergedDetection : NSObject

@property(nonatomic, strong) MPCNormalizedDetection *detection;
// Flat pairs: region id, detection index in that region.
@property(nonatomic, copy) NSArray<NSNumber *> *sources;
@property(nonatomic) NSInteger outcome;
@property(nonatomic) NSInteger overlappedRectIndex;
@property(nonatomic) NSInteger landmarkRoiIndex;

@end

// The palm search record of one frame (the ObjC copy of
// MpHandLandmarkerPalmSearch).
@interface MPCHandLandmarkerPalmSearch : NSObject

@property(nonatomic) int64_t timestampUs;
@property(nonatomic) int64_t frameSequence;
@property(nonatomic) BOOL palmDetectorRan;
@property(nonatomic) int64_t detectorFrameIndex;
@property(nonatomic) NSInteger gridTileCount;
@property(nonatomic) NSInteger tilesPerFrame;
@property(nonatomic) NSInteger cursorBefore;
@property(nonatomic) NSInteger cursorAfter;
@property(nonatomic) int64_t callStartNs;
@property(nonatomic) int64_t callEndNs;
@property(nonatomic) int64_t regionsReadyNs;
@property(nonatomic) int64_t mergeDoneNs;
@property(nonatomic) int64_t landmarkRoisReadyNs;
@property(nonatomic) int64_t landmarksDoneNs;
@property(nonatomic, copy) NSArray<MPCPalmSearchRegion *> *regions;
@property(nonatomic, copy) NSArray<MPCPalmSearchMergedDetection *> *merged;

@end

NS_ASSUME_NONNULL_END
