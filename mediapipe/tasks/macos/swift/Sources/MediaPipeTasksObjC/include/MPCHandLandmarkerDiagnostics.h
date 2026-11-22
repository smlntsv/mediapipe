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

// A CPU copy of the input tensor of one model invocation (the ObjC copy of
// MpTensorCapture).
@interface MPCTensorCapture : NSObject

// NO when no tensor reached the model, i.e. inference was skipped. `shape` is
// then empty and `data` nil.
@property(nonatomic) BOOL hasTensor;
// One of MpTensorElementType.
@property(nonatomic) NSInteger elementType;
@property(nonatomic, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, copy, nullable) NSData *data;

@end

// The tensors given to the two models on one frame (the ObjC copy of
// MpHandLandmarkerTensorCapture).
@interface MPCHandLandmarkerTensorCapture : NSObject

@property(nonatomic) int64_t timestampUs;
@property(nonatomic, strong) MPCTensorCapture *palmDetectorInput;
@property(nonatomic, copy) NSArray<MPCTensorCapture *> *landmarkInputs;

@end

// What became of the palm detector's candidates on one frame (the ObjC copy of
// MpHandLandmarkerPalmCandidates).
@interface MPCHandLandmarkerPalmCandidates : NSObject

@property(nonatomic) int64_t timestampUs;
@property(nonatomic) BOOL palmDetectorRan;
@property(nonatomic, copy) NSArray<MPCNormalizedDetection *> *candidates;
@property(nonatomic) NSInteger aboveThresholdCount;
@property(nonatomic) NSInteger afterSuppressionCount;

@end

NS_ASSUME_NONNULL_END
