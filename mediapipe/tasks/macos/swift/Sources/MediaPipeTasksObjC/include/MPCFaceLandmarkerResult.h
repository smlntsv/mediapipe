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

#import "MPCClassifications.h"
#import "MPCLandmark.h"
#import "MPCMatrix.h"

NS_ASSUME_NONNULL_BEGIN

// Result of face landmark detection. Each top-level array element corresponds
// to one detected face. All data is fully copied out of native memory.
@interface MPCFaceLandmarkerResult : NSObject

// Face landmarks in normalized image coordinates ([0, 1]), per face.
@property(nonatomic, readonly) NSArray<NSArray<MPCLandmark *> *> *landmarks;

// Face blendshapes, per face. Empty unless blendshape output was requested.
@property(nonatomic, readonly) NSArray<MPCClassifications *> *blendshapes;

// Facial transformation matrices, per face. Empty unless matrix output was
// requested.
@property(nonatomic, readonly) NSArray<MPCMatrix *> *transformationMatrixes;

- (instancetype)initWithLandmarks:(NSArray<NSArray<MPCLandmark *> *> *)landmarks
                      blendshapes:(NSArray<MPCClassifications *> *)blendshapes
           transformationMatrixes:(NSArray<MPCMatrix *> *)transformationMatrixes
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
