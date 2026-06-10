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

// Result of pose landmark detection. Each top-level array element corresponds
// to one detected pose. All data is fully copied out of native memory.
// (Segmentation masks are out of scope for milestone 1.)
@interface MPCPoseLandmarkerResult : NSObject

// Pose landmarks in normalized image coordinates ([0, 1]), per pose.
@property(nonatomic, readonly) NSArray<NSArray<MPCLandmark *> *> *landmarks;

// Pose landmarks in world coordinates (meters), per pose.
@property(nonatomic, readonly) NSArray<NSArray<MPCLandmark *> *> *worldLandmarks;

- (instancetype)initWithLandmarks:(NSArray<NSArray<MPCLandmark *> *> *)landmarks
                   worldLandmarks:(NSArray<NSArray<MPCLandmark *> *> *)worldLandmarks
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
