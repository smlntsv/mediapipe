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

#import "MPCCategory.h"
#import "MPCLandmark.h"

NS_ASSUME_NONNULL_BEGIN

// Result of hand landmark detection. Each top-level array element corresponds
// to one detected hand. All data is fully copied out of native memory.
@interface MPCHandLandmarkerResult : NSObject

// Hand landmarks in normalized image coordinates ([0, 1]), per hand.
@property(nonatomic, readonly) NSArray<NSArray<MPCLandmark *> *> *landmarks;

// Hand landmarks in world coordinates (meters), per hand.
@property(nonatomic, readonly) NSArray<NSArray<MPCLandmark *> *> *worldLandmarks;

// Handedness classification, per hand.
@property(nonatomic, readonly) NSArray<NSArray<MPCCategory *> *> *handedness;

- (instancetype)initWithLandmarks:(NSArray<NSArray<MPCLandmark *> *> *)landmarks
                   worldLandmarks:(NSArray<NSArray<MPCLandmark *> *> *)worldLandmarks
                       handedness:(NSArray<NSArray<MPCCategory *> *> *)handedness
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
