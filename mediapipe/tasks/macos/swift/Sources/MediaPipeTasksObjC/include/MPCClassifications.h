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

NS_ASSUME_NONNULL_BEGIN

// A group of categories from one classifier head (e.g. face blendshapes),
// mirroring the MediaPipe Tasks `Classifications` container.
@interface MPCClassifications : NSObject

@property(nonatomic, readonly) NSArray<MPCCategory *> *categories;
@property(nonatomic, readonly) NSInteger headIndex;
@property(nonatomic, readonly, nullable) NSString *headName;

- (instancetype)initWithCategories:(NSArray<MPCCategory *> *)categories
                         headIndex:(NSInteger)headIndex
                          headName:(nullable NSString *)headName
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
