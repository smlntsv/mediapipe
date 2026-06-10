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

// Private Obj-C++ helpers that deep-copy MediaPipe C result containers into
// Objective-C value objects. Kept out of the public `include/` dir because it
// references the C++-style MediaPipe headers.

#import "MPCCategory.h"
#import "MPCLandmark.h"

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"

// Deep-copies a native landmark list into ObjC objects. Works for both
// MpNormalizedLandmarks and MpLandmarks (identical field layout).
template <typename LandmarksT>
static inline NSArray<MPCLandmark *> *MPCCopyLandmarks(const LandmarksT &list) {
  NSMutableArray<MPCLandmark *> *out =
      [NSMutableArray arrayWithCapacity:list.landmarks_count];
  for (uint32_t i = 0; i < list.landmarks_count; ++i) {
    const auto &lm = list.landmarks[i];
    NSString *name = lm.name ? [NSString stringWithUTF8String:lm.name] : nil;
    [out addObject:[[MPCLandmark alloc] initWithX:lm.x
                                                y:lm.y
                                                z:lm.z
                                    hasVisibility:lm.has_visibility ? YES : NO
                                       visibility:lm.visibility
                                      hasPresence:lm.has_presence ? YES : NO
                                         presence:lm.presence
                                             name:name]];
  }
  return out;
}

static inline NSArray<MPCCategory *> *MPCCopyCategories(const MpCategories &list) {
  NSMutableArray<MPCCategory *> *out =
      [NSMutableArray arrayWithCapacity:list.categories_count];
  for (uint32_t i = 0; i < list.categories_count; ++i) {
    const MpCategory &c = list.categories[i];
    NSString *categoryName =
        c.category_name ? [NSString stringWithUTF8String:c.category_name] : nil;
    NSString *displayName =
        c.display_name ? [NSString stringWithUTF8String:c.display_name] : nil;
    [out addObject:[[MPCCategory alloc] initWithIndex:c.index
                                                score:c.score
                                         categoryName:categoryName
                                          displayName:displayName]];
  }
  return out;
}
