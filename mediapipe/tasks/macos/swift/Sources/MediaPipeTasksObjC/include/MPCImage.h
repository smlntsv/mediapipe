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

NS_ASSUME_NONNULL_BEGIN

// Wraps a native MediaPipe image (MpImage). Created from a tightly-packed RGBA8
// buffer (sRGBA, one byte per channel, no row padding). This object exclusively
// owns the underlying native image and frees it exactly once on deallocation.
@interface MPCImage : NSObject

// `rgbaData` must contain exactly width * height * 4 bytes in R,G,B,A order.
- (nullable instancetype)initWithRGBAData:(NSData *)rgbaData
                                    width:(NSInteger)width
                                   height:(NSInteger)height
                                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

// Number of MPCImage instances currently alive (debug/ownership diagnostics).
// Should stay near 0 during steady-state frame processing.
+ (NSInteger)liveInstanceCount;

@end

NS_ASSUME_NONNULL_END
