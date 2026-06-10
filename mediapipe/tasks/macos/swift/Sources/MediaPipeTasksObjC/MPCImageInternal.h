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

// Private bridging header: exposes the native MpImage pointer to other shim
// translation units. This file references MediaPipe's C++-style headers and so
// is intentionally kept OUT of the public `include/` directory (Swift never
// sees it).

#import "MPCImage.h"

#include "mediapipe/tasks/c/vision/core/image.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPCImage (Internal)

// The native image owned by this object. Valid for the lifetime of the MPCImage.
@property(nonatomic, readonly) MpImagePtr imagePtr;

@end

NS_ASSUME_NONNULL_END
