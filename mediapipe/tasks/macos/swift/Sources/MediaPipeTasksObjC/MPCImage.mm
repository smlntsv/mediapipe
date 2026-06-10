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

#import "MPCImageInternal.h"

#import "MPCError.h"

#include <stdatomic.h>

#include "mediapipe/tasks/c/core/common.h"  // MpErrorFree
#include "mediapipe/tasks/c/vision/core/image.h"

// Live-instance counter for ownership diagnostics (see +liveInstanceCount).
static _Atomic(long) gMPCImageLiveCount = 0;

@implementation MPCImage {
  // Exclusively owned native image; freed exactly once in -dealloc.
  MpImagePtr _image;
}

+ (NSInteger)liveInstanceCount {
  return (NSInteger)atomic_load(&gMPCImageLiveCount);
}

- (instancetype)initWithRGBAData:(NSData *)rgbaData
                           width:(NSInteger)width
                          height:(NSInteger)height
                           error:(NSError **)error {
  self = [super init];
  if (!self) {
    return nil;
  }

  const NSInteger expected = width * height * 4;
  if (width <= 0 || height <= 0 || (NSInteger)rgbaData.length != expected) {
    if (error) {
      *error = MPCMakeError(
          kMpInvalidArgument,
          [[NSString stringWithFormat:
                         @"RGBA buffer size %lu does not match %ldx%ld (expected %ld bytes).",
                         (unsigned long)rgbaData.length, (long)width, (long)height,
                         (long)expected] UTF8String]);
    }
    return nil;
  }

  char *errorMsg = NULL;
  MpImagePtr image = NULL;
  MpStatus status = MpImageCreateFromUint8Data(
      kMpImageFormatSrgba, (int)width, (int)height,
      (const uint8_t *)rgbaData.bytes, (int)rgbaData.length, &image, &errorMsg);
  if (status != kMpOk) {
    if (error) {
      *error = MPCMakeError(status, errorMsg);  // copies the message
    }
    if (errorMsg) {
      MpErrorFree(errorMsg);
    }
    return nil;
  }

  _image = image;
  atomic_fetch_add(&gMPCImageLiveCount, 1);
  return self;
}

- (MpImagePtr)imagePtr {
  return _image;
}

- (void)dealloc {
  if (_image) {
    MpImageFree(_image);
    _image = NULL;
    atomic_fetch_sub(&gMPCImageLiveCount, 1);
  }
}

@end
