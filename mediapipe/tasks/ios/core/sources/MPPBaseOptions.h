// Copyright 2022 The MediaPipe Authors.
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

/**
 * The delegate to run MediaPipe. If the delegate is not set, the default
 * delegate CPU is used.
 */
typedef NS_ENUM(NSUInteger, MPPDelegate) {
  MPPDelegateCPU,
  MPPDelegateGPU,
  /**
   * Runs inference through Core ML (Apple Neural Engine when available).
   * Requires models pre-converted with
   * `mediapipe/tasks/macos/experiments/coreml_ane/convert_delegate.py`, looked
   * up as `<sha256-of-tflite>.mlmodelc` in `coreMLModelCacheDirectory`. A
   * model without a converted counterpart falls back to CPU/XNNPACK for that
   * model only, so this delegate is always safe to request.
   */
  MPPDelegateCoreML,
} NS_SWIFT_NAME(Delegate);

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds the base options that is used for creation of any type of task. It has fields with
 * important information acceleration configuration, TFLite model source etc.
 */
NS_SWIFT_NAME(BaseOptions)
@interface MPPBaseOptions : NSObject <NSCopying>

/** The path to the model asset to open and mmap in memory. */
@property(nonatomic, copy) NSString *modelAssetPath;

/** Overrides the default backend to use for the provided model. */
@property(nonatomic) MPPDelegate delegate;

/**
 * MPPDelegateCoreML only: directory containing the converted
 * `<sha256-of-tflite>.mlmodelc` models. When nil (the default), the directory
 * of `modelAssetPath` is used — i.e. converted models placed next to the
 * `.task` file are found automatically.
 */
@property(nonatomic, copy, nullable) NSString *coreMLModelCacheDirectory;

@end

NS_ASSUME_NONNULL_END
