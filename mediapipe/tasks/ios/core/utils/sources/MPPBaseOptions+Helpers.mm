// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

namespace {
using BaseOptionsProto = ::mediapipe::tasks::core::proto::BaseOptions;
using InferenceCalculatorOptionsProto = ::mediapipe::InferenceCalculatorOptions;
}

@implementation MPPBaseOptions (Helpers)

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto withUseStreamMode:(BOOL)useStreamMode {
  [self copyToProto:baseOptionsProto];
  baseOptionsProto->set_use_stream_mode(useStreamMode);
}

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto {
  baseOptionsProto->Clear();

  if (self.modelAssetPath) {
    baseOptionsProto->mutable_model_asset()->set_file_name(self.modelAssetPath.UTF8String);
  }

  if (self.delegate == MPPDelegateGPU) {
    baseOptionsProto->mutable_acceleration()->mutable_gpu()->MergeFrom(
        InferenceCalculatorOptionsProto::Delegate::Gpu());
  } else if (self.delegate == MPPDelegateCoreML) {
    auto *coreml = baseOptionsProto->mutable_acceleration()->mutable_coreml();
    // Default the model cache dir to the model asset's directory so converted
    // models shipped next to the .task file are found with no extra setup
    // (mirrors the macOS wrapper's behavior).
    NSString *cacheDir = self.coreMLModelCacheDirectory;
    if (cacheDir.length == 0 && self.modelAssetPath.length > 0) {
      cacheDir = [self.modelAssetPath stringByDeletingLastPathComponent];
    }
    if (cacheDir.length > 0) {
      coreml->set_model_cache_dir(cacheDir.UTF8String);
    }
    // compute_units stays at its default (ALL): Core ML picks the ANE.
  }
}

@end
