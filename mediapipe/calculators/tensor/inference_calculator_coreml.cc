// Copyright 2026 The MediaPipe Authors.
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
//
// InferenceCalculatorCoreMl: runs inference through Core ML instead of TFLite,
// which is the only route to the Apple Neural Engine (TFLite has no Core ML
// delegate in this tree; its GPU delegate tops out at Metal).
//
// The TFLite flatbuffer arriving via the MODEL side packet (or model_path) is
// used as a lookup key: the calculator computes its SHA-256 and loads
// "<hex>.mlmodelc" from options.delegate().coreml().model_cache_dir(). The
// converted model must follow the "input_<i>"/"output_<i>" feature-naming
// convention in TFLite model I/O order (produced by
// mediapipe/tasks/macos/experiments/coreml_ane/convert_delegate.py, which
// validates numeric equivalence at conversion time).
//
// When no converted model exists for a given flatbuffer, the calculator
// transparently falls back to TFLite CPU/XNNPACK, so tasks whose bundles
// contain a mix of converted and unconverted models keep working.
//
// ANE-specific care (each of these produced silent garbage in the prototype):
//  - MLMultiArray outputs can be stride-padded (non-contiguous); outputs are
//    copied honoring .strides, with a memcpy fast path when dense.
//  - Outputs may come back as float16; both fp16 and fp32 are handled.
//  - MediaPipe output tensors use the TFLite output shapes (not the Core ML
//    shapes), so downstream Tensors* calculators see exactly what they expect.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <CommonCrypto/CommonDigest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace api2 {

namespace {

std::string Sha256Hex(const void* data, size_t size) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data, static_cast<CC_LONG>(size), digest);
  std::string hex;
  hex.reserve(CC_SHA256_DIGEST_LENGTH * 2);
  for (unsigned char byte : digest) {
    absl::StrAppendFormat(&hex, "%02x", byte);
  }
  return hex;
}

MLComputeUnits ToComputeUnits(
    mediapipe::InferenceCalculatorOptions::Delegate::CoreMl::ComputeUnits
        units) {
  switch (units) {
    case mediapipe::InferenceCalculatorOptions::Delegate::CoreMl::
        CPU_AND_NEURAL_ENGINE:
      return MLComputeUnitsCPUAndNeuralEngine;
    case mediapipe::InferenceCalculatorOptions::Delegate::CoreMl::CPU_AND_GPU:
      return MLComputeUnitsCPUAndGPU;
    case mediapipe::InferenceCalculatorOptions::Delegate::CoreMl::CPU_ONLY:
      return MLComputeUnitsCPUOnly;
    case mediapipe::InferenceCalculatorOptions::Delegate::CoreMl::ALL:
      return MLComputeUnitsAll;
  }
  return MLComputeUnitsAll;
}

absl::Status NSErrorToStatus(NSError* error, absl::string_view context) {
  return absl::InternalError(absl::StrCat(
      context, ": ",
      error ? error.localizedDescription.UTF8String : "unknown Core ML error"));
}

// Copies an MLMultiArray into `dst` (float32, row-major) honoring the array's
// strides. ANE-produced outputs are frequently padded/non-contiguous, so a
// linear copy would silently read zeros/garbage.
absl::Status CopyMultiArrayToFloats(MLMultiArray* array, float* dst,
                                    int64_t expected_elements) {
  const int rank = static_cast<int>(array.shape.count);
  std::vector<int64_t> shape(rank), strides(rank);
  int64_t n = 1;
  for (int d = 0; d < rank; ++d) {
    shape[d] = array.shape[d].integerValue;
    strides[d] = array.strides[d].integerValue;
    n *= shape[d];
  }
  RET_CHECK_EQ(n, expected_elements)
      << "Core ML output element count does not match the TFLite output shape.";

  bool dense = true;
  int64_t expect = 1;
  for (int d = rank - 1; d >= 0; --d) {
    if (strides[d] != expect) {
      dense = false;
      break;
    }
    expect *= shape[d];
  }

  const MLMultiArrayDataType data_type = array.dataType;
  RET_CHECK(data_type == MLMultiArrayDataTypeFloat32 ||
            data_type == MLMultiArrayDataTypeFloat16)
      << "Unsupported Core ML output data type: " << data_type;

  __block absl::Status copy_status = absl::OkStatus();
  [array getBytesWithHandler:^(const void* bytes, NSInteger /*size*/) {
    if (dense) {
      if (data_type == MLMultiArrayDataTypeFloat32) {
        std::memcpy(dst, bytes, n * sizeof(float));
      } else {
        const __fp16* src = static_cast<const __fp16*>(bytes);
        for (int64_t i = 0; i < n; ++i) dst[i] = src[i];
      }
      return;
    }
    std::vector<int64_t> idx(rank, 0);
    for (int64_t i = 0; i < n; ++i) {
      int64_t offset = 0;
      for (int d = 0; d < rank; ++d) offset += idx[d] * strides[d];
      if (data_type == MLMultiArrayDataTypeFloat32) {
        dst[i] = static_cast<const float*>(bytes)[offset];
      } else {
        dst[i] = static_cast<const __fp16*>(bytes)[offset];
      }
      for (int d = rank - 1; d >= 0; --d) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
  }];
  return copy_status;
}

}  // namespace

class InferenceCalculatorCoreMlImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorCoreMl,
                                         InferenceCalculatorCoreMlImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::Status InitCoreMlModel(CalculatorContext* cc,
                               const std::string& compiled_model_path);
  absl::Status InitFallbackRunner(CalculatorContext* cc);

  // TfLite requires the model to stay alive while in use; it is also the
  // source of the authoritative I/O shapes for the output tensors.
  Packet<TfLiteModelPtr> model_packet_;

  // Core ML path.
  MLModel* model_ = nil;
  NSMutableArray<NSString*>* input_names_ = nil;
  NSMutableArray<NSString*>* output_names_ = nil;
  // Expected Core ML input shapes (from the model's feature descriptions).
  std::vector<NSArray<NSNumber*>*> input_shapes_;
  std::vector<NSArray<NSNumber*>*> input_strides_;
  std::vector<int64_t> input_elements_;
  // TFLite output shapes; MediaPipe output tensors are created with these.
  std::vector<Tensor::Shape> output_shapes_;

  // TFLite CPU/XNNPACK fallback when no converted model exists.
  std::unique_ptr<InferenceRunner> fallback_runner_;
};

absl::Status InferenceCalculatorCoreMlImpl::UpdateContract(
    CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  RET_CHECK(!kDelegate(cc).IsConnected())
      << "Delegate configuration through side packet is not supported.";
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";
  WarnFeedbackTensorsUnsupported(cc);
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCoreMlImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  const auto& coreml_options = options.delegate().coreml();

  MP_ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const tflite::FlatBufferModel& flatbuffer = *model_packet_.Get();
  const std::string model_hash = Sha256Hex(flatbuffer.allocation()->base(),
                                           flatbuffer.allocation()->bytes());

  std::string compiled_model_path;
  if (!coreml_options.model_cache_dir().empty()) {
    compiled_model_path = absl::StrCat(coreml_options.model_cache_dir(), "/",
                                       model_hash, ".mlmodelc");
  }

  BOOL is_dir = NO;
  const bool model_available =
      !compiled_model_path.empty() &&
      [[NSFileManager defaultManager]
          fileExistsAtPath:@(compiled_model_path.c_str())
               isDirectory:&is_dir];
  if (model_available) {
    MP_RETURN_IF_ERROR(InitCoreMlModel(cc, compiled_model_path));
    // I/O mapping uses the TFLite tensor names; Core ML features are already
    // ordered by the input_<i>/output_<i> convention.
    MP_ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
    MP_ASSIGN_OR_RETURN(
        const auto io_names,
        InferenceIoMapper::GetInputOutputTensorNamesFromModel(
            flatbuffer, op_resolver_packet.Get()));
    return InferenceCalculatorNodeImpl::UpdateIoMapping(cc, io_names);
  }

  ABSL_LOG(WARNING) << absl::StrFormat(
      "InferenceCalculatorCoreMl: no converted Core ML model found for "
      "tflite model with SHA-256 %s (looked for %s). Falling back to TFLite "
      "CPU/XNNPACK for this model. Convert it with "
      "mediapipe/tasks/macos/experiments/coreml_ane/convert_delegate.py to "
      "enable the Neural Engine.",
      model_hash,
      compiled_model_path.empty() ? "<model_cache_dir unset>"
                                  : compiled_model_path);
  MP_RETURN_IF_ERROR(InitFallbackRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, fallback_runner_->GetInputOutputTensorNames());
}

absl::Status InferenceCalculatorCoreMlImpl::InitCoreMlModel(
    CalculatorContext* cc, const std::string& compiled_model_path) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();

  MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
  config.computeUnits =
      ToComputeUnits(options.delegate().coreml().compute_units());

  NSError* error = nil;
  NSURL* url =
      [NSURL fileURLWithPath:@(compiled_model_path.c_str()) isDirectory:YES];
  model_ = [MLModel modelWithContentsOfURL:url configuration:config
                                     error:&error];
  if (!model_) {
    return NSErrorToStatus(
        error, absl::StrCat("Failed to load compiled Core ML model at ",
                            compiled_model_path));
  }

  // The TFLite flatbuffer is the source of truth for the I/O contract that
  // downstream calculators rely on.
  const tflite::Model* tflite_model = model_packet_.Get()->GetModel();
  RET_CHECK(tflite_model->subgraphs() && tflite_model->subgraphs()->size() > 0);
  const tflite::SubGraph& subgraph = *tflite_model->subgraphs()->Get(0);

  const int num_inputs = subgraph.inputs()->size();
  const int num_outputs = subgraph.outputs()->size();
  input_names_ = [NSMutableArray arrayWithCapacity:num_inputs];
  output_names_ = [NSMutableArray arrayWithCapacity:num_outputs];

  NSDictionary<NSString*, MLFeatureDescription*>* input_descriptions =
      model_.modelDescription.inputDescriptionsByName;
  for (int i = 0; i < num_inputs; ++i) {
    const tflite::Tensor& tensor =
        *subgraph.tensors()->Get(subgraph.inputs()->Get(i));
    RET_CHECK(tensor.type() == tflite::TensorType_FLOAT32) << absl::StrFormat(
        "InferenceCalculatorCoreMl supports float32 inputs only; input %d of "
        "the tflite model is of type %s.",
        i, tflite::EnumNameTensorType(tensor.type()));
    int64_t tflite_elements = 1;
    for (int d = 0; d < tensor.shape()->size(); ++d) {
      tflite_elements *= tensor.shape()->Get(d);
    }

    NSString* name = [NSString stringWithFormat:@"input_%d", i];
    MLFeatureDescription* description = input_descriptions[name];
    RET_CHECK(description != nil && description.multiArrayConstraint != nil)
        << absl::StrCat(
               "Converted Core ML model at ", compiled_model_path,
               " does not expose the expected multiarray input feature '",
               name.UTF8String,
               "'. Re-convert the model with convert_delegate.py.");

    NSArray<NSNumber*>* shape = description.multiArrayConstraint.shape;
    int64_t coreml_elements = 1;
    NSMutableArray<NSNumber*>* strides =
        [NSMutableArray arrayWithCapacity:shape.count];
    for (NSUInteger d = 0; d < shape.count; ++d) [strides addObject:@(0)];
    int64_t stride = 1;
    for (NSInteger d = shape.count - 1; d >= 0; --d) {
      strides[d] = @(stride);
      stride *= shape[d].integerValue;
      coreml_elements *= shape[d].integerValue;
    }
    RET_CHECK_EQ(coreml_elements, tflite_elements) << absl::StrCat(
        "Core ML input feature '", name.UTF8String,
        "' element count does not match the tflite model input.");

    [input_names_ addObject:name];
    input_shapes_.push_back(shape);
    input_strides_.push_back(strides);
    input_elements_.push_back(tflite_elements);
  }

  NSSet<NSString*>* output_features = [NSSet
      setWithArray:model_.modelDescription.outputDescriptionsByName.allKeys];
  output_shapes_.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const tflite::Tensor& tensor =
        *subgraph.tensors()->Get(subgraph.outputs()->Get(i));
    RET_CHECK(tensor.type() == tflite::TensorType_FLOAT32) << absl::StrFormat(
        "InferenceCalculatorCoreMl supports float32 outputs only; output %d "
        "of the tflite model is of type %s.",
        i, tflite::EnumNameTensorType(tensor.type()));
    std::vector<int> dims(tensor.shape()->size());
    for (int d = 0; d < tensor.shape()->size(); ++d) {
      dims[d] = tensor.shape()->Get(d);
      RET_CHECK_GT(dims[d], 0) << "Dynamic tflite output shapes are not "
                                  "supported by InferenceCalculatorCoreMl.";
    }
    NSString* name = [NSString stringWithFormat:@"output_%d", i];
    RET_CHECK([output_features containsObject:name]) << absl::StrCat(
        "Converted Core ML model at ", compiled_model_path,
        " does not expose the expected output feature '", name.UTF8String,
        "'. Re-convert the model with convert_delegate.py.");
    [output_names_ addObject:name];
    output_shapes_.emplace_back(Tensor::Shape{dims});
  }
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCoreMlImpl::InitFallbackRunner(
    CalculatorContext* cc) {
  MP_ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads =
      GetXnnpackNumThreads(/*opts_has_delegate=*/false, options.delegate());
  TfLiteDelegatePtr delegate(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                             &TfLiteXNNPackDelegateDelete);
  MP_ASSIGN_OR_RETURN(
      fallback_runner_,
      CreateInferenceInterpreterDelegateRunner(
          model_packet_, op_resolver_packet, std::move(delegate),
          options.cpu_num_thread(), &options.input_output_config()));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorCoreMlImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  if (fallback_runner_) {
    return fallback_runner_->Run(cc, tensor_span);
  }

  RET_CHECK_EQ(tensor_span.size(), static_cast<int>(input_elements_.size()))
      << "Input tensor count does not match the model.";

  // Wrap each input tensor's CPU buffer as an MLMultiArray without copying.
  // The read views must outlive the prediction call.
  std::vector<Tensor::CpuReadView> read_views;
  read_views.reserve(tensor_span.size());
  NSMutableDictionary<NSString*, MLFeatureValue*>* features =
      [NSMutableDictionary dictionaryWithCapacity:tensor_span.size()];
  NSError* error = nil;
  for (int i = 0; i < tensor_span.size(); ++i) {
    const Tensor& tensor = tensor_span[i];
    RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32)
        << "InferenceCalculatorCoreMl expects float32 input tensors.";
    RET_CHECK_EQ(tensor.shape().num_elements(), input_elements_[i])
        << "Input tensor size does not match the model input.";
    read_views.push_back(tensor.GetCpuReadView());
    // MLMultiArray requires a mutable pointer, but prediction only reads it.
    void* data = const_cast<float*>(read_views.back().buffer<float>());
    MLMultiArray* array =
        [[MLMultiArray alloc] initWithDataPointer:data
                                            shape:input_shapes_[i]
                                         dataType:MLMultiArrayDataTypeFloat32
                                          strides:input_strides_[i]
                                      deallocator:nil
                                            error:&error];
    if (!array) {
      return NSErrorToStatus(error, "Failed to wrap input tensor");
    }
    features[input_names_[i]] =
        [MLFeatureValue featureValueWithMultiArray:array];
  }

  MLDictionaryFeatureProvider* provider =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
                                                        error:&error];
  if (!provider) {
    return NSErrorToStatus(error, "Failed to create feature provider");
  }
  id<MLFeatureProvider> outputs = [model_ predictionFromFeatures:provider
                                                           error:&error];
  if (!outputs) {
    return NSErrorToStatus(error, "Core ML prediction failed");
  }

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_shapes_.size());
  for (size_t i = 0; i < output_shapes_.size(); ++i) {
    MLMultiArray* array =
        [outputs featureValueForName:output_names_[i]].multiArrayValue;
    RET_CHECK(array != nil) << absl::StrCat(
        "Core ML prediction is missing output feature '",
        output_names_[i].UTF8String, "'.");
    output_tensors.emplace_back(Tensor::ElementType::kFloat32,
                                output_shapes_[i]);
    auto write_view = output_tensors.back().GetCpuWriteView();
    MP_RETURN_IF_ERROR(CopyMultiArrayToFloats(
        array, write_view.buffer<float>(),
        output_tensors.back().shape().num_elements()));
  }
  return output_tensors;
}

absl::Status InferenceCalculatorCoreMlImpl::Close(CalculatorContext* cc) {
  model_ = nil;
  input_names_ = nil;
  output_names_ = nil;
  input_shapes_.clear();
  input_strides_.clear();
  fallback_runner_ = nullptr;
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
