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

// Private (non-public) helper. Not part of the Swift-importable surface.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSErrorDomain const MPCErrorDomain;

// Builds an NSError from a MediaPipe C status code and (optional) error message.
// The message is COPIED into the NSError before this returns; the caller still
// owns `cMessage` and must free it (with MpErrorFree) afterwards.
NSError *MPCMakeError(int statusCode, const char *_Nullable cMessage);

NS_ASSUME_NONNULL_END
