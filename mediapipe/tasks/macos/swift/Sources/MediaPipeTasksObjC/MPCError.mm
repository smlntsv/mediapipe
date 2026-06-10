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

#import "MPCError.h"

NSErrorDomain const MPCErrorDomain = @"com.google.mediapipe.tasks.macos";

NSError *MPCMakeError(int statusCode, const char *cMessage) {
  NSString *message = nil;
  if (cMessage != NULL) {
    // Copy the C string into an NSString. After this, the native buffer can be
    // safely freed by the caller; we hold an independent copy.
    message = [NSString stringWithUTF8String:cMessage];
  }
  if (message == nil) {
    message = [NSString stringWithFormat:@"MediaPipe task failed with status %d.",
                                         statusCode];
  }
  return [NSError errorWithDomain:MPCErrorDomain
                             code:statusCode
                         userInfo:@{NSLocalizedDescriptionKey : message}];
}
