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

#import "MPCHandLandmarkerPalmSearch.h"

@implementation MPCPalmSearchOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _includeFullFrame = YES;
  }
  return self;
}

@end

@implementation MPCPalmSearchRegionDetection
@end

@implementation MPCPalmSearchRegion

- (instancetype)init {
  self = [super init];
  if (self) {
    _detections = @[];
    _candidates = @[];
  }
  return self;
}

@end

@implementation MPCPalmSearchMergedDetection

- (instancetype)init {
  self = [super init];
  if (self) {
    _sources = @[];
  }
  return self;
}

@end

@implementation MPCHandLandmarkerPalmSearch

- (instancetype)init {
  self = [super init];
  if (self) {
    _regions = @[];
    _merged = @[];
  }
  return self;
}

@end
