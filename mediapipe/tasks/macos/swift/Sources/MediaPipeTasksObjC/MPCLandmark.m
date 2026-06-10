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

#import "MPCLandmark.h"

@implementation MPCLandmark

- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
            hasVisibility:(BOOL)hasVisibility
               visibility:(float)visibility
              hasPresence:(BOOL)hasPresence
                 presence:(float)presence
                     name:(NSString *)name {
  self = [super init];
  if (self) {
    _x = x;
    _y = y;
    _z = z;
    _hasVisibility = hasVisibility;
    _visibility = visibility;
    _hasPresence = hasPresence;
    _presence = presence;
    _name = [name copy];
  }
  return self;
}

@end
