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

// An Objective-C value object mirroring a single MediaPipe landmark. Used for
// both normalized landmarks (coordinates in [0, 1]) and world landmarks
// (coordinates in meters). All data is fully copied out of the MediaPipe C
// result; this object holds no pointers into native result memory.
@interface MPCLandmark : NSObject

@property(nonatomic, readonly) float x;
@property(nonatomic, readonly) float y;
@property(nonatomic, readonly) float z;

@property(nonatomic, readonly) BOOL hasVisibility;
@property(nonatomic, readonly) float visibility;

@property(nonatomic, readonly) BOOL hasPresence;
@property(nonatomic, readonly) float presence;

@property(nonatomic, readonly, nullable) NSString *name;

- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
            hasVisibility:(BOOL)hasVisibility
               visibility:(float)visibility
              hasPresence:(BOOL)hasPresence
                 presence:(float)presence
                     name:(nullable NSString *)name NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
