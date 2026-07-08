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

import Foundation
import MediaPipeTasksObjC

/// A landmark in normalized image coordinates: `x` and `y` are in `[0, 1]`,
/// `z` represents depth with the origin at the image center.
public struct NormalizedLandmark: Sendable, Equatable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var visibility: Float?
    public var presence: Float?
    public var name: String?

    init(_ landmark: MPCLandmark) {
        x = landmark.x
        y = landmark.y
        z = landmark.z
        visibility = landmark.hasVisibility ? landmark.visibility : nil
        presence = landmark.hasPresence ? landmark.presence : nil
        name = landmark.name
    }
}

/// A landmark in world coordinates (meters), with the origin at the geometric
/// center of the detected object.
public struct Landmark: Sendable, Equatable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var visibility: Float?
    public var presence: Float?
    public var name: String?

    init(_ landmark: MPCLandmark) {
        x = landmark.x
        y = landmark.y
        z = landmark.z
        visibility = landmark.hasVisibility ? landmark.visibility : nil
        presence = landmark.hasPresence ? landmark.presence : nil
        name = landmark.name
    }
}

/// A classification category (e.g. handedness).
public struct Category: Sendable, Equatable {
    public var index: Int
    public var score: Float
    public var categoryName: String?
    public var displayName: String?

    init(_ category: MPCCategory) {
        index = category.index
        score = category.score
        categoryName = category.categoryName
        displayName = category.displayName
    }
}
