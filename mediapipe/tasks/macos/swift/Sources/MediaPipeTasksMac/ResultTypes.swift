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

/// A group of `Category` results produced by one classifier head — mirrors the
/// MediaPipe Tasks `Classifications` container (used for face blendshapes).
public struct Classifications: Sendable, Equatable {
    /// The classification categories, ordered by decreasing score.
    public var categories: [Category]
    /// The index of the classifier head these categories came from.
    public var headIndex: Int
    /// The optional name of the classifier head.
    public var headName: String?

    public init(categories: [Category], headIndex: Int = 0, headName: String? = nil) {
        self.categories = categories
        self.headIndex = headIndex
        self.headName = headName
    }
}

/// A dense matrix stored in column-major order, mirroring the MediaPipe Web
/// `Matrix` shape (`rows`, `columns`, `data`). Used for facial transformation
/// matrices.
public struct Matrix: Sendable, Equatable {
    public var rows: Int
    public var columns: Int
    /// `rows * columns` values in column-major order.
    public var data: [Float]

    public init(rows: Int, columns: Int, data: [Float]) {
        self.rows = rows
        self.columns = columns
        self.data = data
    }
}

/// A segmentation mask, mirroring the MediaPipe Web/iOS `MPMask` shape.
///
/// Milestone 1 does not produce masks (Pose `segmentationMasks` is `nil`); this
/// type exists so result shapes match MediaPipe and masks can be populated
/// later without an API change.
public struct MPMask: Sendable {
    public var width: Int
    public var height: Int
    /// Per-pixel float confidence mask in row-major order (`width * height`).
    public var float32Data: [Float]

    public init(width: Int, height: Int, float32Data: [Float]) {
        self.width = width
        self.height = height
        self.float32Data = float32Data
    }
}
