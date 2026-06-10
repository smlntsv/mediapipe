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

/// Standard MediaPipe edge lists for drawing skeletal connections.
enum Connections {
    /// 21-point hand connections.
    static let hand: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4),          // thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          // index
        (5, 9), (9, 10), (10, 11), (11, 12),     // middle
        (9, 13), (13, 14), (14, 15), (15, 16),   // ring
        (13, 17), (17, 18), (18, 19), (19, 20),  // pinky
        (0, 17),                                 // palm base
    ]

    /// 33-point pose connections (torso + limbs; face mesh omitted).
    static let pose: [(Int, Int)] = [
        (11, 12),                                          // shoulders
        (11, 13), (13, 15), (12, 14), (14, 16),            // arms
        (15, 17), (15, 19), (15, 21), (17, 19),            // left hand
        (16, 18), (16, 20), (16, 22), (18, 20),            // right hand
        (11, 23), (12, 24), (23, 24),                      // torso
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  // left leg
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  // right leg
    ]
}
