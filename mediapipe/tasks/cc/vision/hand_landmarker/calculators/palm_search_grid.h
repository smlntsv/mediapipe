/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_GRID_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_GRID_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// Region generation maths of the palm search path. Pure functions, no state.

// Size of one tile along an axis of length 1 covered by `count` tiles, each
// sharing the fraction `overlap` of its size with its neighbour.
inline float PalmSearchTileSize(int count, float overlap) {
  return 1.0f / (1.0f + (count - 1) * (1.0f - overlap));
}

// The tiles of a `rows` x `columns` grid over the frame, row by row, as
// regions with id, grid position and requested box filled in. The first and
// last tile of each axis touch the frame edges. Empty when rows or columns
// is not positive; `overlap` is clamped to [0, 0.95].
inline std::vector<PalmSearchRegion> PalmSearchGridTiles(int rows, int columns,
                                                         float overlap) {
  std::vector<PalmSearchRegion> tiles;
  if (rows <= 0 || columns <= 0) return tiles;
  overlap = std::min(std::max(overlap, 0.0f), 0.95f);
  const float width = PalmSearchTileSize(columns, overlap);
  const float height = PalmSearchTileSize(rows, overlap);
  tiles.reserve(rows * columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      PalmSearchRegion tile;
      tile.id = kPalmSearchGridRegionIdBase + row * columns + column;
      tile.kind = PalmSearchRegionKind::kGrid;
      tile.grid_row = row;
      tile.grid_column = column;
      tile.requested_width = width;
      tile.requested_height = height;
      tile.requested_x_center = column * width * (1.0f - overlap) + width / 2;
      tile.requested_y_center = row * height * (1.0f - overlap) + height / 2;
      tiles.push_back(tile);
    }
  }
  return tiles;
}

// Indices of the grid tiles to run on a detector frame whose schedule cursor
// is `cursor`, and through `cursor_after` the cursor of the next detector
// frame. `per_frame` <= 0 or >= `tile_count`: every tile, cursor stays 0.
inline std::vector<int> PalmSearchScheduledTiles(int tile_count, int per_frame,
                                                 int cursor,
                                                 int* cursor_after) {
  std::vector<int> indices;
  if (tile_count <= 0) {
    *cursor_after = 0;
    return indices;
  }
  if (per_frame <= 0 || per_frame >= tile_count) {
    for (int i = 0; i < tile_count; ++i) indices.push_back(i);
    *cursor_after = 0;
    return indices;
  }
  cursor = ((cursor % tile_count) + tile_count) % tile_count;
  for (int i = 0; i < per_frame; ++i) {
    indices.push_back((cursor + i) % tile_count);
  }
  *cursor_after = (cursor + per_frame) % tile_count;
  return indices;
}

// Fills `region`'s sampled window from its requested box: image-to-tensor
// grows the box around its center to a square in pixels, because the model
// input is square and the detector graph keeps the aspect ratio.
inline void SetPalmSearchSampledWindow(int image_width, int image_height,
                                       PalmSearchRegion* region) {
  const float side_px = std::max(region->requested_width * image_width,
                                 region->requested_height * image_height);
  region->sampled_x_center = region->requested_x_center;
  region->sampled_y_center = region->requested_y_center;
  region->sampled_width = side_px / image_width;
  region->sampled_height = side_px / image_height;
}

// The rect to give image-to-tensor so that it samples `region`'s requested
// box of the frame with the task's `rotation` (radians, a multiple of 90
// degrees). Image-to-tensor multiplies a rect's width and height by the
// image's, then rotates it; for a quarter turn the two therefore swap, as in
// BaseVisionTaskApi::ConvertToNormalizedRect.
inline mediapipe::NormalizedRect PalmSearchRegionRect(
    const PalmSearchRegion& region, float rotation, int image_width,
    int image_height) {
  mediapipe::NormalizedRect rect;
  rect.set_x_center(region.requested_x_center);
  rect.set_y_center(region.requested_y_center);
  rect.set_rotation(rotation);
  const bool quarter_turn = std::abs(std::sin(rotation)) > 0.5f;
  if (quarter_turn) {
    rect.set_width(region.requested_height * image_height / image_width);
    rect.set_height(region.requested_width * image_width / image_height);
  } else {
    rect.set_width(region.requested_width);
    rect.set_height(region.requested_height);
  }
  return rect;
}

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_PALM_SEARCH_GRID_H_
