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

#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/palm_search_grid.h"

#include <cmath>
#include <vector>

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {
namespace {

using ::testing::ElementsAre;

TEST(PalmSearchGridTest, TilesWithoutOverlapPartitionTheFrame) {
  const auto tiles = PalmSearchGridTiles(2, 4, 0.0f);
  ASSERT_EQ(tiles.size(), 8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(tiles[i].id, 1 + i);
    EXPECT_EQ(tiles[i].kind, PalmSearchRegionKind::kGrid);
    EXPECT_EQ(tiles[i].grid_row, i / 4);
    EXPECT_EQ(tiles[i].grid_column, i % 4);
    EXPECT_FLOAT_EQ(tiles[i].requested_width, 0.25f);
    EXPECT_FLOAT_EQ(tiles[i].requested_height, 0.5f);
    EXPECT_FLOAT_EQ(tiles[i].requested_x_center, 0.125f + 0.25f * (i % 4));
    EXPECT_FLOAT_EQ(tiles[i].requested_y_center, 0.25f + 0.5f * (i / 4));
  }
}

// With overlap the tiles grow, still reach both frame edges, and each
// shares the configured fraction of its size with its neighbour.
TEST(PalmSearchGridTest, OverlapGrowsTilesAndKeepsTheEdges) {
  for (float overlap : {0.1f, 0.2f, 0.3f}) {
    for (int count : {2, 3, 4}) {
      const auto tiles = PalmSearchGridTiles(1, count, overlap);
      const float width = 1.0f / (1.0f + (count - 1) * (1.0f - overlap));
      ASSERT_EQ(tiles.size(), count);
      EXPECT_NEAR(tiles[0].requested_x_center - width / 2, 0.0f, 1e-6);
      EXPECT_NEAR(tiles[count - 1].requested_x_center + width / 2, 1.0f, 1e-6);
      for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(tiles[i].requested_width, width, 1e-6);
        EXPECT_FLOAT_EQ(tiles[i].requested_height, 1.0f);
        if (i > 0) {
          const float shared =
              (tiles[i - 1].requested_x_center + width / 2) -
              (tiles[i].requested_x_center - width / 2);
          EXPECT_NEAR(shared, overlap * width, 1e-6);
        }
      }
    }
  }
  // The documented magnifications: 4 columns at overlap 0.2 is 3.4x.
  EXPECT_NEAR(1.0f / PalmSearchTileSize(4, 0.2f), 3.4f, 1e-5);
  EXPECT_NEAR(1.0f / PalmSearchTileSize(2, 0.1f), 1.9f, 1e-5);
}

TEST(PalmSearchGridTest, NoGrid) {
  EXPECT_TRUE(PalmSearchGridTiles(0, 4, 0.0f).empty());
  EXPECT_TRUE(PalmSearchGridTiles(3, 0, 0.0f).empty());
}

TEST(PalmSearchGridTest, ScheduleAllTiles) {
  int cursor_after = -1;
  EXPECT_THAT(PalmSearchScheduledTiles(4, 0, 0, &cursor_after),
              ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(cursor_after, 0);
  EXPECT_THAT(PalmSearchScheduledTiles(4, 9, 3, &cursor_after),
              ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(cursor_after, 0);
  EXPECT_TRUE(PalmSearchScheduledTiles(0, 1, 0, &cursor_after).empty());
}

// Every tile is visited exactly once per `ceil(tiles / k)` detector frames
// when k divides the tile count, and the cursor wraps.
TEST(PalmSearchGridTest, ScheduleRoundRobin) {
  int cursor = 0;
  std::vector<int> visited;
  for (int frame = 0; frame < 6; ++frame) {
    for (int index : PalmSearchScheduledTiles(6, 1, cursor, &cursor)) {
      visited.push_back(index);
    }
  }
  EXPECT_THAT(visited, ElementsAre(0, 1, 2, 3, 4, 5));
  EXPECT_EQ(cursor, 0);

  cursor = 0;
  EXPECT_THAT(PalmSearchScheduledTiles(5, 2, cursor, &cursor),
              ElementsAre(0, 1));
  EXPECT_THAT(PalmSearchScheduledTiles(5, 2, cursor, &cursor),
              ElementsAre(2, 3));
  EXPECT_THAT(PalmSearchScheduledTiles(5, 2, cursor, &cursor),
              ElementsAre(4, 0));
  EXPECT_EQ(cursor, 1);
}

// A region is sampled as a square in pixels around its center.
TEST(PalmSearchGridTest, SampledWindowIsSquareInPixels) {
  PalmSearchRegion region;
  region.requested_x_center = 0.125f;
  region.requested_y_center = 0.125f;
  region.requested_width = 0.25f;   // 320 px of 1280
  region.requested_height = 0.25f;  // 180 px of 720
  SetPalmSearchSampledWindow(1280, 720, &region);
  EXPECT_FLOAT_EQ(region.sampled_width * 1280, 320.0f);
  EXPECT_FLOAT_EQ(region.sampled_height * 720, 320.0f);
  EXPECT_FLOAT_EQ(region.sampled_x_center, 0.125f);
  EXPECT_FLOAT_EQ(region.sampled_y_center, 0.125f);

  // Portrait frame: the height is the longer side of the tile.
  region.requested_width = 0.5f;   // 360 px of 720
  region.requested_height = 0.5f;  // 640 px of 1280
  SetPalmSearchSampledWindow(720, 1280, &region);
  EXPECT_FLOAT_EQ(region.sampled_width * 720, 640.0f);
  EXPECT_FLOAT_EQ(region.sampled_height * 1280, 640.0f);
}

// The rect covers the requested box of the frame at every rotation: its
// width and height, multiplied by the image's and rotated, give the box.
TEST(PalmSearchGridTest, RegionRectAtFourRotations) {
  PalmSearchRegion region;
  region.requested_x_center = 0.3f;
  region.requested_y_center = 0.6f;
  region.requested_width = 0.25f;  // 320 px
  region.requested_height = 0.5f;  // 360 px
  const int width = 1280;
  const int height = 720;
  for (int quarter = 0; quarter < 4; ++quarter) {
    const float rotation = -quarter * M_PI / 2;
    const NormalizedRect rect =
        PalmSearchRegionRect(region, rotation, width, height);
    EXPECT_FLOAT_EQ(rect.x_center(), 0.3f);
    EXPECT_FLOAT_EQ(rect.y_center(), 0.6f);
    EXPECT_FLOAT_EQ(rect.rotation(), rotation);
    // Extent of the rotated rect along the frame's axes, in pixels.
    const float w = rect.width() * width;
    const float h = rect.height() * height;
    const float extent_x =
        std::abs(w * std::cos(rotation)) + std::abs(h * std::sin(rotation));
    const float extent_y =
        std::abs(w * std::sin(rotation)) + std::abs(h * std::cos(rotation));
    EXPECT_NEAR(extent_x, 320.0f, 1e-2) << "quarter turns: " << quarter;
    EXPECT_NEAR(extent_y, 360.0f, 1e-2) << "quarter turns: " << quarter;
  }
}

}  // namespace
}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
