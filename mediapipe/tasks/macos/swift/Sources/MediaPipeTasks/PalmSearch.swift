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

import CoreGraphics
import Foundation
import MediaPipeTasksObjC

// Palm search regions: the palm detection model is run on a list of regions
// of the frame, not only on the whole frame, and what the regions found is
// merged. A far hand is a few pixels wide when the whole frame is scaled into
// the model's 192x192 input; inside a region it is larger.
//
// Nothing here is a recommended configuration. The options exist so that
// configurations can be compared on the same frames.

/// Which grid tiles run on a detector frame. A detector frame is a frame on
/// which the palm detector runs at all: in video mode it is skipped while
/// `numHands` hands are tracked.
public enum PalmSearchSchedule: Sendable, Equatable {
    /// Every grid tile on every detector frame.
    case allRegions
    /// `perFrame` grid tiles per detector frame, cycling through the grid.
    /// The cycle advances only on detector frames.
    case roundRobin(perFrame: Int)

    var tilesPerFrame: Int {
        switch self {
        case .allRegions: return 0
        case .roundRobin(let perFrame): return max(1, perFrame)
        }
    }
}

/// A grid of tiles over the frame. Each tile is one search region.
///
/// A region is sampled as a square window: the tile grown around its center
/// to a square in pixels. On a frame that is not square, the tiles of one
/// axis therefore overlap more than `overlap` says, and the magnification a
/// tile gives is `frame long side / tile long side`.
public struct PalmSearchGrid: Sendable, Equatable {
    public var rows: Int
    public var columns: Int
    /// Fraction of a tile's width (height) shared with its horizontal
    /// (vertical) neighbour, in `0 ..< 1`. More overlap means larger tiles
    /// and less magnification.
    public var overlap: Float

    public init(rows: Int, columns: Int, overlap: Float = 0) {
        self.rows = rows
        self.columns = columns
        self.overlap = overlap
    }

    /// Width and height of one tile as fractions of the frame.
    public var tileSize: (width: Float, height: Float) {
        let shared = 1 - min(max(overlap, 0), 0.95)
        return (1 / (1 + Float(max(columns, 1) - 1) * shared),
                1 / (1 + Float(max(rows, 1) - 1) * shared))
    }

    /// How much larger a palm is in the model input of one tile than in the
    /// model input of the whole frame, for a frame of this size.
    public func magnification(frameSize: CGSize) -> Float {
        let tile = tileSize
        let frameLong = Float(max(frameSize.width, frameSize.height))
        let tileLong = max(tile.width * Float(frameSize.width),
                           tile.height * Float(frameSize.height))
        return tileLong > 0 ? frameLong / tileLong : 1
    }

    /// The smallest grid that gives at least `targetMagnification` on a frame
    /// of this size with this overlap. Arithmetic only; it is not a default.
    ///
    /// Because a region is sampled as a square window, the short axis of the
    /// frame needs fewer tiles than the long one: 1280x720 at 4x is 4 columns
    /// by 3 rows, not 4 by 4.
    public static func recommended(frameSize: CGSize,
                                   targetMagnification: Float,
                                   overlap: Float = 0) -> PalmSearchGrid {
        let shared = 1 - min(max(overlap, 0), 0.95)
        let magnification = max(targetMagnification, 1)
        let width = Float(frameSize.width)
        let height = Float(frameSize.height)
        let longSide = max(width, height)
        let shortSide = min(width, height)
        // Side of the square window that gives the magnification.
        let side = longSide / magnification
        func count(_ length: Float) -> Int {
            // Smallest n with length / (1 + (n - 1) * shared) <= side.
            max(1, Int((1 + (length / side - 1) / shared - 1e-4).rounded(.up)))
        }
        let longCount = count(longSide)
        let shortCount = count(shortSide)
        return width >= height
            ? PalmSearchGrid(rows: shortCount, columns: longCount, overlap: overlap)
            : PalmSearchGrid(rows: longCount, columns: shortCount, overlap: overlap)
    }
}

/// How detections of different regions that overlap become one.
public enum PalmSearchMerge: Int, Sendable {
    /// Boxes and keypoints averaged by score, as in the detector's own
    /// non-max suppression.
    case weighted = 0
    /// The highest scored detection, unchanged.
    case bestScore = 1
}

/// Configuration of the palm search regions path. Set
/// `HandLandmarkerOptions.palmSearch` to use it; nil leaves the pipeline
/// exactly as it is without this feature.
public struct PalmSearchOptions: Sendable, Equatable {
    /// The whole frame is one of the regions on every detector frame.
    public var includeFullFrame: Bool
    /// Grid tiles as regions. Nil: no grid.
    public var grid: PalmSearchGrid?
    /// Which grid tiles run on a detector frame. The full frame and external
    /// regions are not scheduled: they run on every detector frame.
    public var schedule: PalmSearchSchedule
    /// Allows `searchRegions` on the detect methods.
    public var acceptsExternalRegions: Bool
    /// Drops a detection whose box touches an edge of the window its region
    /// was sampled from, unless that edge is a frame edge. Never applied to
    /// the full frame.
    public var rejectTruncatedDetections: Bool
    /// Drops a detection of a grid tile or external region whose box is wider
    /// or taller than this fraction of the frame. Nil: off.
    public var maxPalmFraction: Float?
    public var merge: PalmSearchMerge
    /// Clips the detector's hand rects to `numHands` before the association
    /// with tracked hands, as the pipeline without search regions does. When
    /// false the only clip is the one after the association.
    public var clipBeforeAssociation: Bool

    public init(includeFullFrame: Bool = true,
                grid: PalmSearchGrid? = nil,
                schedule: PalmSearchSchedule = .allRegions,
                acceptsExternalRegions: Bool = false,
                rejectTruncatedDetections: Bool = false,
                maxPalmFraction: Float? = nil,
                merge: PalmSearchMerge = .weighted,
                clipBeforeAssociation: Bool = false) {
        self.includeFullFrame = includeFullFrame
        self.grid = grid
        self.schedule = schedule
        self.acceptsExternalRegions = acceptsExternalRegions
        self.rejectTruncatedDetections = rejectTruncatedDetections
        self.maxPalmFraction = maxPalmFraction
        self.merge = merge
        self.clipBeforeAssociation = clipBeforeAssociation
    }

    var bridged: MPCPalmSearchOptions {
        let out = MPCPalmSearchOptions()
        out.includeFullFrame = includeFullFrame
        out.gridRows = grid?.rows ?? 0
        out.gridColumns = grid?.columns ?? 0
        out.gridOverlap = grid?.overlap ?? 0
        out.roundRobinTilesPerFrame = grid == nil ? 0 : schedule.tilesPerFrame
        out.acceptsExternalRegions = acceptsExternalRegions
        out.rejectTruncatedDetections = rejectTruncatedDetections
        out.maxPalmFraction = maxPalmFraction ?? 0
        out.merge = merge.rawValue
        out.clipBeforeAssociation = clipBeforeAssociation
        return out
    }
}

/// Where a search region came from.
public enum PalmSearchRegionKind: Int, Sendable {
    case fullFrame = 0
    case grid = 1
    case external = 2
}

/// What became of one detection of one region.
public enum PalmSearchDetectionOutcome: Int, Sendable {
    /// Went into the merge.
    case merged = 0
    /// Dropped by `rejectTruncatedDetections`.
    case rejectedTruncated = 1
    /// Dropped by `maxPalmFraction`.
    case rejectedSize = 2
}

/// What became of the hand rect of one merged detection.
public enum PalmSearchHandOutcome: Int, Sendable {
    /// Given to the landmark model.
    case toLandmarks = 0
    /// Dropped by the association: it overlapped a rect carried over from the
    /// previous frame.
    case overlapsTrackedRect = 1
    /// Dropped by the association: it overlapped the hand rect of an earlier
    /// (higher scored) detection of this frame.
    case overlapsEarlierRect = 2
    /// Kept by the association, cut off by the clip to `numHands` after it.
    case clippedByNumHands = 3
    /// Cut off by the clip to `numHands` before the association
    /// (`clipBeforeAssociation`).
    case clippedBeforeAssociation = 4
}

/// One detection of one region.
public struct PalmSearchRegionDetection: Sendable, Equatable {
    /// As it left the region's own non-max suppression, in frame coordinates.
    public var detection: StageDetection
    public var outcome: PalmSearchDetectionOutcome
    /// Index into `HandLandmarkerPalmSearch.merged` of the detection it
    /// became or was folded into. Nil when it was dropped.
    public var mergedIndex: Int?
}

/// One region the palm model was run on, and what it produced.
public struct PalmSearchRegion: Sendable, Equatable {
    /// Stable for a given grid: 0 is the full frame, `1 + row * columns +
    /// column` a grid tile, `1000 + i` the i-th external region of the call.
    public var id: Int
    public var kind: PalmSearchRegionKind
    /// Grid tiles only.
    public var gridRow: Int?
    public var gridColumn: Int?
    /// The region as asked for, normalized to the frame.
    public var requested: CGRect
    /// The window the model input was sampled from: the requested box grown
    /// around its center to a square in pixels. It can reach outside the
    /// frame; those pixels are black.
    public var sampled: CGRect
    public var detections: [PalmSearchRegionDetection]
    /// Only with `palmCandidateLimit` > 0: this region's highest scored boxes
    /// whatever their score, and how many boxes passed the score threshold.
    public var candidates: [StageDetection]
    public var aboveThresholdCount: Int
    /// Monotonic clock, nanoseconds: when the region entered the loop and
    /// when its detections left it. Regions can overlap in time, so these do
    /// not add up to the detector's cost.
    public var cropStartNanoseconds: Int64
    public var detectionsReadyNanoseconds: Int64
}

/// Identifies one detection of one region on this frame.
public struct PalmSearchSource: Sendable, Equatable {
    public var regionID: Int
    /// Index into that region's `detections`.
    public var detectionIndex: Int
}

/// One detection after the merge across regions.
public struct PalmSearchMergedDetection: Sendable, Equatable {
    public var detection: StageDetection
    /// The region detections it was made of, highest scored first.
    public var sources: [PalmSearchSource]
    public var outcome: PalmSearchHandOutcome
    /// For the two overlap outcomes: index of the overlapped rect in the
    /// association's output, which lists `HandLandmarkerStages.previousRects`
    /// first and the kept detector rects after them.
    public var overlappedRectIndex: Int?
    /// For `.toLandmarks`: index into `HandLandmarkerStages.landmarkRois`.
    public var landmarkRoiIndex: Int?
}

/// The palm search record of one frame. Present on every result when
/// `HandLandmarkerOptions.palmSearch` is set and `stageOutputs` is true.
///
/// Raw values of one frame. Coordinates are normalized to the image as
/// passed in. Clock values are a monotonic clock in nanoseconds and are 0
/// where nothing was measured.
public struct HandLandmarkerPalmSearch: Sendable, Equatable {
    /// Timestamp, in microseconds, of the graph output this record was read
    /// from.
    public var timestampMicroseconds: Int64
    /// The number of detect calls on this landmarker before this one.
    public var frameSequence: Int64
    /// Whether the palm detector ran on this frame. In video mode it is
    /// skipped while the rects carried over from the previous frame number
    /// `numHands`; the schedule, `regions` and `merged` are then empty.
    public var palmDetectorRan: Bool

    /// The number of detector frames before this one.
    public var detectorFrameIndex: Int64
    /// Number of grid tiles, and how many run per detector frame (0: all).
    public var gridTileCount: Int
    public var tilesPerFrame: Int
    /// Index of the first grid tile run on this detector frame and on the
    /// next one.
    public var cursorBefore: Int
    public var cursorAfter: Int

    /// When the detect call was entered and when the graph had produced its
    /// outputs.
    public var callStartNanoseconds: Int64
    public var callEndNanoseconds: Int64
    /// When the region list and the merged detections were emitted.
    public var regionsReadyNanoseconds: Int64
    public var mergeDoneNanoseconds: Int64
    /// When the landmark model's rects were emitted and when its landmarks
    /// were collected.
    public var landmarkRoisReadyNanoseconds: Int64
    public var landmarksDoneNanoseconds: Int64

    /// In the order the model was run on them.
    public var regions: [PalmSearchRegion]
    /// The merged detections, highest score first. These are
    /// `HandLandmarkerStages.palmDetections`.
    public var merged: [PalmSearchMergedDetection]

    init(_ search: MPCHandLandmarkerPalmSearch) {
        timestampMicroseconds = search.timestampUs
        frameSequence = search.frameSequence
        palmDetectorRan = search.palmDetectorRan
        detectorFrameIndex = search.detectorFrameIndex
        gridTileCount = search.gridTileCount
        tilesPerFrame = search.tilesPerFrame
        cursorBefore = search.cursorBefore
        cursorAfter = search.cursorAfter
        callStartNanoseconds = search.callStartNs
        callEndNanoseconds = search.callEndNs
        regionsReadyNanoseconds = search.regionsReadyNs
        mergeDoneNanoseconds = search.mergeDoneNs
        landmarkRoisReadyNanoseconds = search.landmarkRoisReadyNs
        landmarksDoneNanoseconds = search.landmarksDoneNs
        regions = search.regions.map { region in
            func rect(_ x: Float, _ y: Float, _ w: Float, _ h: Float) -> CGRect {
                CGRect(x: CGFloat(x - w / 2), y: CGFloat(y - h / 2),
                       width: CGFloat(w), height: CGFloat(h))
            }
            let kind = PalmSearchRegionKind(rawValue: region.kind) ?? .external
            return PalmSearchRegion(
                id: region.regionId,
                kind: kind,
                gridRow: kind == .grid ? region.gridRow : nil,
                gridColumn: kind == .grid ? region.gridColumn : nil,
                requested: rect(region.requestedXCenter, region.requestedYCenter,
                                region.requestedWidth, region.requestedHeight),
                sampled: rect(region.sampledXCenter, region.sampledYCenter,
                              region.sampledWidth, region.sampledHeight),
                detections: region.detections.map {
                    PalmSearchRegionDetection(
                        detection: StageDetection($0.detection),
                        outcome: PalmSearchDetectionOutcome(rawValue: $0.outcome) ?? .merged,
                        mergedIndex: $0.mergedIndex >= 0 ? $0.mergedIndex : nil)
                },
                candidates: region.candidates.map(StageDetection.init),
                aboveThresholdCount: region.aboveThresholdCount,
                cropStartNanoseconds: region.cropStartNs,
                detectionsReadyNanoseconds: region.detectionsReadyNs)
        }
        merged = search.merged.map { merged in
            let flat = merged.sources
            return PalmSearchMergedDetection(
                detection: StageDetection(merged.detection),
                sources: stride(from: 0, to: flat.count - 1, by: 2).map {
                    PalmSearchSource(regionID: flat[$0].intValue,
                                     detectionIndex: flat[$0 + 1].intValue)
                },
                outcome: PalmSearchHandOutcome(rawValue: merged.outcome) ?? .toLandmarks,
                overlappedRectIndex: merged.overlappedRectIndex >= 0
                    ? merged.overlappedRectIndex : nil,
                landmarkRoiIndex: merged.landmarkRoiIndex >= 0
                    ? merged.landmarkRoiIndex : nil)
        }
    }
}
