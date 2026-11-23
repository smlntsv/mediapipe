// Runs the C++ HandLandmarker on raw BGRA frames read from stdin and writes
// one JSON line per frame: palm detections, stage records, the palm search
// record and the landmarks, with floats printed so that they round-trip.
//
// Used to compare configurations on identical frames, value for value.
//
//   ffmpeg -i clip.mov -f rawvideo -pix_fmt bgra - | palm_search_dump \
//       --model hand_landmarker.task --width 1280 --height 720 --mode video

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

namespace {

using ::mediapipe::Detection;
using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::core::BaseOptions;
using namespace ::mediapipe::tasks::vision::hand_landmarker;  // NOLINT

std::string F(double v) { return absl::StrFormat("%.9g", v); }

std::string DetectionJson(const Detection& d) {
  const auto& box = d.location_data().relative_bounding_box();
  std::string out = "[" + F(d.score_size() > 0 ? d.score(0) : 0) + "," +
                    F(box.xmin()) + "," + F(box.ymin()) + "," +
                    F(box.width()) + "," + F(box.height());
  for (const auto& k : d.location_data().relative_keypoints()) {
    out += "," + F(k.x()) + "," + F(k.y());
  }
  return out + "]";
}

std::string DetectionsJson(const std::vector<Detection>& detections) {
  std::string out = "[";
  for (int i = 0; i < detections.size(); ++i) {
    if (i) out += ",";
    out += DetectionJson(detections[i]);
  }
  return out + "]";
}

std::string RectsJson(const std::vector<NormalizedRect>& rects) {
  std::string out = "[";
  for (int i = 0; i < rects.size(); ++i) {
    if (i) out += ",";
    const auto& r = rects[i];
    out += "[" + F(r.x_center()) + "," + F(r.y_center()) + "," + F(r.width()) +
           "," + F(r.height()) + "," + F(r.rotation()) + "]";
  }
  return out + "]";
}

template <typename T>
std::string NumbersJson(const std::vector<T>& values) {
  std::string out = "[";
  for (int i = 0; i < values.size(); ++i) {
    if (i) out += ",";
    out += F(static_cast<double>(values[i]));
  }
  return out + "]";
}

std::string RegionJson(const PalmSearchRegionResult& result) {
  const PalmSearchRegion& r = result.region;
  std::string out = absl::StrFormat(
      "{\"id\":%d,\"kind\":%d,\"row\":%d,\"col\":%d,", r.id,
      static_cast<int>(r.kind), r.grid_row, r.grid_column);
  out += "\"requested\":[" + F(r.requested_x_center) + "," +
         F(r.requested_y_center) + "," + F(r.requested_width) + "," +
         F(r.requested_height) + "],";
  out += "\"sampled\":[" + F(r.sampled_x_center) + "," + F(r.sampled_y_center) +
         "," + F(r.sampled_width) + "," + F(r.sampled_height) + "],";
  out += absl::StrFormat("\"crop_start_ns\":%d,\"ready_ns\":%d,",
                         result.crop_start_ns, result.detections_ready_ns);
  out += "\"det\":[";
  for (int i = 0; i < result.detections.size(); ++i) {
    if (i) out += ",";
    out += absl::StrFormat("{\"outcome\":%d,\"merged\":%d,\"d\":",
                           static_cast<int>(result.detections[i].outcome),
                           result.detections[i].merged_index) +
           DetectionJson(result.detections[i].detection) + "}";
  }
  out += "]";
  if (!result.candidates.empty()) {
    out += absl::StrFormat(",\"above\":%d,\"cand\":",
                           result.above_threshold_count) +
           DetectionsJson(result.candidates);
  }
  return out + "}";
}

std::string PalmSearchJson(const HandLandmarkerPalmSearch& search) {
  std::string out = absl::StrFormat(
      "{\"ran\":%d,\"seq\":%d,\"call_start_ns\":%d,\"call_end_ns\":%d,"
      "\"rois_ready_ns\":%d,\"landmarks_done_ns\":%d",
      search.palm_detector_ran ? 1 : 0, search.frame_sequence,
      search.call_start_ns, search.call_end_ns, search.landmark_rois_ready_ns,
      search.landmarks_done_ns);
  const auto& schedule = search.record.schedule;
  out += absl::StrFormat(
      ",\"schedule\":{\"detector_frame\":%d,\"tiles\":%d,\"per_frame\":%d,"
      "\"cursor_before\":%d,\"cursor_after\":%d}",
      schedule.detector_frame_index, schedule.grid_tile_count,
      schedule.tiles_per_frame, schedule.cursor_before, schedule.cursor_after);
  out += absl::StrFormat(",\"regions_ready_ns\":%d,\"merge_done_ns\":%d",
                         search.record.regions_ready_ns,
                         search.record.merge_done_ns);
  out += ",\"regions\":[";
  for (int i = 0; i < search.record.regions.size(); ++i) {
    if (i) out += ",";
    out += RegionJson(search.record.regions[i]);
  }
  out += "],\"merged\":[";
  for (int i = 0; i < search.record.merged.size(); ++i) {
    if (i) out += ",";
    const auto& merged = search.record.merged[i];
    out += absl::StrFormat(
        "{\"outcome\":%d,\"overlapped\":%d,\"roi\":%d,\"src\":[",
        static_cast<int>(search.merged_outcomes[i]),
        search.merged_overlapped_rect_index[i],
        search.merged_landmark_roi_index[i]);
    for (int k = 0; k < merged.sources.size(); ++k) {
      if (k) out += ",";
      out += absl::StrFormat("[%d,%d]", merged.sources[k].region_id,
                             merged.sources[k].detection_index);
    }
    out += "],\"d\":" + DetectionJson(merged.detection) + "}";
  }
  return out + "]}";
}

std::string ResultJson(int index, const HandLandmarkerResult& result) {
  std::string out = absl::StrFormat("{\"i\":%d,\"hands\":[", index);
  for (int h = 0; h < result.hand_landmarks.size(); ++h) {
    if (h) out += ",";
    const auto& handedness = result.handedness[h].categories;
    out += "{\"h\":\"" +
           (handedness.empty() ? std::string()
                               : handedness[0].category_name.value_or("")) +
           "\",\"hs\":" + F(handedness.empty() ? 0 : handedness[0].score) +
           ",\"lm\":[";
    const auto& landmarks = result.hand_landmarks[h].landmarks;
    for (int k = 0; k < landmarks.size(); ++k) {
      if (k) out += ",";
      out += F(landmarks[k].x) + "," + F(landmarks[k].y) + "," +
             F(landmarks[k].z);
    }
    out += "],\"wlm\":[";
    const auto& world = result.hand_world_landmarks[h].landmarks;
    for (int k = 0; k < world.size(); ++k) {
      if (k) out += ",";
      out += F(world[k].x) + "," + F(world[k].y) + "," + F(world[k].z);
    }
    out += "]}";
  }
  out += "]";
  if (result.stages) {
    const HandLandmarkerStages& s = *result.stages;
    out += absl::StrFormat(",\"ts\":%d,\"ran\":%d", s.timestamp_us,
                           s.palm_detector_ran ? 1 : 0);
    out += ",\"prev\":" + RectsJson(s.previous_rects);
    out += ",\"det\":" + DetectionsJson(s.palm_detections);
    out += ",\"palm_rects\":" + RectsJson(s.palm_rects);
    out += ",\"det_rects\":" + RectsJson(s.detector_hand_rects);
    out += ",\"rois\":" + RectsJson(s.landmark_rois);
    out += ",\"presence\":" + NumbersJson(s.presence_scores);
    out += ",\"next\":" + RectsJson(s.hand_rects_next_frame);
    out += ",\"miss\":" + NumbersJson(s.keep_alive_miss_counts);
  }
  if (result.palm_candidates) {
    out += absl::StrFormat(",\"above\":%d,\"kept\":%d,\"cand\":",
                           result.palm_candidates->above_threshold_count,
                           result.palm_candidates->after_suppression_count) +
           DetectionsJson(result.palm_candidates->candidates);
  }
  if (result.palm_search) {
    out += ",\"search\":" + PalmSearchJson(*result.palm_search);
  }
  return out + "}\n";
}

int Fail(const std::string& message) {
  fprintf(stderr, "palm_search_dump: %s\n", message.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (strncmp(argv[i], "--", 2) != 0) return Fail("bad argument");
    args[argv[i] + 2] = argv[i + 1];
  }
  auto get = [&](const std::string& key, const std::string& fallback) {
    auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
  };
  const int width = atoi(get("width", "0").c_str());
  const int height = atoi(get("height", "0").c_str());
  const std::string mode = get("mode", "image");
  const std::string delegate = get("delegate", "cpu");
  const std::string search = get("search", "off");
  const double fps = atof(get("fps", "30").c_str());
  const int rotation = atoi(get("rotation", "0").c_str());
  if (width <= 0 || height <= 0 || get("model", "").empty()) {
    return Fail("--model, --width and --height are required");
  }

  // --external "left,top,right,bottom;left,top,right,bottom": the same
  // regions on every frame.
  std::vector<::mediapipe::tasks::components::containers::RectF> external;
  {
    const std::string spec = get("external", "");
    const char* p = spec.c_str();
    float l, t, r, b;
    int used = 0;
    while (sscanf(p, "%f,%f,%f,%f%n", &l, &t, &r, &b, &used) == 4) {
      external.push_back({l, t, r, b});
      p += used;
      if (*p == ';') ++p;
    }
  }

  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path = get("model", "");
  if (delegate == "gpu") {
    options->base_options.delegate = BaseOptions::GPU;
  } else if (delegate == "coreml") {
    options->base_options.delegate = BaseOptions::COREML;
    BaseOptions::CoreMlOptions coreml;
    coreml.model_cache_dir = get("coreml-cache", "");
    options->base_options.delegate_options = coreml;
  } else if (delegate != "cpu") {
    return Fail("--delegate is cpu, gpu or coreml");
  }
  // As the C API: the GPU service only for the GPU delegate.
  options->base_options.disable_default_service = delegate != "gpu";
  options->running_mode =
      mode == "video" ? ::mediapipe::tasks::vision::core::RunningMode::VIDEO
                      : ::mediapipe::tasks::vision::core::RunningMode::IMAGE;
  options->num_hands = atoi(get("num-hands", "2").c_str());
  options->tracking_grace_frames = atoi(get("grace", "0").c_str());
  options->debug_stage_outputs = get("stages", "1") != "0";
  options->debug_palm_candidates = atoi(get("candidates", "0").c_str());
  if (search != "off") {
    PalmSearchOptions palm_search;
    palm_search.include_full_frame = get("full-frame", "1") != "0";
    palm_search.clip_before_association = get("clip-before", "0") != "0";
    // --grid ROWSxCOLUMNS
    const std::string grid = get("grid", "");
    if (!grid.empty()) {
      if (sscanf(grid.c_str(), "%dx%d", &palm_search.grid_rows,
                 &palm_search.grid_columns) != 2) {
        return Fail("--grid is ROWSxCOLUMNS");
      }
    }
    palm_search.grid_overlap = atof(get("overlap", "0").c_str());
    palm_search.round_robin_tiles_per_frame = atoi(get("rr", "0").c_str());
    palm_search.reject_truncated_detections =
        get("reject-truncated", "0") != "0";
    palm_search.max_palm_fraction = atof(get("max-palm", "0").c_str());
    if (get("merge", "weighted") == "best") {
      palm_search.merge = PalmSearchOptions::Merge::kBestScore;
    }
    palm_search.accepts_external_regions = !external.empty();
    options->palm_search = palm_search;
  }

  auto landmarker = HandLandmarker::Create(std::move(options));
  if (!landmarker.ok()) return Fail(std::string(landmarker.status().message()));

  FILE* out = stdout;
  if (!get("out", "").empty()) {
    out = fopen(get("out", "").c_str(), "w");
    if (!out) return Fail("cannot open --out");
  }

  const size_t frame_bytes = static_cast<size_t>(width) * height * 4;
  std::vector<uint8_t> bgra(frame_bytes);
  for (int index = 0;; ++index) {
    size_t got = 0;
    while (got < frame_bytes) {
      const size_t n = fread(bgra.data() + got, 1, frame_bytes - got, stdin);
      if (n == 0) break;
      got += n;
    }
    if (got < frame_bytes) break;

    auto frame = std::make_shared<::mediapipe::ImageFrame>(
        ::mediapipe::ImageFormat::SRGBA, width, height,
        ::mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    for (int y = 0; y < height; ++y) {
      const uint8_t* src = bgra.data() + static_cast<size_t>(y) * width * 4;
      uint8_t* dst = frame->MutablePixelData() + y * frame->WidthStep();
      for (int x = 0; x < width; ++x) {
        dst[4 * x + 0] = src[4 * x + 2];
        dst[4 * x + 1] = src[4 * x + 1];
        dst[4 * x + 2] = src[4 * x + 0];
        dst[4 * x + 3] = src[4 * x + 3];
      }
    }
    ::mediapipe::Image image(std::move(frame));
    std::optional<::mediapipe::tasks::vision::core::ImageProcessingOptions>
        processing;
    if (rotation != 0) {
      processing.emplace();
      processing->rotation_degrees = rotation;
    }
    auto result = mode == "video"
                      ? (*landmarker)
                            ->DetectForVideo(
                                image,
                                static_cast<int64_t>(index * 1000.0 / fps),
                                processing, false, external)
                      : (*landmarker)->Detect(image, processing, false,
                                              external);
    if (!result.ok()) {
      return Fail(absl::StrFormat("frame %d: %s", index,
                                  std::string(result.status().message())));
    }
    const std::string line = ResultJson(index, *result);
    fwrite(line.data(), 1, line.size(), out);
  }
  if (out != stdout) fclose(out);
  return (*landmarker)->Close().ok() ? 0 : 1;
}
