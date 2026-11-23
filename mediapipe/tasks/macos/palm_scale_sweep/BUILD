# Measurement tools for the palm search regions work. Not part of any library.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

# Runs the C++ HandLandmarker on raw BGRA frames from stdin and writes one
# JSON line per frame with the palm detections, stage records and landmarks
# at full float precision.
cc_binary(
    name = "palm_search_dump",
    srcs = ["palm_search_dump.cc"],
    deps = [
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/components/containers:rect",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/hand_landmarker",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
