# Description:
#   OpenCV libraries for video/image processing on MacOS

load("@bazel_skylib//lib:paths.bzl", "paths")

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# Example configurations:
#
# # OpenCV 3
# To configure OpenCV 3, obtain the path of OpenCV 3 from Homebrew. The
# following commands show the output of the command with version 3.4.16_10:
#
# $ brew ls opencv@3 | grep version.hpp
# $ /opt/homebrew/Cellar/opencv@3/3.4.16_10/include/opencv2/core/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/3.4.16_10" for the example above).
#
# # OpenCV 4
# To configure OpenCV 4, obtain the path of OpenCV 4 from Homebrew. The
# following commands show the output of the command with version 4.10.0_12:
#
# $ brew ls opencv | grep version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/core/version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/dnn/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/4.10.0_12" for the example above). For OpenCV 4, you will also need to
# adjust the include paths. The header search path should be
# "include/opencv4/opencv2/**/*.h*" and the include prefix needs to be set to
# "include/opencv4".

# Configured for MacPorts OpenCV 3 (`sudo port install opencv3`).
# With the WORKSPACE `macos_opencv` repository rooted at "/opt", the MacPorts
# layout is:
#   /opt/local/lib/opencv3/libopencv_*.dylib
#   /opt/local/include/opencv3/opencv2/**
PREFIX = "local"

OPENCV_SUBDIR = "opencv3"

cc_library(
    name = "opencv",
    srcs = glob(
        [
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_core.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_calib3d.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_features2d.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_highgui.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_imgcodecs.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_imgproc.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_video.dylib"),
            paths.join(PREFIX, "lib", OPENCV_SUBDIR, "libopencv_videoio.dylib"),
        ],
    ),
    hdrs = glob([paths.join(PREFIX, "include", OPENCV_SUBDIR, "opencv2/**/*.h*")]),
    includes = [paths.join(PREFIX, "include", OPENCV_SUBDIR)],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
