# Release packaging & portability (macOS)

## TL;DR

- **Local development:** build without `MP_BUNDLE_DEPS` (the default). The
  artifact links MacPorts OpenCV from `/opt/local` and is **not portable**, but
  that's fine on a dev machine that has `opencv3` installed.
- **Release / distribution:** build with `MP_BUNDLE_DEPS=1` to bundle all
  non-system dylibs into the framework (Option A, below). ✅ implemented &
  validated — but first read the **license implications**: the current MacPorts
  OpenCV tree pulls in GPL codecs (x264/x265 via ffmpeg). Pursue **Option C**
  (trim `videoio`/`highgui`) before shipping a closed-source product.

## The dependency (why it isn't portable by default)

`otool -L` on the framework binary
(`MediaPipeTasksC.xcframework/macos-arm64/MediaPipeTasksC.framework/MediaPipeTasksC`)
shows that **every** dynamic dependency is a macOS system library **except** the
OpenCV dylibs, which are linked by **absolute MacPorts paths**:

```
@rpath/MediaPipeTasksC.framework/Versions/A/MediaPipeTasksC   (self)
/usr/lib/libc++.1.dylib                                       (system)
/usr/lib/libSystem.B.dylib, /usr/lib/libobjc.A.dylib         (system)
/System/Library/Frameworks/*                                  (Metal, MetalKit, CoreVideo,
                                                               CoreMedia, AVFoundation, OpenGL,
                                                               Accelerate, AppKit, …)  (system)
/opt/local/lib/opencv3/libopencv_core.3.4.dylib              ← MacPorts, NOT portable
/opt/local/lib/opencv3/libopencv_calib3d.3.4.dylib          ← MacPorts, NOT portable
/opt/local/lib/opencv3/libopencv_features2d.3.4.dylib       ← …
/opt/local/lib/opencv3/libopencv_highgui.3.4.dylib
/opt/local/lib/opencv3/libopencv_imgcodecs.3.4.dylib
/opt/local/lib/opencv3/libopencv_imgproc.3.4.dylib
/opt/local/lib/opencv3/libopencv_video.3.4.dylib
/opt/local/lib/opencv3/libopencv_videoio.3.4.dylib
```

**Consequence:** the artifact only runs on a machine that has MacPorts `opencv3`
installed at `/opt/local`. It cannot be shipped to other Macs as-is.

The OpenCV dylibs themselves pull in a **tree** of further MacPorts libraries
(e.g. `libopencv_core` → `/opt/local/lib/libz.1.dylib`; `libopencv_imgcodecs`
has ~14 `/opt/local` dependencies: libpng, libjpeg, libtiff, libwebp,
libopenjp2, …). So making the artifact portable means handling that whole tree,
not just the 8 direct dylibs.

To re-run the inspection:

```bash
FW=mediapipe/tasks/macos/swift/Artifacts/MediaPipeTasksC.xcframework/macos-arm64/MediaPipeTasksC.framework/MediaPipeTasksC
otool -L "$FW" | grep -v '/System/\|/usr/lib/\|MediaPipeTasksC.framework'   # non-system deps
```

## Proposed fixes

### Option A — Bundle the dylibs into the framework ✅ IMPLEMENTED

`build_macos_xcframework.sh MP_BUNDLE_DEPS=1` produces a portable framework. It
uses [`dylibbundler`](https://github.com/auriamg/macdylibbundler)
(`brew install dylibbundler` or `sudo port install dylibbundler`; the script
fails with install instructions if it's missing) to:

1. recursively copy every non-system dynamic dependency into
   `MediaPipeTasksC.framework/Versions/A/Libraries`;
2. rewrite all install names (the framework's references to OpenCV, and the
   bundled libs' references to each other) to `@rpath/<lib>`;
3. anchor `@rpath` with `@loader_path` rpaths — the framework binary gets
   `@loader_path/Libraries`, each bundled lib gets `@loader_path` (siblings) —
   and strip the spurious `@rpath/` rpath dylibbundler leaves behind (which
   would otherwise be a duplicate `LC_RPATH` the linker rejects);
4. ad-hoc re-sign every bundled dylib and the framework binary;
5. **verify** the framework binary and every bundled dylib reference only
   `/System`, `/usr/lib`, `@rpath`, `@loader_path`, or `@executable_path` —
   the build **fails** if any `/opt/local` (or other non-portable) path remains.

```bash
MP_BUNDLE_DEPS=1 ./mediapipe/tasks/macos/build_macos_xcframework.sh
```

Validated: `swift build`, `swift test` (16 tests), and the macOS `.app`
(`ParitySmokeTest`) all run against the bundled artifact, and `otool -L` shows
**0** `/opt/local` references anywhere in the framework or the `.app`.

#### Size impact

The MacPorts `opencv3` dependency tree is large: **95 dylibs, ~112 MB**. The
xcframework grows from ~25 MB (unbundled) to ~140 MB. Most of the weight is the
video stack pulled in by OpenCV `videoio` (ffmpeg + codecs) — see below.

#### ⚠️ Third-party license implications (READ BEFORE DISTRIBUTING)

The bundled tree is **not** all permissive. Notable members:

- **OpenCV** (BSD-3-Clause) and **libjpeg/libpng/libtiff/libwebp/openjp2/
  freetype/harfbuzz/zlib/lzma/zstd** — permissive (BSD/MIT/zlib-like); require
  attribution only.
- **ffmpeg** (`libavcodec/avformat/avutil/swscale/swresample`) — LGPL-2.1+ at
  minimum, pulled in by OpenCV `videoio`.
- **x264** and **x265** and parts of the ffmpeg build — **GPL**. Bundling these
  would impose **GPL** obligations on a redistributed binary.

**Do not ship the fully-bundled artifact as-is for a closed-source product.**
The video codecs (ffmpeg/x264/x265/libvpx) are only present because OpenCV's
`videoio`/`highgui` modules are linked, and the vision landmarker tasks do not
use them at runtime. The right fix before release is **Option C** (drop
`videoio`/`highgui`), which removes ffmpeg/x264/x265 entirely — eliminating both
the GPL concern and the bulk of the size — leaving only BSD/permissive OpenCV
core + image codecs to bundle.

### Option B — Static-link OpenCV (robust, long-term)

Link OpenCV statically so there is no runtime OpenCV dependency at all:

- `--define OPENCV=source` builds OpenCV from source and links it statically.
  This is the cleanest result, but currently **fails on this machine** due to a
  `rules_foreign_cc` + `apple_support` `cc_wrapper` relative-path bug during
  OpenCV's CMake compiler check (see the OpenCV section history). Fixing that
  unlocks a fully self-contained binary.
- Alternatively, provide static OpenCV `.a` archives and point
  `third_party/opencv_macos.BUILD` at them with `linkstatic = 1` (MacPorts does
  not ship a static `opencv3` variant by default, so this means building static
  OpenCV separately).
- Pros: single self-contained binary, no bundled dylib tree, simplest to ship.
- Cons: the source build needs the foreign_cc fix; static OpenCV must be sourced.

### Option C — Trim the OpenCV surface (complementary)

The vision landmarker tasks likely do not need all of OpenCV
(`highgui`/`videoio`/`calib3d` are probably unused at runtime for
Hand/Pose/Face). Auditing and dropping unused OpenCV modules shrinks whatever
must be bundled or statically linked. Out of scope here, but worth doing before
a real release.

## Recommendation

- **Now:** Option A (`MP_BUNDLE_DEPS=1`) is implemented and gives a portable
  artifact for internal testing / non-redistributed use.
- **Before any external release:** apply **Option C** (drop OpenCV
  `videoio`/`highgui`) so the bundle no longer contains ffmpeg/x264/x265 — this
  removes the GPL exposure and ~most of the 112 MB — then bundle the remaining
  BSD/permissive libs with Option A.
- **Long-term:** Option B (static OpenCV via a fixed `OPENCV=source` build) for a
  single self-contained binary with no bundled dylib tree.
