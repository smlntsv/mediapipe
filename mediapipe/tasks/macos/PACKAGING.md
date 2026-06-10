# Release packaging & portability (macOS)

## Current state: NOT portable yet

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

### Option A — Bundle the dylibs into the framework (recommended, near-term)

Recursively copy all non-system dependencies into the framework and rewrite
their install names to `@rpath`/`@loader_path`, then re-sign. The standard tool
is [`dylibbundler`](https://github.com/auriamg/macdylibbundler)
(`brew install dylibbundler` or `sudo port install dylibbundler`):

```bash
FW_DIR=…/MediaPipeTasksC.framework/Versions/A
dylibbundler -of -cd -b \
  -x "$FW_DIR/MediaPipeTasksC" \
  -d "$FW_DIR/Libraries" \
  -p "@loader_path/Libraries"
codesign --force --deep --sign - …/MediaPipeTasksC.framework
```

- Pros: keeps the existing MacPorts-based build; produces a self-contained,
  shippable framework.
- Cons: bundles a tree of third-party dylibs (tens of MB) — track their licenses
  (OpenCV is BSD; libjpeg/png/tiff/webp each have their own); must re-sign.
- This would be added to `build_macos_xcframework.sh` behind an opt-in
  (e.g. `MP_BUNDLE_DEPS=1`) once validated. It is **not** wired in yet because it
  needs `dylibbundler` installed and its own validation pass.

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

Ship-blocking for distribution. Near-term: **Option A (bundle with
`dylibbundler`)** to get a portable artifact quickly. Long-term: **Option B
(static OpenCV via a fixed `OPENCV=source` build)** for a clean self-contained
binary, optionally combined with **Option C** to minimize size.
