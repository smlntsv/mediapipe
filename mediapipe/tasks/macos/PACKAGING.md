# Release packaging & portability (macOS)

## TL;DR

- **Local development:** build without `MP_BUNDLE_DEPS` (the default). The
  artifact links MacPorts OpenCV from `/opt/local` and is **not portable**, but
  that's fine on a dev machine that has `opencv3` installed.
- **Release / distribution:** build with `MP_BUNDLE_DEPS=1` to bundle all
  non-system dylibs into the framework (Option A, below). ✅ implemented &
  validated.
- OpenCV is **trimmed to `core` + `imgproc`** (Option C, below), so the bundle
  is just **3 dylibs / 4.7 MB** (`libopencv_core`, `libopencv_imgproc`, `libz`)
  and contains **no GPL/LGPL** code — no ffmpeg/x264/x265/libvpx. The bundled
  xcframework is **~29 MB** total.

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

#### Size & license (with the trimmed OpenCV from Option C)

With OpenCV trimmed to `core` + `imgproc` (Option C), bundling pulls in just:

| Bundled dylib | License |
| --- | --- |
| `libopencv_core.3.4.dylib` | OpenCV BSD-3-Clause |
| `libopencv_imgproc.3.4.dylib` | OpenCV BSD-3-Clause |
| `libz.1.dylib` | zlib (permissive) |

**3 dylibs, ~4.7 MB**; the full bundled xcframework is **~29 MB**. **No GPL or
LGPL** code is bundled — no ffmpeg/`libav*`/x264/x265/libvpx. (For comparison,
the *untrimmed* OpenCV tree was 95 dylibs / ~112 MB and **did** pull GPL codecs;
see Option C for why.)

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

### Option C — Trim the OpenCV surface ✅ IMPLEMENTED

The Hand/Pose/Face vision tasks only use OpenCV `core` (cv::Mat) and `imgproc`
(resize/cvtColor/warpAffine). `third_party/opencv_macos.BUILD` is trimmed to
exactly those two modules — deliberately excluding `videoio`, `highgui`,
`video`, `calib3d`, and `features2d`. `imgcodecs` is **not** needed either,
because the wrapper feeds raw RGBA (`MpImageCreateFromUint8Data`) rather than
decoding image files.

Result: the GPU-capable artifact links **only** `libopencv_core` +
`libopencv_imgproc`, and bundling drops from 95 dylibs/112 MB to **3 dylibs/
4.7 MB** with **no GPL/LGPL** codecs. Verified: `swift test` (16),
the macOS `.app`, GPU delegate, and Hand/Pose/Face parity all still pass; `otool`
shows no ffmpeg/`libav*`/x264/x265/libvpx and no `/opt/local` paths.

To add a module back (only if a future feature needs it), uncomment it in
`third_party/opencv_macos.BUILD` — but avoid `videoio`/`highgui` to keep the
artifact GPL-free.

## Recommendation

- **Ship-ready (internal / permissive):** Option C (trimmed OpenCV) + Option A
  (`MP_BUNDLE_DEPS=1`) — implemented. Produces a portable, ~29 MB, BSD/zlib-only
  GPU-capable artifact. Include the OpenCV (BSD-3) and zlib license texts in your
  distribution's third-party notices.
- **Long-term:** Option B (static OpenCV via a fixed `OPENCV=source` build) for a
  single self-contained binary with no bundled dylib tree.
