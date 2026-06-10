# Webcam demo memory investigation — GPU leak found and fixed

## Summary

The runaway memory growth (reported reaching ~90 GB until macOS force-quit) was a
**per-frame IOSurface leak in MediaPipe's macOS Metal GPU inference path**, not in
the demo app, the Swift/Objective-C++ wrapper, the CVPixelBuffer conversion, or the
C result ownership.

It is now **fixed** with two source-level changes in `mediapipe/gpu` (a per-frame
texture-cache flush on the CVPixelBuffer-backed GPU path). After the fix the GPU
path is **bounded**: memory ramps once to a small working set and then stays flat.

| Task | CPU Δ/1000 frames | GPU Δ/1000 frames (before) | GPU Δ/12000 frames (before) | GPU steady-state (after fix) |
| --- | --- | --- | --- | --- |
| hand | +1.5 MB | **+1542 MB** (~1.54 MB/frame) | ~+18 GB (linear → OOM) | **−0.05 MB/frame (bounded)** |
| pose | +3.2 MB | +767 MB (~0.77 MB/frame) | linear | **−0.01 MB/frame (bounded)** |
| face | +6.3 MB | +776 MB (~0.78 MB/frame) | linear | **−0.005 MB/frame (bounded)** |

"Steady-state" = growth measured *after* a 1500-frame warmup that saturates the
texture-cache working set; the negative numbers mean the working set even shrinks
during the measured window. Both delegates are now stable.

## How it was diagnosed

1. **Isolation.** A non-camera stress test (`MemoryStressTests` in the package)
   runs `detectForVideo(pixelBuffer:)` on one buffer N times and measures resident
   memory (`phys_footprint`), with each iteration in an `autoreleasepool`. CPU was
   flat; GPU grew ~0.8–1.5 MB/frame. `autoreleasepool` did not change the GPU
   numbers → genuinely-retained memory, not autorelease accumulation. The wrapper
   code is identical for both delegates and `MPCImage.liveInstanceCount` stays ~0,
   so the leak was inside the dylib's GPU graph.

2. **Attribution.** `vmmap --summary` on a live GPU stress process showed the
   growing region was **IOSurface**, climbing monotonically past 10,000 surfaces /
   7.6 GB while MALLOC stayed flat (~75 MB). So ~1–2 IOSurfaces leaked per
   inference and were never released. (Hand leaks ~2× pose/face because it runs two
   inference passes per frame — palm detection + landmarks.)

## Root cause

On the GPU path each input frame becomes a CVPixelBuffer-backed `GpuBuffer`, and a
GPU texture is created from its IOSurface — twice:

- a **GL** texture in `GpuBufferStorageCvPixelBuffer::GetTexture`
  (`CVOpenGLTextureCacheCreateTextureFromImage`), used by `Image::ConvertToGpu`, and
- a **Metal** texture in `MPPMetalHelper`’s `copyCVMetalTextureWithGpuBuffer:`
  (`CVMetalTextureCacheCreateTextureFromImage`), used by
  `image_to_tensor_converter_metal.cc`.

Both `CVOpenGLTextureCache` and `CVMetalTextureCache` keep an **internal reference
to the backing IOSurface** for every texture they vend, releasing them only when
the cache is flushed. Upstream relies on
`CvTextureCacheManager::FlushTextureCaches()`, which (a) is only wired for the
**OpenGL** caches and (b) is only invoked when a *pooled* buffer allocation hits its
threshold (`CvPixelBufferPoolWrapper::GetBuffer`).

On desktop macOS, the CPU→GPU upload path creates input pixel buffers **without a
pool** (`CreateCVPixelBufferForImageFrame`), so that pool-pressure flush never
fires. And the Metal cache was never flushed at all
(`metal_shared_resources.cc`: *“TODO: register and flush metal caches too.”*). With
both caches pinning every frame's IOSurface and nothing ever flushing them, memory
grew ~1 buffer/frame without bound — the ~90 GB OOM. This path is force-enabled on
macOS by this package (`MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER`, gated to
`!TARGET_OS_OSX` upstream), so upstream never exercises it on desktop macOS.

## The fix

Flush each texture cache once per frame, right before vending that frame's texture,
so the **previous** frame's now-unreferenced textures are recycled. This is safe to
call every frame — the flush never invalidates textures still in use (their
`CVMetalTexture`/`CVOpenGLTexture`/`MTLTexture` keep them alive), and the cache's
age-based retention keeps in-flight async GPU work valid. The IOSurface is pinned by
**both** caches, so both flushes are required for its refcount to reach zero.

- `mediapipe/gpu/MPPMetalHelper.cc` — `CVMetalTextureCacheFlush` at the top of
  `copyCVMetalTextureWithGpuBuffer:plane:`.
- `mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.cc` — `CVOpenGLTextureCacheFlush`
  at the top of `GetTexture` (macOS branch).

These two changes are kept as a **tracked patch** rather than direct edits to the
upstream files:
`mediapipe/tasks/macos/patches/mediapipe_macos_gpu_texture_cache_flush.patch`.
Because they target files in the mediapipe **root** repo (not an external Bazel
module), the `MODULE.bazel` `single_version_override` mechanism used for
`apple_support_lc_uuid.patch` can't apply them, so
`mediapipe/tasks/macos/build_macos_xcframework.sh` applies the patch idempotently
(`git apply`, with a reverse `--check` guard) before the Bazel build. A clean
checkout therefore builds the bounded-memory dylib deterministically.

Both are scoped to the macOS CVPixelBuffer path and don't change iOS behavior.
We deliberately did **not** set `kCVMetalTextureCacheMaximumTextureAgeKey = 0`:
that would shrink the working set further but risks releasing a texture while async
GPU work still references it. The age-based retention is what keeps it correct; the
per-frame flush is what makes it bounded.

### Result

The leak is gone. The stress test confirms growth is **constant regardless of frame
count** — the signature of a bounded working set, not a leak:

| frames | hand/GPU growth (before) | hand/GPU growth (after) |
| --- | --- | --- |
| 1,000 | +1542 MB | +229 MB |
| 12,000 | linear (~+18 GB) | **+227 MB** (~0.02 MB/frame, falling) |

The residual ~230 MB at full stress speed is the bounded texture-cache working set
(≈1 second of in-flight IOSurfaces; the stress loop runs at ~130 fps). At a real
30 fps webcam it is a few tens of MB. The `MemoryStressTests` GPU tests now assert a
near-zero **steady-state** rate (after a warmup that absorbs the one-time ramp) and
**pass** for hand/pose/face, while still catching the old ~1 MB/frame regression.

## What was also changed in the demo (hardening, still useful)

1. **Diagnostics:** resident-memory MB + frame count in the UI; a stderr log every
   60 frames (`task / delegate / stage / fps / inference ms / detections / rss MB /
   MPCImage.live`).
2. **Debug stages** (`Stage` picker): `Preview` / `Convert` / `Full` to bisect.
3. **`autoreleasepool`** around every captured-frame block.
4. **No unbounded queueing:** `alwaysDiscardsLateVideoFrames = true`, synchronous
   detection on the serial capture queue, coalesced (≤1 pending) UI publishing.
5. **Latest-only results**, `[weak self]`, no buffers/images retained beyond a frame.
6. **CVPixelBuffer conversion** locks read-only with a balanced `defer` unlock.
7. **C ownership:** every result freed once via `Mp*CloseResult`; every `MpImage`
   freed once in `MPCImage.dealloc` (verified by `MPCImage.liveInstanceCount`).

## Reproduce / verify

```bash
# All six pass now (CPU + GPU, hand/pose/face):
MP_*_MODEL=… MP_*_IMAGE=… swift test --filter MemoryStressTests
# Reproduce the raw (pre-warmup) growth shape without the per-iteration pool:
MP_STRESS_NO_POOL=1 swift test --filter MemoryStressTests
```

To re-attribute with `vmmap` (should now show IOSurface oscillating/bounded, not
climbing): run a long GPU stress and `vmmap --summary <pid>` a few times.
