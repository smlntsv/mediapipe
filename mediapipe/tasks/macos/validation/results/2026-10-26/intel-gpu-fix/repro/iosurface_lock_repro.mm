// Repro of MediaPipe's CPU->GPU image upload on macOS
// (mediapipe/objc/util.cc CreateCVPixelBufferForImageFrame, SRGBA case):
// CVPixelBufferCreate with {IOSurfaceProperties: {}, OpenGLCompatibility: true},
// lock with kCVPixelBufferLock_ReadOnly, WRITE pixels with the CPU, unlock
// ReadOnly, then read the pixels on the GPU through CVMetalTextureCache.
// Compares lock flags ReadOnly (MediaPipe) vs 0 (read-write), per Metal device.
//   clang++ -std=c++17 -fobjc-arc -arch x86_64 -arch arm64 -mmacosx-version-min=13.0 -framework Metal -framework CoreVideo -framework Foundation iosurface_lock_repro.mm -o iosurface_lock_repro
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

static CFDictionaryRef Attrs() {
  NSDictionary* d = @{
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (id)kCVPixelFormatOpenGLCompatibility : @YES,
  };
  return (__bridge_retained CFDictionaryRef)d;
}

static int Run(id<MTLDevice> dev, CVOptionFlags lock_flags, int frame) {
  const int w = 384, h = 512;
  CVPixelBufferRef pb = nullptr;
  CVReturn st = CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, Attrs(), &pb);
  if (st != kCVReturnSuccess) { printf("create failed %d\n", st); return 2; }

  CVMetalTextureCacheRef cache = nullptr;
  CVMetalTextureCacheCreate(nullptr, nullptr, dev, nullptr, &cache);

  // A different color per frame, so a stale GPU copy would show.
  const uint8_t b = 10 + frame * 40, g = 128, r = 200;
  CVPixelBufferLockBaseAddress(pb, lock_flags);
  uint8_t* base = (uint8_t*)CVPixelBufferGetBaseAddress(pb);
  const size_t bpr = CVPixelBufferGetBytesPerRow(pb);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = base + y * bpr + x * 4;
      p[0] = b; p[1] = g; p[2] = r; p[3] = 255;
    }
  CVPixelBufferUnlockBaseAddress(pb, lock_flags);

  CVMetalTextureRef cvtex = nullptr;
  st = CVMetalTextureCacheCreateTextureFromImage(nullptr, cache, pb, nullptr, MTLPixelFormatBGRA8Unorm, w, h, 0, &cvtex);
  if (st != kCVReturnSuccess) { printf("texture failed %d\n", st); return 2; }
  id<MTLTexture> tex = CVMetalTextureGetTexture(cvtex);

  id<MTLBuffer> out = [dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
  memset(out.contents, 0, w * h * 4);
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(w, h, 1) toBuffer:out destinationOffset:0
destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  const uint8_t* o = (const uint8_t*)out.contents;
  size_t good = 0;
  for (int i = 0; i < w * h; ++i)
    if (o[i * 4] == b && o[i * 4 + 1] == g && o[i * 4 + 2] == r) ++good;
  printf("  lock=%-9s frame %d: GPU sees %6zu/%d expected pixels, px0 BGRA=(%d,%d,%d,%d) want (%d,%d,%d,255)\n",
         lock_flags ? "ReadOnly" : "0 (rw)", frame, good, w * h, o[0], o[1], o[2], o[3], b, g, r);
  CFRelease(cvtex);
  CFRelease(cache);
  CVPixelBufferRelease(pb);
  return good == (size_t)(w * h) ? 0 : 1;
}

int main() {
  @autoreleasepool {
    int rc = 0;
    for (id<MTLDevice> dev in MTLCopyAllDevices()) {
      printf("%s (apple1=%d, unified=%d)\n", dev.name.UTF8String,
             [dev supportsFamily:MTLGPUFamilyApple1], dev.hasUnifiedMemory);
      for (int f = 0; f < 2; ++f) {
        rc |= Run(dev, kCVPixelBufferLock_ReadOnly, f) << 1;
        rc |= Run(dev, 0, f);
      }
    }
    return rc;
  }
}
