// Minimal repro of MediaPipe's image->tensor Metal pattern: draw a full-screen
// quad (constant color) into an RGBA32Float texture that ALIASES a shared
// MTLBuffer (newTextureWithDescriptor:offset:bytesPerRow:, usage RenderTarget),
// then read the buffer on the CPU. Compares with: draw into a private texture,
// then blit texture -> buffer. Runs on every Metal device in the machine.
//   clang++ -std=c++17 -fobjc-arc -arch x86_64 -arch arm64 -mmacosx-version-min=13.0 -framework Metal -framework Foundation linear_rt_repro.mm -o linear_rt_repro
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <initializer_list>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; };
vertex V vs(uint vid [[vertex_id]]) {
  float2 q[6] = {{-1,1},{1,1},{1,-1},{-1,1},{1,-1},{-1,-1}};
  V v; v.p = float4(q[vid], 0, 1); return v;
}
fragment float4 fs(V in [[stage_in]]) { return float4(0.25, 0.5, 0.75, 0); }
)";

static int Run(id<MTLDevice> dev, bool via_blit, int w, int h) {
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [lib newFunctionWithName:@"vs"];
  pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!ps) { printf("pipeline error\n"); return 2; }

  const NSUInteger bpr = w * 16;
  const NSUInteger len = bpr * h;
  id<MTLBuffer> buf = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
  memset(buf.contents, 0, len);

  MTLTextureDescriptor* td = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:w height:h mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> tex;
  if (via_blit) {
    td.storageMode = MTLStorageModePrivate;
    tex = [dev newTextureWithDescriptor:td];
  } else {
    // Exactly what image_to_tensor_converter_metal.cc MTLTextureWithBuffer does.
    tex = [buf newTextureWithDescriptor:td offset:0 bytesPerRow:bpr];
  }
  if (!tex) { printf("texture nil\n"); return 2; }

  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = tex;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
  [enc setRenderPipelineState:ps];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
  [enc endEncoding];
  if (via_blit) {
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf destinationOffset:0
destinationBytesPerRow:bpr destinationBytesPerImage:len];
    [b endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];

  const float* f = (const float*)buf.contents;
  size_t good = 0, n = (size_t)w * h;
  for (size_t i = 0; i < n; ++i) {
    if (f[i * 4] == 0.25f && f[i * 4 + 1] == 0.5f && f[i * 4 + 2] == 0.75f) ++good;
  }
  printf("  %-26s %dx%d: %zu/%zu pixels correct, px0=(%g,%g,%g,%g) status=%ld\n",
         via_blit ? "private texture + blit" : "buffer-aliased texture", w, h, good, n,
         f[0], f[1], f[2], f[3], (long)cb.status);
  return good == n ? 0 : 1;
}

int main() {
  @autoreleasepool {
    int rc = 0;
    for (id<MTLDevice> dev in MTLCopyAllDevices()) {
      printf("%s (apple1=%d, unified=%d, lowPower=%d)\n", dev.name.UTF8String,
             [dev supportsFamily:MTLGPUFamilyApple1], dev.hasUnifiedMemory, dev.isLowPower);
      for (int s : {192, 224}) {
        rc |= Run(dev, false, s, s) << 1;
        rc |= Run(dev, true, s, s);
      }
    }
    return rc;
  }
}
