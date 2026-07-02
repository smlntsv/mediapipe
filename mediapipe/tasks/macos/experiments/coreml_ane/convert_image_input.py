#!/usr/bin/env python3
"""Reconvert both hand models with ct.ImageType inputs so Core ML accepts a
CVPixelBuffer directly (the production marshaling path): Core ML performs the
uint8->float [0,1] normalization internally (ANE-side), no MLMultiArray fill.

The torch models (from onnx via tf2onnx) expect NHWC; ImageType feeds NCHW, so
each model is wrapped with a permute. PReLU is monkeypatched to the elementwise
form (see README: onnx2torch's masked-assignment lowering emits non_zero ops
that are ANE-ineligible).

Validation: CoreML(image) vs TFLite on an identical uint8 image.
"""
import json
import sys

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import coremltools as ct
import tensorflow as tf
import onnx2torch.node_converters.activations as act
from onnx2torch import convert


def _prelu_forward(self, input_tensor, slope):
    return F.relu(input_tensor) - slope * F.relu(-input_tensor)


act.OnnxPReLU.forward = _prelu_forward


class ImageWrapper(torch.nn.Module):
    """NCHW image input (CoreML ImageType) -> NHWC for the tf2onnx graph."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x.permute(0, 2, 3, 1))


def build(onnx_path, tflite_path, out_path, side):
    m = onnx.load(onnx_path)
    wrapped = ImageWrapper(convert(m).eval())
    example = torch.rand(1, 3, side, side)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    mlm = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, side, side),
                             scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlm.save(out_path)

    # Validate against tflite on the same uint8 image.
    rng = np.random.default_rng(7)
    img_u8 = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    x = (img_u8.astype(np.float32) / 255.0)[None, ...]  # NHWC [0,1]

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    ref = {d["name"]: interp.get_tensor(d["index"]).copy() for d in outs}

    from PIL import Image
    pred = list(mlm.predict({"image": Image.fromarray(img_u8)}).values())
    diffs = {}
    for name, r in ref.items():
        cands = [p for p in pred if tuple(p.shape) == tuple(r.shape)]
        diffs[name] = min(float(np.abs(np.asarray(c) - r).max()) for c in cands)
    print(f"  {out_path}: max|Δ| vs tflite:", json.dumps(diffs))
    # output feature names in CoreML order (needed by the Swift side)
    spec = mlm.get_spec()
    print("  outputs:", [(o.name, [int(d) for d in o.type.multiArrayType.shape]) for o in spec.description.output])


print("== landmark model (224) ==")
build("out/hand_landmarks_detector.onnx", "models/hand_landmarks_detector.tflite",
      "out/hand_landmarks_img_fp16.mlpackage", 224)
print("== palm model (192) ==")
build("out_palm/hand_landmarks_detector.onnx", "models/hand_detector.tflite",
      "out_palm/palm_img_fp16.mlpackage", 192)
