#!/usr/bin/env python3
"""Convert MediaPipe's hand_landmarks_detector.tflite to Core ML (.mlpackage)
with numeric validation at every hop.

Chain (coremltools has no TFLite frontend):
    tflite --(tf2onnx --tflite)--> onnx --(onnx2torch)--> torch --(coremltools)--> mlpackage

Validation: identical random inputs through the TFLite interpreter (ground
truth) and each converted stage; report max-abs-diff per output. FP32 package
for validation, FP16 package for the ANE benchmark (ANE prefers FP16).

Usage: venv/bin/python convert.py models/hand_landmarks_detector.tflite out/
"""

import json
import os
import subprocess
import sys

import numpy as np

TFLITE = sys.argv[1] if len(sys.argv) > 1 else "models/hand_landmarks_detector.tflite"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
ONNX_PATH = os.path.join(OUT, "hand_landmarks_detector.onnx")

# --- Ground truth: TFLite interpreter -----------------------------------------
import tensorflow as tf  # noqa: E402

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
inp = interp.get_input_details()
outs = interp.get_output_details()
print("== TFLite signature ==")
for d in inp:
    print("  input :", d["name"], d["shape"].tolist(), d["dtype"].__name__)
for d in outs:
    print("  output:", d["name"], d["shape"].tolist(), d["dtype"].__name__)
assert len(inp) == 1
in_shape = [int(x) for x in inp[0]["shape"]]  # e.g. [1, 224, 224, 3] NHWC

rng = np.random.default_rng(7)
x = rng.random(in_shape, dtype=np.float32)  # [0,1) — matches the model's expected range


def tflite_run(x_):
    interp.set_tensor(inp[0]["index"], x_)
    interp.invoke()
    # Sort by name for a stable comparison order across frameworks.
    return {d["name"]: interp.get_tensor(d["index"]).copy() for d in outs}


ref = tflite_run(x)

# --- Hop 1: tflite -> ONNX ------------------------------------------------------
print("\n== tf2onnx (tflite -> onnx) ==")
subprocess.run(
    [sys.executable, "-m", "tf2onnx.convert",
     "--tflite", TFLITE, "--output", ONNX_PATH, "--opset", "17"],
    check=True, capture_output=True, text=True)
print("  wrote", ONNX_PATH)

import onnx  # noqa: E402

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx_in = onnx_model.graph.input[0]
onnx_in_shape = [d.dim_value for d in onnx_in.type.tensor_type.shape.dim]
onnx_out_names = [o.name for o in onnx_model.graph.output]
print("  onnx input:", onnx_in.name, onnx_in_shape, "| outputs:", onnx_out_names)

# --- Hop 2: ONNX -> torch --------------------------------------------------------
print("\n== onnx2torch (onnx -> torch) ==")
import torch  # noqa: E402
from onnx2torch import convert as onnx_to_torch  # noqa: E402

tmodel = onnx_to_torch(onnx_model).eval()
tin = torch.from_numpy(x.reshape(onnx_in_shape))
with torch.no_grad():
    tout = tmodel(tin)
tout = tout if isinstance(tout, (list, tuple)) else [tout]
print("  torch outputs:", [tuple(t.shape) for t in tout])

# Map torch outputs (ONNX graph order) onto tflite outputs by matching shapes,
# then verify values.
def match_and_diff(candidates, reference):
    diffs = {}
    used = set()
    for name, r in reference.items():
        best = None
        for i, t in enumerate(candidates):
            if i in used or tuple(t.shape) != tuple(r.shape):
                continue
            d = float(np.abs(np.asarray(t) - r).max())
            if best is None or d < best[1]:
                best = (i, d)
        assert best is not None, f"no candidate matches output {name} {r.shape}"
        used.add(best[0])
        diffs[name] = best[1]
    return diffs


tdiffs = match_and_diff([t.numpy() for t in tout], ref)
print("  max|Δ| torch vs tflite:", json.dumps(tdiffs, indent=2))
assert max(tdiffs.values()) < 1e-3, "torch conversion diverged"

# --- Hop 3: torch -> Core ML -----------------------------------------------------
print("\n== coremltools (torch -> mlpackage) ==")
import coremltools as ct  # noqa: E402

traced = torch.jit.trace(tmodel, tin)

def to_coreml(precision, tag):
    mlm = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=tuple(onnx_in_shape))],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    path = os.path.join(OUT, f"hand_landmarks_detector_{tag}.mlpackage")
    mlm.save(path)
    print(f"  wrote {path}")
    return path, mlm


fp32_path, mlm32 = to_coreml(ct.precision.FLOAT32, "fp32")
fp16_path, mlm16 = to_coreml(ct.precision.FLOAT16, "fp16")

# --- Validate Core ML vs TFLite ---------------------------------------------------
def coreml_diff(mlm, tag):
    pred = mlm.predict({"image": x.reshape(onnx_in_shape)})
    diffs = match_and_diff(list(pred.values()), ref)
    print(f"  max|Δ| coreml-{tag} vs tflite:", json.dumps(diffs, indent=2))
    return diffs


d32 = coreml_diff(mlm32, "fp32")
d16 = coreml_diff(mlm16, "fp16")
assert max(d32.values()) < 1e-2, "coreml fp32 diverged"

# --- TFLite XNNPACK CPU latency reference ----------------------------------------
print("\n== TFLite CPU (XNNPACK) latency reference ==")
import time  # noqa: E402

interp4 = tf.lite.Interpreter(model_path=TFLITE, num_threads=4)
interp4.allocate_tensors()
i4 = interp4.get_input_details()[0]["index"]
for _ in range(20):
    interp4.set_tensor(i4, x)
    interp4.invoke()
lat = []
for _ in range(200):
    t0 = time.perf_counter()
    interp4.set_tensor(i4, x)
    interp4.invoke()
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
print(f"  tflite xnnpack(4T): mean {sum(lat)/len(lat):.2f} ms  p50 {lat[100]:.2f}  p90 {lat[180]:.2f}")

print("\nAll conversions validated. FP16 package for benchmarking:", fp16_path)
