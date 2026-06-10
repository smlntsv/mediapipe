// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Parity reference: runs MediaPipe Tasks-Vision (Hand/Pose/Face) on a shared
// test image in the browser and saves the results as ../shared/output/<out> in
// the same envelope produced by the native macOS smoke test.
//
// Query params: ?image=<file in /assets>&out=<filename in shared/output>
// Defaults: image=test_image.jpg, out=web.json

import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

const WASM_PATH = "/wasm";
const ASSETS = "/assets";
const MIN_DETECTION = 0.5;
const MIN_PRESENCE = 0.5;
const MIN_TRACKING = 0.5;
const NUM_HANDS = 2;
const NUM_POSES = 1;
const NUM_FACES = 1;

const params = new URLSearchParams(location.search);
const IMAGE_NAME = params.get("image") ?? "test_image.jpg";
const OUT_NAME = params.get("out") ?? "web.json";
const RUNNING_MODE: "IMAGE" | "VIDEO" =
  (params.get("mode") ?? "").toUpperCase() === "VIDEO" ? "VIDEO" : "IMAGE";
const DELEGATE: "CPU" | "GPU" =
  (params.get("delegate") ?? "").toUpperCase() === "GPU" ? "GPU" : "CPU";
const VIDEO_TS = 0; // single still frame in VIDEO mode

const statusEl = document.getElementById("status") as HTMLParagraphElement;
const outputEl = document.getElementById("output") as HTMLPreElement;
const previewEl = document.getElementById("preview") as HTMLImageElement;

const setStatus = (t: string) => (statusEl.textContent = t);

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${url}`));
    img.src = url;
  });
}

async function fetchModel(name: string): Promise<{ buffer: Uint8Array; sha256: string }> {
  const ab = await (await fetch(`${ASSETS}/${name}`)).arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", ab);
  const sha256 = Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return { buffer: new Uint8Array(ab), sha256 };
}

// Web-compatible landmark serialization: visibility/presence always numbers.
function lm(p: { x: number; y: number; z: number; visibility?: number }) {
  return {
    x: p.x,
    y: p.y,
    z: p.z,
    visibility: p.visibility ?? 0,
    presence: (p as { presence?: number }).presence ?? 0,
  };
}
const lms = (arr: any[][]) => arr.map((inst) => inst.map(lm));
const cats = (arr: any[][]) =>
  arr.map((inst) =>
    inst.map((c) => ({
      index: c.index,
      score: c.score,
      categoryName: c.categoryName ?? null,
      displayName: c.displayName ?? null,
    })),
  );

async function main() {
  try {
    setStatus("Initializing WASM fileset…");
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    setStatus(`Loading ${IMAGE_NAME}…`);
    const image = await loadImage(`${ASSETS}/${IMAGE_NAME}`);
    previewEl.src = image.src;

    setStatus("Hashing + loading models…");
    const handModel = await fetchModel("hand_landmarker.task");
    const poseModel = await fetchModel("pose_landmarker.task");
    const faceModel = await fetchModel("face_landmarker.task");

    const hand = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: handModel.buffer, delegate: DELEGATE },
      runningMode: RUNNING_MODE,
      numHands: NUM_HANDS,
      minHandDetectionConfidence: MIN_DETECTION,
      minHandPresenceConfidence: MIN_PRESENCE,
      minTrackingConfidence: MIN_TRACKING,
    });
    const pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: poseModel.buffer, delegate: DELEGATE },
      runningMode: RUNNING_MODE,
      numPoses: NUM_POSES,
      minPoseDetectionConfidence: MIN_DETECTION,
      minPosePresenceConfidence: MIN_PRESENCE,
      minTrackingConfidence: MIN_TRACKING,
      outputSegmentationMasks: false,
    });
    const face = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: faceModel.buffer, delegate: DELEGATE },
      runningMode: RUNNING_MODE,
      numFaces: NUM_FACES,
      minFaceDetectionConfidence: MIN_DETECTION,
      minFacePresenceConfidence: MIN_PRESENCE,
      minTrackingConfidence: MIN_TRACKING,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });

    setStatus(`Running detection (${RUNNING_MODE})…`);
    const h =
      RUNNING_MODE === "IMAGE" ? hand.detect(image) : hand.detectForVideo(image, VIDEO_TS);
    const p =
      RUNNING_MODE === "IMAGE" ? pose.detect(image) : pose.detectForVideo(image, VIDEO_TS);
    const f =
      RUNNING_MODE === "IMAGE" ? face.detect(image) : face.detectForVideo(image, VIDEO_TS);

    const envelope = {
      source: "web",
      image: IMAGE_NAME,
      models: { hand: handModel.sha256, pose: poseModel.sha256, face: faceModel.sha256 },
      options: {
        hand: {
          numHands: NUM_HANDS,
          minHandDetectionConfidence: MIN_DETECTION,
          minHandPresenceConfidence: MIN_PRESENCE,
          minTrackingConfidence: MIN_TRACKING,
          runningMode: RUNNING_MODE,
          delegate: DELEGATE,
        },
        pose: {
          numPoses: NUM_POSES,
          minPoseDetectionConfidence: MIN_DETECTION,
          minPosePresenceConfidence: MIN_PRESENCE,
          minTrackingConfidence: MIN_TRACKING,
          outputSegmentationMasks: false,
          runningMode: RUNNING_MODE,
          delegate: DELEGATE,
        },
        face: {
          numFaces: NUM_FACES,
          minFaceDetectionConfidence: MIN_DETECTION,
          minFacePresenceConfidence: MIN_PRESENCE,
          minTrackingConfidence: MIN_TRACKING,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
          runningMode: RUNNING_MODE,
          delegate: DELEGATE,
        },
      },
      hand: {
        landmarks: lms(h.landmarks),
        worldLandmarks: lms(h.worldLandmarks),
        handedness: cats(h.handedness),
        handednesses: cats(
          (h as { handednesses?: any[][] }).handednesses ?? h.handedness,
        ),
      },
      pose: {
        landmarks: lms(p.landmarks),
        worldLandmarks: lms(p.worldLandmarks),
        segmentationMasks: null,
      },
      face: {
        faceLandmarks: lms(f.faceLandmarks),
        faceBlendshapes: (f.faceBlendshapes ?? []).map((c) => ({
          categories: c.categories.map((x) => ({
            index: x.index,
            score: x.score,
            categoryName: x.categoryName ?? null,
            displayName: x.displayName ?? null,
          })),
          headIndex: c.headIndex ?? 0,
          headName: c.headName ?? null,
        })),
        facialTransformationMatrixes: (f.facialTransformationMatrixes ?? []).map((m) => ({
          rows: m.rows,
          columns: m.columns,
          data: Array.from(m.data),
        })),
      },
    };

    const json = JSON.stringify(envelope, null, 2);
    outputEl.textContent = json;

    setStatus(`Saving ../shared/output/${OUT_NAME}…`);
    const resp = await fetch(`/save-web-json?out=${encodeURIComponent(OUT_NAME)}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: json,
    });
    const saved = await resp.json();

    const counts = `hand=${envelope.hand.landmarks.length}×${
      envelope.hand.landmarks[0]?.length ?? 0
    }, pose=${envelope.pose.landmarks.length}×${
      envelope.pose.landmarks[0]?.length ?? 0
    }, face=${envelope.face.faceLandmarks.length}×${
      envelope.face.faceLandmarks[0]?.length ?? 0
    }`;
    setStatus(`Done (${counts}). Saved to ${saved.saved ?? OUT_NAME}`);
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }
}

void main();
