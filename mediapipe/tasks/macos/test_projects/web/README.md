# Web parity reference (`@mediapipe/tasks-vision`)

Runs HandLandmarker, PoseLandmarker, and FaceLandmarker in the browser on the
shared test image and writes `../shared/output/web.json` in the parity envelope
(see [../README.md](../README.md)).

## Run

```bash
# 1. Get the shared models + image (once):
../shared/download_models.sh

# 2. Install deps and stage assets/wasm locally:
npm install
npm run setup

# 3. Start the dev server (opens the browser):
npm run dev
```

On load the page runs all three landmarkers and POSTs the result to the dev
server, which writes `../shared/output/web.json`. The status line shows the
detected counts; the JSON is also shown on the page.

### Choosing image / output

Query params let you target a different image and output file (e.g. for
full-body pose parity):

```
http://localhost:5173/?image=test_image_full_body.jpg&out=web_pose.json
```

Notes:
- WASM and `.task` models are served locally from `public/` (no CDN needed).
- Options used (matched to the native side): `numHands: 2`, `numPoses: 1`,
  `numFaces: 1`, confidences `0.5`, `runningMode: "IMAGE"`, **`delegate: "CPU"`**,
  and face blendshapes + transformation matrices enabled.
- Model `.task` files are loaded via `modelAssetBuffer` so their SHA256 can be
  recorded in the JSON (`models`) for cross-engine verification.
