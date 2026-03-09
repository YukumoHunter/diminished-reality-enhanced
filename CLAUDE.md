# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diminished Reality — a mobile web app that uses real-time object detection to visually de-emphasize unhealthy grocery items via AR effects. A DETR-based ONNX model runs on a Python WebSocket backend; a React/Vite frontend renders effects on a canvas overlay.

## Commands

### Full app
```bash
# Run to start Linux/macOS:
./start-backend.sh
./start-frontend.sh
# Or on Windows:
start-backend.bat
start-frontend.bat
```

### Frontend (`frontend/`)
```bash
pnpm install       # install dependencies
pnpm dev           # dev server with auto HTTPS + WS proxy
pnpm build         # production build → dist/
```

### Backend (`backend/`)
```bash
uv sync            # install Python dependencies
uv run python server.py   # start WS server on 0.0.0.0:5174
```

No test framework is configured in either package.

## Architecture

**Data flow:**
1. Frontend captures JPEG frames from native camera via `canvas.toBlob()`
2. Sends `[4-byte uint32 requestId BE] + [JPEG bytes]` binary over WebSocket
3. Vite proxies `/ws` → `ws://localhost:5174` (backend runs plain WS, no SSL)
4. Backend: TurboJPEG decode → ImageOps.pad → ONNX inference (TensorRT/FP16)
5. ByteTrack tracker assigns persistent `tracker_id` per detection
6. Returns JSON `{detections: [{class, confidence, bbox, in_schijf_van_vijf, tracker_id}]}`
7. Frontend EMA-smooths bounding boxes per `tracker_id`, renders at 60fps via rAF

**Key files:**
- `backend/server.py` — WebSocket server, ONNX inference, ByteTrack tracking, coordinate math
- `frontend/src/App.jsx` — Camera, WebSocket, adaptive frame capture, EMA bbox smoothing, rAF loop
- `frontend/src/effects.js` — Canvas rendering; 4 effects (None/Blur/Overlay/Desaturate); iOS manual pixel paths; coordinate scaling for `object-fit: cover`
- `frontend/src/constants.js` — `SCHIJF_VAN_VIJF` health status for 60+ Dutch food products
- `frontend/src/Settings.jsx` — Dark glassmorphism settings panel; per-product health overrides

**Performance:**
- Backend: `asyncio.Queue(maxsize=1)` drops stale frames; `ThreadPoolExecutor(max_workers=1)` for non-blocking inference
- Frontend: adaptive send rate (waits for response or 500ms timeout); EMA smoothing with fade in/out; 60fps rendering decoupled from inference
- Binary WebSocket protocol avoids base64 overhead
- Pure numpy preprocessing (no torch dependency, saves ~858MB)

## Network / SSL

- Backend runs plain `ws://` on port 5174 (no SSL needed)
- Frontend uses `@vitejs/plugin-basic-ssl` for auto self-signed HTTPS
- Vite proxies `/ws` → `ws://localhost:5174`
- Phones on local hotspot connect to `https://<host-ip>:5173`, accept cert warning once

## iOS Quirks

`effects.js` detects iOS via UA string. On iOS: OffscreenCanvas + manual pixel blur/desaturation (CSS filters unavailable). Fullscreen uses `webkitRequestFullscreen`.

## Model

`backend/model/model.onnx` (~130 MB) is not tracked in git. The model is a DETR-based detector with 60 product classes + background (61 output channels). Place it at `backend/model/model.onnx`.
