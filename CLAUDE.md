# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diminished Reality — a mobile web app that uses real-time object detection to visually de-emphasize unhealthy grocery items via AR effects. A DETR-based ONNX model runs on a Python WebSocket backend; a React/Vite frontend renders effects on a canvas overlay.

## Commands

### Frontend (`diminished-frontend/`)
```bash
pnpm install       # install dependencies
pnpm dev           # dev server with HTTPS (requires ../cert/)
pnpm build         # production build → dist/
pnpm lint          # ESLint
pnpm preview       # preview production build
```

### Backend (`back-end/`)
```bash
uv sync            # install Python dependencies
uv run python back-end.py   # start WSS server on 0.0.0.0:5174
```

No test framework is configured in either package.

## Architecture

**Data flow:**
1. Frontend captures 560×560 JPEG frames from webcam every 200ms
2. Sends `{image: base64, requestId}` over WebSocket to `wss://diminish.soaratorium.com:5174`
3. Backend: TurboJPEG decode → ImageOps.pad(560×560) → ONNX inference (TensorRT/FP16)
4. Model outputs `dets` (cx,cy,w,h normalized) + `labels` (61-class logits)
5. Backend filters (confidence > 0.5, class < 60), transforms coordinates, looks up `SCHIJF_VAN_VIJF` health status
6. Returns `{detections: [{class, confidence, bbox, in_schijf_van_vijf}]}`
7. Frontend draws canvas effects per detection via `DiminishObject.jsx`

**Key files:**
- [back-end/back-end.py](back-end/back-end.py) — WebSocket server, ONNX inference, coordinate math, `SCHIJF_VAN_VIJF` health dict (currently all `False` for testing)
- [diminished-frontend/src/App.jsx](diminished-frontend/src/App.jsx) — WebSocket client, webcam capture, motion detection, settings state
- [diminished-frontend/src/DiminishObject.jsx](diminished-frontend/src/DiminishObject.jsx) — Canvas rendering; 4 effect types: None/Blur/Overlay/Desaturate; iOS-specific implementations
- [diminished-frontend/src/constants.js](diminished-frontend/src/constants.js) — `SCHIJF_VAN_VIJF_DEFAULTS` health status for 60 Dutch food products
- [diminished-frontend/src/Settings.jsx](diminished-frontend/src/Settings.jsx) — Settings panel; per-product health overrides

**Performance:**
- Backend uses a queue (maxsize=1) that drops stale frames when GPU is busy; inference runs in `ThreadPoolExecutor`
- Frontend measures RTT in `rttArray` (max 1000 samples)
- Motion detection (DeviceMotionEvent) skips inference during fast phone movement (accel > 20 m/s², rotation > 270 deg/s)

## SSL / Certificates

Both backend and frontend require HTTPS. Certificates live at `cert/cert.pem` and `cert/key.pem` (relative to repo root). The `ca/` directory holds the CA for the `diminish.soaratorium.com` domain. The backend also reads from that path at startup.

## iOS Quirks

`DiminishObject.jsx` has separate code paths for iOS: OffscreenCanvas is used differently, blur/desaturation are implemented manually (CSS filters are unavailable), and fullscreen uses the webkit API.

## Model

`back-end/model/model.onnx` (~130 MB) is not tracked in git. The model is a DETR-based detector with 60 product classes + background (61 output channels).
