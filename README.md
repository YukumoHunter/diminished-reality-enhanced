# Diminished Reality

Mobile web app that detects grocery items in real time and visually de-emphasizes unhealthy ones with AR effects.

## Requirements

- `pnpm`
- `uv`
- `Python 3.11+`
- `backend/model/model.onnx`

## Install

```bash
cd frontend
pnpm install

cd ../backend
uv sync
```

Place the ONNX model at `backend/model/model.onnx`.

## Run

Linux/macOS:

```bash
./start-backend.sh
./start-frontend.sh
```

Windows:

```bat
start-backend.bat
start-frontend.bat
```

Frontend runs on `https://localhost:5173` and proxies `/ws` to the backend on `ws://localhost:5174`.
