import { useRef, useEffect, useState, useCallback } from 'react';
import { prepareRenderDetections, renderEffects, EFFECT, EFFECT_STRENGTH, OUTLINE } from './effects';
import Settings from './Settings';

const CAPTURE_LONGEST = 640;
const SEND_TIMEOUT = 30;
const MIN_SEND_INTERVAL = 10;
const JPEG_QUALITY = 0.7;

function supportsNativeBlur() {
  return typeof CSS !== 'undefined' && typeof CSS.supports === 'function' && (
    CSS.supports('backdrop-filter', 'blur(1px)') ||
    CSS.supports('-webkit-backdrop-filter', 'blur(1px)')
  );
}

function syncNativeBlurLayer(layer, detections) {
  if (!layer) return;

  while (layer.childElementCount > detections.length) {
    layer.lastElementChild.remove();
  }

  while (layer.childElementCount < detections.length) {
    const box = document.createElement('div');
    box.className = 'native-blur-box';
    layer.appendChild(box);
  }

  layer.style.display = detections.length ? 'block' : 'none';

  for (let i = 0; i < detections.length; i++) {
    const det = detections[i];
    const box = layer.children[i];
    const blur = `blur(${Math.max(0, EFFECT_STRENGTH[EFFECT.BLUR] * det.opacity)}px)`;

    box.style.transform = `translate3d(${det.x}px, ${det.y}px, 0)`;
    box.style.width = `${det.w}px`;
    box.style.height = `${det.h}px`;
    box.style.opacity = `${det.opacity}`;
    box.style.backdropFilter = blur;
    box.style.webkitBackdropFilter = blur;
  }
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const blurLayerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [effect, setEffect] = useState(EFFECT.BLUR);
  const [outlineMode, setOutlineMode] = useState(OUTLINE.OFF);
  const [outlineColor, setOutlineColor] = useState('health_based');
  const [classOverrides, setClassOverrides] = useState({});

  // Refs for rAF-accessible state
  const settingsRef = useRef({ effect, outlineMode, outlineColor, classOverrides });
  const detectionsRef = useRef([]);
  const captureSize = useRef({ w: 0, h: 0 });
  const wsRef = useRef(null);
  const pendingRef = useRef(false);
  const sendTimerRef = useRef(null);
  const requestIdRef = useRef(0);
  const lastSendAtRef = useRef(0);
  const nativeBlurSupportedRef = useRef(supportsNativeBlur());

  // Sync settings to ref
  useEffect(() => {
    settingsRef.current = { effect, outlineMode, outlineColor, classOverrides };
  }, [effect, outlineMode, outlineColor, classOverrides]);

  // Send overrides to backend when user changes them
  const skipNextSyncRef = useRef(false);
  const mountedRef = useRef(false);
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (skipNextSyncRef.current) { skipNextSyncRef.current = false; return; }
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'save_settings', overrides: classOverrides }));
    }
  }, [classOverrides]);

  // --- Camera ---
  useEffect(() => {
    let stream = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.play();
        }
      } catch (e) {
        console.error('Camera error:', e);
      }
    })();
    return () => stream?.getTracks().forEach(t => t.stop());
  }, []);

  // Size capture canvas once video dimensions are known
  const onVideoReady = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const { videoWidth: vw, videoHeight: vh } = video;
    if (!vw || !vh) return;
    const scale = CAPTURE_LONGEST / Math.max(vw, vh);
    const cw = Math.round(vw * scale);
    const ch = Math.round(vh * scale);
    captureSize.current = { w: cw, h: ch };
    if (captureCanvasRef.current) {
      captureCanvasRef.current.width = cw;
      captureCanvasRef.current.height = ch;
    }
  }, []);

  // --- WebSocket ---
  useEffect(() => {
    let ws;
    let reconnectTimer;

    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WS open');
        setConnected(true);
        // Request saved overrides from backend
        ws.send(JSON.stringify({ type: 'get_settings' }));
      };
      ws.onclose = (event) => {
        console.log('WS close', { code: event.code, reason: event.reason });
        setConnected(false);
        detectionsRef.current = [];
        wsRef.current = null;
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = (event) => {
        console.error('WS error', event);
        ws.close();
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'settings') {
            // Load saved overrides from backend without echoing back
            if (data.overrides) {
              skipNextSyncRef.current = true;
              setClassOverrides(data.overrides);
            }
            return;
          }
          detectionsRef.current = data.detections || [];
        } catch (e) {
          console.error('WS message error:', e);
        }

        // Unblock next send
        pendingRef.current = false;
        clearTimeout(sendTimerRef.current);
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  // --- rAF loop: capture + render ---
  useEffect(() => {
    let rafId;

    const loop = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const capCanvas = captureCanvasRef.current;
      const blurLayer = blurLayerRef.current;

      if (video && canvas && video.readyState >= 2) {
        // Match canvas to display size
        const rect = video.getBoundingClientRect();
        if (canvas.width !== rect.width) canvas.width = rect.width;
        if (canvas.height !== rect.height) canvas.height = rect.height;

        // Send frame if WS is ready and not waiting for response
        if (wsRef.current?.readyState === WebSocket.OPEN && !pendingRef.current && capCanvas) {
          const { w: cw, h: ch } = captureSize.current;
          const now = performance.now();
          if (cw > 0 && ch > 0 && now - lastSendAtRef.current >= MIN_SEND_INTERVAL) {
            pendingRef.current = true;
            lastSendAtRef.current = now;

            const cCtx = capCanvas.getContext('2d');
            cCtx.drawImage(video, 0, 0, cw, ch);

            capCanvas.toBlob((blob) => {
              if (!blob || wsRef.current?.readyState !== WebSocket.OPEN) {
                pendingRef.current = false;
                return;
              }

              requestIdRef.current = (requestIdRef.current + 1) & 0xFFFFFFFF;
              const id = requestIdRef.current;

              blob.arrayBuffer().then((buf) => {
                const msg = new ArrayBuffer(4 + buf.byteLength);
                new DataView(msg).setUint32(0, id, false);
                new Uint8Array(msg, 4).set(new Uint8Array(buf));
                wsRef.current?.send(msg);

                // Timeout: unblock if server is slow
                sendTimerRef.current = setTimeout(() => { pendingRef.current = false; }, SEND_TIMEOUT);
              });
            }, 'image/jpeg', JPEG_QUALITY);
          }
        }

        // Render effects
        const s = settingsRef.current;
        const renderDetections = prepareRenderDetections(
          detectionsRef.current,
          s.classOverrides,
          captureSize.current.w,
          captureSize.current.h,
          video.videoWidth,
          video.videoHeight,
          rect.width,
          rect.height
        );
        const useNativeBlur = nativeBlurSupportedRef.current && s.effect === EFFECT.BLUR;

        syncNativeBlurLayer(
          blurLayer,
          useNativeBlur
            ? renderDetections.filter((det) => !det.isHealthy && det.opacity > 0 && det.w > 0 && det.h > 0)
            : []
        );

        renderEffects(
          canvas,
          video,
          renderDetections,
          useNativeBlur ? EFFECT.NONE : s.effect,
          s.outlineMode,
          s.outlineColor
        );
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, []);

  // --- Fullscreen ---
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else if (el.webkitRequestFullscreen) {
      el.webkitRequestFullscreen();
    } else {
      el.requestFullscreen();
    }
  }, []);

  return (
    <div className="app" ref={containerRef}>
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        onLoadedMetadata={onVideoReady}
      />
      <div ref={blurLayerRef} className="native-blur-layer" />
      <canvas ref={canvasRef} className="overlay-canvas" />
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

      <div className="controls">
        <button className="ctrl-btn" onClick={toggleFullscreen} aria-label="Fullscreen">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </button>
        <button className="ctrl-btn" onClick={() => setSettingsOpen(true)} aria-label="Settings">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        effect={effect}
        setEffect={setEffect}
        outlineMode={outlineMode}
        setOutlineMode={setOutlineMode}
        outlineColor={outlineColor}
        setOutlineColor={setOutlineColor}
        classOverrides={classOverrides}
        setClassOverrides={setClassOverrides}
      />
    </div>
  );
}

export default App;
