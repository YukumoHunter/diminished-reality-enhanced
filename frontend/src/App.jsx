import { useRef, useEffect, useState, useCallback } from 'react';
import { renderEffects, EFFECT, OUTLINE } from './effects';
import Settings from './Settings';

const CAPTURE_LONGEST = 640;
const SEND_TIMEOUT = 30;
const MIN_SEND_INTERVAL = 10;
const JPEG_QUALITY = 0.7;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const containerRef = useRef(null);

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

  // Sync settings to ref
  useEffect(() => {
    settingsRef.current = { effect, outlineMode, outlineColor, classOverrides };
  }, [effect, outlineMode, outlineColor, classOverrides]);

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
            const cCtx = capCanvas.getContext('2d');
            cCtx.drawImage(video, 0, 0, cw, ch);

            capCanvas.toBlob((blob) => {
              if (!blob || wsRef.current?.readyState !== WebSocket.OPEN) return;

              requestIdRef.current = (requestIdRef.current + 1) & 0xFFFFFFFF;
              const id = requestIdRef.current;

              blob.arrayBuffer().then((buf) => {
                const msg = new ArrayBuffer(4 + buf.byteLength);
                new DataView(msg).setUint32(0, id, false);
                new Uint8Array(msg, 4).set(new Uint8Array(buf));
                wsRef.current?.send(msg);
                pendingRef.current = true;
                lastSendAtRef.current = performance.now();

                // Timeout: unblock if server is slow
                sendTimerRef.current = setTimeout(() => { pendingRef.current = false; }, SEND_TIMEOUT);
              });
            }, 'image/jpeg', JPEG_QUALITY);
          }
        }

        // Render effects
        const s = settingsRef.current;
        renderEffects(
          canvas, video, detectionsRef.current,
          s.effect, s.outlineMode, s.outlineColor, s.classOverrides,
          captureSize.current.w, captureSize.current.h
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
