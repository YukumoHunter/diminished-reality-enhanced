export const EFFECT = { NONE: 0, BLUR: 1, OVERLAY: 2, DESATURATE: 3 };
export const OUTLINE = { OFF: 0, HEALTHY: 1, ALL: 2 };

const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

// --- Coordinate transform: capture coords → display coords (object-fit: cover) ---

function captureToDisplay(bbox, captureW, captureH, videoW, videoH, displayW, displayH) {
  // 1. Capture → native video coords
  const scaleX = videoW / captureW;
  const scaleY = videoH / captureH;
  let [x, y, w, h] = bbox;
  x *= scaleX;
  y *= scaleY;
  w *= scaleX;
  h *= scaleY;

  // 2. Compute object-fit: cover crop rect in native video coords
  const displayAR = displayW / displayH;
  const videoAR = videoW / videoH;
  let sx, sy, sw, sh;
  if (videoAR > displayAR) {
    // Video wider than display — crop sides
    sh = videoH;
    sw = videoH * displayAR;
    sx = (videoW - sw) / 2;
    sy = 0;
  } else {
    // Video taller than display — crop top/bottom
    sw = videoW;
    sh = videoW / displayAR;
    sx = 0;
    sy = (videoH - sh) / 2;
  }

  // 3. Native → display
  const dx = (x - sx) * (displayW / sw);
  const dy = (y - sy) * (displayH / sh);
  const dw = w * (displayW / sw);
  const dh = h * (displayH / sh);

  return [dx, dy, dw, dh];
}

// --- iOS manual pixel effects ---

function manualBlur(imageData, radius) {
  const { data, width, height } = imageData;
  const orig = new Uint8ClampedArray(data);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const i = (ny * width + nx) * 4;
            r += orig[i]; g += orig[i + 1]; b += orig[i + 2]; a += orig[i + 3];
            n++;
          }
        }
      }
      const i = (y * width + x) * 4;
      data[i] = r / n; data[i + 1] = g / n; data[i + 2] = b / n; data[i + 3] = a / n;
    }
  }
}

function manualDesaturate(imageData, strength) {
  const { data } = imageData;
  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] += (lum - data[i]) * strength;
    data[i + 1] += (lum - data[i + 1]) * strength;
    data[i + 2] += (lum - data[i + 2]) * strength;
  }
}

// --- Effect applicators ---

function applyBlur(ctx, video, x, y, w, h, strength) {
  if (strength <= 0) return;
  const ix = Math.max(0, Math.round(x));
  const iy = Math.max(0, Math.round(y));
  const iw = Math.round(w);
  const ih = Math.round(h);
  if (iw <= 0 || ih <= 0) return;

  if (isIOS) {
    const off = new OffscreenCanvas(iw, ih);
    const oc = off.getContext('2d');
    oc.drawImage(ctx.canvas, ix, iy, iw, ih, 0, 0, iw, ih);
    const imgData = oc.getImageData(0, 0, iw, ih);
    manualBlur(imgData, Math.min(strength, 8));
    oc.putImageData(imgData, 0, 0);
    ctx.drawImage(off, ix, iy);
  } else {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ix, iy, iw, ih);
    ctx.clip();
    ctx.filter = `blur(${strength}px)`;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                  0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  }
}

function applyOverlay(ctx, x, y, w, h, opacity) {
  if (opacity <= 0) return;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.fillStyle = 'white';
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

function applyDesaturate(ctx, video, x, y, w, h, strength) {
  if (strength <= 0) return;
  const ix = Math.max(0, Math.round(x));
  const iy = Math.max(0, Math.round(y));
  const iw = Math.round(w);
  const ih = Math.round(h);
  if (iw <= 0 || ih <= 0) return;

  if (isIOS) {
    const off = new OffscreenCanvas(iw, ih);
    const oc = off.getContext('2d');
    oc.drawImage(ctx.canvas, ix, iy, iw, ih, 0, 0, iw, ih);
    const imgData = oc.getImageData(0, 0, iw, ih);
    manualDesaturate(imgData, strength);
    oc.putImageData(imgData, 0, 0);
    ctx.drawImage(off, ix, iy);
  } else {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ix, iy, iw, ih);
    ctx.clip();
    ctx.filter = `saturate(${1 - strength})`;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                  0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  }
}

function drawOutline(ctx, x, y, w, h, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

// --- Strength maps ---

const EFFECT_STRENGTH = {
  [EFFECT.NONE]: 0,
  [EFFECT.BLUR]: 20,
  [EFFECT.OVERLAY]: 1,
  [EFFECT.DESATURATE]: 1,
};

// --- Main render function ---

export function renderEffects(canvas, video, detections, effectType, outlineMode, outlineColor, classOverrides, captureW, captureH) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const displayW = canvas.width;
  const displayH = canvas.height;
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  if (!videoW || !videoH || !captureW || !captureH) return;

  for (const det of detections) {
    const isHealthy = classOverrides[det.class] !== undefined
      ? classOverrides[det.class]
      : det.in_schijf_van_vijf;

    const [dx, dy, dw, dh] = captureToDisplay(
      det.bbox, captureW, captureH, videoW, videoH, displayW, displayH
    );

    const opacity = det.opacity ?? 1;

    // Outline
    if (outlineMode === OUTLINE.ALL || (outlineMode === OUTLINE.HEALTHY && isHealthy)) {
      const oColor = outlineColor === 'health_based'
        ? (isHealthy ? '#22c55e' : '#ef4444')
        : outlineColor;
      ctx.globalAlpha = opacity;
      drawOutline(ctx, dx, dy, dw, dh, oColor);
      ctx.globalAlpha = 1;
    }

    // Diminish effect (only for unhealthy)
    if (!isHealthy) {
      const strength = EFFECT_STRENGTH[effectType] * opacity;
      switch (effectType) {
        case EFFECT.BLUR:
          applyBlur(ctx, video, dx, dy, dw, dh, strength);
          break;
        case EFFECT.OVERLAY:
          applyOverlay(ctx, dx, dy, dw, dh, strength);
          break;
        case EFFECT.DESATURATE:
          applyDesaturate(ctx, video, dx, dy, dw, dh, strength);
          break;
      }
    }
  }
}
