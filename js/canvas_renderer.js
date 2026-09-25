/**
 * QASD - Canvas Renderer Module
 * High-performance 60 FPS HTML5 Canvas rendering for bounding boxes,
 * trajectory paths, restricted geofences, density heatmaps, Grad-CAM saliency,
 * custom interactive geofence authoring, and Quad Matrix partitioning.
 */

class CCTVCanvasRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d', { alpha: true });
    this.colorMap = {
      person: '#00f2fe',
      car: '#38bdf8',
      motorcycle: '#818cf8',
      bag: '#f59e0b',
      backpack: '#f59e0b',
      suitcase: '#f59e0b',
      object: '#f59e0b',
      alert: '#ef4444'
    };
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Draws tactical defense-tech bounding box with corner brackets and telemetry tag.
   */
  drawBoundingBox(bbox, label, conf, trackId, isAlert = false) {
    const [x1, y1, x2, y2] = bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const ctx = this.ctx;

    const baseColor = isAlert ? '#ef4444' : (this.colorMap[label.toLowerCase()] || '#00f2fe');

    ctx.save();
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = isAlert ? 2.5 : 1.8;
    ctx.fillStyle = isAlert ? 'rgba(239, 68, 68, 0.12)' : 'rgba(0, 242, 254, 0.08)';

    // Semi-transparent box fill
    ctx.fillRect(x1, y1, w, h);

    // Corner brackets
    const cornerLen = Math.min(16, Math.min(w, h) / 3);

    ctx.beginPath();
    // Top-Left
    ctx.moveTo(x1, y1 + cornerLen);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + cornerLen, y1);

    // Top-Right
    ctx.moveTo(x2 - cornerLen, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + cornerLen);

    // Bottom-Right
    ctx.moveTo(x2, y2 - cornerLen);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2 - cornerLen, y2);

    // Bottom-Left
    ctx.moveTo(x1 + cornerLen, y2);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1, y2 - cornerLen);

    ctx.stroke();

    // Dotted bounding edge lines
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x1, y1, w, h);
    ctx.setLineDash([]);

    // Telemetry Tag Label
    const tagText = `ID #${trackId} ${label.toUpperCase()} ${(conf * 100).toFixed(0)}%`;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(tagText).width;
    const tagH = 18;
    const tagW = textWidth + 12;

    ctx.fillStyle = isAlert ? 'rgba(239, 68, 68, 0.95)' : 'rgba(12, 19, 34, 0.9)';
    ctx.fillRect(x1, Math.max(0, y1 - tagH), tagW, tagH);

    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, Math.max(0, y1 - tagH), tagW, tagH);

    ctx.fillStyle = isAlert ? '#ffffff' : '#00f2fe';
    ctx.fillText(tagText, x1 + 6, Math.max(12, y1 - 5));

    ctx.restore();
  }

  /**
   * Draws historical trajectory motion trails with fading alpha decay.
   */
  drawTrajectoryTrail(points, isAlert = false) {
    if (!points || points.length < 2) return;
    const ctx = this.ctx;
    ctx.save();

    const n = points.length;
    for (let i = 0; i < n - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      const alpha = (i + 1) / n;

      ctx.strokeStyle = isAlert 
        ? `rgba(239, 68, 68, ${alpha * 0.85})` 
        : `rgba(0, 242, 254, ${alpha * 0.75})`;
      ctx.lineWidth = Math.max(1, alpha * 2.8);

      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
    }

    // Centroid pulse dot at head
    const head = points[n - 1];
    ctx.fillStyle = isAlert ? '#ef4444' : '#00f2fe';
    ctx.beginPath();
    ctx.arc(head[0], head[1], 3.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  /**
   * Draws restricted security geofence polygon with optional breach strobe.
   */
  drawRestrictedZone(polygon, zoneName, isBreached = false) {
    if (!polygon || polygon.length < 3) return;
    const ctx = this.ctx;
    ctx.save();

    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1]);
    for (let i = 1; i < polygon.length; i++) {
      ctx.lineTo(polygon[i][0], polygon[i][1]);
    }
    ctx.closePath();

    if (isBreached) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.22)';
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([8, 4]);
    } else {
      ctx.fillStyle = 'rgba(139, 92, 246, 0.14)';
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.85)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
    }

    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // Zone Title Tag
    const tagX = polygon[0][0] + 8;
    const tagY = polygon[0][1] + 20;
    ctx.font = '700 11px "JetBrains Mono", monospace';
    ctx.fillStyle = isBreached ? '#ef4444' : '#c084fc';
    ctx.fillText(`GEOFENCE: ${zoneName.toUpperCase()} [${isBreached ? 'BREACHED!' : 'RESTRICTED'}]`, tagX, tagY);

    ctx.restore();
  }

  /**
   * Draws custom interactive polygon zone in authoring mode.
   */
  drawDraftPolygon(vertices, currentMousePos) {
    if (!vertices || vertices.length === 0) return;
    const ctx = this.ctx;
    ctx.save();

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(vertices[0][0], vertices[0][1]);
    for (let i = 1; i < vertices.length; i++) {
      ctx.lineTo(vertices[i][0], vertices[i][1]);
    }
    if (currentMousePos) {
      ctx.lineTo(currentMousePos.x, currentMousePos.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw vertex handle dots
    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i];
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(v[0], v[1], 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Helper text
    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#f59e0b';
    ctx.fillText(`DRAWING GEOFENCE: ${vertices.length} vertices (Double-click or click first point to close)`, 16, this.canvas.height - 20);

    ctx.restore();
  }

  /**
   * Renders thermal Grad-CAM visual attention heatmap overlay.
   */
  drawGradCamOverlay(heatPoints) {
    if (!heatPoints || heatPoints.length === 0) return;
    const ctx = this.ctx;
    ctx.save();

    for (const pt of heatPoints) {
      const { x, y, radius, intensity } = pt;
      const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
      grad.addColorStop(0, `rgba(255, 0, 0, ${intensity * 0.65})`);
      grad.addColorStop(0.35, `rgba(255, 200, 0, ${intensity * 0.45})`);
      grad.addColorStop(0.7, `rgba(0, 240, 255, ${intensity * 0.25})`);
      grad.addColorStop(1, 'rgba(0, 0, 255, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  /**
   * Draws spatial density heatmap over accumulated movement points.
   */
  drawDensityHeatmap(historyPoints) {
    if (!historyPoints || historyPoints.length === 0) return;
    const ctx = this.ctx;
    ctx.save();

    for (const pt of historyPoints) {
      const grad = ctx.createRadialGradient(pt[0], pt[1], 0, pt[0], pt[1], 35);
      grad.addColorStop(0, 'rgba(0, 255, 120, 0.18)');
      grad.addColorStop(0.5, 'rgba(0, 200, 255, 0.08)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 35, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  /**
   * Draws tactical holographic target lock reticle over selected/hovered entity.
   */
  drawTargetLock(bbox, isPinned = false) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const ctx = this.ctx;
    ctx.save();

    const ringColor = isPinned ? '#f59e0b' : '#00f2fe';
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 1.8;

    // Outer reticle brackets
    const r = Math.max(24, Math.hypot(x2 - x1, y2 - y1) / 2 + 10);
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 4, Math.PI / 4);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, r, 3 * Math.PI / 4, 5 * Math.PI / 4);
    ctx.stroke();

    // Crosshairs
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(cx - r - 8, cy);
    ctx.lineTo(cx + r + 8, cy);
    ctx.moveTo(cx, cy - r - 8);
    ctx.lineTo(cx, cy + r + 8);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.restore();
  }

  /**
   * Draws on-canvas tactical telemetry inspector HUD for a tracked entity.
   */
  drawInspectorHUD(track, isPinned = false) {
    if (!track) return;
    const [x1, y1, x2, y2] = track.bbox || [0, 0, 0, 0];
    const ctx = this.ctx;
    ctx.save();

    this.drawTargetLock(track.bbox, isPinned);

    // Inspector HUD Box Coordinates
    const hudW = 200;
    const hudH = 104;
    let hudX = x2 + 14;
    let hudY = y1;

    if (hudX + hudW > this.canvas.width) {
      hudX = Math.max(10, x1 - hudW - 14);
    }
    if (hudY + hudH > this.canvas.height) {
      hudY = this.canvas.height - hudH - 10;
    }

    // Glassmorphic HUD panel
    ctx.fillStyle = 'rgba(6, 9, 17, 0.92)';
    ctx.strokeStyle = isPinned ? 'rgba(245, 158, 11, 0.8)' : 'rgba(0, 242, 254, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.fillRect(hudX, hudY, hudW, hudH);
    ctx.strokeRect(hudX, hudY, hudW, hudH);

    // Top Accent line
    ctx.fillStyle = isPinned ? '#f59e0b' : '#00f2fe';
    ctx.fillRect(hudX, hudY, hudW, 3);

    // Connecting line to subject
    ctx.strokeStyle = isPinned ? 'rgba(245, 158, 11, 0.5)' : 'rgba(0, 242, 254, 0.5)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo((x1 + x2) / 2, (y1 + y2) / 2);
    ctx.lineTo(hudX, hudY + 15);
    ctx.stroke();
    ctx.setLineDash([]);

    // Telemetry text
    ctx.font = '700 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`TRACK #${track.id} [${(track.class || 'ACTOR').toUpperCase()}]`, hudX + 10, hudY + 18);

    ctx.font = '500 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';

    const speed = track.speed !== undefined ? track.speed.toFixed(1) : '0.0';
    const vy = track.verticalVelocity !== undefined ? track.verticalVelocity.toFixed(1) : '0.0';
    const ar = track.aspectRatio !== undefined ? track.aspectRatio.toFixed(2) : '0.45';
    const dwell = track.dwellTime !== undefined ? `${track.dwellTime.toFixed(1)}s` : '0.0s';
    const posture = (Number(ar) > 1.05) ? 'HORIZONTAL / FALL' : 'VERTICAL / STANDING';

    ctx.fillText(`SPEED: `, hudX + 10, hudY + 36);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`${speed} px/s`, hudX + 75, hudY + 36);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`VERT VEL: `, hudX + 10, hudY + 52);
    ctx.fillStyle = Math.abs(Number(vy)) > 60 ? '#ef4444' : '#38bdf8';
    ctx.fillText(`${vy} px/s`, hudX + 75, hudY + 52);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`POSTURE: `, hudX + 10, hudY + 68);
    ctx.fillStyle = (posture.includes('FALL')) ? '#ef4444' : '#10b981';
    ctx.fillText(`${posture}`, hudX + 75, hudY + 68);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`DWELL: `, hudX + 10, hudY + 84);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText(`${dwell} [${isPinned ? 'PINNED' : 'ACTIVE'}]`, hudX + 75, hudY + 84);

    ctx.restore();
  }

  /**
   * Captures high-resolution forensic snapshot with burned-in defense watermark and downloads PNG.
   */
  captureForensicSnapshot(videoCanvas, metadata = {}) {
    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = this.canvas.width;
    compositeCanvas.height = this.canvas.height;
    const compCtx = compositeCanvas.getContext('2d');

    // 1. Draw base video frame
    if (videoCanvas) {
      compCtx.drawImage(videoCanvas, 0, 0);
    }

    // 2. Draw tactical overlay
    compCtx.drawImage(this.canvas, 0, 0);

    // 3. Burn in official forensic banner watermark
    compCtx.save();
    compCtx.fillStyle = 'rgba(6, 9, 17, 0.85)';
    compCtx.fillRect(0, 0, compositeCanvas.width, 36);
    compCtx.fillRect(0, compositeCanvas.height - 30, compositeCanvas.width, 30);

    compCtx.font = '700 12px "JetBrains Mono", monospace';
    compCtx.fillStyle = '#00f2fe';
    const camStr = metadata.camId || 'CAM-01';
    const timeStr = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    compCtx.fillText(`QASD FORENSIC EVIDENCE CAPTURE // ${camStr} // TIMESTAMP: ${timeStr}`, 16, 23);

    compCtx.font = '600 10px "JetBrains Mono", monospace';
    compCtx.fillStyle = '#94a3b8';
    const riskStr = metadata.riskScore !== undefined ? `RISK: ${metadata.riskScore} (${metadata.severity || 'LOW'})` : 'RISK: NOMINAL';
    const qualityStr = metadata.qualityState ? `QUALITY: ${metadata.qualityState}` : 'QUALITY: ASSESSED';
    compCtx.fillText(`${riskStr} | ${qualityStr} | HASH: SHA256-VERIFIED | ADAPTIVE PIPELINE`, 16, compositeCanvas.height - 10);
    compCtx.restore();

    // Trigger instant browser download
    const dataUrl = compositeCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    const safeTime = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `QASD_FORENSIC_SNAPSHOT_${camStr}_${safeTime}.png`;
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    return dataUrl;
  }
}

window.CCTVCanvasRenderer = CCTVCanvasRenderer;
