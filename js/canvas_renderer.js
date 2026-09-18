/**
 * QASD - Canvas Renderer Module
 * High-performance 60 FPS HTML5 Canvas rendering for bounding boxes,
 * trajectory paths, restricted geofences, density heatmaps, and Grad-CAM saliency.
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

    // Draw active centroid pulse dot at head
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
      ctx.fillStyle = 'rgba(139, 92, 246, 0.12)';
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
      ctx.lineWidth = 1.8;
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
    ctx.fillText(`ZONE: ${zoneName.toUpperCase()} [${isBreached ? 'BREACHED!' : 'RESTRICTED'}]`, tagX, tagY);

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
  }
}

window.CCTVCanvasRenderer = CCTVCanvasRenderer;
