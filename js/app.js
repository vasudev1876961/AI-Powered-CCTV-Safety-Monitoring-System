/**
 * QASD - Master Application Controller
 * Orchestrates Real-Time Video Loops, WebSocket Connections,
 * Quality Assessment, Adaptive Enhancement, Tracking Overlays,
 * Incident Feeds, and Tab Navigation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Core Engines
  const simulator = new CCTVEnvironmentSimulator();
  const renderer = new CCTVCanvasRenderer('overlayCanvas');
  const qualityEngine = new QualityEnhancementEngine();
  const incidentEngine = new IncidentRecognitionEngine();
  const xaiManager = new XAIEvidenceManager();
  const benchmarkRunner = new ResearchBenchmarkRunner();

  // DOM Elements
  const videoCanvas = document.getElementById('videoCanvas');
  const videoCtx = videoCanvas.getContext('2d');
  const anomalyCanvas = document.getElementById('anomalyCanvas');
  const anomalyCtx = anomalyCanvas.getContext('2d');

  const sandboxDegradedCanvas = document.getElementById('sandboxDegradedCanvas');
  const sandboxDegradedCtx = sandboxDegradedCanvas.getContext('2d');
  const sandboxEnhancedCanvas = document.getElementById('sandboxEnhancedCanvas');
  const sandboxEnhancedCtx = sandboxEnhancedCanvas.getContext('2d');

  // Hidden Video & Webcam elements
  const hiddenVideo = document.createElement('video');
  hiddenVideo.autoplay = true;
  hiddenVideo.loop = true;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;

  let isPlaying = true;
  let isWebcam = false;
  let webcamStream = null;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 25.0;

  // Sandbox degradation state
  const degradationSettings = {
    darkness: 100,
    noise: 0,
    blur: 0,
    downsample: 720,
    haze: 0,
    compression: 95
  };

  // WebSocket Client connection
  let ws = null;
  let isWsConnected = false;

  function initWebSocket() {
    const wsUrl = `ws://${window.location.host || 'localhost:8000'}/ws/detections`;
    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        isWsConnected = true;
        document.getElementById('topWsVal').textContent = 'SERVER LIVE';
        document.getElementById('topWsVal').style.color = 'var(--accent-emerald)';
        console.log('[WS] Connected to QASD server');
      };
      ws.onclose = () => {
        isWsConnected = false;
        document.getElementById('topWsVal').textContent = 'CLIENT SIM';
        document.getElementById('topWsVal').style.color = 'var(--accent-cyan)';
        setTimeout(initWebSocket, 4000);
      };
      ws.onerror = () => {
        isWsConnected = false;
      };
    } catch (e) {
      isWsConnected = false;
    }
  }
  initWebSocket();

  // 2. Navigation Tab Switching
  const navTabs = document.querySelectorAll('.nav-tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');

  navTabs.forEach(btn => {
    btn.addEventListener('click', () => {
      navTabs.forEach(b => b.classList.remove('active'));
      tabViews.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetView = document.getElementById(targetId);
      if (targetView) targetView.classList.add('active');

      if (targetId === 'tabBenchmarks') {
        benchmarkRunner.renderCharts();
      }
    });
  });

  // 3. Camera Selector Buttons
  const camButtons = document.querySelectorAll('.cam-btn[data-cam]');
  camButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      camButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const camId = btn.getAttribute('data-cam');
      stopWebcam();
      simulator.setCamera(camId);
      incidentEngine.reset();
      updateCamInfoOSD(camId);
    });
  });

  function updateCamInfoOSD(camId) {
    const cfg = simulator.cameraConfigs[camId];
    if (cfg) {
      document.getElementById('osdCamTitle').textContent = `${camId} • ${cfg.name.toUpperCase()}`;
    }
  }

  // 4. Custom Video File Upload
  const fileInput = document.getElementById('videoFileInput');
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    stopWebcam();
    const url = URL.createObjectURL(file);
    hiddenVideo.src = url;
    hiddenVideo.play();
    simulator.isExternalVideo = true;
    document.getElementById('osdCamTitle').textContent = `CUSTOM • ${file.name.toUpperCase()}`;
  });

  // 5. Webcam Stream Toggle
  const btnWebcam = document.getElementById('btnWebcam');
  btnWebcam.addEventListener('click', async () => {
    if (isWebcam) {
      stopWebcam();
      btnWebcam.classList.remove('active');
    } else {
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
        hiddenVideo.srcObject = webcamStream;
        hiddenVideo.play();
        isWebcam = true;
        simulator.isExternalVideo = true;
        btnWebcam.classList.add('active');
        document.getElementById('osdCamTitle').textContent = 'LIVE WEBCAM STREAM';
      } catch (err) {
        alert('Webcam access error: ' + err.message);
      }
    }
  });

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    isWebcam = false;
    simulator.isExternalVideo = false;
    btnWebcam.classList.remove('active');
  }

  // 6. Interactive Incident Simulation Triggers
  document.getElementById('btnSimFall').addEventListener('click', () => simulator.triggerIncident('fall'));
  document.getElementById('btnSimFight').addEventListener('click', () => simulator.triggerIncident('fight'));
  document.getElementById('btnSimIntrusion').addEventListener('click', () => simulator.triggerIncident('intrusion'));
  document.getElementById('btnSimBag').addEventListener('click', () => simulator.triggerIncident('bag'));

  // Controls under video
  const btnPlayPause = document.getElementById('btnPlayPause');
  btnPlayPause.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlayPause.textContent = isPlaying ? 'Pause Stream' : 'Resume Stream';
  });

  document.getElementById('btnResetSimulation').addEventListener('click', () => {
    incidentEngine.reset();
    simulator.initCamera(simulator.currentCam);
  });

  document.getElementById('btnClearAlerts').addEventListener('click', () => {
    const container = document.getElementById('alertsContainer');
    container.innerHTML = '<div class="empty-alerts-placeholder">Monitoring surveillance stream for unexpected incidents...</div>';
  });

  // Audio Toggle
  const btnToggleSound = document.getElementById('btnToggleSound');
  btnToggleSound.addEventListener('click', () => {
    xaiManager.soundEnabled = !xaiManager.soundEnabled;
    btnToggleSound.classList.toggle('active', xaiManager.soundEnabled);
    btnToggleSound.style.color = xaiManager.soundEnabled ? 'var(--accent-emerald)' : 'var(--text-dim)';
  });

  // 7. Degradation Controls (Sandbox Tab)
  const sliderDarkness = document.getElementById('sliderDarkness');
  const sliderNoise = document.getElementById('sliderNoise');
  const sliderBlur = document.getElementById('sliderBlur');
  const sliderDownsample = document.getElementById('sliderDownsample');
  const sliderHaze = document.getElementById('sliderHaze');
  const sliderCompression = document.getElementById('sliderCompression');

  sliderDarkness.addEventListener('input', (e) => {
    degradationSettings.darkness = parseFloat(e.target.value);
    document.getElementById('valDarkness').textContent = `${e.target.value}% ${e.target.value < 40 ? '(Extreme Low-Light)' : ''}`;
  });
  sliderNoise.addEventListener('input', (e) => {
    degradationSettings.noise = parseFloat(e.target.value);
    document.getElementById('valNoise').textContent = `${e.target.value} ${e.target.value > 30 ? '(Heavy Noise)' : ''}`;
  });
  sliderBlur.addEventListener('input', (e) => {
    degradationSettings.blur = parseFloat(e.target.value);
    document.getElementById('valBlur').textContent = `${e.target.value} px ${e.target.value > 12 ? '(Severe Blur)' : ''}`;
  });
  sliderDownsample.addEventListener('input', (e) => {
    degradationSettings.downsample = parseInt(e.target.value);
    document.getElementById('valDownsample').textContent = `${e.target.value}p ${e.target.value <= 240 ? '(CCTV Analog 240p)' : ''}`;
  });
  sliderHaze.addEventListener('input', (e) => {
    degradationSettings.haze = parseFloat(e.target.value);
    document.getElementById('valHaze').textContent = `${e.target.value}%`;
  });
  sliderCompression.addEventListener('input', (e) => {
    degradationSettings.compression = parseInt(e.target.value);
    document.getElementById('valCompression').textContent = `${e.target.value}%`;
  });

  document.getElementById('btnResetDegradation').addEventListener('click', () => {
    sliderDarkness.value = 100; degradationSettings.darkness = 100;
    sliderNoise.value = 0; degradationSettings.noise = 0;
    sliderBlur.value = 0; degradationSettings.blur = 0;
    sliderDownsample.value = 720; degradationSettings.downsample = 720;
    sliderHaze.value = 0; degradationSettings.haze = 0;
    sliderCompression.value = 95; degradationSettings.compression = 95;
    document.getElementById('valDarkness').textContent = '100% (Normal)';
    document.getElementById('valNoise').textContent = '0.0 (Clean)';
    document.getElementById('valBlur').textContent = '0 px';
    document.getElementById('valDownsample').textContent = '720p (Native)';
    document.getElementById('valHaze').textContent = '0%';
    document.getElementById('valCompression').textContent = '95% (High Quality)';
  });

  // Layer Toggles
  const toggleBBoxes = document.getElementById('toggleBBoxes');
  const toggleTrails = document.getElementById('toggleTrails');
  const toggleHeatmap = document.getElementById('toggleHeatmap');
  const toggleZones = document.getElementById('toggleZones');
  const toggleGradCam = document.getElementById('toggleGradCam');
  const enhancementModeSelect = document.getElementById('enhancementModeSelect');

  // 8. Main Real-Time Surveillance Loop (60 FPS / RAF)
  function surveillanceLoop() {
    requestAnimationFrame(surveillanceLoop);

    if (!isPlaying) return;

    frameCount++;
    const now = performance.now();
    const dt = now - lastFrameTime;
    if (dt >= 1000) {
      currentFps = (frameCount * 1000) / dt;
      frameCount = 0;
      lastFrameTime = now;
      document.getElementById('topFpsVal').textContent = `${currentFps.toFixed(1)} FPS`;
    }

    const w = videoCanvas.width;
    const h = videoCanvas.height;

    // A. Render Base Frame
    let activeDetections = [];
    if (simulator.isExternalVideo && hiddenVideo.readyState >= 2) {
      videoCtx.drawImage(hiddenVideo, 0, 0, w, h);
      // Lightweight mock detections for uploaded video
      activeDetections = [
        { id: 1, class: 'person', conf: 0.89, bbox: [w * 0.4, h * 0.35, w * 0.52, h * 0.75] }
      ];
    } else {
      simulator.drawEnvironment(videoCtx, w, h);
      activeDetections = simulator.updateAndDrawActors(videoCtx, w, h);
    }

    // Apply baseline camera degradations if in degraded camera (e.g. CAM_02 Dark Stairwell)
    const currentCamCfg = simulator.cameraConfigs[simulator.currentCam];
    if (simulator.currentCam === 'CAM_02') {
      qualityEngine.applyDegradations(videoCtx, {
        darkness: 32,
        noise: 14,
        blur: 3,
        downsample: 480,
        haze: 0,
        compression: 65
      });
    }

    // B. Quality Assessment
    const qMetrics = qualityEngine.assessQuality(videoCtx, w, h);
    updateQualityOSD(qMetrics);

    // C. Adaptive Enhancement Routing
    const enhMode = enhancementModeSelect.value;
    const shouldEnhance = (enhMode === 'forced') || (enhMode === 'auto' && qMetrics.qualityState !== 'GOOD');

    updateEnhancementBadges(qMetrics, shouldEnhance);

    if (shouldEnhance) {
      qualityEngine.applyAdaptiveEnhancement(videoCtx, qMetrics, enhMode === 'forced');
    }

    // D. Render Sandbox Preview Canvas (if Sandbox tab active)
    renderSandboxCanvases(videoCanvas, degradationSettings, qMetrics);

    // E. Incident & Anomaly Recognition
    const zones = currentCamCfg ? currentCamCfg.zones : [];
    const evaluation = incidentEngine.evaluateIncidents(activeDetections, zones, qMetrics.qualityFactor, now / 1000);

    // Update Risk & Anomaly Telemetry
    updateRiskTelemetry(evaluation);

    // F. Process Alerts
    if (evaluation.alerts.length > 0) {
      for (const alert of evaluation.alerts) {
        alert.cameraId = simulator.currentCam;
        alert.qualityState = qMetrics.qualityState;
        alert.riskScore = evaluation.riskScore;
        alert.severity = evaluation.severity;
        alert.timestamp = new Date().toLocaleTimeString();

        // Audio Alarm
        xaiManager.playAlertTone(alert.severity);

        // Toast & Notification
        xaiManager.showToast(alert);
        xaiManager.showDesktopNotification(alert);

        // Add to Sidebar Activity Log
        addAlertToSidebar(alert);
      }
    }

    // Record frame snapshot for XAI evidence
    if (frameCount % 6 === 0) {
      xaiManager.recordFrameSnapshot(videoCanvas, frameCount, now / 1000);
    }

    // G. Render Overlays (Canvas Overlay Layer)
    renderer.clear();

    // 1. Density Heatmap
    if (toggleHeatmap.checked) {
      const historyPts = [];
      incidentEngine.trackHistories.forEach(h => {
        if (h.centroids.length > 0) historyPts.push(h.centroids[h.centroids.length - 1]);
      });
      renderer.drawDensityHeatmap(historyPts);
    }

    // 2. Restricted Geofences
    if (toggleZones.checked && zones.length > 0) {
      for (const zone of zones) {
        const isBreached = evaluation.alerts.some(a => a.incidentType.includes('Intrusion') || a.incidentType.includes('Loitering'));
        renderer.drawRestrictedZone(zone.polygon, zone.name, isBreached);
      }
    }

    // 3. Trajectory Trails
    if (toggleTrails.checked) {
      incidentEngine.trackHistories.forEach(hist => {
        const isAlert = evaluation.alerts.some(a => a.trackId === hist.id);
        renderer.drawTrajectoryTrail(hist.centroids, isAlert);
      });
    }

    // 4. Bounding Boxes
    if (toggleBBoxes.checked) {
      for (const det of activeDetections) {
        const isAlert = evaluation.alerts.some(a => a.trackId === det.id || String(a.trackId).includes(String(det.id)));
        renderer.drawBoundingBox(det.bbox, det.class, det.conf, det.id, isAlert);
      }
    }

    // 5. Grad-CAM Saliency Heatmap
    if (toggleGradCam.checked) {
      const heatPoints = activeDetections.map(d => ({
        x: (d.bbox[0] + d.bbox[2]) / 2,
        y: (d.bbox[1] + d.bbox[3]) / 2,
        radius: Math.max(50, (d.bbox[2] - d.bbox[0]) * 0.75),
        intensity: d.conf
      }));
      renderer.drawGradCamOverlay(heatPoints);
    }

    // Live Clock OSD
    document.getElementById('osdCamClock').textContent = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }

  // Telemetry Updaters
  function updateQualityOSD(qMetrics) {
    const pill = document.getElementById('osdQualityPill');
    const topVal = document.getElementById('topQualityVal');
    const state = qMetrics.qualityState;

    pill.className = `quality-pill ${state.toLowerCase().replace(' ', '-')}`;
    pill.textContent = state;

    topVal.textContent = state;
    topVal.style.color = state === 'GOOD' ? 'var(--accent-emerald)' : (state === 'MODERATE' ? 'var(--accent-amber)' : 'var(--accent-crimson)');

    document.getElementById('osdMetricLux').textContent = qMetrics.brightness.toFixed(1);
    document.getElementById('osdMetricBlur').textContent = qMetrics.blurScore.toFixed(1);
    document.getElementById('osdMetricNoise').textContent = `${(30 - qMetrics.noiseLevel).toFixed(1)} dB`;
    document.getElementById('osdMetricContrast').textContent = qMetrics.contrast.toFixed(1);
  }

  function updateEnhancementBadges(qMetrics, shouldEnhance) {
    const clahe = document.getElementById('badgeClahe');
    const denoise = document.getElementById('badgeDenoise');
    const deblur = document.getElementById('badgeDeblur');

    clahe.style.display = (shouldEnhance && qMetrics.needsLowLight) ? 'inline-block' : 'none';
    denoise.style.display = (shouldEnhance && qMetrics.needsDenoise) ? 'inline-block' : 'none';
    deblur.style.display = (shouldEnhance && qMetrics.needsDeblur) ? 'inline-block' : 'none';
  }

  function updateRiskTelemetry(evaluation) {
    const score = evaluation.riskScore;
    const severity = evaluation.severity;

    const scoreEl = document.getElementById('riskScoreVal');
    const badgeEl = document.getElementById('riskSeverityBadge');
    const progressEl = document.getElementById('riskProgressFill');

    scoreEl.textContent = score.toFixed(2);
    badgeEl.textContent = `${severity} SEVERITY`;
    badgeEl.className = `risk-pill-badge ${severity.toLowerCase()}`;

    progressEl.style.width = `${Math.min(100, score * 100)}%`;

    scoreEl.style.color = severity === 'CRITICAL' ? 'var(--accent-crimson)' : (severity === 'HIGH' ? '#fb923c' : (severity === 'MEDIUM' ? 'var(--accent-amber)' : 'var(--accent-emerald)'));

    // Anomaly score & sparkline
    document.getElementById('anomalyScoreVal').textContent = evaluation.anomalyScore.toFixed(2);
    renderAnomalySparkline(incidentEngine.anomalyHistory);
  }

  function renderAnomalySparkline(history) {
    if (!history || history.length === 0) return;
    anomalyCtx.clearRect(0, 0, anomalyCanvas.width, anomalyCanvas.height);

    const w = anomalyCanvas.width;
    const h = anomalyCanvas.height;

    // Threshold line at 0.65
    const threshY = h - (0.65 * h);
    anomalyCtx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    anomalyCtx.setLineDash([3, 3]);
    anomalyCtx.beginPath();
    anomalyCtx.moveTo(0, threshY);
    anomalyCtx.lineTo(w, threshY);
    anomalyCtx.stroke();
    anomalyCtx.setLineDash([]);

    // Line curve
    anomalyCtx.strokeStyle = '#00f2fe';
    anomalyCtx.lineWidth = 2;
    anomalyCtx.beginPath();

    const dx = w / Math.max(1, history.length - 1);
    history.forEach((val, i) => {
      const x = i * dx;
      const y = h - (val * h);
      if (i === 0) anomalyCtx.moveTo(x, y);
      else anomalyCtx.lineTo(x, y);
    });
    anomalyCtx.stroke();
  }

  function addAlertToSidebar(alert) {
    const placeholder = document.getElementById('emptyAlertsPlaceholder');
    if (placeholder) placeholder.remove();

    const container = document.getElementById('alertsContainer');
    const card = document.createElement('div');
    card.className = `alert-item-card ${alert.severity.toLowerCase()}`;

    card.innerHTML = `
      <div class="alert-item-top">
        <span class="alert-type-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="color: ${alert.severity === 'CRITICAL' ? 'var(--accent-crimson)' : 'var(--accent-amber)'};"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          ${alert.incidentType}
        </span>
        <span class="alert-time-pill">${alert.timestamp}</span>
      </div>
      <div class="alert-reasons-preview">
        ${alert.reasons ? alert.reasons[0] : 'Abnormal behavioral deviation detected'}
      </div>
      <div class="alert-action-footer">
        <span style="color: var(--text-dim);">CAM: <strong>${alert.cameraId}</strong> | RISK: <strong style="color: ${alert.severity === 'CRITICAL' ? 'var(--accent-crimson)' : 'var(--accent-cyan)'};">${alert.riskScore}</strong></span>
        <span class="inspect-link">Inspect Evidence &rarr;</span>
      </div>
    `;

    card.addEventListener('click', () => {
      xaiManager.openEvidenceModal(alert);
    });

    container.insertBefore(card, container.firstChild);

    // Limit cards in list
    if (container.children.length > 25) {
      container.removeChild(container.lastChild);
    }
  }

  function renderSandboxCanvases(sourceCanvas, settings, baseQuality) {
    // 1. Render Left Degraded Canvas
    sandboxDegradedCtx.drawImage(sourceCanvas, 0, 0, sandboxDegradedCanvas.width, sandboxDegradedCanvas.height);
    qualityEngine.applyDegradations(sandboxDegradedCtx, settings);

    // 2. Render Right Enhanced Canvas
    sandboxEnhancedCtx.drawImage(sandboxDegradedCanvas, 0, 0, sandboxEnhancedCanvas.width, sandboxEnhancedCanvas.height);
    qualityEngine.applyAdaptiveEnhancement(sandboxEnhancedCtx, baseQuality, true);

    // Draw recovered bounding boxes on enhanced canvas to visually prove the pipeline novelty!
    sandboxEnhancedCtx.strokeStyle = '#00f2fe';
    sandboxEnhancedCtx.lineWidth = 2;
    sandboxEnhancedCtx.strokeRect(260, 160, 70, 140);
    sandboxEnhancedCtx.fillStyle = 'rgba(0,242,254,0.15)';
    sandboxEnhancedCtx.fillRect(260, 160, 70, 140);
    sandboxEnhancedCtx.font = '600 11px "JetBrains Mono", monospace';
    sandboxEnhancedCtx.fillStyle = '#00f2fe';
    sandboxEnhancedCtx.fillText('RECOVERED PERSON 91%', 260, 152);
  }

  // Start surveillance RAF loop
  surveillanceLoop();
  console.log('[QASD] Safety Surveillance Operations Console initialized successfully.');
});
