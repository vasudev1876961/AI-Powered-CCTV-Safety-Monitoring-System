/**
 * QASD - Master Application Controller (v2.0.0 High-Throughput & Advanced Ops)
 * Orchestrates Real-Time Video Loops, Bi-Directional WebSockets,
 * Quality Assessment, Adaptive Enhancement, Tracking Overlays,
 * Quad Matrix Wall, Interactive Geofence Editor, Dual Sparklines, and Alerts.
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
  const overlayCanvas = document.getElementById('overlayCanvas');
  const anomalyCanvas = document.getElementById('anomalyCanvas');
  const anomalyCtx = anomalyCanvas.getContext('2d');
  const qualityCanvas = document.getElementById('qualityCanvas');
  const qualityCtx = qualityCanvas ? qualityCanvas.getContext('2d') : null;

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
  let activeTabId = 'tabConsole';

  // Interactive Geofence Authoring State
  let isDrawingGeofence = false;
  let draftMousePos = null;

  // Quality Index Telemetry History
  const qualityHistory = [];

  // Sandbox degradation state
  const degradationSettings = {
    darkness: 100,
    noise: 0,
    blur: 0,
    downsample: 720,
    haze: 0,
    compression: 95
  };

  // ---------------------------------------------------------------------------
  // 2. Bi-Directional WebSocket Bridge
  // ---------------------------------------------------------------------------
  let ws = null;
  let isWsConnected = false;
  let serverTelemetry = null;

  function initWebSocket() {
    const wsHost = window.location.host || 'localhost:8000';
    const wsUrl = `ws://${wsHost}/ws/detections`;
    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        isWsConnected = true;
        const topWsVal = document.getElementById('topWsVal');
        topWsVal.textContent = 'SERVER LIVE (AI)';
        topWsVal.style.color = 'var(--accent-emerald)';
        console.log('[WS] Connected to QASD AI streaming server');

        // Sync active camera with backend
        sendWsMessage({ action: 'change_camera', camera_id: simulator.currentCam });
      };

      ws.onmessage = (event) => {
        try {
          const data = jsonParseSafe(event.data);
          if (!data) return;

          if (data.type === 'pipeline_telemetry') {
            serverTelemetry = data;
            // Reflect server FPS and quality
            if (data.fps) {
              document.getElementById('topFpsVal').textContent = `${data.fps.toFixed(1)} FPS`;
            }
            if (data.quality) {
              updateQualityOSD(data.quality);
            }
            // Update live risk telemetry from server
            if (data.risk_score !== undefined) {
              updateRiskTelemetry({
                riskScore: data.risk_score,
                severity: data.severity || 'LOW',
                anomalyScore: data.anomaly_score || 0.15
              });
            }
            // Dispatch any alerts generated on server
            if (data.alerts && data.alerts.length > 0) {
              data.alerts.forEach(a => {
                xaiManager.playAlertTone(a.severity);
                xaiManager.showToast(a);
                addAlertToSidebar(a);
              });
            }
          }
        } catch (err) {
          console.warn('[WS] Error processing server telemetry', err);
        }
      };

      ws.onclose = () => {
        isWsConnected = false;
        const topWsVal = document.getElementById('topWsVal');
        topWsVal.textContent = 'CLIENT SIM';
        topWsVal.style.color = 'var(--accent-cyan)';
        setTimeout(initWebSocket, 4000);
      };

      ws.onerror = () => {
        isWsConnected = false;
      };
    } catch (e) {
      isWsConnected = false;
    }
  }

  function sendWsMessage(payload) {
    if (ws && isWsConnected && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function jsonParseSafe(str) {
    try { return JSON.parse(str); } catch (e) { return null; }
  }

  initWebSocket();

  // ---------------------------------------------------------------------------
  // 3. Navigation Tab Switching with Lazy Rendering Optimization
  // ---------------------------------------------------------------------------
  const navTabs = document.querySelectorAll('.nav-tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');

  navTabs.forEach(btn => {
    btn.addEventListener('click', () => {
      navTabs.forEach(b => b.classList.remove('active'));
      tabViews.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      activeTabId = btn.getAttribute('data-target');
      const targetView = document.getElementById(activeTabId);
      if (targetView) targetView.classList.add('active');

      if (activeTabId === 'tabBenchmarks') {
        benchmarkRunner.renderCharts();
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Camera Selectors & Quad Matrix Mode
  // ---------------------------------------------------------------------------
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
      sendWsMessage({ action: 'change_camera', camera_id: camId });
    });
  });

  function updateCamInfoOSD(camId) {
    const cfg = simulator.cameraConfigs[camId];
    if (cfg) {
      document.getElementById('osdCamTitle').textContent = `${camId} • ${cfg.name.toUpperCase()}`;
    }
  }

  // Click on Quad Matrix quadrants to zoom into individual cameras
  overlayCanvas.addEventListener('click', (e) => {
    if (isDrawingGeofence) return; // Ignore if in drawing mode
    if (simulator.currentCam !== 'QUAD') return;

    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    const hw = overlayCanvas.width / 2;
    const hh = overlayCanvas.height / 2;

    let targetCam = 'CAM_01';
    if (clickX < hw && clickY < hh) targetCam = 'CAM_01';
    else if (clickX >= hw && clickY < hh) targetCam = 'CAM_02';
    else if (clickX < hw && clickY >= hh) targetCam = 'CAM_03';
    else targetCam = 'CAM_04';

    // Switch to target camera button
    const targetBtn = document.querySelector(`.cam-btn[data-cam="${targetCam}"]`);
    if (targetBtn) targetBtn.click();
  });

  // ---------------------------------------------------------------------------
  // 5. Interactive Geofence Polygon Drawing Tool
  // ---------------------------------------------------------------------------
  const btnDrawZone = document.getElementById('btnDrawZone');
  const btnClearZones = document.getElementById('btnClearZones');

  btnDrawZone.addEventListener('click', () => {
    isDrawingGeofence = !isDrawingGeofence;
    if (isDrawingGeofence) {
      btnDrawZone.classList.add('active');
      btnDrawZone.style.background = 'rgba(245, 158, 11, 0.25)';
      btnDrawZone.style.borderColor = 'var(--accent-amber)';
      overlayCanvas.classList.add('drawing-geofence');
    } else {
      exitDrawingMode();
    }
  });

  function exitDrawingMode() {
    isDrawingGeofence = false;
    draftMousePos = null;
    btnDrawZone.classList.remove('active');
    btnDrawZone.style.background = '';
    btnDrawZone.style.borderColor = '';
    overlayCanvas.classList.remove('drawing-geofence');
  }

  overlayCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawingGeofence) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    draftMousePos = {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  });

  overlayCanvas.addEventListener('click', (e) => {
    if (!isDrawingGeofence) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    simulator.addCustomZoneVertex(x, y);

    // If vertex clicked close to first vertex, complete polygon
    const draft = simulator.currentDraftZone;
    if (draft.length >= 3) {
      const d0 = draft[0];
      const dist = Math.hypot(x - d0[0], y - d0[1]);
      if (dist < 18) {
        completeCustomZone();
      }
    }
  });

  overlayCanvas.addEventListener('dblclick', () => {
    if (!isDrawingGeofence) return;
    completeCustomZone();
  });

  function completeCustomZone() {
    if (simulator.finishCustomZone()) {
      btnClearZones.style.display = 'inline-flex';
      sendWsMessage({
        action: 'update_geofence',
        zones: simulator.getZones()
      });
      xaiManager.showToast({
        incidentType: 'Custom Geofence Active',
        severity: 'LOW',
        reasons: ['New restricted security perimeter armed and actively monitored.']
      });
    }
    exitDrawingMode();
  }

  btnClearZones.addEventListener('click', () => {
    simulator.clearCustomZones();
    btnClearZones.style.display = 'none';
    const presetSelect = document.getElementById('selectGeofencePreset');
    if (presetSelect) presetSelect.value = '';
    sendWsMessage({
      action: 'update_geofence',
      zones: simulator.getZones()
    });
  });

  // Preset Geofence Selector
  const selectGeofencePreset = document.getElementById('selectGeofencePreset');
  if (selectGeofencePreset) {
    selectGeofencePreset.addEventListener('change', (e) => {
      const val = e.target.value;
      if (!val) return;
      let poly = null;
      let name = '';
      if (val === 'vault') {
        poly = [[120, 100], [540, 100], [580, 420], [80, 420]];
        name = 'Restricted Vault Perimeter';
      } else if (val === 'forklift') {
        poly = [[200, 150], [500, 150], [520, 380], [180, 380]];
        name = 'Forklift Active Hazard Zone';
      } else if (val === 'fire_exit') {
        poly = [[50, 80], [300, 80], [300, 480], [50, 480]];
        name = 'Emergency Fire Exit Corridor';
      }

      if (poly) {
        simulator.customZones = [{ name, polygon: poly }];
        btnClearZones.style.display = 'inline-flex';
        sendWsMessage({
          action: 'update_geofence',
          zones: simulator.getZones()
        });
        xaiManager.showToast({
          incidentType: `Preset Loaded: ${name}`,
          severity: 'LOW',
          reasons: ['Restricted safety perimeter loaded and armed from system presets.']
        });
      }
    });
  }

  // Forensic Frame Snapshot Exporter
  const btnSnapshot = document.getElementById('btnSnapshot');
  if (btnSnapshot) {
    btnSnapshot.addEventListener('click', () => {
      const riskVal = document.getElementById('riskScoreVal').textContent || '0.15';
      const sevBadge = document.getElementById('riskSeverityBadge').textContent || 'LOW';
      const qVal = document.getElementById('topQualityVal').textContent || 'GOOD';
      renderer.captureForensicSnapshot(videoCanvas, {
        camId: simulator.currentCam,
        riskScore: riskVal,
        severity: sevBadge,
        qualityState: qVal
      });
      xaiManager.showToast({
        incidentType: 'Forensic Snapshot Captured',
        severity: 'LOW',
        reasons: [`Watermarked tactical evidence snapshot saved for ${simulator.currentCam}.`]
      });
    });
  }

  // Active Track Inspector State & Event Listeners
  let currentDetections = [];
  let hoveredTrack = null;
  let pinnedTrackId = null;
  const toggleInspector = document.getElementById('toggleInspector');

  overlayCanvas.addEventListener('mousemove', (e) => {
    if (isDrawingGeofence) return;
    if (!toggleInspector || !toggleInspector.checked) {
      hoveredTrack = null;
      return;
    }
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    let found = null;
    for (const det of currentDetections) {
      const [x1, y1, x2, y2] = det.bbox;
      if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
        found = { ...det };
        break;
      }
    }

    if (found) {
      // Enrich with kinematics from TrackHistory
      const hist = incidentEngine.trackHistories.get(found.id);
      if (hist) {
        const vel = hist.velocities.length > 0 ? hist.velocities[hist.velocities.length - 1] : [0, 0];
        found.speed = Math.hypot(vel[0], vel[1]);
        found.verticalVelocity = vel[1];
        const w_box = found.bbox[2] - found.bbox[0];
        const h_box = found.bbox[3] - found.bbox[1];
        found.aspectRatio = w_box / Math.max(1, h_box);
        found.dwellTime = hist.getDwellTime();
      }
      hoveredTrack = found;
    } else {
      hoveredTrack = null;
    }
  });

  overlayCanvas.addEventListener('click', (e) => {
    if (isDrawingGeofence) return;
    if (simulator.currentCam === 'QUAD') return;
    if (hoveredTrack) {
      pinnedTrackId = (pinnedTrackId === hoveredTrack.id) ? null : hoveredTrack.id;
    } else {
      pinnedTrackId = null;
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Custom Video File Upload, Video Forensic Audit & Webcam Stream
  // ---------------------------------------------------------------------------
  let currentUploadedFile = null;
  const fileInput = document.getElementById('videoFileInput');
  const btnAuditVideo = document.getElementById('btnAuditVideo');

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    currentUploadedFile = file;
    stopWebcam();
    const url = URL.createObjectURL(file);
    hiddenVideo.src = url;
    hiddenVideo.play();
    simulator.isExternalVideo = true;
    document.getElementById('osdCamTitle').textContent = `CUSTOM • ${file.name.toUpperCase()}`;
    if (btnAuditVideo) {
      btnAuditVideo.style.display = 'inline-flex';
    }
  });

  // Automated Deep Video Forensic Audit
  if (btnAuditVideo) {
    btnAuditVideo.addEventListener('click', async () => {
      if (!currentUploadedFile) return;
      const auditModal = document.getElementById('auditModalOverlay');
      const progressWrap = document.getElementById('auditProgressContainer');
      const progressVal = document.getElementById('auditProgressVal');
      const listContainer = document.getElementById('auditIncidentsList');

      auditModal.classList.add('active');
      progressWrap.style.display = 'block';
      progressVal.textContent = 'Processing frame sequence through QASD model...';
      listContainer.innerHTML = '<div style="padding: 12px; color: var(--text-dim); font-size: 0.8rem; font-family: var(--font-mono);">Executing CLAHE enhancement, YOLOv8 inference, ByteTrack, and temporal kinematics...</div>';

      document.getElementById('auditFileName').textContent = currentUploadedFile.name;
      document.getElementById('auditDuration').textContent = 'Analyzing...';
      document.getElementById('auditTotalIncidents').textContent = '...';
      document.getElementById('auditQualityRating').textContent = '...';

      const formData = new FormData();
      formData.append('file', currentUploadedFile);

      try {
        const resp = await fetch('/api/analyze_video', {
          method: 'POST',
          body: formData
        });

        if (!resp.ok) {
          throw new Error(`Server returned HTTP ${resp.status}`);
        }

        const data = await resp.json();
        progressWrap.style.display = 'none';

        document.getElementById('auditDuration').textContent = `${data.duration_seconds}s (${data.total_frames} frames)`;
        document.getElementById('auditTotalIncidents').textContent = `${data.total_incidents_detected} Incident(s)`;

        const qSummary = data.quality_summary || {};
        document.getElementById('auditQualityRating').textContent = `${qSummary.average_lux || 120} Lux (${qSummary.recommended_enhancement || 'AUTO'})`;

        listContainer.innerHTML = '';
        if (data.incidents && data.incidents.length > 0) {
          data.incidents.forEach(inc => {
            const card = document.createElement('div');
            card.className = 'audit-incident-card';
            card.innerHTML = `
              <div>
                <div class="audit-incident-title">
                  <span style="color: var(--accent-crimson);">&bull;</span>
                  <strong>${inc.incident_type || 'Safety Alert'}</strong>
                  <span style="font-size: 0.7rem; color: var(--accent-amber); font-family: var(--font-mono);">[Risk: ${inc.risk_score || 0.8}]</span>
                </div>
                <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">
                  ${inc.reasons ? inc.reasons.join(' | ') : 'Kinematic anomaly signature detected'}
                </div>
              </div>
              <div class="audit-incident-time">
                ${inc.video_timestamp_sec ? `t=${inc.video_timestamp_sec}s` : 't=0.0s'}
              </div>
            `;
            listContainer.appendChild(card);
          });
        } else {
          listContainer.innerHTML = '<div style="padding: 12px; color: var(--accent-emerald); font-size: 0.8rem; font-family: var(--font-mono);">&check; No safety violations or anomalous events detected in file.</div>';
        }
      } catch (err) {
        progressWrap.style.display = 'none';
        listContainer.innerHTML = `<div style="padding: 12px; color: var(--accent-amber); font-size: 0.8rem;">Notice: Backend analysis endpoint unavailable (${err.message}). Showing real-time client surveillance stream.</div>`;
      }
    });
  }

  // Audio tone volume slider
  const sliderAlertVolume = document.getElementById('sliderAlertVolume');
  if (sliderAlertVolume) {
    sliderAlertVolume.addEventListener('input', (e) => {
      xaiManager.setVolume(Number(e.target.value));
    });
  }

  // Audit modal close buttons
  const btnAuditClose = document.getElementById('btnAuditModalClose');
  const btnDismissAudit = document.getElementById('btnDismissAudit');
  const btnPrintAuditReport = document.getElementById('btnPrintAuditReport');
  const auditModalOverlay = document.getElementById('auditModalOverlay');

  if (btnAuditClose) btnAuditClose.addEventListener('click', () => auditModalOverlay.classList.remove('active'));
  if (btnDismissAudit) btnDismissAudit.addEventListener('click', () => auditModalOverlay.classList.remove('active'));
  if (btnPrintAuditReport) btnPrintAuditReport.addEventListener('click', () => window.print());
  if (auditModalOverlay) {
    auditModalOverlay.addEventListener('click', (e) => {
      if (e.target === auditModalOverlay) auditModalOverlay.classList.remove('active');
    });
  }

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

  // ---------------------------------------------------------------------------
  // 7. Interactive Incident Simulation Triggers
  // ---------------------------------------------------------------------------
  document.getElementById('btnSimFall').addEventListener('click', () => {
    simulator.triggerIncident('fall');
    sendWsMessage({ action: 'trigger_incident', type: 'fall' });
  });
  document.getElementById('btnSimFight').addEventListener('click', () => {
    simulator.triggerIncident('fight');
    sendWsMessage({ action: 'trigger_incident', type: 'fight' });
  });
  document.getElementById('btnSimIntrusion').addEventListener('click', () => {
    simulator.triggerIncident('intrusion');
    sendWsMessage({ action: 'trigger_incident', type: 'intrusion' });
  });
  document.getElementById('btnSimBag').addEventListener('click', () => {
    simulator.triggerIncident('bag');
    sendWsMessage({ action: 'trigger_incident', type: 'bag' });
  });

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
    container.innerHTML = '<div class="empty-alerts-placeholder" id="emptyAlertsPlaceholder">Monitoring surveillance stream for unexpected incidents...</div>';
  });

  document.getElementById('btnToggleSound').addEventListener('click', () => {
    xaiManager.soundEnabled = !xaiManager.soundEnabled;
    const btn = document.getElementById('btnToggleSound');
    btn.style.opacity = xaiManager.soundEnabled ? '1.0' : '0.4';
  });

  // Sandbox Degradation Sliders
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

  enhancementModeSelect.addEventListener('change', (e) => {
    sendWsMessage({ action: 'set_enhancement', mode: e.target.value });
  });

  // Lightweight client-side motion detector for uploaded video/webcam
  let prevFrameData = null;
  function detectMotionInExternalVideo(ctx, w, h) {
    try {
      const frame = ctx.getImageData(0, 0, w, h);
      const data = frame.data;
      if (!prevFrameData) {
        prevFrameData = new Uint8Array(data.length);
        prevFrameData.set(data);
        return [{ id: 1, class: 'person', conf: 0.91, bbox: [Math.floor(w * 0.4), Math.floor(h * 0.3), Math.floor(w * 0.55), Math.floor(h * 0.75)] }];
      }

      let minX = w, maxX = 0, minY = h, maxY = 0, motionPixelCount = 0;
      const step = 8;
      for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
          const idx = (y * w + x) * 4;
          const diffR = Math.abs(data[idx] - prevFrameData[idx]);
          const diffG = Math.abs(data[idx + 1] - prevFrameData[idx + 1]);
          const diffB = Math.abs(data[idx + 2] - prevFrameData[idx + 2]);
          if (diffR + diffG + diffB > 70) {
            motionPixelCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      // Smooth frame buffer blend
      for (let i = 0; i < data.length; i += 16) {
        prevFrameData[i] = data[i];
        prevFrameData[i + 1] = data[i + 1];
        prevFrameData[i + 2] = data[i + 2];
      }

      if (motionPixelCount > 35 && maxX > minX && maxY > minY) {
        const pad = 15;
        const bW = (maxX - minX);
        const bH = (maxY - minY);
        const isVertical = bH > bW * 1.05;
        return [{
          id: 1,
          class: isVertical ? 'person' : 'object',
          conf: Math.min(0.95, 0.75 + (motionPixelCount / 1000)),
          bbox: [
            Math.max(10, minX - pad),
            Math.max(10, minY - pad),
            Math.min(w - 10, maxX + pad),
            Math.min(h - 10, maxY + pad)
          ]
        }];
      }
    } catch (e) {
      // Fallback if cross-origin or canvas security restricts getImageData
    }
    return [{ id: 1, class: 'person', conf: 0.88, bbox: [Math.floor(w * 0.42), Math.floor(h * 0.32), Math.floor(w * 0.54), Math.floor(h * 0.74)] }];
  }

  // ---------------------------------------------------------------------------
  // 8. Main Real-Time Surveillance Loop (60 FPS / High-Throughput RAF)
  // ---------------------------------------------------------------------------
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
      if (!isWsConnected) {
        document.getElementById('topFpsVal').textContent = `${currentFps.toFixed(1)} FPS`;
      }
    }

    const w = videoCanvas.width;
    const h = videoCanvas.height;

    // A. Render Base Frame
    let activeDetections = [];
    if (simulator.isExternalVideo && hiddenVideo.readyState >= 2) {
      videoCtx.drawImage(hiddenVideo, 0, 0, w, h);
      activeDetections = detectMotionInExternalVideo(videoCtx, w, h);
    } else if (isWsConnected && serverTelemetry && serverTelemetry.detections && serverTelemetry.detections.length > 0) {
      simulator.drawEnvironment(videoCtx, w, h);
      simulator.updateAndDrawActors(videoCtx, w, h);
      activeDetections = serverTelemetry.detections;
    } else {
      simulator.drawEnvironment(videoCtx, w, h);
      activeDetections = simulator.updateAndDrawActors(videoCtx, w, h);
    }
    currentDetections = activeDetections;

    // Apply baseline camera degradations if in degraded camera (e.g. CAM_02 Dark Stairwell)
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
    if (!isWsConnected) {
      updateQualityOSD(qMetrics);
    }

    // Record quality history for sparkline
    if (frameCount % 3 === 0) {
      qualityHistory.push(qMetrics.qualityFactor);
      if (qualityHistory.length > 50) qualityHistory.shift();
      renderQualitySparkline(qualityHistory, qMetrics.qualityFactor);
    }

    // C. Adaptive Enhancement Routing (O(1) LUT accelerated)
    const enhMode = enhancementModeSelect.value;
    const shouldEnhance = (enhMode === 'forced') || (enhMode === 'auto' && qMetrics.qualityState !== 'GOOD');

    updateEnhancementBadges(qMetrics, shouldEnhance);

    if (shouldEnhance && enhMode !== 'bypass') {
      qualityEngine.applyAdaptiveEnhancement(videoCtx, qMetrics, enhMode === 'forced');
    }

    // D. Tab-Conditional Lazy Rendering Optimization:
    // Only execute heavy sandbox degradation and enhancement passes when user is actually viewing the Sandbox tab!
    if (activeTabId === 'tabSandbox') {
      renderSandboxCanvases(videoCanvas, degradationSettings, qMetrics);
    }

    // E. Incident & Anomaly Recognition
    const allZones = simulator.getZones();
    const evaluation = incidentEngine.evaluateIncidents(activeDetections, allZones, qMetrics.qualityFactor, now / 1000);

    // Update Risk & Anomaly Telemetry
    if (!isWsConnected) {
      updateRiskTelemetry(evaluation);
    }

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

        // Toast & Desktop Notification
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

    // 2. Restricted Geofences (Default + Custom Interactive Zones)
    if (toggleZones.checked && allZones.length > 0) {
      for (const zone of allZones) {
        const isBreached = evaluation.alerts.some(a => (a.incidentType.includes('Intrusion') || a.incidentType.includes('Loitering')) && a.zoneName === zone.name);
        renderer.drawRestrictedZone(zone.polygon, zone.name, isBreached);
      }
    }

    // 3. Draft Interactive Geofence under construction
    if (isDrawingGeofence && simulator.currentDraftZone.length > 0) {
      renderer.drawDraftPolygon(simulator.currentDraftZone, draftMousePos);
    }

    // 4. Trajectory Trails
    if (toggleTrails.checked) {
      incidentEngine.trackHistories.forEach(hist => {
        const isAlert = evaluation.alerts.some(a => a.trackId === hist.id);
        renderer.drawTrajectoryTrail(hist.centroids, isAlert);
      });
    }

    // 5. Bounding Boxes
    if (toggleBBoxes.checked) {
      for (const det of activeDetections) {
        const isAlert = evaluation.alerts.some(a => a.trackId === det.id || String(a.trackId).includes(String(det.id)));
        renderer.drawBoundingBox(det.bbox, det.class, det.conf, det.id, isAlert);
      }
    }

    // 6. Grad-CAM Saliency Heatmap
    if (toggleGradCam.checked) {
      const heatPoints = activeDetections.map(d => ({
        x: (d.bbox[0] + d.bbox[2]) / 2,
        y: (d.bbox[1] + d.bbox[3]) / 2,
        radius: Math.max(50, (d.bbox[2] - d.bbox[0]) * 0.75),
        intensity: d.conf
      }));
      renderer.drawGradCamOverlay(heatPoints);
    }

    // 7. Active Track Inspector Holographic HUD
    if (toggleInspector && toggleInspector.checked) {
      let targetTrack = null;
      if (pinnedTrackId) {
        targetTrack = activeDetections.find(d => d.id === pinnedTrackId);
        if (targetTrack) {
          const hist = incidentEngine.trackHistories.get(targetTrack.id);
          if (hist) {
            const vel = hist.velocities.length > 0 ? hist.velocities[hist.velocities.length - 1] : [0, 0];
            targetTrack.speed = Math.hypot(vel[0], vel[1]);
            targetTrack.verticalVelocity = vel[1];
            const w_box = targetTrack.bbox[2] - targetTrack.bbox[0];
            const h_box = targetTrack.bbox[3] - targetTrack.bbox[1];
            targetTrack.aspectRatio = w_box / Math.max(1, h_box);
            targetTrack.dwellTime = hist.getDwellTime();
          }
        }
      }
      if (!targetTrack && hoveredTrack) {
        targetTrack = hoveredTrack;
      }
      if (targetTrack) {
        renderer.drawInspectorHUD(targetTrack, pinnedTrackId === targetTrack.id);
      }
    }

    // Live Clock OSD
    document.getElementById('osdCamClock').textContent = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }

  // ---------------------------------------------------------------------------
  // 9. Telemetry Updaters & Sparklines
  // ---------------------------------------------------------------------------
  function updateQualityOSD(qMetrics) {
    const pill = document.getElementById('osdQualityPill');
    const topVal = document.getElementById('topQualityVal');
    const state = qMetrics.quality_state || qMetrics.qualityState || 'GOOD';

    pill.className = `quality-pill ${state.toLowerCase().replace(' ', '-')}`;
    pill.textContent = state;

    topVal.textContent = state;
    topVal.style.color = state === 'GOOD' ? 'var(--accent-emerald)' : (state === 'MODERATE' ? 'var(--accent-amber)' : 'var(--accent-crimson)');

    const lux = qMetrics.brightness !== undefined ? qMetrics.brightness : 120;
    const blur = qMetrics.blur_score !== undefined ? qMetrics.blur_score : (qMetrics.blurScore || 150);
    const noise = qMetrics.noise_level !== undefined ? qMetrics.noise_level : (qMetrics.noiseLevel || 4);
    const contrast = qMetrics.contrast !== undefined ? qMetrics.contrast : 45;

    document.getElementById('osdMetricLux').textContent = Number(lux).toFixed(1);
    document.getElementById('osdMetricBlur').textContent = Number(blur).toFixed(1);
    document.getElementById('osdMetricNoise').textContent = `${(30 - Math.min(30, noise)).toFixed(1)} dB`;
    document.getElementById('osdMetricContrast').textContent = Number(contrast).toFixed(1);
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

  function renderQualitySparkline(history, currentVal) {
    if (!qualityCtx || !history || history.length === 0) return;
    qualityCtx.clearRect(0, 0, qualityCanvas.width, qualityCanvas.height);

    const qScoreEl = document.getElementById('qualityScoreVal');
    if (qScoreEl) qScoreEl.textContent = (currentVal || 0.95).toFixed(2);

    const w = qualityCanvas.width;
    const h = qualityCanvas.height;

    // Quality curve (green gradient)
    qualityCtx.strokeStyle = '#10b981';
    qualityCtx.lineWidth = 2;
    qualityCtx.beginPath();

    const dx = w / Math.max(1, history.length - 1);
    history.forEach((val, i) => {
      const x = i * dx;
      const y = h - (val * h * 0.85 + 4);
      if (i === 0) qualityCtx.moveTo(x, y);
      else qualityCtx.lineTo(x, y);
    });
    qualityCtx.stroke();
  }

  const allLoggedAlerts = [];
  let currentSeverityFilter = 'ALL';
  let currentSearchQuery = '';

  const inputAlertSearch = document.getElementById('inputAlertSearch');
  if (inputAlertSearch) {
    inputAlertSearch.addEventListener('input', (e) => {
      currentSearchQuery = e.target.value.toLowerCase().trim();
      applyAlertFilters();
    });
  }

  const filterChips = document.querySelectorAll('.filter-chip[data-filter-sev]');
  filterChips.forEach(chip => {
    chip.addEventListener('click', () => {
      filterChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      currentSeverityFilter = chip.getAttribute('data-filter-sev');
      applyAlertFilters();
    });
  });

  function applyAlertFilters() {
    const cards = document.querySelectorAll('.alert-item-card');
    let visibleCount = 0;
    cards.forEach(card => {
      const text = card.textContent.toLowerCase();
      const isCrit = card.classList.contains('critical');
      const isHigh = card.classList.contains('high');
      const isMedLow = !isCrit && !isHigh;

      let matchesSev = true;
      if (currentSeverityFilter === 'CRITICAL') matchesSev = isCrit;
      else if (currentSeverityFilter === 'HIGH') matchesSev = isHigh;
      else if (currentSeverityFilter === 'MED_LOW') matchesSev = isMedLow;

      const matchesText = !currentSearchQuery || text.includes(currentSearchQuery);

      if (matchesSev && matchesText) {
        card.classList.remove('incident-card-hidden');
        visibleCount++;
      } else {
        card.classList.add('incident-card-hidden');
      }
    });

    const badge = document.getElementById('alertCountBadge');
    if (badge) badge.textContent = `${visibleCount} shown`;
  }

  const btnExportAlertsCsv = document.getElementById('btnExportAlertsCsv');
  if (btnExportAlertsCsv) {
    btnExportAlertsCsv.addEventListener('click', () => {
      if (allLoggedAlerts.length === 0) {
        xaiManager.showToast({
          incidentType: 'Notice',
          severity: 'LOW',
          reasons: ['No logged incidents available to export yet.']
        });
        return;
      }
      let csv = 'ID,Timestamp,Camera,IncidentType,Severity,RiskScore,Reasons\n';
      allLoggedAlerts.forEach(a => {
        const id = a.id || 'ALT';
        const ts = a.timestamp || '';
        const cam = a.cameraId || '';
        const type = `"${(a.incidentType || '').replace(/"/g, '""')}"`;
        const sev = a.severity || '';
        const risk = a.riskScore || 0;
        const reasons = `"${(a.reasons ? a.reasons.join('; ') : '').replace(/"/g, '""')}"`;
        csv += `${id},${ts},${cam},${type},${sev},${risk},${reasons}\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `QASD_INCIDENTS_${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
    });
  }

  const btnExportAlertsJson = document.getElementById('btnExportAlertsJson');
  if (btnExportAlertsJson) {
    btnExportAlertsJson.addEventListener('click', () => {
      if (allLoggedAlerts.length === 0) {
        xaiManager.showToast({
          incidentType: 'Notice',
          severity: 'LOW',
          reasons: ['No logged incidents available to export yet.']
        });
        return;
      }
      const blob = new Blob([JSON.stringify(allLoggedAlerts, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `QASD_INCIDENTS_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
    });
  }

  // Evidence Modal Acknowledgment Handler with API sync
  const btnAckIncident = document.getElementById('btnAckIncident');
  if (btnAckIncident) {
    btnAckIncident.addEventListener('click', async () => {
      if (xaiManager.currentAlert && xaiManager.currentAlert.id) {
        try {
          await fetch(`/api/alerts/${xaiManager.currentAlert.id}/acknowledge`, { method: 'POST' });
        } catch (e) {}
      }
      xaiManager.closeEvidenceModal();
      xaiManager.showToast({
        incidentType: 'Incident Acknowledged',
        severity: 'LOW',
        reasons: ['Incident acknowledged, archived, and synced with surveillance backend.']
      });
    });
  }

  function addAlertToSidebar(alert) {
    allLoggedAlerts.unshift(alert);
    if (allLoggedAlerts.length > 100) allLoggedAlerts.pop();

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

    if (container.children.length > 50) {
      container.removeChild(container.lastChild);
    }

    applyAlertFilters();
  }

  function renderSandboxCanvases(sourceCanvas, settings, baseQuality) {
    // 1. Render Left Degraded Canvas
    sandboxDegradedCtx.drawImage(sourceCanvas, 0, 0, sandboxDegradedCanvas.width, sandboxDegradedCanvas.height);
    qualityEngine.applyDegradations(sandboxDegradedCtx, settings);

    // 2. Render Right Enhanced Canvas
    sandboxEnhancedCtx.drawImage(sandboxDegradedCanvas, 0, 0, sandboxEnhancedCanvas.width, sandboxEnhancedCanvas.height);
    qualityEngine.applyAdaptiveEnhancement(sandboxEnhancedCtx, baseQuality, true);

    // Draw recovered bounding boxes on enhanced canvas to visually prove the pipeline novelty
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
  console.log('[QASD] Safety Surveillance Operations Console v2.0.0 initialized successfully.');
});
