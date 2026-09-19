/**
 * QASD - CCTV Video Simulation & Camera Matrix Engine
 * Procedurally generates realistic surveillance environments and dynamic actors
 * with authentic physics, plus support for webcam, uploaded video files,
 * and 4-Camera Synchronized Quad Matrix Wall (2x2 View).
 */

class CCTVEnvironmentSimulator {
  constructor() {
    this.currentCam = 'CAM_01';
    this.actors = [];
    this.frameCounter = 0;
    this.externalVideo = null;
    this.isExternalVideo = false;

    // Custom Geofence state
    this.customZones = [];
    this.currentDraftZone = [];

    this.cameraConfigs = {
      CAM_01: {
        name: 'Main Entrance Corridor',
        lux: 145,
        defaultQuality: 'GOOD',
        zones: [],
      },
      CAM_02: {
        name: 'Dark Stairwell & Basement',
        lux: 26,
        defaultQuality: 'DEGRADED',
        zones: [],
      },
      CAM_03: {
        name: 'Perimeter Security Zone',
        lux: 62,
        defaultQuality: 'MODERATE',
        zones: [
          {
            name: 'Restricted Vault Perimeter',
            polygon: [[180, 140], [780, 140], [840, 480], [120, 480]],
          }
        ],
      },
      CAM_04: {
        name: 'Industrial Warehouse Floor',
        lux: 88,
        defaultQuality: 'MODERATE',
        zones: [
          {
            name: 'Forklift Active Hazard Zone',
            polygon: [[280, 180], [680, 180], [720, 440], [240, 440]],
          }
        ],
      },
      QUAD: {
        name: 'Quad Matrix Multi-Camera Wall (2x2)',
        lux: 80,
        defaultQuality: 'MULTI',
        zones: [],
      }
    };

    this.initCamera(this.currentCam);
  }

  setCamera(camId) {
    if (this.cameraConfigs[camId]) {
      this.currentCam = camId;
      this.isExternalVideo = false;
      this.initCamera(camId);
    }
  }

  getZones() {
    const baseZones = (this.cameraConfigs[this.currentCam] && this.cameraConfigs[this.currentCam].zones) ? [...this.cameraConfigs[this.currentCam].zones] : [];
    return [...baseZones, ...this.customZones];
  }

  addCustomZoneVertex(x, y) {
    this.currentDraftZone.push([Math.round(x), Math.round(y)]);
  }

  finishCustomZone(name = "Custom Restricted Geofence") {
    if (this.currentDraftZone.length >= 3) {
      this.customZones.push({
        name: `${name} #${this.customZones.length + 1}`,
        polygon: [...this.currentDraftZone],
        isCustom: true
      });
      this.currentDraftZone = [];
      return true;
    }
    this.currentDraftZone = [];
    return false;
  }

  clearCustomZones() {
    this.customZones = [];
    this.currentDraftZone = [];
  }

  initCamera(camId) {
    this.actors = [];

    if (camId === 'CAM_01') {
      // Pedestrian corridor flow
      this.actors = [
        { id: 10, class: 'person', x: 260, y: 260, w: 42, h: 105, vx: 1.2, vy: 0.3, state: 'walking', conf: 0.94 },
        { id: 14, class: 'person', x: 620, y: 310, w: 45, h: 112, vx: -1.0, vy: -0.2, state: 'walking', conf: 0.91 },
        { id: 22, class: 'bag', x: 270, y: 350, w: 28, h: 22, vx: 0, vy: 0, state: 'stationary', conf: 0.86 }
      ];
    } else if (camId === 'CAM_02') {
      // Dark basement stairwell
      this.actors = [
        { id: 17, class: 'person', x: 440, y: 220, w: 44, h: 108, vx: 0.6, vy: 0.8, state: 'walking', conf: 0.82 }
      ];
    } else if (camId === 'CAM_03') {
      // Perimeter fence
      this.actors = [
        { id: 31, class: 'person', x: 80, y: 320, w: 46, h: 110, vx: 1.5, vy: 0.1, state: 'approaching_zone', conf: 0.88 }
      ];
    } else if (camId === 'CAM_04') {
      // Industrial warehouse
      this.actors = [
        { id: 41, class: 'person', x: 320, y: 300, w: 48, h: 115, vx: 0.8, vy: 0.4, state: 'walking', conf: 0.92 },
        { id: 45, class: 'suitcase', x: 460, y: 340, w: 34, h: 26, vx: 0, vy: 0, state: 'stationary', conf: 0.89 }
      ];
    } else if (camId === 'QUAD') {
      // Synchronized actors across all 4 quadrants (960x540 split into four 480x270 boxes)
      this.actors = [
        // Q1: CAM_01 (0,0 to 480,270)
        { id: 101, class: 'person', x: 140, y: 130, w: 24, h: 58, vx: 0.8, vy: 0.2, state: 'walking', conf: 0.93, quad: 1 },
        { id: 102, class: 'person', x: 310, y: 140, w: 24, h: 60, vx: -0.7, vy: -0.1, state: 'walking', conf: 0.90, quad: 1 },
        // Q2: CAM_02 (480,0 to 960,270)
        { id: 201, class: 'person', x: 680, y: 110, w: 24, h: 58, vx: 0.4, vy: 0.5, state: 'walking', conf: 0.81, quad: 2 },
        // Q3: CAM_03 (0,270 to 480,540)
        { id: 301, class: 'person', x: 90, y: 400, w: 25, h: 60, vx: 0.9, vy: 0.1, state: 'approaching_zone', conf: 0.87, quad: 3 },
        // Q4: CAM_04 (480,270 to 960,540)
        { id: 401, class: 'person', x: 640, y: 390, w: 26, h: 62, vx: 0.6, vy: 0.3, state: 'walking', conf: 0.91, quad: 4 },
        { id: 402, class: 'suitcase', x: 740, y: 430, w: 20, h: 16, vx: 0, vy: 0, state: 'stationary', conf: 0.88, quad: 4 },
      ];
    }
  }

  /**
   * Triggers explicit incident actions for interactive demonstrations.
   */
  triggerIncident(type) {
    if (this.actors.length === 0) return;

    if (type === 'fall') {
      const person = this.actors.find(a => a.class === 'person') || this.actors[0];
      person.state = 'falling';
      person.fallProgress = 0;
    } else if (type === 'fight') {
      if (this.currentCam === 'QUAD') {
        const p1 = this.actors.find(a => a.quad === 1);
        if (p1) {
          p1.state = 'fighting';
          p1.fightPhase = 0;
        }
      } else {
        this.actors = [
          { id: 51, class: 'person', x: 420, y: 310, w: 44, h: 110, vx: 1.8, vy: 0.5, state: 'fighting', conf: 0.92, fightPhase: 0 },
          { id: 52, class: 'person', x: 470, y: 312, w: 45, h: 108, vx: -1.6, vy: -0.4, state: 'fighting', conf: 0.89, fightPhase: Math.PI }
        ];
      }
    } else if (type === 'intrusion') {
      const intruder = this.actors.find(a => a.class === 'person') || this.actors[0];
      if (this.currentCam === 'QUAD') {
        intruder.x = 220;
        intruder.y = 390;
      } else {
        intruder.x = 420;
        intruder.y = 300;
      }
      intruder.state = 'intruding';
    } else if (type === 'bag') {
      if (this.currentCam !== 'QUAD') {
        this.actors = [
          { id: 60, class: 'person', x: 680, y: 280, w: 45, h: 110, vx: 2.2, vy: -0.2, state: 'walking', conf: 0.92 },
          { id: 61, class: 'backpack', x: 260, y: 360, w: 32, h: 26, vx: 0, vy: 0, state: 'stationary', conf: 0.87 }
        ];
      }
    }
  }

  /**
   * Renders background architectural scenery matching selected camera.
   */
  drawEnvironment(ctx, width, height) {
    this.frameCounter++;
    const cam = this.currentCam;

    if (cam === 'QUAD') {
      this.drawQuadEnvironment(ctx, width, height);
      return;
    }

    if (cam === 'CAM_01') {
      // Main Entrance Corridor
      ctx.fillStyle = '#101626';
      ctx.fillRect(0, 0, width, height);

      // Floor tiles perspective
      ctx.fillStyle = '#0a0e1a';
      ctx.beginPath();
      ctx.moveTo(0, height * 0.52);
      ctx.lineTo(width, height * 0.52);
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.fill();

      // Ceiling lights lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(width * 0.2, 0); ctx.lineTo(width * 0.35, height * 0.52);
      ctx.moveTo(width * 0.8, 0); ctx.lineTo(width * 0.65, height * 0.52);
      ctx.stroke();

      // Doorway frame at back
      ctx.fillStyle = '#070a12';
      ctx.fillRect(width * 0.42, height * 0.28, width * 0.16, height * 0.24);

    } else if (cam === 'CAM_02') {
      // Dark Stairwell & Basement
      ctx.fillStyle = '#05070d';
      ctx.fillRect(0, 0, width, height);

      // Concrete steps
      ctx.fillStyle = '#090d18';
      const steps = 7;
      for (let s = 0; s < steps; s++) {
        const sy = height * 0.45 + (s * (height * 0.55 / steps));
        ctx.fillRect(0, sy, width, (height * 0.55 / steps) - 2);
      }

      // Flickering dim emergency bulb
      const flicker = 0.65 + Math.sin(this.frameCounter * 0.3) * 0.15 + (Math.random() < 0.05 ? -0.3 : 0);
      const lampGrad = ctx.createRadialGradient(width * 0.5, 60, 0, width * 0.5, 60, 240);
      lampGrad.addColorStop(0, `rgba(255, 230, 160, ${flicker * 0.4})`);
      lampGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = lampGrad;
      ctx.beginPath();
      ctx.arc(width * 0.5, 60, 240, 0, Math.PI * 2);
      ctx.fill();

    } else if (cam === 'CAM_03') {
      // Perimeter Security Zone (Night)
      ctx.fillStyle = '#03050a';
      ctx.fillRect(0, 0, width, height);

      // Night ground
      ctx.fillStyle = '#070a14';
      ctx.fillRect(0, height * 0.38, width, height * 0.62);

      // Chain-link fence
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, height * 0.38);
        ctx.lineTo(x + 20, height * 0.95);
        ctx.stroke();
      }

      // Tower floodlight beam
      const floodGrad = ctx.createRadialGradient(width * 0.75, 40, 10, width * 0.55, height * 0.65, 320);
      floodGrad.addColorStop(0, 'rgba(200, 240, 255, 0.35)');
      floodGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = floodGrad;
      ctx.beginPath();
      ctx.moveTo(width * 0.75, 40);
      ctx.lineTo(width * 0.3, height);
      ctx.lineTo(width * 0.9, height);
      ctx.closePath();
      ctx.fill();

    } else if (cam === 'CAM_04') {
      // Industrial Warehouse
      ctx.fillStyle = '#080c16';
      ctx.fillRect(0, 0, width, height);

      // Storage racks
      ctx.fillStyle = '#141d2f';
      ctx.fillRect(40, 60, 140, height * 0.8);
      ctx.fillRect(width - 180, 60, 140, height * 0.8);

      // Hazard warning floor striping
      ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
      for (let x = 200; x < width - 200; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, height * 0.88);
        ctx.lineTo(x + 25, height * 0.88);
        ctx.lineTo(x - 15, height);
        ctx.lineTo(x - 40, height);
        ctx.fill();
      }
    }
  }

  /**
   * Renders the 4-camera Quad Matrix view with borders and individual camera scenery.
   */
  drawQuadEnvironment(ctx, width, height) {
    const hw = width / 2;
    const hh = height / 2;

    // Q1: CAM-01 Entrance (Top-Left)
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, hw, hh);
    ctx.clip();
    ctx.fillStyle = '#101626';
    ctx.fillRect(0, 0, hw, hh);
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, hh * 0.55, hw, hh * 0.45);
    ctx.restore();

    // Q2: CAM-02 Dark Stairwell (Top-Right)
    ctx.save();
    ctx.beginPath();
    ctx.rect(hw, 0, hw, hh);
    ctx.clip();
    ctx.fillStyle = '#05070d';
    ctx.fillRect(hw, 0, hw, hh);
    ctx.fillStyle = '#090d18';
    for (let s = 0; s < 5; s++) {
      ctx.fillRect(hw, hh * 0.45 + s * (hh * 0.55 / 5), hw, (hh * 0.55 / 5) - 2);
    }
    ctx.restore();

    // Q3: CAM-03 Perimeter Fence (Bottom-Left)
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, hh, hw, hh);
    ctx.clip();
    ctx.fillStyle = '#03050a';
    ctx.fillRect(0, hh, hw, hh);
    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, hh + hh * 0.4, hw, hh * 0.6);
    ctx.restore();

    // Q4: CAM-04 Warehouse (Bottom-Right)
    ctx.save();
    ctx.beginPath();
    ctx.rect(hw, hh, hw, hh);
    ctx.clip();
    ctx.fillStyle = '#080c16';
    ctx.fillRect(hw, hh, hw, hh);
    ctx.fillStyle = '#141d2f';
    ctx.fillRect(hw + 20, hh + 30, 60, hh * 0.7);
    ctx.fillRect(width - 80, hh + 30, 60, hh * 0.7);
    ctx.restore();

    // Quad Grid Partition Dividers
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(hw, 0); ctx.lineTo(hw, height);
    ctx.moveTo(0, hh); ctx.lineTo(width, hh);
    ctx.stroke();

    // Mini Quadrant Headings
    ctx.font = '700 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00f2fe';
    ctx.fillText('CAM-01 [ENTRANCE - GOOD]', 12, 22);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('CAM-02 [STAIRWELL - DEGRADED]', hw + 12, 22);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('CAM-03 [PERIMETER - MODERATE]', 12, hh + 22);
    ctx.fillStyle = '#10b981';
    ctx.fillText('CAM-04 [WAREHOUSE - MODERATE]', hw + 12, hh + 22);
  }

  /**
   * Updates actor physics and draws actors.
   */
  updateAndDrawActors(ctx, width, height) {
    const activeDetections = [];
    const isQuad = this.currentCam === 'QUAD';
    const hw = width / 2;
    const hh = height / 2;

    for (const a of this.actors) {
      // 1. Physics & Animation Updates
      if (a.state === 'walking' || a.state === 'approaching_zone') {
        a.x += a.vx;
        a.y += a.vy;

        if (isQuad) {
          // Keep actor confined to its quadrant
          const minX = (a.quad === 2 || a.quad === 4) ? hw + 20 : 20;
          const maxX = (a.quad === 2 || a.quad === 4) ? width - 30 : hw - 30;
          const minY = (a.quad === 3 || a.quad === 4) ? hh + 30 : 30;
          const maxY = (a.quad === 3 || a.quad === 4) ? height - 30 : hh - 30;

          if (a.x < minX || a.x + a.w > maxX) a.vx *= -1;
          if (a.y < minY || a.y + a.h > maxY) a.vy *= -1;
        } else {
          // Normal camera bounds
          if (a.x < 60 || a.x + a.w > width - 60) a.vx *= -1;
          if (a.y < height * 0.35 || a.y + a.h > height - 40) a.vy *= -1;
        }

      } else if (a.state === 'falling') {
        a.fallProgress = (a.fallProgress || 0) + 0.08;
        a.y += isQuad ? 2.5 : 4.5;
        if (a.fallProgress > 1.0) {
          a.state = 'collapsed';
          a.w = isQuad ? 52 : 95;
          a.h = isQuad ? 20 : 36;
        }

      } else if (a.state === 'fighting') {
        a.fightPhase += 0.25;
        a.x += Math.sin(a.fightPhase) * 3.5;
        a.y += Math.cos(a.fightPhase * 1.3) * 1.5;

      } else if (a.state === 'intruding') {
        const targetX = isQuad ? 240 : 480;
        const targetY = isQuad ? 400 : 320;
        a.x += (targetX - a.x) * 0.03;
        a.y += (targetY - a.y) * 0.03;
      }

      // 2. Draw Actor Visual Representation on Screen
      ctx.save();
      if (a.class === 'person') {
        if (a.state === 'collapsed') {
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.roundRect(a.x, a.y, a.w, a.h, 6);
          ctx.fill();

          ctx.fillStyle = '#f8fafc';
          ctx.beginPath();
          ctx.arc(a.x + (isQuad ? 8 : 14), a.y + a.h / 2, isQuad ? 6 : 10, 0, Math.PI * 2);
          ctx.fill();

        } else {
          const headR = Math.min(isQuad ? 8 : 14, a.w * 0.3);
          const headX = a.x + a.w / 2;
          const headY = a.y + headR + 2;

          ctx.fillStyle = '#f8fafc';
          ctx.beginPath();
          ctx.arc(headX, headY, headR, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = a.state === 'fighting' ? '#ef4444' : '#00f2fe';
          ctx.beginPath();
          ctx.roundRect(a.x + 3, headY + headR, a.w - 6, a.h - (headR * 2 + 4), isQuad ? 4 : 6);
          ctx.fill();
        }
      } else {
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.roundRect(a.x, a.y, a.w, a.h, 3);
        ctx.fill();
        ctx.strokeStyle = '#d97706';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(a.x + a.w * 0.3, a.y - 3, a.w * 0.4, 3);
      }
      ctx.restore();

      // 3. Register Detection for pipeline
      activeDetections.push({
        id: a.id,
        class: a.class,
        conf: a.conf,
        bbox: [Math.floor(a.x), Math.floor(a.y), Math.floor(a.x + a.w), Math.floor(a.y + a.h)]
      });
    }

    return activeDetections;
  }
}

window.CCTVEnvironmentSimulator = CCTVEnvironmentSimulator;
