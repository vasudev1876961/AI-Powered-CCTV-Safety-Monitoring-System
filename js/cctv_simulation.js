/**
 * QASD - CCTV Video Simulation & Camera Matrix Engine
 * Procedurally generates realistic surveillance environments and dynamic actors
 * with authentic physics, plus support for webcam and uploaded video files.
 */

class CCTVEnvironmentSimulator {
  constructor() {
    this.currentCam = 'CAM_01';
    this.actors = [];
    this.frameCounter = 0;
    this.externalVideo = null;
    this.isExternalVideo = false;

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

  initCamera(camId) {
    this.actors = [];
    const now = Date.now() / 1000;

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
      // Spawn two interacting persons close together
      this.actors = [
        { id: 51, class: 'person', x: 420, y: 310, w: 44, h: 110, vx: 1.8, vy: 0.5, state: 'fighting', conf: 0.92, fightPhase: 0 },
        { id: 52, class: 'person', x: 470, y: 312, w: 45, h: 108, vx: -1.6, vy: -0.4, state: 'fighting', conf: 0.89, fightPhase: Math.PI }
      ];
    } else if (type === 'intrusion') {
      const intruder = this.actors.find(a => a.class === 'person') || this.actors[0];
      intruder.x = 420;
      intruder.y = 300;
      intruder.state = 'intruding';
    } else if (type === 'bag') {
      // Person walks away leaving bag
      this.actors = [
        { id: 60, class: 'person', x: 680, y: 280, w: 45, h: 110, vx: 2.2, vy: -0.2, state: 'walking', conf: 0.92 },
        { id: 61, class: 'backpack', x: 260, y: 360, w: 32, h: 26, vx: 0, vy: 0, state: 'stationary', conf: 0.87 }
      ];
    }
  }

  /**
   * Renders background architectural scenery matching selected camera.
   */
  drawEnvironment(ctx, width, height) {
    this.frameCounter++;
    const cam = this.currentCam;

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

      // Chain-link fence representation
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
   * Updates actor physics and draws actors.
   */
  updateAndDrawActors(ctx, width, height) {
    const activeDetections = [];

    for (const a of this.actors) {
      // 1. Physics & Animation Updates
      if (a.state === 'walking' || a.state === 'approaching_zone') {
        a.x += a.vx;
        a.y += a.vy;

        // Bounce within boundaries
        if (a.x < 60 || a.x + a.w > width - 60) a.vx *= -1;
        if (a.y < height * 0.35 || a.y + a.h > height - 40) a.vy *= -1;

      } else if (a.state === 'falling') {
        a.fallProgress = (a.fallProgress || 0) + 0.08;
        // Rapid downward displacement and aspect ratio flip
        a.y += 4.5;
        if (a.fallProgress > 1.0) {
          a.state = 'collapsed';
          // Transformed to horizontal body
          const temp = a.w;
          a.w = 95;
          a.h = 36;
        }

      } else if (a.state === 'fighting') {
        a.fightPhase += 0.25;
        // Rapid erratic jitter
        a.x += Math.sin(a.fightPhase) * 3.5;
        a.y += Math.cos(a.fightPhase * 1.3) * 1.5;

      } else if (a.state === 'intruding') {
        // Move towards restricted zone center
        a.x += (480 - a.x) * 0.03;
        a.y += (320 - a.y) * 0.03;
      }

      // 2. Draw Actor Visual Representation on Screen
      ctx.save();
      if (a.class === 'person') {
        if (a.state === 'collapsed') {
          // Fallen horizontal body
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.roundRect(a.x, a.y, a.w, a.h, 8);
          ctx.fill();

          // Head on floor
          ctx.fillStyle = '#f8fafc';
          ctx.beginPath();
          ctx.arc(a.x + 14, a.y + a.h / 2, 10, 0, Math.PI * 2);
          ctx.fill();

        } else {
          // Standing pedestrian silhouette
          const headR = Math.min(14, a.w * 0.3);
          const headX = a.x + a.w / 2;
          const headY = a.y + headR + 2;

          // Head
          ctx.fillStyle = '#f8fafc';
          ctx.beginPath();
          ctx.arc(headX, headY, headR, 0, Math.PI * 2);
          ctx.fill();

          // Torso & Limbs
          ctx.fillStyle = a.state === 'fighting' ? '#ef4444' : '#00f2fe';
          ctx.beginPath();
          ctx.roundRect(a.x + 4, headY + headR, a.w - 8, a.h - (headR * 2 + 4), 6);
          ctx.fill();
        }
      } else {
        // Luggage / bag object
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.roundRect(a.x, a.y, a.w, a.h, 4);
        ctx.fill();
        // Handle
        ctx.strokeStyle = '#d97706';
        ctx.lineWidth = 2;
        ctx.strokeRect(a.x + a.w * 0.3, a.y - 4, a.w * 0.4, 4);
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
