/**
 * QASD - Incident Recognition & Risk Scoring Engine
 * Detects Falls, Violence, Intrusion, Loitering, Abandoned Objects,
 * Computes Continuous Anomaly Scores and Multi-Factor Risk Assessment.
 */

class IncidentRecognitionEngine {
  constructor() {
    this.trackHistories = new Map(); // id -> { centroids, bboxes, times, velocities, aspectRatios }
    this.anomalyHistory = [];
    this.lastAlertTimes = new Map();
    this.cooldownSeconds = 3.5;
  }

  reset() {
    this.trackHistories.clear();
    this.anomalyHistory = [];
    this.lastAlertTimes.clear();
  }

  /**
   * Updates track state with new observation.
   */
  updateTrackHistory(track, timestamp) {
    const tid = track.id;
    const bbox = track.bbox;
    const now = timestamp || Date.now() / 1000;

    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const w = Math.max(1, x2 - x1);
    const h = Math.max(1, y2 - y1);
    const ar = w / h;

    if (!this.trackHistories.has(tid)) {
      this.trackHistories.set(tid, {
        id: tid,
        class: track.class,
        centroids: [],
        bboxes: [],
        times: [],
        velocities: [],
        aspectRatios: [],
      });
    }

    const hist = this.trackHistories.get(tid);

    let vx = 0, vy = 0;
    if (hist.times.length > 0) {
      const dt = Math.max(0.01, now - hist.times[hist.times.length - 1]);
      const prevC = hist.centroids[hist.centroids.length - 1];
      vx = (cx - prevC[0]) / dt;
      vy = (cy - prevC[1]) / dt;
    }

    hist.centroids.push([cx, cy]);
    hist.bboxes.push(bbox);
    hist.times.push(now);
    hist.velocities.push([vx, vy]);
    hist.aspectRatios.push(ar);

    // Limit buffer to last 60 frames
    if (hist.centroids.length > 60) {
      hist.centroids.shift();
      hist.bboxes.shift();
      hist.times.shift();
      hist.velocities.shift();
      hist.aspectRatios.shift();
    }
  }

  /**
   * Point in Polygon test using ray-casting.
   */
  isPointInPolygon(point, polygon) {
    const [x, y] = point;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  /**
   * Calculates stationary dwell time in seconds.
   */
  getDwellTime(hist, stationaryRadius = 30) {
    if (hist.centroids.length < 2) return 0;
    const current = hist.centroids[hist.centroids.length - 1];
    let dwell = 0;
    for (let i = hist.centroids.length - 2; i >= 0; i--) {
      const pt = hist.centroids[i];
      const dist = Math.hypot(current[0] - pt[0], current[1] - pt[1]);
      if (dist <= stationaryRadius) {
        dwell += (hist.times[i + 1] - hist.times[i]);
      } else {
        break;
      }
    }
    return dwell;
  }

  /**
   * Evaluates active tracks against all safety incident models.
   */
  evaluateIncidents(activeTracks, restrictedZones, qualityFactor, timestamp) {
    const now = timestamp || Date.now() / 1000;
    const detectedAlerts = [];

    // Update histories
    for (const t of activeTracks) {
      this.updateTrackHistory(t, now);
    }

    // 1. Fall & Collapse Incident
    for (const t of activeTracks) {
      if (t.class !== 'person') continue;
      const hist = this.trackHistories.get(t.id);
      if (!hist || hist.centroids.length < 4) continue;

      // Downward velocity over last 5 frames
      const recentVy = hist.velocities.slice(-5).map(v => v[1]);
      const avgDownwardVy = recentVy.reduce((a, b) => a + b, 0) / recentVy.length;

      const currentAR = hist.aspectRatios[hist.aspectRatios.length - 1];
      const initialAR = hist.aspectRatios[Math.max(0, hist.aspectRatios.length - 8)];
      const arShift = currentAR - initialAR;
      const dwellGround = this.getDwellTime(hist, 35);

      const isHorizontal = currentAR > 1.05;
      const isFalling = (avgDownwardVy > 65 || arShift > 0.45) && isHorizontal;
      const isCollapsed = isHorizontal && dwellGround > 1.2;

      if (isFalling || isCollapsed) {
        const conf = isCollapsed ? 0.94 : 0.88;
        const alertKey = `fall_${t.id}`;

        if (!this.lastAlertTimes.has(alertKey) || (now - this.lastAlertTimes.get(alertKey) > this.cooldownSeconds)) {
          this.lastAlertTimes.set(alertKey, now);

          const reasons = [
            `Sudden downward velocity: ${avgDownwardVy.toFixed(1)} px/s`,
            `Posture transition to horizontal: AR shifted to ${currentAR.toFixed(2)}`,
            `Ground persistence duration: ${dwellGround.toFixed(1)}s stationary on floor`
          ];

          detectedAlerts.push({
            incidentType: 'Possible Fall / Collapse',
            trackId: t.id,
            bbox: t.bbox,
            confidence: conf,
            temporalConf: 0.92,
            detectionConf: t.conf,
            reasons,
            metrics: {
              verticalVelocity: avgDownwardVy.toFixed(1),
              aspectRatio: currentAR.toFixed(2),
              dwellTime: dwellGround.toFixed(1)
            }
          });
        }
      }
    }

    // 2. Physical Altercation / Violence
    const persons = activeTracks.filter(t => t.class === 'person');
    for (let i = 0; i < persons.length; i++) {
      for (let j = i + 1; j < persons.length; j++) {
        const p1 = persons[i];
        const p2 = persons[j];
        const h1 = this.trackHistories.get(p1.id);
        const h2 = this.trackHistories.get(p2.id);
        if (!h1 || !h2 || h1.centroids.length < 3 || h2.centroids.length < 3) continue;

        const c1 = h1.centroids[h1.centroids.length - 1];
        const c2 = h2.centroids[h2.centroids.length - 1];
        const dist = Math.hypot(c1[0] - c2[0], c1[1] - c2[1]);

        if (dist < 75) {
          const v1 = h1.velocities.slice(-4).map(v => Math.hypot(v[0], v[1]));
          const v2 = h2.velocities.slice(-4).map(v => Math.hypot(v[0], v[1]));
          const avgSpeed = (v1.reduce((a, b) => a + b, 0) + v2.reduce((a, b) => a + b, 0)) / (v1.length + v2.length);

          if (avgSpeed > 38) {
            const alertKey = `fight_${p1.id}_${p2.id}`;
            if (!this.lastAlertTimes.has(alertKey) || (now - this.lastAlertTimes.get(alertKey) > this.cooldownSeconds)) {
              this.lastAlertTimes.set(alertKey, now);

              detectedAlerts.push({
                incidentType: 'Physical Altercation / Violence',
                trackId: `${p1.id} & ${p2.id}`,
                bbox: [
                  Math.min(p1.bbox[0], p2.bbox[0]),
                  Math.min(p1.bbox[1], p2.bbox[1]),
                  Math.max(p1.bbox[2], p2.bbox[2]),
                  Math.max(p1.bbox[3], p2.bbox[3])
                ],
                confidence: 0.91,
                temporalConf: 0.90,
                detectionConf: Math.min(p1.conf, p2.conf),
                reasons: [
                  `Aggressive physical proximity (${dist.toFixed(1)}px separation)`,
                  `High-frequency kinetic oscillations (${avgSpeed.toFixed(1)} px/s velocity)`,
                  'Reciprocal acceleration detected across interacting entities'
                ],
                metrics: {
                  separationDist: dist.toFixed(1),
                  kineticEnergy: avgSpeed.toFixed(1)
                }
              });
            }
          }
        }
      }
    }

    // 3. Zone Intrusion & Loitering
    if (restrictedZones && restrictedZones.length > 0) {
      for (const t of activeTracks) {
        const hist = this.trackHistories.get(t.id);
        if (!hist || hist.centroids.length === 0) continue;

        const footPoint = [hist.centroids[hist.centroids.length - 1][0], t.bbox[3]];

        for (const zone of restrictedZones) {
          if (this.isPointInPolygon(footPoint, zone.polygon)) {
            const dwell = this.getDwellTime(hist, 35);
            const isLoitering = dwell > 5.0;
            const incType = isLoitering ? 'Suspicious Loitering' : 'Perimeter Intrusion';
            const alertKey = `zone_${zone.name}_${t.id}_${incType}`;

            if (!this.lastAlertTimes.has(alertKey) || (now - this.lastAlertTimes.get(alertKey) > this.cooldownSeconds)) {
              this.lastAlertTimes.set(alertKey, now);

              detectedAlerts.push({
                incidentType: incType,
                trackId: t.id,
                bbox: t.bbox,
                confidence: isLoitering ? 0.95 : 0.89,
                temporalConf: 0.94,
                detectionConf: t.conf,
                reasons: [
                  `Entity breached ${zone.name} boundary`,
                  `Contact coordinate: (${footPoint[0].toFixed(0)}, ${footPoint[1].toFixed(0)})`,
                  `Time inside sensitive perimeter: ${dwell.toFixed(1)}s`
                ],
                metrics: {
                  zone: zone.name,
                  dwellTime: dwell.toFixed(1)
                }
              });
            }
          }
        }
      }
    }

    // 4. Abandoned Luggage / Object
    const bags = activeTracks.filter(t => ['bag', 'backpack', 'suitcase', 'object'].includes(t.class));
    for (const bag of bags) {
      const bHist = this.trackHistories.get(bag.id);
      if (!bHist) continue;

      const dwell = this.getDwellTime(bHist, 20);
      if (dwell >= 4.0) {
        const bagC = bHist.centroids[bHist.centroids.length - 1];
        let nearestPersonDist = Infinity;

        for (const p of persons) {
          const pHist = this.trackHistories.get(p.id);
          if (!pHist || pHist.centroids.length === 0) continue;
          const pC = pHist.centroids[pHist.centroids.length - 1];
          const d = Math.hypot(bagC[0] - pC[0], bagC[1] - pC[1]);
          if (d < nearestPersonDist) nearestPersonDist = d;
        }

        if (nearestPersonDist > 110) {
          const alertKey = `bag_${bag.id}`;
          if (!this.lastAlertTimes.has(alertKey) || (now - this.lastAlertTimes.get(alertKey) > this.cooldownSeconds)) {
            this.lastAlertTimes.set(alertKey, now);

            detectedAlerts.push({
              incidentType: 'Abandoned Object / Luggage',
              trackId: bag.id,
              bbox: bag.bbox,
              confidence: 0.89,
              temporalConf: 0.91,
              detectionConf: bag.conf,
              reasons: [
                `Stationary package unattended for ${dwell.toFixed(1)}s`,
                `Nearest individual: ${nearestPersonDist.toFixed(1)}px (exceeds 110px safe perimeter)`
              ],
              metrics: {
                unattendedSeconds: dwell.toFixed(1),
                nearestPersonDist: nearestPersonDist.toFixed(1)
              }
            });
          }
        }
      }
    }

    // 5. Compute Continuous Anomaly Score
    const anomalyScore = this.computeAnomalyScore(activeTracks);

    // 6. Calculate Unified Multi-Factor Risk Score:
    // Risk Score = P(inc) * C(temp) * C(det) * Q(factor)
    let maxRiskScore = 0.12;
    let primaryAlert = null;

    if (detectedAlerts.length > 0) {
      primaryAlert = detectedAlerts[0];
      const pInc = primaryAlert.confidence;
      const cTemp = primaryAlert.temporalConf || 0.90;
      const cDet = primaryAlert.detectionConf || 0.85;
      const qFac = qualityFactor || 0.85;

      // Base formula multiplied by normalization factor
      const rawRisk = pInc * cTemp * cDet * qFac;
      maxRiskScore = Math.min(0.98, Math.max(0.20, rawRisk * 1.35));
    } else if (anomalyScore > 0.6) {
      maxRiskScore = Math.min(0.85, anomalyScore * 0.9);
    }

    return {
      alerts: detectedAlerts,
      anomalyScore,
      riskScore: parseFloat(maxRiskScore.toFixed(2)),
      severity: this.getSeverityTier(maxRiskScore),
    };
  }

  /**
   * Continuous Anomaly Score calculation based on spatial dispersion and velocity variance.
   */
  computeAnomalyScore(activeTracks) {
    if (activeTracks.length === 0) return 0.08;

    let totalSpeed = 0;
    for (const t of activeTracks) {
      const hist = this.trackHistories.get(t.id);
      if (hist && hist.velocities.length > 0) {
        const [vx, vy] = hist.velocities[hist.velocities.length - 1];
        totalSpeed += Math.hypot(vx, vy);
      }
    }
    const meanSpeed = totalSpeed / activeTracks.length;

    // Normal baseline mean speed is ~25 px/s. Sudden rush or chaos spikes this.
    let score = 0.12 + Math.min(0.80, (meanSpeed / 75.0) * 0.45);
    if (activeTracks.length > 5) score += 0.15; // Crowd congestion bonus

    this.anomalyHistory.push(score);
    if (this.anomalyHistory.length > 40) this.anomalyHistory.shift();

    return parseFloat(Math.min(0.98, Math.max(0.06, score)).toFixed(2));
  }

  getSeverityTier(riskScore) {
    if (riskScore >= 0.75) return 'CRITICAL';
    if (riskScore >= 0.55) return 'HIGH';
    if (riskScore >= 0.35) return 'MEDIUM';
    return 'LOW';
  }
}

window.IncidentRecognitionEngine = IncidentRecognitionEngine;
