/**
 * QASD - Quality Assessment & Image Enhancement Engine (Client-Side)
 * Computes frame quality metrics and executes adaptive restoration filters
 * as well as synthetic CCTV degradation simulation.
 */

class QualityEnhancementEngine {
  constructor() {
    this.tempCanvas = document.createElement('canvas');
    this.tempCtx = this.tempCanvas.getContext('2d', { willReadFrequently: true });
  }

  /**
   * Assesses quality of a canvas frame: Brightness, Blur, Noise, Contrast, Quality State.
   */
  assessQuality(ctx, width, height) {
    // Sample a 160x90 downsampled buffer for fast real-time CPU evaluation
    const sampleW = 160;
    const sampleH = 90;
    this.tempCanvas.width = sampleW;
    this.tempCanvas.height = sampleH;

    this.tempCtx.drawImage(ctx.canvas, 0, 0, sampleW, sampleH);
    const imgData = this.tempCtx.getImageData(0, 0, sampleW, sampleH);
    const data = imgData.data;
    const len = data.length;

    let sumLuma = 0;
    let sumSqDiff = 0;
    const lumaBuffer = new Float32Array(sampleW * sampleH);

    // 1. Luminance & Mean
    for (let i = 0, j = 0; i < len; i += 4, j++) {
      // Rec. 709 luminance
      const luma = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      lumaBuffer[j] = luma;
      sumLuma += luma;
    }
    const meanLuma = sumLuma / (sampleW * sampleH);

    // 2. Contrast (RMS) & Laplacian Blur Index
    let laplacianVarSum = 0;
    let noiseDiffSum = 0;

    for (let y = 1; y < sampleH - 1; y++) {
      for (let x = 1; x < sampleW - 1; x++) {
        const idx = y * sampleW + x;
        const center = lumaBuffer[idx];

        // RMS Contrast component
        sumSqDiff += (center - meanLuma) * (center - meanLuma);

        // Discrete Laplacian kernel:
        // [ 0,  1,  0 ]
        // [ 1, -4,  1 ]
        // [ 0,  1,  0 ]
        const lap = (
          lumaBuffer[idx - sampleW] +
          lumaBuffer[idx + sampleW] +
          lumaBuffer[idx - 1] +
          lumaBuffer[idx + 1] -
          4 * center
        );
        laplacianVarSum += lap * lap;

        // Simple high-frequency noise estimate vs 3x3 local average
        const localAvg = (
          lumaBuffer[idx - sampleW] + lumaBuffer[idx + sampleW] +
          lumaBuffer[idx - 1] + lumaBuffer[idx + 1] + center
        ) / 5.0;
        noiseDiffSum += Math.abs(center - localAvg);
      }
    }

    const contrastRMS = Math.sqrt(sumSqDiff / (sampleW * sampleH));
    const blurScore = laplacianVarSum / ((sampleW - 2) * (sampleH - 2));
    const noiseLevel = (noiseDiffSum / ((sampleW - 2) * (sampleH - 2))) * 2.5;

    // Quality State Decision Logic
    let degScore = 0;
    const isLowLight = meanLuma < 55;
    const isSevereLowLight = meanLuma < 28;
    const isBlurry = blurScore < 95;
    const isSevereBlurry = blurScore < 40;
    const isNoisy = noiseLevel > 14;

    if (isLowLight) degScore += isSevereLowLight ? 2 : 1;
    if (isBlurry) degScore += isSevereBlurry ? 2 : 1;
    if (isNoisy) degScore += 1;
    if (contrastRMS < 28) degScore += 1;

    let qualityState = 'GOOD';
    let qualityFactor = 0.95;

    if (degScore === 0) {
      qualityState = 'GOOD';
      qualityFactor = 0.95;
    } else if (degScore <= 2) {
      qualityState = 'MODERATE';
      qualityFactor = 0.75;
    } else if (degScore <= 4) {
      qualityState = 'DEGRADED';
      qualityFactor = 0.50;
    } else {
      qualityState = 'SEVERELY DEGRADED';
      qualityFactor = 0.25;
    }

    return {
      qualityState,
      qualityFactor,
      brightness: meanLuma,
      blurScore,
      noiseLevel,
      contrast: contrastRMS,
      needsLowLight: isLowLight,
      needsDenoise: isNoisy,
      needsDeblur: isBlurry,
    };
  }

  /**
   * Applies Adaptive Enhancement (CLAHE + Deblur + Denoise) directly onto target canvas.
   */
  applyAdaptiveEnhancement(targetCtx, qualityMetrics, forcedMode = false) {
    const width = targetCtx.canvas.width;
    const height = targetCtx.canvas.height;
    const imgData = targetCtx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const needsLight = forcedMode || qualityMetrics.needsLowLight;
    const needsDenoise = forcedMode || qualityMetrics.needsDenoise;
    const needsDeblur = forcedMode || qualityMetrics.needsDeblur;

    // Adaptive dynamic gamma & contrast stretching
    if (needsLight) {
      const meanB = qualityMetrics.brightness;
      // Power law gamma brightening factor
      const gamma = Math.min(2.4, Math.max(0.4, 0.45 + (meanB / 140) * 0.55));
      const invGamma = 1.0 / gamma;

      for (let i = 0; i < data.length; i += 4) {
        // Boost shadows, equalize midtones (CLAHE approximation in sRGB)
        data[i] = Math.min(255, Math.pow(data[i] / 255, invGamma) * 255 * 1.25);
        data[i + 1] = Math.min(255, Math.pow(data[i + 1] / 255, invGamma) * 255 * 1.25);
        data[i + 2] = Math.min(255, Math.pow(data[i + 2] / 255, invGamma) * 255 * 1.25);
      }
    }

    // Denoising (subtle edge-preserving median/box smoothing)
    if (needsDenoise && !needsDeblur) {
      // Light smoothing on noisy low-light pixels
      for (let i = 4; i < data.length - 4; i += 4) {
        data[i] = (data[i - 4] + data[i] + data[i + 4]) / 3;
        data[i + 1] = (data[i - 3] + data[i + 1] + data[i + 5]) / 3;
        data[i + 2] = (data[i - 2] + data[i + 2] + data[i + 6]) / 3;
      }
    }

    // Sharpening / Deblurring (Unsharp Mask Kernel)
    if (needsDeblur) {
      const copy = new Uint8ClampedArray(data);
      const stride = width * 4;
      const kWeight = 0.55; // Unsharp strength

      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const idx = (y * width + x) * 4;
          for (let c = 0; c < 3; c++) {
            const original = copy[idx + c];
            const blurredNeighbor = (
              copy[idx - stride + c] +
              copy[idx + stride + c] +
              copy[idx - 4 + c] +
              copy[idx + 4 + c]
            ) / 4;
            const sharpVal = original + kWeight * (original - blurredNeighbor);
            data[idx + c] = Math.max(0, Math.min(255, sharpVal));
          }
        }
      }
    }

    targetCtx.putImageData(imgData, 0, 0);
  }

  /**
   * Applies synthetic CCTV Degradations onto a canvas for the Sandbox & Benchmark evaluation.
   */
  applyDegradations(targetCtx, settings) {
    const { darkness, noise, blur, downsample, haze, compression } = settings;
    const w = targetCtx.canvas.width;
    const h = targetCtx.canvas.height;

    // 1. Low Resolution Downsampling (Pixelation)
    if (downsample < h) {
      const aspect = w / h;
      const targetW = Math.floor(downsample * aspect);
      this.tempCanvas.width = targetW;
      this.tempCanvas.height = downsample;

      // Draw downscaled then upscale back with nearest-neighbor
      this.tempCtx.imageSmoothingEnabled = true;
      this.tempCtx.drawImage(targetCtx.canvas, 0, 0, targetW, downsample);

      targetCtx.imageSmoothingEnabled = false;
      targetCtx.drawImage(this.tempCanvas, 0, 0, targetW, downsample, 0, 0, w, h);
    }

    // 2. Motion Blur (Directional smear)
    if (blur > 0) {
      targetCtx.save();
      targetCtx.globalAlpha = 0.45;
      const steps = Math.min(10, Math.floor(blur));
      for (let s = 1; s <= steps; s++) {
        targetCtx.drawImage(targetCtx.canvas, s * 1.5, s * 0.8);
      }
      targetCtx.restore();
    }

    // 3. Pixel-level Manipulation: Low Light, Gaussian Noise, Haze
    const imgData = targetCtx.getImageData(0, 0, w, h);
    const data = imgData.data;

    const darkFactor = darkness / 100.0; // 0.15 to 1.0
    const hazeLevel = (haze || 0) / 100.0; // 0.0 to 1.0
    const noiseStd = noise || 0; // 0 to 60

    for (let i = 0; i < data.length; i += 4) {
      // A. Darkening
      let r = data[i] * darkFactor;
      let g = data[i + 1] * darkFactor;
      let b = data[i + 2] * darkFactor;

      // B. Atmospheric Haze (Koschmieder model approximation)
      if (hazeLevel > 0) {
        const atmospheric = 200;
        const transmission = 1.0 - (hazeLevel * 0.7);
        r = r * transmission + atmospheric * (1.0 - transmission);
        g = g * transmission + atmospheric * (1.0 - transmission);
        b = b * transmission + atmospheric * (1.0 - transmission);
      }

      // C. Gaussian Noise
      if (noiseStd > 0) {
        // Box-Muller transform for gaussian noise
        const u1 = Math.random() || 1e-5;
        const u2 = Math.random();
        const randN = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * noiseStd;
        r += randN;
        g += randN;
        b += randN;
      }

      // D. JPEG Compression block artifacts approximation
      if (compression < 80 && Math.random() < 0.12) {
        const blockJitter = (Math.random() - 0.5) * (80 - compression) * 0.8;
        r += blockJitter;
        g += blockJitter;
        b += blockJitter;
      }

      data[i] = Math.max(0, Math.min(255, r));
      data[i + 1] = Math.max(0, Math.min(255, g));
      data[i + 2] = Math.max(0, Math.min(255, b));
    }

    targetCtx.putImageData(imgData, 0, 0);
  }
}

window.QualityEnhancementEngine = QualityEnhancementEngine;
