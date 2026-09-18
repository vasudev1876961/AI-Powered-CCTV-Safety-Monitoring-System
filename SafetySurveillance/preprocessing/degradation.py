"""
CCTV Degradation Simulation Engine
Synthetically injects realistic CCTV defects: low light, sensor noise,
motion blur, atmospheric haze, downsampling, and JPEG compression.
Used for controlled experimental evaluation and robustness benchmarks.
"""

import cv2
import numpy as np


class CCTVVideoDegrader:
    """Simulates real-world poor CCTV environmental challenges."""

    @staticmethod
    def apply_low_light(frame: np.ndarray, factor: float = 0.35, gamma: float = 2.2) -> np.ndarray:
        """
        Attenuates illumination and applies power-law darkening.
        factor: multiplier in [0.1, 1.0] (lower is darker)
        """
        attenuated = frame.astype(np.float32) * factor
        # Nonlinear gamma drop in shadow regions
        normalized = attenuated / 255.0
        darkened = np.power(normalized, gamma) * 255.0
        return np.clip(darkened, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_gaussian_noise(frame: np.ndarray, std: float = 25.0) -> np.ndarray:
        """Adds zero-mean additive Gaussian sensor thermal noise."""
        noise = np.random.normal(0, std, frame.shape)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(frame: np.ndarray, kernel_size: int = 15, angle: float = 45.0) -> np.ndarray:
        """Simulates camera shake or fast motion with directional linear blur."""
        if kernel_size <= 1:
            return frame
        # Construct directional motion kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        radian = np.deg2rad(angle)
        dx = np.cos(radian)
        dy = np.sin(radian)
        center = kernel_size // 2

        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * dx)
            y = int(center + offset * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0

        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def apply_downsampling(frame: np.ndarray, target_height: int = 240) -> np.ndarray:
        """Simulates low-resolution analog CCTV cameras (e.g. 720p -> 240p -> upscaled)."""
        h, w = frame.shape[:2]
        if h <= target_height:
            return frame
        aspect = w / h
        target_width = int(target_height * aspect)
        downscaled = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
        return upscaled

    @staticmethod
    def apply_haze(frame: np.ndarray, atmospheric_light: float = 210.0, beta: float = 0.015) -> np.ndarray:
        """Simulates fog, smoke, or outdoor haze using the Koschmieder optical scattering model."""
        h, w = frame.shape[:2]
        # Depth map approximation from top to bottom
        depth = np.linspace(0.2, 1.0, h)[:, None]
        depth = np.tile(depth, (1, w))

        transmission = np.exp(-beta * depth * 100.0)[:, :, None]
        frame_float = frame.astype(np.float32)
        hazed = frame_float * transmission + atmospheric_light * (1.0 - transmission)
        return np.clip(hazed, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_compression(frame: np.ndarray, quality: int = 20) -> np.ndarray:
        """Simulates high-compression artifacts from low-bandwidth RTSP/NVR encoders."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(5, quality)]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encimg, 1)

    def apply_composite_degradation(
        self,
        frame: np.ndarray,
        severity: str = "DEGRADED",
        darkness: float = 0.4,
        noise_level: float = 20.0,
        blur_kernel: int = 9,
        target_res: int = 360,
        haze_factor: float = 0.4,
        jpeg_quality: int = 30
    ) -> np.ndarray:
        """Applies customizable combined degradation."""
        out = frame.copy()
        if darkness < 0.95:
            out = self.apply_low_light(out, factor=darkness)
        if blur_kernel > 1:
            out = self.apply_motion_blur(out, kernel_size=blur_kernel)
        if noise_level > 2.0:
            out = self.apply_gaussian_noise(out, std=noise_level)
        if target_res < out.shape[0]:
            out = self.apply_downsampling(out, target_height=target_res)
        if haze_factor > 0.05:
            out = self.apply_haze(out, beta=0.01 * haze_factor)
        if jpeg_quality < 85:
            out = self.apply_compression(out, quality=jpeg_quality)
        return out
