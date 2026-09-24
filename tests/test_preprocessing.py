"""
Unit Tests for CCTV Quality Preprocessing & Adaptive Enhancement Modules
"""

import unittest
import numpy as np
import cv2

from SafetySurveillance.preprocessing.quality_assessment import CCTVQualityEstimator
from SafetySurveillance.preprocessing.low_light import LowLightEnhancer
from SafetySurveillance.preprocessing.denoise import FrameDenoiser
from SafetySurveillance.preprocessing.deblur import FrameDeblurrer
from SafetySurveillance.preprocessing.degradation import CCTVVideoDegrader


class TestPreprocessingModules(unittest.TestCase):
    def setUp(self):
        # Create standard test frames
        self.clean_frame = np.full((240, 320, 3), 140, dtype=np.uint8)
        # Add high-contrast edges to simulate sharp detail
        cv2.rectangle(self.clean_frame, (50, 50), (200, 180), (255, 255, 255), -1)
        cv2.circle(self.clean_frame, (100, 100), 30, (0, 0, 0), -1)

        self.dark_frame = (self.clean_frame * 0.15).astype(np.uint8)
        self.blurry_frame = cv2.GaussianBlur(self.clean_frame, (25, 25), 0)

        self.quality_estimator = CCTVQualityEstimator()
        self.low_light = LowLightEnhancer()
        self.denoiser = FrameDenoiser()
        self.deblurrer = FrameDeblurrer()
        self.degrader = CCTVVideoDegrader()

    def test_quality_assessment_clean_frame(self):
        metrics = self.quality_estimator.assess_frame(self.clean_frame)
        self.assertIn("quality_state", metrics)
        self.assertIn("quality_factor", metrics)
        self.assertIn("brightness", metrics)
        self.assertIn("blur_score", metrics)
        self.assertGreater(metrics["quality_factor"], 0.5)
        self.assertGreater(metrics["brightness"], 50.0)

    def test_quality_assessment_dark_frame(self):
        metrics = self.quality_estimator.assess_frame(self.dark_frame)
        self.assertTrue(metrics["needs_low_light"])
        self.assertLess(metrics["brightness"], 50.0)
        self.assertIn(metrics["quality_state"], ["DEGRADED", "SEVERELY DEGRADED", "MODERATE"])

    def test_quality_assessment_empty_frame(self):
        metrics = self.quality_estimator.assess_frame(np.array([]))
        self.assertEqual(metrics["quality_state"], "SEVERELY DEGRADED")
        self.assertEqual(metrics["quality_factor"], 0.1)

    def test_low_light_enhancement(self):
        enhanced = self.low_light.enhance(self.dark_frame)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Enhanced frame must have higher mean brightness than original dark frame
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(self.dark_frame)))

    def test_frame_denoiser(self):
        noisy_frame = self.clean_frame.copy()
        noise = np.random.normal(0, 25, noisy_frame.shape).astype(np.int16)
        noisy_frame = np.clip(noisy_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        denoised = self.denoiser.denoise(noisy_frame, noise_level=20.0)
        self.assertEqual(denoised.shape, noisy_frame.shape)
        self.assertEqual(denoised.dtype, np.uint8)

    def test_frame_deblurrer(self):
        deblurred = self.deblurrer.deblur(self.blurry_frame, blur_score=30.0)
        self.assertEqual(deblurred.shape, self.blurry_frame.shape)
        self.assertEqual(deblurred.dtype, np.uint8)

    def test_video_degradation_pipeline(self):
        degraded = self.degrader.apply_composite_degradation(
            self.clean_frame,
            darkness=0.4,
            noise_level=12.0,
            blur_kernel=5,
            target_res=180
        )
        self.assertEqual(degraded.shape, self.clean_frame.shape)
        self.assertEqual(degraded.dtype, np.uint8)
        self.assertLess(float(np.mean(degraded)), float(np.mean(self.clean_frame)))


if __name__ == "__main__":
    unittest.main()
