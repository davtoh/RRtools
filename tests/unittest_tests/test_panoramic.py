"""
Unit and integration tests for RRtoolbox 360° Spherical Panoramic Stitcher.
Consolidated into tests/unittest_tests/ for unified test runner discovery.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from RRtoolbox.geometry.features import (
    FeatureEngine,
    KeypointMatches,
    get_neural_matcher,
    register_neural_matcher,
)
from RRtoolbox.geometry.panoramic import (
    BlendMode,
    PanoramicStitcher,
    PanoramaResult,
    ProjectionType,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    crop_panorama_canvas,
    cylindrical_warp,
    estimate_focal_length,
    feather_blend,
    multiband_blend,
    reconstruct_from_laplacian,
    spherical_warp,
)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "demo"
PANO_IMG1 = ASSETS_DIR / "panorama_input_1.jpg"
PANO_IMG2 = ASSETS_DIR / "panorama_input_2.jpg"


def _create_synthetic_textured_image(width=400, height=300):
    """Generate a rich synthetic image with circles, rectangles, and text for feature testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        img[y, :, :] = int(y / height * 100)

    cv2.circle(img, (100, 100), 40, (255, 200, 0), -1)
    cv2.rectangle(img, (200, 50), (320, 180), (0, 255, 100), -1)
    cv2.putText(img, "RRtools", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.circle(img, (280, 230), 25, (0, 100, 255), -1)
    return img


class TestPanoramicGeometry(unittest.TestCase):
    """Unit tests for focal length estimation, remapping, and feature matching."""

    def test_focal_length_estimation(self):
        """Verify focal length estimation across standard horizontal fields of view."""
        f_65 = estimate_focal_length((600, 1000), fov_degrees=65.0)
        expected_f = 1000.0 / (2.0 * np.tan(np.radians(32.5)))
        self.assertTrue(np.isclose(f_65, expected_f, rtol=1e-3))
        self.assertGreater(f_65, 0)

        f_90 = estimate_focal_length((600, 1000), fov_degrees=90.0)
        self.assertTrue(np.isclose(f_90, 500.0, rtol=1e-3))

    def test_spherical_and_cylindrical_warp(self):
        """Verify spherical and cylindrical remapping functions generate valid warped outputs and masks."""
        img = _create_synthetic_textured_image(width=400, height=300)
        f = estimate_focal_length(img.shape[:2], fov_degrees=65.0)

        w_sph, m_sph = spherical_warp(img, f)
        self.assertEqual(w_sph.shape, img.shape)
        self.assertEqual(m_sph.shape, img.shape[:2])
        self.assertGreater(np.sum(m_sph > 0), 0.5 * img.shape[0] * img.shape[1])
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        self.assertEqual(m_sph[cy, cx], 255)

        w_cyl, m_cyl = cylindrical_warp(img, f)
        self.assertEqual(w_cyl.shape, img.shape)
        self.assertEqual(m_cyl.shape, img.shape[:2])
        self.assertGreater(np.sum(m_cyl > 0), 0.5 * img.shape[0] * img.shape[1])
        self.assertEqual(m_cyl[cy, cx], 255)

    def test_feature_engine_detection_and_matching(self):
        """Test SIFT/FLANN feature extraction and synthetic translation recovery."""
        img1 = _create_synthetic_textured_image(width=500, height=400)
        dx, dy = 30.0, 15.0
        M_true = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        img2 = cv2.warpAffine(img1, M_true, (500, 400))

        engine = FeatureEngine(feature_type="sift", matcher_type="flann", ransac_thresh=3.0)
        matches = engine.match_images(img1, img2, estimate_transform="affine_partial")

        self.assertTrue(matches.is_valid)
        self.assertGreaterEqual(matches.inlier_count, 10)
        self.assertGreater(matches.inlier_ratio, 0.3)
        self.assertIsNotNone(matches.transform_matrix)
        est_dx = matches.transform_matrix[0, 2]
        est_dy = matches.transform_matrix[1, 2]
        self.assertTrue(np.isclose(est_dx, -dx, atol=2.0))
        self.assertTrue(np.isclose(est_dy, -dy, atol=2.0))

    def test_feature_engine_neural_hook(self):
        """Verify pluggable neural correspondence hook registration and dispatch."""
        called = {}

        def mock_neural_matcher(img1, img2, config):
            called["hook_invoked"] = True
            return KeypointMatches(
                src_points=np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32),
                dst_points=np.array([[15, 25], [35, 45], [55, 65], [75, 85]], dtype=np.float32),
                inliers=np.ones(4, dtype=bool),
                inlier_count=4,
                inlier_ratio=1.0,
                transform_matrix=np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]),
                transform_type="mock_neural",
                status="ok",
                confidence=0.95,
                raw_matches_count=4,
                elapsed_ms=12.5,
                metadata={"model": "mock_neural_v1"},
            )

        register_neural_matcher("mock_neural", mock_neural_matcher)
        self.assertIsNotNone(get_neural_matcher("mock_neural"))

        engine = FeatureEngine(neural_backend="mock_neural")
        dummy1 = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy2 = np.zeros((100, 100, 3), dtype=np.uint8)
        res = engine.match_images(dummy1, dummy2)

        self.assertTrue(called.get("hook_invoked"))
        self.assertTrue(res.is_valid)
        self.assertEqual(res.inlier_count, 4)
        self.assertEqual(res.confidence, 0.95)

    def test_crop_panorama_canvas(self):
        """Verify cropping canvas trims empty outer black borders."""
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        canvas[40:160, 50:250] = 128
        mask[40:160, 50:250] = 255

        cropped_img, cropped_mask = crop_panorama_canvas(canvas, mask)
        self.assertEqual(cropped_img.shape, (120, 200, 3))
        self.assertEqual(cropped_mask.shape, (120, 200))


class TestPanoramicBlendingAndStitching(unittest.TestCase):
    """Integration and algorithmic tests for multi-band blending and panoramic stitching."""

    def test_laplacian_pyramid_lossless_reconstruction(self):
        """Verify that Gaussian + Laplacian pyramid decomposition perfectly reconstructs original image."""
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        gp = build_gaussian_pyramid(img, levels=4)
        lp = build_laplacian_pyramid(gp)
        rec = reconstruct_from_laplacian(lp)

        diff = np.max(np.abs(img.astype(np.float32) - rec))
        self.assertLess(diff, 1e-4)

    def test_multiband_blending_exposure_continuity(self):
        """Verify multi-band Laplacian blending eliminates step discontinuity between different exposures."""
        h, w = 200, 300
        img_a = np.full((h, w, 3), 50, dtype=np.uint8)
        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_a[:, :200] = 255

        img_b = np.full((h, w, 3), 200, dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        mask_b[:, 100:] = 255

        blended = multiband_blend(img_a, img_b, mask_a, mask_b, levels=3)

        self.assertEqual(blended.shape, (h, w, 3))
        self.assertLess(np.mean(blended[:, :50]), 60)
        self.assertGreater(np.mean(blended[:, 250:]), 180)

        val_120 = float(np.mean(blended[:, 120]))
        val_150 = float(np.mean(blended[:, 150]))
        val_180 = float(np.mean(blended[:, 180]))
        self.assertLessEqual(val_120, val_150)
        self.assertLessEqual(val_150, val_180)

    def test_feather_blend(self):
        """Verify feather linear blending between two overlapping images."""
        h, w = 150, 200
        img_a = np.full((h, w, 3), 40, dtype=np.uint8)
        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_a[:, :130] = 255

        img_b = np.full((h, w, 3), 180, dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        mask_b[:, 70:] = 255

        blended = feather_blend(img_a, img_b, mask_a, mask_b, blur_ksize=15)
        self.assertEqual(blended.shape, (h, w, 3))
        self.assertLess(np.mean(blended[:, :50]), 50)
        self.assertGreater(np.mean(blended[:, 150:]), 150)

    def test_panoramic_stitcher_real_dataset_spherical(self):
        """Integration test: stitch real indoor laboratory views using spherical projection and multi-band blending."""
        if not PANO_IMG1.exists() or not PANO_IMG2.exists():
            self.skipTest("Demo panorama assets missing")

        stitcher = PanoramicStitcher(
            projection=ProjectionType.SPHERICAL,
            blend_mode=BlendMode.MULTIBAND,
            fov_degrees=65.0,
            pyramid_levels=4,
            crop_borders=True,
        )
        result = stitcher.stitch([PANO_IMG1, PANO_IMG2])

        self.assertIsInstance(result, PanoramaResult)
        self.assertEqual(result.projection, "spherical")
        self.assertEqual(result.blend_mode, "multiband")
        self.assertEqual(result.num_images, 2)
        self.assertGreater(result.output_shape[1], 4000)
        self.assertGreater(result.output_shape[0], 2000)
        self.assertGreaterEqual(result.inlier_count, 30)
        self.assertGreater(result.inlier_ratio, 0.20)
        self.assertGreater(result.elapsed_ms, 0)
        self.assertGreater(np.mean(result.image), 10.0)

    def test_panoramic_stitcher_cylindrical_and_feather(self):
        """Integration test: stitch real indoor views using cylindrical projection and feather blending."""
        if not PANO_IMG1.exists() or not PANO_IMG2.exists():
            self.skipTest("Demo panorama assets missing")

        stitcher = PanoramicStitcher(
            projection="cylindrical",
            blend_mode="feather",
            fov_degrees=65.0,
            crop_borders=True,
        )
        result = stitcher.stitch([PANO_IMG1, PANO_IMG2])

        self.assertEqual(result.projection, "cylindrical")
        self.assertEqual(result.blend_mode, "feather")
        self.assertGreater(result.output_shape[1], 4000)
        self.assertGreaterEqual(result.inlier_count, 30)

    def test_panoramic_stitcher_validation_errors(self):
        """Test error handling for invalid input configurations."""
        stitcher = PanoramicStitcher()

        with self.assertRaises(ValueError):
            stitcher.stitch(["nonexistent.jpg"])

        with self.assertRaises(FileNotFoundError):
            stitcher.stitch(["nonexistent1.jpg", "nonexistent2.jpg"])

    def test_cli_panorama_execution(self):
        """Test CLI subcommand 'panorama' executes and produces output."""
        if not PANO_IMG1.exists() or not PANO_IMG2.exists():
            self.skipTest("Demo panorama assets missing")

        with tempfile.TemporaryDirectory(prefix="rrtools_cli_pano_") as tmp_dir:
            out_file = Path(tmp_dir) / "test_pano_out.jpg"
            cmd = [
                sys.executable,
                "-m",
                "RRtoolbox",
                "panorama",
                str(PANO_IMG1),
                str(PANO_IMG2),
                "-o",
                str(out_file),
                "-p",
                "spherical",
                "-b",
                "multiband",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI command failed:\n{proc.stderr}\n{proc.stdout}")
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 100000)


if __name__ == "__main__":
    unittest.main()

