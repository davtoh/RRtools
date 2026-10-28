"""
Unit and integration tests for RRtoolbox 360° Spherical Panoramic Stitcher (Milestone 2 - Path A.1).
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

from RRtoolbox.geometry.features import (
    FeatureEngine,
    FeatureType,
    KeypointMatches,
    MatcherType,
    register_neural_matcher,
    get_neural_matcher,
)
from RRtoolbox.geometry.panoramic import (
    PanoramicStitcher,
    PanoramaResult,
    ProjectionType,
    BlendMode,
    estimate_focal_length,
    spherical_warp,
    cylindrical_warp,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian,
    multiband_blend,
    feather_blend,
    crop_panorama_canvas,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo"
PANO_IMG1 = ASSETS_DIR / "panorama_input_1.jpg"
PANO_IMG2 = ASSETS_DIR / "panorama_input_2.jpg"


def _create_synthetic_textured_image(width=400, height=300):
    """Generate a rich synthetic image with circles, rectangles, and text for feature testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add gradient background
    for y in range(height):
        img[y, :, :] = int(y / height * 100)

    # Add geometric shapes with distinct textures
    cv2.circle(img, (100, 100), 40, (255, 200, 0), -1)
    cv2.rectangle(img, (200, 50), (320, 180), (0, 255, 100), -1)
    cv2.putText(img, "RRtools", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.circle(img, (280, 230), 25, (0, 100, 255), -1)
    return img


def test_focal_length_estimation():
    """Verify focal length estimation across standard horizontal fields of view."""
    # 65 deg FOV, width = 1000
    f_65 = estimate_focal_length((600, 1000), fov_degrees=65.0)
    expected_f = 1000.0 / (2.0 * np.tan(np.radians(32.5)))
    assert np.isclose(f_65, expected_f, rtol=1e-3)
    assert f_65 > 0

    # 90 deg FOV: tan(45 deg) = 1.0 => f = W / 2
    f_90 = estimate_focal_length((600, 1000), fov_degrees=90.0)
    assert np.isclose(f_90, 500.0, rtol=1e-3)


def test_spherical_and_cylindrical_warp():
    """Verify spherical and cylindrical remapping functions generate valid warped outputs and masks."""
    img = _create_synthetic_textured_image(width=400, height=300)
    f = estimate_focal_length(img.shape[:2], fov_degrees=65.0)

    # Spherical warp
    w_sph, m_sph = spherical_warp(img, f)
    assert w_sph.shape == img.shape
    assert m_sph.shape == img.shape[:2]
    assert np.sum(m_sph > 0) > 0.5 * img.shape[0] * img.shape[1]
    # Check that central pixel is active
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    assert m_sph[cy, cx] == 255

    # Cylindrical warp
    w_cyl, m_cyl = cylindrical_warp(img, f)
    assert w_cyl.shape == img.shape
    assert m_cyl.shape == img.shape[:2]
    assert np.sum(m_cyl > 0) > 0.5 * img.shape[0] * img.shape[1]
    assert m_cyl[cy, cx] == 255


def test_feature_engine_detection_and_matching():
    """Test SIFT/FLANN feature extraction and synthetic translation recovery."""
    img1 = _create_synthetic_textured_image(width=500, height=400)
    # Create shifted version with known translation dx=30, dy=15
    dx, dy = 30.0, 15.0
    M_true = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    img2 = cv2.warpAffine(img1, M_true, (500, 400))

    engine = FeatureEngine(feature_type="sift", matcher_type="flann", ransac_thresh=3.0)
    # match img1 (target) and img2 (source) to recover transform mapping img2 -> img1: dx=-30, dy=-15
    matches = engine.match_images(img1, img2, estimate_transform="affine_partial")

    assert matches.is_valid
    assert matches.inlier_count >= 10
    assert matches.inlier_ratio > 0.3
    assert matches.transform_matrix is not None
    # Verify estimated translation is close to (-dx, -dy)
    est_dx = matches.transform_matrix[0, 2]
    est_dy = matches.transform_matrix[1, 2]
    assert np.isclose(est_dx, -dx, atol=2.0)
    assert np.isclose(est_dy, -dy, atol=2.0)


def test_feature_engine_neural_hook():
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
    assert get_neural_matcher("mock_neural") is not None

    engine = FeatureEngine(neural_backend="mock_neural")
    dummy1 = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy2 = np.zeros((100, 100, 3), dtype=np.uint8)
    res = engine.match_images(dummy1, dummy2)

    assert called.get("hook_invoked") is True
    assert res.is_valid
    assert res.inlier_count == 4
    assert res.confidence == 0.95


def test_laplacian_pyramid_lossless_reconstruction():
    """Verify that Gaussian + Laplacian pyramid decomposition perfectly reconstructs original image."""
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    gp = build_gaussian_pyramid(img, levels=4)
    lp = build_laplacian_pyramid(gp)
    rec = reconstruct_from_laplacian(lp)

    diff = np.max(np.abs(img.astype(np.float32) - rec))
    # Mathematical reconstruction should have zero error
    assert diff < 1e-4


def test_multiband_blending_exposure_continuity():
    """Verify multi-band Laplacian blending eliminates step discontinuity between different exposures."""
    h, w = 200, 300
    # Image A: dark gray patch (50) on left
    img_a = np.full((h, w, 3), 50, dtype=np.uint8)
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_a[:, :200] = 255  # covers x: 0..200

    # Image B: bright white patch (200) on right
    img_b = np.full((h, w, 3), 200, dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    mask_b[:, 100:] = 255  # covers x: 100..300 (overlap is 100..200)

    blended = multiband_blend(img_a, img_b, mask_a, mask_b, levels=3)

    assert blended.shape == (h, w, 3)
    # Check left region retains dark tone
    assert np.mean(blended[:, :50]) < 60
    # Check right region retains bright tone
    assert np.mean(blended[:, 250:]) > 180
    # In overlap region, verify smooth monotonic gradient across the seam (x=150)
    val_120 = float(np.mean(blended[:, 120]))
    val_150 = float(np.mean(blended[:, 150]))
    val_180 = float(np.mean(blended[:, 180]))
    assert val_120 <= val_150 <= val_180


def test_feather_blend():
    """Verify feather linear blending between two overlapping images."""
    h, w = 150, 200
    img_a = np.full((h, w, 3), 40, dtype=np.uint8)
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_a[:, :130] = 255

    img_b = np.full((h, w, 3), 180, dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    mask_b[:, 70:] = 255

    blended = feather_blend(img_a, img_b, mask_a, mask_b, blur_ksize=15)
    assert blended.shape == (h, w, 3)
    assert np.mean(blended[:, :50]) < 50
    assert np.mean(blended[:, 150:]) > 150


def test_crop_panorama_canvas():
    """Verify cropping canvas trims empty outer black borders."""
    canvas = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    canvas[40:160, 50:250] = 128
    mask[40:160, 50:250] = 255

    cropped_img, cropped_mask = crop_panorama_canvas(canvas, mask)
    assert cropped_img.shape == (120, 200, 3)
    assert cropped_mask.shape == (120, 200)


@pytest.mark.skipif(not PANO_IMG1.exists() or not PANO_IMG2.exists(), reason="Demo panorama assets missing")
def test_panoramic_stitcher_real_dataset_spherical():
    """Integration test: stitch real indoor laboratory views using spherical projection and multi-band blending."""
    stitcher = PanoramicStitcher(
        projection=ProjectionType.SPHERICAL,
        blend_mode=BlendMode.MULTIBAND,
        fov_degrees=65.0,
        pyramid_levels=4,
        crop_borders=True,
    )
    result = stitcher.stitch([PANO_IMG1, PANO_IMG2])

    assert isinstance(result, PanoramaResult)
    assert result.projection == "spherical"
    assert result.blend_mode == "multiband"
    assert result.num_images == 2
    # Verify wide composite dimensions
    assert result.output_shape[1] > 4000  # Wide panoramic width (> 4k pixels)
    assert result.output_shape[0] > 2000  # Height (> 2k pixels)
    assert result.inlier_count >= 30      # Strong RANSAC correspondence
    assert result.inlier_ratio > 0.20
    assert result.elapsed_ms > 0
    # Check that image is non-empty
    assert np.mean(result.image) > 10.0


@pytest.mark.skipif(not PANO_IMG1.exists() or not PANO_IMG2.exists(), reason="Demo panorama assets missing")
def test_panoramic_stitcher_cylindrical_and_feather():
    """Integration test: stitch real indoor views using cylindrical projection and feather blending."""
    stitcher = PanoramicStitcher(
        projection="cylindrical",
        blend_mode="feather",
        fov_degrees=65.0,
        crop_borders=True,
    )
    result = stitcher.stitch([PANO_IMG1, PANO_IMG2])

    assert result.projection == "cylindrical"
    assert result.blend_mode == "feather"
    assert result.output_shape[1] > 4000
    assert result.inlier_count >= 30


def test_panoramic_stitcher_validation_errors(tmp_path):
    """Test error handling for invalid input configurations."""
    stitcher = PanoramicStitcher()

    # Fewer than 2 images
    with pytest.raises(ValueError, match="at least 2 images"):
        stitcher.stitch(["nonexistent.jpg"])

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        stitcher.stitch(["nonexistent1.jpg", "nonexistent2.jpg"])


def test_cli_panorama_execution(tmp_path):
    """Test CLI subcommand 'panorama' executes and produces output."""
    import subprocess

    out_file = tmp_path / "test_pano_out.jpg"
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
    assert proc.returncode == 0, f"CLI command failed:\n{proc.stderr}\n{proc.stdout}"
    assert out_file.exists()
    assert out_file.stat().st_size > 100000
