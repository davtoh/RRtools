"""
Unit and regression tests for RRtools Super-Resolution & Hardware Detection (Phase 1).
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

from RRtoolbox.hardware import detect_hardware, get_onnx_execution_providers
from RRtoolbox.restoration.super_resolution import (
    SuperResolutionEngine,
    MultiFrameSuperResolution,
    SingleImageSuperResolution,
    Backend,
    compute_sharpness,
    compute_contrast,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="rrtools_test_sr_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_pattern():
    """Create a high-contrast synthetic test pattern with text and grid."""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    # Add grid lines
    img[::10, :] = 255
    img[:, ::10] = 255
    # Add circle
    cv2.circle(img, (80, 60), 30, (0, 255, 255), -1)
    # Add text
    cv2.putText(img, "RRTOOLS", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def test_hardware_detection():
    """Verify hardware detection runs cleanly and provides non-empty profile."""
    profile = detect_hardware()
    assert profile is not None
    assert isinstance(profile.device_name, str)
    assert isinstance(profile.vram_mb, int)
    assert profile.recommended_device in ["cuda", "directml", "cpu"]
    summary_text = profile.summary()
    assert "Hardware Profile:" in summary_text
    assert "Active Provider:" in summary_text


def test_onnx_execution_providers():
    """Verify ONNX execution provider resolution always has CPU fallback."""
    providers = get_onnx_execution_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1
    assert "CPUExecutionProvider" in providers


def test_compute_sharpness_and_contrast(synthetic_pattern):
    """Verify sharpness and contrast metrics behave monotonically."""
    sharp_pattern = compute_sharpness(synthetic_pattern)
    contrast_pattern = compute_contrast(synthetic_pattern)
    assert sharp_pattern > 0
    assert contrast_pattern > 0

    # Blurring should strictly reduce sharpness
    blurred = cv2.GaussianBlur(synthetic_pattern, (9, 9), 3.0)
    sharp_blurred = compute_sharpness(blurred)
    assert sharp_blurred < sharp_pattern

    # Empty image safety
    assert compute_sharpness(np.zeros((0, 0), dtype=np.uint8)) == 0.0
    assert compute_contrast(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_single_image_super_resolution(synthetic_pattern):
    """Test SISR edge-directed upscaling across scales 2x and 3x."""
    sisr_2x = SingleImageSuperResolution(scale=2, backend=Backend.EDGE_DIRECTED)
    out_2x = sisr_2x.process(synthetic_pattern)
    h, w = synthetic_pattern.shape[:2]
    assert out_2x.shape == (h * 2, w * 2, 3)

    sisr_3x = SingleImageSuperResolution(scale=3, backend=Backend.EDGE_DIRECTED)
    out_3x = sisr_3x.process(synthetic_pattern)
    assert out_3x.shape == (h * 3, w * 3, 3)


def test_single_image_super_resolution_grayscale():
    """Verify SISR correctly handles grayscale single-channel images."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(gray, (50, 50), 25, 255, -1)
    sisr = SingleImageSuperResolution(scale=2)
    out = sisr.process(gray)
    assert out.shape == (200, 200)
    assert compute_sharpness(out) > 0


def test_multi_frame_super_resolution_synthetic(synthetic_pattern):
    """Test MFSR with sub-pixel shifted optical frames."""
    # Simulate camera optical point spread function (lens blur)
    base_optical = cv2.GaussianBlur(synthetic_pattern, (3, 3), 0.8)
    h, w = base_optical.shape[:2]
    frames = [base_optical]

    # Generate frames with sub-pixel translations (handheld jitter)
    shifts = [(0.5, 0.25), (-0.3, 0.4), (0.2, -0.6)]
    for dx, dy in shifts:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base_optical, M, (w, h), flags=cv2.INTER_LANCZOS4)
        frames.append(shifted)

    mfsr = MultiFrameSuperResolution(scale=2, detail_boost=1.4)
    fused = mfsr.process(frames, reference_idx=0)
    assert fused.shape == (h * 2, w * 2, 3)

    # Sharpness should be reconstructed and higher than the blurred reference upscale
    bicubic_ref = cv2.resize(base_optical, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    assert compute_sharpness(fused) > compute_sharpness(bicubic_ref)


def test_super_resolution_engine_routing(temp_dir):
    """Test SuperResolutionEngine end-to-end routing between SISR and MFSR."""
    # Write two small test images
    img1 = np.full((80, 80, 3), 128, dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (60, 60), (255, 0, 0), -1)
    img2 = np.full((80, 80, 3), 128, dtype=np.uint8)
    cv2.rectangle(img2, (21, 21), (61, 61), (255, 0, 0), -1)

    p1 = temp_dir / "frame1.png"
    p2 = temp_dir / "frame2.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    engine = SuperResolutionEngine(scale=2)

    # Test Single Image routing
    res_single = engine.restore(str(p1))
    assert res_single.mode == "sisr"
    assert res_single.output_shape == (160, 160)
    assert res_single.sharpness_gain_pct >= 0

    # Test Multi-Frame routing
    res_multi = engine.restore([str(p1), str(p2)])
    assert res_multi.mode == "mfsr"
    assert res_multi.output_shape == (160, 160)
    assert res_multi.metadata["source_count"] == 2

    # Test save
    out_file = temp_dir / "out_restored.png"
    saved_dest = res_multi.save(out_file)
    assert saved_dest.exists()
    assert saved_dest.stat().st_size > 0


def test_real_dataset_multi_frame():
    """Verify MFSR on real test images in tests/ directory."""
    p_left = Path("tests/good_left.jpg")
    p_right = Path("tests/good_right.jpg")

    if not (p_left.exists() and p_right.exists()):
        pytest.skip("Test image assets not found.")

    engine = SuperResolutionEngine(scale=2)
    res = engine.restore([p_left, p_right])
    assert res.mode == "mfsr"
    assert res.output_shape[0] == res.input_shape[0] * 2
    assert res.output_shape[1] == res.input_shape[1] * 2
    assert res.sharpness_gain_pct > 0
    assert res.elapsed_ms < 5000  # Must complete rapidly under 5s
