"""
Sub-Pixel ECC and Phase Correlation Image Registration for RRtools.

Provides:
- align_ecc_subpixel: Sub-pixel 2D transformation matrix estimation using ECC with phase correlation fallback.
- accumulate_warped_frame: High-resolution inverse warping and weight accumulation.
- boost_high_frequency_details: Bilateral high-frequency edge detail decomposition.
- align_and_fuse_frames: Multi-frame sub-pixel alignment and fusion pipeline.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def align_ecc_subpixel(
    ref_gray: np.ndarray,
    target_gray: np.ndarray,
    motion_model: int = cv2.MOTION_EUCLIDEAN,
    max_iterations: int = 50,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Estimate sub-pixel 2D transformation matrix using Enhanced Correlation Coefficient (ECC).
    Falls back to 2D phase correlation if displacement is large or nonlinear.
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, target_gray, warp_matrix, motion_model, criteria
        )
        return warp_matrix
    except cv2.error:
        shift, _ = cv2.phaseCorrelate(
            ref_gray.astype(np.float32), target_gray.astype(np.float32)
        )
        warp_matrix[0, 2] = float(shift[0])
        warp_matrix[1, 2] = float(shift[1])
        return warp_matrix


def accumulate_warped_frame(
    target_img: np.ndarray,
    warp_mat: np.ndarray,
    scale: int,
    hr_shape: Tuple[int, int],
    is_color: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp target frame to high-resolution coordinate frame and return weighted frame and mask."""
    hr_h, hr_w = hr_shape
    scale_mat = np.array(
        [
            [warp_mat[0, 0], warp_mat[0, 1], warp_mat[0, 2] * scale],
            [warp_mat[1, 0], warp_mat[1, 1], warp_mat[1, 2] * scale],
        ],
        dtype=np.float32,
    )
    tgt_hr = cv2.resize(target_img, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)
    warped = cv2.warpAffine(
        tgt_hr, scale_mat, (hr_w, hr_h), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
    )
    ones = np.ones((hr_h, hr_w), dtype=np.float32)
    warped_mask = cv2.warpAffine(
        ones, scale_mat, (hr_w, hr_h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
    )
    if is_color:
        warped_mask = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)

    return warped.astype(np.float32) * warped_mask, warped_mask


def boost_high_frequency_details(fused_hr: np.ndarray, detail_boost: float) -> np.ndarray:
    """
    Bilateral high-frequency detail decomposition.
    Separates true optical edge transitions from sensor noise.
    """
    smooth = cv2.bilateralFilter(fused_hr, 9, 75, 75)
    detail = cv2.subtract(fused_hr, smooth)
    return cv2.addWeighted(fused_hr, 1.0, detail, detail_boost, 0)


def align_and_fuse_frames(
    images: List[np.ndarray],
    reference_idx: int = 0,
    scale: int = 2,
    motion_model: int = cv2.MOTION_EUCLIDEAN,
    max_ecc_iterations: int = 50,
    ecc_eps: float = 1e-4,
    detail_boost: float = 1.2,
) -> np.ndarray:
    """
    Fuse multiple frames into a super-resolved image via sub-pixel registration.
    """
    ref_img = images[reference_idx]
    h, w = ref_img.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    is_color = len(ref_img.shape) == 3
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if is_color else ref_img

    hr_accum = cv2.resize(ref_img, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    weight_shape = (hr_h, hr_w, 3) if is_color else (hr_h, hr_w)
    weight_accum = np.ones(weight_shape, dtype=np.float32)

    for i, target_img in enumerate(images):
        if i == reference_idx:
            continue
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        if target_gray.shape != ref_gray.shape:
            target_gray = cv2.resize(target_gray, (w, h), interpolation=cv2.INTER_AREA)
            target_img = cv2.resize(target_img, (w, h), interpolation=cv2.INTER_AREA)

        warp_mat = align_ecc_subpixel(ref_gray, target_gray, motion_model, max_ecc_iterations, ecc_eps)
        warped_weighted, warped_mask = accumulate_warped_frame(
            target_img, warp_mat, scale, (hr_h, hr_w), is_color
        )
        hr_accum += warped_weighted
        weight_accum += warped_mask

    fused_hr = (hr_accum / np.maximum(weight_accum, 1e-5)).clip(0, 255).astype(np.uint8)
    return boost_high_frequency_details(fused_hr, detail_boost)

