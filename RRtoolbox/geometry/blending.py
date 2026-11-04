"""
Multi-Band Laplacian and Feathered Seam Blending for RRtools.

Provides:
- build_gaussian_pyramid: Multi-scale octave decomposition.
- build_laplacian_pyramid: Band-pass differential pyramid computation.
- reconstruct_from_laplacian: Lossless pyramid reconstruction.
- multiband_blend: Seam blending across spatial frequency octaves.
- feather_blend: Distance-transform weighted linear smooth blending.
- crop_panorama_canvas: Content-bounding margin trimming.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """Construct a Gaussian pyramid for an image or weight mask."""
    gp = [image.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(gaussian_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """Construct a Laplacian pyramid from a Gaussian pyramid."""
    lp = [gaussian_pyr[-1]]
    for i in range(len(gaussian_pyr) - 1, 0, -1):
        target_size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        up = cv2.pyrUp(gaussian_pyr[i], dstsize=target_size)
        lp.append(cv2.subtract(gaussian_pyr[i - 1], up))
    return lp[::-1]


def reconstruct_from_laplacian(laplacian_pyr: List[np.ndarray]) -> np.ndarray:
    """Reconstruct full-resolution image from its Laplacian pyramid."""
    curr = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        target_size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        curr = cv2.add(cv2.pyrUp(curr, dstsize=target_size), laplacian_pyr[i])
    return curr


def _compute_voronoi_seam_weight(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    is_color: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distance-transform Voronoi seam partition in overlap zone."""
    has_a = mask_a > 0
    has_b = mask_b > 0
    overlap = has_a & has_b

    d_a = cv2.distanceTransform(has_a.astype(np.uint8), cv2.DIST_L2, 5)
    d_b = cv2.distanceTransform(has_b.astype(np.uint8), cv2.DIST_L2, 5)

    weight_a = np.zeros(mask_a.shape[:2], dtype=np.float32)
    weight_a[has_a & (~has_b)] = 1.0
    weight_a[has_b & (~has_a)] = 0.0

    if np.any(overlap):
        weight_a[overlap] = (d_a[overlap] >= d_b[overlap]).astype(np.float32)

    w_pyr_input = cv2.merge([weight_a, weight_a, weight_a]) if is_color else weight_a
    return w_pyr_input, (has_a | has_b)


def multiband_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    Multi-Band Laplacian Pyramid Seam Blending (Burt & Adelson).
    Blends low frequencies across wide bands and high frequencies tightly at the seam.
    """
    is_color = len(image_a.shape) == 3
    w_pyr_input, union_mask = _compute_voronoi_seam_weight(mask_a, mask_b, is_color)

    min_dim = min(image_a.shape[0], image_a.shape[1])
    actual_levels = max(1, min(levels, int(np.floor(np.log2(max(1, min_dim // 16))))))

    gp_a = build_gaussian_pyramid(image_a, actual_levels)
    gp_b = build_gaussian_pyramid(image_b, actual_levels)
    gp_w = build_gaussian_pyramid(w_pyr_input, actual_levels)

    lp_a = build_laplacian_pyramid(gp_a)
    lp_b = build_laplacian_pyramid(gp_b)

    lp_blend = [
        lp_a[l] * gp_w[l] + lp_b[l] * (1.0 - gp_w[l])
        for l in range(len(lp_a))
    ]

    blended = reconstruct_from_laplacian(lp_blend)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    blended[~union_mask] = 0
    return blended


def feather_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    blur_ksize: int = 31,
) -> np.ndarray:
    """Feathered linear blending using distance transform and Gaussian smoothing."""
    has_a, has_b = mask_a > 0, mask_b > 0
    overlap = has_a & has_b

    d_a = cv2.distanceTransform(has_a.astype(np.uint8), cv2.DIST_L2, 5)
    d_b = cv2.distanceTransform(has_b.astype(np.uint8), cv2.DIST_L2, 5)
    sum_d = np.where((d_a + d_b) == 0, 1.0, d_a + d_b)

    alpha = np.zeros_like(d_a, dtype=np.float32)
    alpha[has_a & (~has_b)] = 1.0
    alpha[has_b & (~has_a)] = 0.0
    if np.any(overlap):
        alpha[overlap] = d_a[overlap] / sum_d[overlap]

    if blur_ksize > 1 and blur_ksize % 2 == 1:
        alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)

    if len(image_a.shape) == 3:
        alpha = cv2.merge([alpha, alpha, alpha])

    blended = image_a.astype(np.float32) * alpha + image_b.astype(np.float32) * (1.0 - alpha)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    blended[~(has_a | has_b)] = 0
    return blended


def crop_panorama_canvas(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim empty outer black borders around the panorama canvas."""
    if mask is None or np.sum(mask > 0) == 0:
        return image, mask

    has_content = mask > 0
    y_indices, x_indices = np.where(has_content)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, mask

    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())

    cropped_img = image[y_min : y_max + 1, x_min : x_max + 1]
    cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
    return cropped_img, cropped_mask
