"""
Spherical and Cylindrical Image Warping Transformations for RRtools.

Provides:
- estimate_focal_length: Focal length calculation from horizontal FOV and resolution.
- spherical_warp: Backward equirectangular spherical coordinate remapping.
- cylindrical_warp: Backward cylindrical coordinate remapping.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def estimate_focal_length(image_shape: Tuple[int, int], fov_degrees: float = 65.0) -> float:
    """
    Estimate focal length in pixels from image width and horizontal field of view.
    Formula: f = W / (2 * tan(fov / 2))
    """
    width = image_shape[1]
    fov_rad = np.radians(fov_degrees)
    return float(width / (2.0 * np.tan(fov_rad / 2.0)))


def _apply_warp_and_mask(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    safe_mask: np.ndarray,
    w: int,
    h: int,
    interpolation: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to remap image and enforce valid bounds mask."""
    valid = (
        safe_mask
        & (map_x >= 0.0)
        & (map_x < (w - 1.0))
        & (map_y >= 0.0)
        & (map_y < (h - 1.0))
    )
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = (valid.astype(np.uint8)) * 255
    warped[mask == 0] = 0
    return warped, mask


def _init_warp_coords(
    shape: Tuple[int, int],
    focal_length: float,
    cx: Optional[float],
    cy: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute grid coordinates, angle theta, and pole safety mask."""
    h, w = shape[:2]
    cx_val = float(cx if cx is not None else w / 2.0)
    cy_val = float(cy if cy is not None else h / 2.0)
    f = float(focal_length)
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    theta = (x_idx - cx_val) / f
    cos_theta = np.cos(theta)
    safe_mask = (np.abs(theta) < (np.pi / 2.0 - 0.05)) & (cos_theta > 0.01)
    return x_idx, y_idx, theta, safe_mask, cx_val, cy_val


def spherical_warp(
    image: np.ndarray,
    focal_length: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project planar perspective image onto an equirectangular spherical surface."""
    h, w = image.shape[:2]
    x_idx, y_idx, theta, safe_mask, cx_val, cy_val = _init_warp_coords(
        image.shape, focal_length, cx, cy
    )
    phi = (y_idx - cy_val) / float(focal_length)
    cos_theta = np.cos(theta)

    map_x = np.zeros_like(x_idx, dtype=np.float32)
    map_y = np.zeros_like(y_idx, dtype=np.float32)
    f = float(focal_length)
    map_x[safe_mask] = f * np.tan(theta[safe_mask]) + cx_val
    map_y[safe_mask] = f * (np.tan(phi[safe_mask]) / cos_theta[safe_mask]) + cy_val

    return _apply_warp_and_mask(image, map_x, map_y, safe_mask, w, h, interpolation)


def cylindrical_warp(
    image: np.ndarray,
    focal_length: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project planar perspective image onto a cylinder."""
    h, w = image.shape[:2]
    x_idx, y_idx, theta, safe_mask, cx_val, cy_val = _init_warp_coords(
        image.shape, focal_length, cx, cy
    )
    h_norm = (y_idx - cy_val) / float(focal_length)
    cos_theta = np.cos(theta)

    map_x = np.zeros_like(x_idx, dtype=np.float32)
    map_y = np.zeros_like(y_idx, dtype=np.float32)
    f = float(focal_length)
    map_x[safe_mask] = f * np.tan(theta[safe_mask]) + cx_val
    map_y[safe_mask] = f * (h_norm[safe_mask] / cos_theta[safe_mask]) + cy_val

    return _apply_warp_and_mask(image, map_x, map_y, safe_mask, w, h, interpolation)
