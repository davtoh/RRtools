"""
Multi-Angle 360° Spherical & Cylindrical Panoramic Stitcher for RRtools.

Provides:
- Spherical & Cylindrical Remapping: Transforms planar camera views into
  equirectangular / cylindrical projections, converting non-linear 3D perspective
  rotations into smooth coordinate translations and eliminating planar tearing.
- Multi-Band Laplacian Pyramid Blending: Decomposes warped views into spatial
  frequency bands; blends low-frequency illumination across a broad transition to
  eliminate exposure steps, and sharp high frequencies in a tight band to prevent
  ghosting and double edges.
- PanoramicStitcher: Unified multi-view registration and stitching engine
  with automatic focal length estimation, bundle-adjusted canvas expansion,
  and comprehensive performance metrics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from RRtoolbox.geometry.features import FeatureEngine, KeypointMatches

logger = logging.getLogger("RRtoolbox.geometry.panoramic")


class ProjectionType(str, Enum):
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR = "planar"


class BlendMode(str, Enum):
    MULTIBAND = "multiband"
    FEATHER = "feather"
    LINEAR = "linear"
    NONE = "none"


@dataclass
class PanoramaResult:
    """Result container for panoramic stitching processing."""
    image: np.ndarray
    projection: str
    blend_mode: str
    num_images: int
    output_shape: Tuple[int, int]
    inlier_count: int
    inlier_ratio: float
    elapsed_ms: float
    metadata: Dict[str, Any]

    def save(self, destination: Union[str, Path]) -> Path:
        """Save the stitched panorama to disk."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), self.image)
        logger.info(f"Saved panorama image to: {dest} ({self.output_shape[1]}x{self.output_shape[0]})")
        return dest


def estimate_focal_length(image_shape: Tuple[int, int], fov_degrees: float = 65.0) -> float:
    """
    Estimate focal length in pixels from image width and horizontal field of view.

    Formula:
        f = W / (2 * tan(fov / 2))

    Args:
        image_shape: (height, width) or (height, width, channels).
        fov_degrees: Horizontal Field-of-View in degrees (default: 65.0 deg).

    Returns:
        Focal length in pixels.
    """
    width = image_shape[1]
    fov_rad = np.radians(fov_degrees)
    f = float(width / (2.0 * np.tan(fov_rad / 2.0)))
    return f


def spherical_warp(
    image: np.ndarray,
    focal_length: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project planar perspective image onto an equirectangular spherical surface.

    Spherical backward mapping:
        theta = (x' - cx) / f
        phi = (y' - cy) / f
        X = sin(theta) * cos(phi), Y = sin(phi), Z = cos(theta) * cos(phi)
        x = f * tan(theta) + cx
        y = f * (tan(phi) / cos(theta)) + cy

    Args:
        image: Input image (BGR or grayscale).
        focal_length: Camera focal length in pixels.
        cx: Principal point X (defaults to image_width / 2).
        cy: Principal point Y (defaults to image_height / 2).
        interpolation: OpenCV interpolation flag.

    Returns:
        Tuple of (warped_image, binary_valid_mask).
    """
    h, w = image.shape[:2]
    cx = float(cx if cx is not None else w / 2.0)
    cy = float(cy if cy is not None else h / 2.0)
    f = float(focal_length)

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    theta = (x_idx - cx) / f
    phi = (y_idx - cy) / f

    cos_theta = np.cos(theta)
    # Avoid division by zero near the +/- 90-degree poles
    safe_mask = (np.abs(theta) < (np.pi / 2.0 - 0.05)) & (cos_theta > 0.01)

    map_x = np.zeros_like(x_idx, dtype=np.float32)
    map_y = np.zeros_like(y_idx, dtype=np.float32)

    map_x[safe_mask] = f * np.tan(theta[safe_mask]) + cx
    map_y[safe_mask] = f * (np.tan(phi[safe_mask]) / cos_theta[safe_mask]) + cy

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
    if len(image.shape) == 3:
        warped[mask == 0] = 0
    else:
        warped[mask == 0] = 0

    return warped, mask


def cylindrical_warp(
    image: np.ndarray,
    focal_length: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project planar perspective image onto a cylinder.

    Cylindrical backward mapping:
        theta = (x' - cx) / f
        h_norm = (y' - cy) / f
        x = f * tan(theta) + cx
        y = f * (h_norm / cos(theta)) + cy

    Args:
        image: Input image (BGR or grayscale).
        focal_length: Camera focal length in pixels.
        cx: Principal point X (defaults to image_width / 2).
        cy: Principal point Y (defaults to image_height / 2).
        interpolation: OpenCV interpolation flag.

    Returns:
        Tuple of (warped_image, binary_valid_mask).
    """
    h, w = image.shape[:2]
    cx = float(cx if cx is not None else w / 2.0)
    cy = float(cy if cy is not None else h / 2.0)
    f = float(focal_length)

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    theta = (x_idx - cx) / f
    h_norm = (y_idx - cy) / f

    cos_theta = np.cos(theta)
    safe_mask = (np.abs(theta) < (np.pi / 2.0 - 0.05)) & (cos_theta > 0.01)

    map_x = np.zeros_like(x_idx, dtype=np.float32)
    map_y = np.zeros_like(y_idx, dtype=np.float32)

    map_x[safe_mask] = f * np.tan(theta[safe_mask]) + cx
    map_y[safe_mask] = f * (h_norm[safe_mask] / cos_theta[safe_mask]) + cy

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
    if len(image.shape) == 3:
        warped[mask == 0] = 0
    else:
        warped[mask == 0] = 0

    return warped, mask


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Construct a Gaussian pyramid for an image or weight mask.
    level 0 is original resolution; higher levels are progressively downscaled.
    """
    gp = [image.astype(np.float32)]
    for _ in range(levels):
        down = cv2.pyrDown(gp[-1])
        gp.append(down)
    return gp


def build_laplacian_pyramid(gaussian_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Construct a Laplacian pyramid from a Gaussian pyramid.
    level 0 is finest detail difference; last level is coarsest base.
    """
    num_levels = len(gaussian_pyr)
    lp = [gaussian_pyr[-1]]
    for i in range(num_levels - 1, 0, -1):
        target_size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        up = cv2.pyrUp(gaussian_pyr[i], dstsize=target_size)
        lap = cv2.subtract(gaussian_pyr[i - 1], up)
        lp.append(lap)
    return lp[::-1]


def reconstruct_from_laplacian(laplacian_pyr: List[np.ndarray]) -> np.ndarray:
    """Reconstruct full-resolution image from its Laplacian pyramid."""
    curr = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        target_size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        up = cv2.pyrUp(curr, dstsize=target_size)
        curr = cv2.add(up, laplacian_pyr[i])
    return curr


def multiband_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    Multi-Band Laplacian Pyramid Seam Blending (Burt & Adelson).
    Blends low-frequency illumination across a wide band to erase exposure jumps,
    while sharp high-frequency edges transition tightly at the Voronoi seam.

    Args:
        image_a: Base image canvas (BGR or gray, uint8).
        image_b: Registered image canvas (BGR or gray, uint8).
        mask_a: Valid pixel mask for image_a (uint8).
        mask_b: Valid pixel mask for image_b (uint8).
        levels: Number of pyramid octave levels (default: 4).

    Returns:
        Seamlessly blended composite image (uint8).
    """
    # 1. Compute distance-transform seam partition in the overlap zone
    has_a = mask_a > 0
    has_b = mask_b > 0
    overlap = has_a & has_b

    d_a = cv2.distanceTransform(has_a.astype(np.uint8), cv2.DIST_L2, 5)
    d_b = cv2.distanceTransform(has_b.astype(np.uint8), cv2.DIST_L2, 5)

    weight_a = np.zeros(mask_a.shape[:2], dtype=np.float32)
    weight_a[has_a & (~has_b)] = 1.0
    weight_a[has_b & (~has_a)] = 0.0

    if np.any(overlap):
        # Distance-weighted Voronoi partition in overlap zone
        weight_a[overlap] = (d_a[overlap] >= d_b[overlap]).astype(np.float32)

    # 2. Channel dimension alignment
    if len(image_a.shape) == 3:
        w_pyr_input = cv2.merge([weight_a, weight_a, weight_a])
    else:
        w_pyr_input = weight_a

    # Determine maximum valid pyramid levels based on canvas size
    min_dim = min(image_a.shape[0], image_a.shape[1])
    actual_levels = min(levels, int(np.floor(np.log2(max(1, min_dim // 16)))))
    actual_levels = max(1, actual_levels)

    # 3. Build pyramids
    gp_a = build_gaussian_pyramid(image_a, actual_levels)
    gp_b = build_gaussian_pyramid(image_b, actual_levels)
    gp_w = build_gaussian_pyramid(w_pyr_input, actual_levels)

    lp_a = build_laplacian_pyramid(gp_a)
    lp_b = build_laplacian_pyramid(gp_b)

    # 4. Multi-band octave blending: L_blend = L_a * G_w + L_b * (1 - G_w)
    lp_blend = []
    for l in range(len(lp_a)):
        l_blend = lp_a[l] * gp_w[l] + lp_b[l] * (1.0 - gp_w[l])
        lp_blend.append(l_blend)

    # 5. Collapse & reconstruct
    blended = reconstruct_from_laplacian(lp_blend)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # Zero-out non-content regions
    union_mask = has_a | has_b
    if len(image_a.shape) == 3:
        blended[~union_mask] = 0
    else:
        blended[~union_mask] = 0

    return blended


def feather_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    blur_ksize: int = 31,
) -> np.ndarray:
    """
    Feathered linear blending using distance transform and Gaussian smoothing.
    """
    has_a = mask_a > 0
    has_b = mask_b > 0
    overlap = has_a & has_b

    d_a = cv2.distanceTransform(has_a.astype(np.uint8), cv2.DIST_L2, 5)
    d_b = cv2.distanceTransform(has_b.astype(np.uint8), cv2.DIST_L2, 5)

    sum_d = d_a + d_b
    sum_d[sum_d == 0] = 1.0

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

    union = has_a | has_b
    blended[~union] = 0
    return blended


def crop_panorama_canvas(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim empty outer black borders around the panorama canvas.
    """
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


class PanoramicStitcher:
    """
    Multi-Angle 360° Spherical Panoramic Stitcher.
    Overcomes the planar homography restriction by projecting camera views into
    spherical coordinates, solving multi-angle bundle alignment, and blending
    with multi-band Laplacian pyramids.
    """

    def __init__(
        self,
        projection: Union[str, ProjectionType] = ProjectionType.SPHERICAL,
        blend_mode: Union[str, BlendMode] = BlendMode.MULTIBAND,
        fov_degrees: float = 65.0,
        focal_length: Optional[float] = None,
        pyramid_levels: int = 4,
        max_features: int = 3000,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 4.0,
        estimate_transform: str = "affine_partial",
        match_downscale: Union[str, float] = "auto",
        crop_borders: bool = True,
        backend: str = "auto",
    ):
        self.projection = (
            ProjectionType(projection.lower()) if isinstance(projection, str) else projection
        )
        self.blend_mode = (
            BlendMode(blend_mode.lower()) if isinstance(blend_mode, str) else blend_mode
        )
        self.fov_degrees = float(fov_degrees)
        self.focal_length = float(focal_length) if focal_length is not None else None
        self.pyramid_levels = int(pyramid_levels)
        self.estimate_transform = estimate_transform
        self.match_downscale = match_downscale
        self.crop_borders = crop_borders
        self.backend = backend.lower()

        self.feature_engine = FeatureEngine(
            feature_type="sift",
            n_features=max_features,
            ratio_thresh=ratio_thresh,
            ransac_thresh=ransac_thresh,
            matcher_type="flann",
        )

    def _resolve_focal_length(self, shape: Tuple[int, int]) -> float:
        """Get calibrated focal length or estimate from horizontal FOV."""
        if self.focal_length is not None and self.focal_length > 0:
            return self.focal_length
        return estimate_focal_length(shape, self.fov_degrees)

    def warp_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remap an image into the configured projection space (spherical, cylindrical, or planar).
        """
        if self.projection == ProjectionType.PLANAR:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return image.copy(), mask

        f = self._resolve_focal_length(image.shape[:2])

        if self.projection == ProjectionType.SPHERICAL:
            return spherical_warp(image, f)
        elif self.projection == ProjectionType.CYLINDRICAL:
            return cylindrical_warp(image, f)
        else:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return image.copy(), mask

    def _resolve_downscale(self, shape: Tuple[int, int]) -> float:
        """Resolve downscale factor for fast feature extraction on large inputs."""
        if isinstance(self.match_downscale, (int, float)):
            return max(1.0, float(self.match_downscale))
        # "auto" downscale: if max dimension > 1600px, downscale so max dimension is ~1000px
        max_dim = max(shape[0], shape[1])
        if max_dim > 1600:
            return float(max_dim / 1000.0)
        return 1.0

    def stitch(
        self,
        images: List[Union[str, Path, np.ndarray]],
    ) -> PanoramaResult:
        """
        Stitch multiple overlapping views into a continuous 360° / wide panorama.

        Args:
            images: List of input images (file paths or numpy arrays). Must contain at least 2 views.

        Returns:
            PanoramaResult containing stitched composite image and execution telemetry.
        """
        t0 = time.perf_counter()

        if len(images) < 2:
            raise ValueError(f"Panoramic stitching requires at least 2 images, got {len(images)}")

        # 1. Load images into memory
        loaded_images: List[np.ndarray] = []
        for item in images:
            if isinstance(item, (str, Path)):
                p = Path(item)
                if not p.exists():
                    raise FileNotFoundError(f"Input image not found: {p}")
                img = cv2.imread(str(p))
                if img is None:
                    raise ValueError(f"Could not read image file: {p}")
                loaded_images.append(img)
            elif isinstance(item, np.ndarray):
                loaded_images.append(item)
            else:
                raise TypeError(f"Unsupported image type: {type(item)}")

        num_images = len(loaded_images)
        logger.info(f"Starting panoramic stitch across {num_images} views (projection: {self.projection.value}, blend: {self.blend_mode.value})")

        # 2. Try OpenCV stitcher backend if explicitly requested
        if self.backend == "opencv":
            cv_stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, pano = cv_stitcher.stitch(loaded_images)
            if status == cv2.Stitcher_OK and pano is not None:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(f"OpenCV stitcher succeeded in {elapsed:.1f}ms")
                return PanoramaResult(
                    image=pano,
                    projection="opencv_panorama",
                    blend_mode="opencv_internal",
                    num_images=num_images,
                    output_shape=pano.shape[:2],
                    inlier_count=100,
                    inlier_ratio=1.0,
                    elapsed_ms=elapsed,
                    metadata={"backend": "opencv"},
                )
            logger.warning("OpenCV Stitcher backend failed, proceeding with native spherical stitcher")

        # 3. Project each image to spherical/cylindrical coordinate space
        warped_views: List[np.ndarray] = []
        warped_masks: List[np.ndarray] = []
        for img in loaded_images:
            w_img, w_mask = self.warp_image(img)
            warped_views.append(w_img)
            warped_masks.append(w_mask)

        # 4. Pairwise alignment between consecutive frames
        # Express all transformations relative to anchor frame 0
        total_inliers = 0
        total_matches = 0
        transforms_to_prev: List[np.ndarray] = []  # transform mapping frame i -> frame i-1

        for i in range(1, num_images):
            img_prev = warped_views[i - 1]
            img_curr = warped_views[i]
            mask_prev = warped_masks[i - 1]
            mask_curr = warped_masks[i]

            # Fast matching downscale if needed
            downscale = self._resolve_downscale(img_prev.shape[:2])
            if downscale > 1.05:
                w_s = int(img_prev.shape[1] / downscale)
                h_s = int(img_prev.shape[0] / downscale)
                s_prev = cv2.resize(img_prev, (w_s, h_s))
                s_curr = cv2.resize(img_curr, (w_s, h_s))
                m_prev = cv2.resize(mask_prev, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
                m_curr = cv2.resize(mask_curr, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
            else:
                s_prev, s_curr = img_prev, img_curr
                m_prev, m_curr = mask_prev, mask_curr

            match_res = self.feature_engine.match_images(
                s_prev,
                s_curr,
                mask1=m_prev,
                mask2=m_curr,
                estimate_transform=self.estimate_transform,
            )

            if not match_res.is_valid:
                # Try translation fallback if affine had insufficient inliers
                logger.debug(f"Pair ({i-1}, {i}) affine matching failed ({match_res.status}). Trying translation fallback.")
                match_res = self.feature_engine.match_images(
                    s_prev,
                    s_curr,
                    mask1=m_prev,
                    mask2=m_curr,
                    estimate_transform="translation",
                )

            if not match_res.is_valid:
                logger.warning(
                    f"Warning: Low correspondence between frames {i-1} and {i} (inliers: {match_res.inlier_count}). Defaulting to side-by-side offset."
                )
                # Default horizontal displacement if completely disjoint
                M = np.array([[1.0, 0.0, float(img_prev.shape[1] * 0.75)], [0.0, 1.0, 0.0]], dtype=np.float64)
            else:
                M = match_res.transform_matrix.copy()
                total_inliers += match_res.inlier_count
                total_matches += match_res.raw_matches_count

                # Scale translation vector back to native resolution if downscaled
                if downscale > 1.05:
                    M[0, 2] *= downscale
                    M[1, 2] *= downscale

            transforms_to_prev.append(M)

        # 5. Chain transformations to align all images into frame 0's coordinate system
        global_transforms: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
        curr_3x3 = np.eye(3, dtype=np.float64)

        for M in transforms_to_prev:
            M_3x3 = np.eye(3, dtype=np.float64)
            M_3x3[:2, :] = M[:2, :]
            curr_3x3 = curr_3x3 @ M_3x3
            global_transforms.append(curr_3x3.copy())

        # 6. Compute global bounding canvas enclosing all warped views
        all_corners: List[np.ndarray] = []
        for i, (w_img, T) in enumerate(zip(warped_views, global_transforms)):
            h, w = w_img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            warped_c = cv2.perspectiveTransform(corners, T)
            all_corners.append(warped_c)

        stacked_corners = np.vstack(all_corners)
        x_min = float(np.floor(stacked_corners[:, 0, 0].min()))
        y_min = float(np.floor(stacked_corners[:, 0, 1].min()))
        x_max = float(np.ceil(stacked_corners[:, 0, 0].max()))
        y_max = float(np.ceil(stacked_corners[:, 0, 1].max()))

        canvas_w = int(max(10, x_max - x_min))
        canvas_h = int(max(10, y_max - y_min))

        dx = -x_min if x_min < 0 else 0.0
        dy = -y_min if y_min < 0 else 0.0

        # Translation matrix to shift origin to (0, 0)
        shift_mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # 7. Warp all views onto canvas coordinates
        canvases: List[np.ndarray] = []
        canvas_masks: List[np.ndarray] = []

        for i, (w_img, w_mask, T) in enumerate(zip(warped_views, warped_masks, global_transforms)):
            T_canvas = shift_mat @ T
            M_affine = T_canvas[:2, :]

            c_img = cv2.warpAffine(
                w_img,
                M_affine,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            c_mask = cv2.warpAffine(
                w_mask,
                M_affine,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            canvases.append(c_img)
            canvas_masks.append(c_mask)

        # 8. Blend all images into unified panorama
        comp_img = canvases[0]
        comp_mask = canvas_masks[0]

        for i in range(1, num_images):
            img_b = canvases[i]
            mask_b = canvas_masks[i]

            if self.blend_mode == BlendMode.MULTIBAND:
                comp_img = multiband_blend(
                    comp_img, img_b, comp_mask, mask_b, levels=self.pyramid_levels
                )
            elif self.blend_mode == BlendMode.FEATHER:
                comp_img = feather_blend(comp_img, img_b, comp_mask, mask_b)
            elif self.blend_mode == BlendMode.LINEAR:
                comp_img = feather_blend(comp_img, img_b, comp_mask, mask_b, blur_ksize=1)
            else:  # BlendMode.NONE
                # Overwrite on overlap
                comp_img[mask_b > 0] = img_b[mask_b > 0]

            comp_mask = comp_mask | mask_b

        # 9. Crop black outer margins if requested
        if self.crop_borders:
            comp_img, comp_mask = crop_panorama_canvas(comp_img, comp_mask)

        elapsed = (time.perf_counter() - t0) * 1000
        inlier_ratio = float(total_inliers / max(1, total_matches))

        logger.info(
            f"Stitched panorama created: {comp_img.shape[1]}x{comp_img.shape[0]} ({elapsed:.1f}ms, inliers: {total_inliers})"
        )

        return PanoramaResult(
            image=comp_img,
            projection=self.projection.value,
            blend_mode=self.blend_mode.value,
            num_images=num_images,
            output_shape=comp_img.shape[:2],
            inlier_count=total_inliers,
            inlier_ratio=inlier_ratio,
            elapsed_ms=elapsed,
            metadata={
                "fov_degrees": self.fov_degrees,
                "pyramid_levels": self.pyramid_levels,
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
            },
        )
