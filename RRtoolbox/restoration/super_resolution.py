"""
Super-Resolution and Multi-Frame Super-Image Enhancement Engine for RRtools.

Provides:
- MultiFrameSuperResolution: Sub-pixel registration and bilateral detail boost.
- SingleImageSuperResolution: Neural upscaler with edge-directed fallback.
- SuperResolutionEngine: Unified orchestrator managing single/burst modes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from RRtoolbox.hardware import (
    configure_opencv_dnn,
    detect_hardware,
    get_onnx_execution_providers,
)
from RRtoolbox.restoration.registration import (
    align_and_fuse_frames,
    align_ecc_subpixel,
)

logger = logging.getLogger("RRtoolbox.restoration.super_resolution")


class Backend(str, Enum):
    AUTO = "auto"
    EDGE_DIRECTED = "edge_directed"
    DNN_SUPERRES = "dnn_superres"
    ONNX = "onnx"


@dataclass
class SuperResolutionResult:
    """Result container for super-resolution processing."""
    image: np.ndarray
    scale: int
    mode: str
    backend: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    sharpness_input: float
    sharpness_output: float
    sharpness_gain_pct: float
    elapsed_ms: float
    metadata: dict

    def save(self, destination: Union[str, Path]) -> Path:
        """Save restored super-image to disk."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), self.image)
        logger.info(f"Saved super-resolution image to: {dest} ({self.output_shape[1]}x{self.output_shape[0]})")
        return dest


def compute_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness metric using variance of Laplacian."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_contrast(image: np.ndarray) -> float:
    """Compute standard deviation of pixel intensities as an RMS contrast metric."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(gray.std())


def _load_engine_images(
    inputs: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]]
) -> Tuple[List[np.ndarray], List[str]]:
    """Parse, load, and validate input image list from paths or arrays."""
    raw_items = inputs if isinstance(inputs, list) else [inputs]
    image_list, source_paths = [], []
    for item in raw_items:
        if isinstance(item, (str, Path)):
            p = Path(item)
            if not p.exists():
                raise FileNotFoundError(f"Input image file not found: {p}")
            arr = cv2.imread(str(p))
            if arr is None:
                raise ValueError(f"Failed to decode image from: {p}")
            image_list.append(arr)
            source_paths.append(str(p))
        elif isinstance(item, np.ndarray):
            image_list.append(item)
            source_paths.append("<memory_array>")
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")
    if not image_list:
        raise ValueError("No valid images supplied for restoration.")
    return image_list, source_paths


def _dispatch_restoration(
    engine: SuperResolutionEngine,
    image_list: List[np.ndarray],
    ref_img: np.ndarray,
    scale: int,
    ref_idx: int,
) -> Tuple[np.ndarray, str, str]:
    """Route processing to multi-frame burst or single-image engine."""
    if len(image_list) > 1:
        if scale != engine.multi_sr.scale:
            engine.multi_sr = MultiFrameSuperResolution(scale=scale)
        return engine.multi_sr.process(image_list, reference_idx=ref_idx), "mfsr", "multi_frame_ecc_lanczos4"

    if scale != engine.single_sr.scale:
        engine.single_sr = SingleImageSuperResolution(
            scale=scale, backend=engine.single_sr.backend,
            model_name=engine.single_sr.model_name, model_path=engine.single_sr.model_path,
        )
    return engine.single_sr.process(ref_img), "sisr", engine.single_sr.active_backend


class MultiFrameSuperResolution:
    """Multi-Frame Super-Resolution (MFSR) engine."""

    def __init__(
        self,
        scale: int = 2,
        motion_model: int = cv2.MOTION_EUCLIDEAN,
        max_ecc_iterations: int = 50,
        ecc_eps: float = 1e-4,
        detail_boost: float = 1.2,
    ):
        self.scale = scale
        self.motion_model = motion_model
        self.max_ecc_iterations = max_ecc_iterations
        self.ecc_eps = ecc_eps
        self.detail_boost = detail_boost

    def _align_ecc(self, ref_gray: np.ndarray, target_gray: np.ndarray) -> np.ndarray:
        """Estimate sub-pixel 2D transformation matrix using ECC or Phase Correlation."""
        return align_ecc_subpixel(
            ref_gray, target_gray, self.motion_model, self.max_ecc_iterations, self.ecc_eps
        )

    def process(self, images: List[np.ndarray], reference_idx: int = 0) -> np.ndarray:
        """Fuse multiple frames into a super-resolved image."""
        if not images:
            raise ValueError("No images provided for multi-frame super-resolution.")
        if len(images) == 1:
            h, w = images[0].shape[:2]
            return cv2.resize(images[0], (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)

        return align_and_fuse_frames(
            images=images, reference_idx=reference_idx, scale=self.scale,
            motion_model=self.motion_model, max_ecc_iterations=self.max_ecc_iterations,
            ecc_eps=self.ecc_eps, detail_boost=self.detail_boost,
        )


class SingleImageSuperResolution:
    """Single-Image Super-Resolution (SISR) engine."""

    def __init__(
        self,
        scale: int = 2,
        backend: Backend = Backend.AUTO,
        model_name: str = "fsrcnn",
        model_path: Optional[Union[str, Path]] = None,
        tile_size: int = 1024,
    ):
        self.scale = scale
        self.backend = backend
        self.model_name = model_name.lower()
        self.model_path = Path(model_path) if model_path else None
        self.tile_size = tile_size
        self._dnn_sr = None
        self._onnx_session = None
        self._resolved_backend = Backend.EDGE_DIRECTED
        self._initialize_backend()

    def _initialize_backend(self):
        """Detect and load the appropriate super-resolution model/backend."""
        if self.backend == Backend.ONNX and self.model_path and self.model_path.exists():
            try:
                import onnxruntime as ort
                providers = get_onnx_execution_providers()
                self._onnx_session = ort.InferenceSession(str(self.model_path), providers=providers)
                self._resolved_backend = Backend.ONNX
                return
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX SR backend: {e}. Falling back.")

        if (self.backend in [Backend.AUTO, Backend.DNN_SUPERRES]) and hasattr(cv2, "dnn_superres"):
            if self.model_path and self.model_path.exists():
                try:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(str(self.model_path))
                    sr.setModel(self.model_name, self.scale)
                    configure_opencv_dnn(sr)
                    self._dnn_sr = sr
                    self._resolved_backend = Backend.DNN_SUPERRES
                    return
                except Exception as e:
                    logger.warning(f"Failed to load OpenCV dnn_superres model: {e}. Using edge-directed fallback.")

        self._resolved_backend = Backend.EDGE_DIRECTED

    @property
    def active_backend(self) -> str:
        return self._resolved_backend.value

    def process(self, image: np.ndarray) -> np.ndarray:
        """Upscale image by self.scale factor."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        h, w = image.shape[:2]
        target_w, target_h = w * self.scale, h * self.scale

        if self._resolved_backend == Backend.DNN_SUPERRES and self._dnn_sr is not None:
            try:
                return self._dnn_sr.upsample(image)
            except Exception as e:
                logger.warning(f"DNN upsample failed: {e}. Falling back to edge-directed.")

        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        smooth = cv2.bilateralFilter(upscaled, 7, 50, 50)
        residual = cv2.subtract(upscaled, smooth)
        return cv2.addWeighted(upscaled, 1.0, residual, 1.15, 0)


class SuperResolutionEngine:
    """Unified Super-Resolution Engine coordinating single-image and multi-frame burst restoration."""

    def __init__(
        self,
        scale: int = 2,
        backend: Backend = Backend.AUTO,
        model_name: str = "fsrcnn",
        model_path: Optional[Union[str, Path]] = None,
    ):
        self.scale = scale
        self.hardware = detect_hardware()
        self.single_sr = SingleImageSuperResolution(
            scale=scale, backend=backend, model_name=model_name, model_path=model_path,
        )
        self.multi_sr = MultiFrameSuperResolution(scale=scale)

    def restore(
        self,
        inputs: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
        scale: Optional[int] = None,
        reference_idx: int = 0,
    ) -> SuperResolutionResult:
        """Execute super-resolution restoration on single image or list of images."""
        t0 = time.time()
        active_scale = scale or self.scale
        image_list, source_paths = _load_engine_images(inputs)
        ref_img = image_list[reference_idx if reference_idx < len(image_list) else 0]
        h_in, w_in = ref_img.shape[:2]

        result_img, mode, used_backend = _dispatch_restoration(
            self, image_list, ref_img, active_scale, reference_idx
        )

        h_out, w_out = result_img.shape[:2]
        sharp_out = compute_sharpness(result_img)
        baseline_bicubic = cv2.resize(ref_img, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
        sharp_base = compute_sharpness(baseline_bicubic)
        gain_pct = ((sharp_out - sharp_base) / max(sharp_base, 1e-5)) * 100.0

        return SuperResolutionResult(
            image=result_img, scale=active_scale, mode=mode, backend=used_backend,
            input_shape=(h_in, w_in), output_shape=(h_out, w_out),
            sharpness_input=sharp_base, sharpness_output=sharp_out,
            sharpness_gain_pct=gain_pct, elapsed_ms=(time.time() - t0) * 1000.0,
            metadata={
                "source_count": len(image_list), "source_paths": source_paths,
                "hardware_device": self.hardware.recommended_device,
                "contrast_input": compute_contrast(ref_img),
                "contrast_output": compute_contrast(result_img),
            },
        )


def main():
    """Command-line interface for RRtools Super-Resolution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RRtools Super-Resolution & Multi-Frame Super-Image Enhancement"
    )
    parser.add_argument("images", nargs="+", help="Input image file paths or glob")
    parser.add_argument("-s", "--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor (default: 2)")
    parser.add_argument("-o", "--output", type=str, default="super_restored.jpg", help="Output file path")
    parser.add_argument("-b", "--backend", type=str, default="auto", choices=["auto", "edge_directed", "dnn_superres", "onnx"])
    parser.add_argument("-m", "--model-path", type=str, default=None, help="Path to optional ONNX or OpenCV superres model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    engine = SuperResolutionEngine(
        scale=args.scale, backend=Backend(args.backend), model_path=args.model_path,
    )
    print(f"Executing Super-Resolution (Scale: {args.scale}x, Inputs: {len(args.images)})...")
    res = engine.restore(args.images, scale=args.scale)
    saved_path = res.save(args.output)
    print(f"Result: {res.mode.upper()} ({res.backend}) | Output: {res.output_shape} | Sharpness: {res.sharpness_output:.2f} (+{res.sharpness_gain_pct:.1f}%) | Saved: {saved_path}")


__all__ = [
    "Backend",
    "SuperResolutionResult",
    "compute_sharpness",
    "compute_contrast",
    "MultiFrameSuperResolution",
    "SingleImageSuperResolution",
    "SuperResolutionEngine",
    "align_ecc_subpixel",
    "main",
]


if __name__ == "__main__":
    main()
