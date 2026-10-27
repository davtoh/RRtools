"""
Hardware Detection and Acceleration Management for RRtools.

Provides cross-platform GPU discovery, ONNX Runtime execution provider selection,
and graceful CPU fallbacks compliant with the davtoh ecosystem standards.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("RRtoolbox.hardware")


@dataclass
class HardwareProfile:
    """Represents detected compute capabilities and recommended acceleration."""
    has_gpu: bool = False
    device_name: str = "CPU"
    vram_mb: int = 0
    cuda_driver_version: str = ""
    cuda_available_torch: bool = False
    cuda_available_cv2: bool = False
    onnx_providers: List[str] = field(default_factory=list)
    recommended_device: str = "cpu"
    recommended_provider: str = "CPUExecutionProvider"

    def summary(self) -> str:
        gpu_stat = f"{self.device_name} ({self.vram_mb} MB VRAM)" if self.has_gpu else "None (CPU only)"
        return (
            f"Hardware Profile:\n"
            f"  - Detected GPU: {gpu_stat}\n"
            f"  - CUDA Driver: {self.cuda_driver_version or 'N/A'}\n"
            f"  - PyTorch CUDA: {self.cuda_available_torch}\n"
            f"  - OpenCV CUDA: {self.cuda_available_cv2}\n"
            f"  - ONNX Providers: {', '.join(self.onnx_providers) or 'None'}\n"
            f"  - Recommended Device: {self.recommended_device.upper()}\n"
            f"  - Active Provider: {self.recommended_provider}"
        )


def _probe_nvidia_smi() -> tuple[bool, str, int, str]:
    """Query nvidia-smi for physical GPU presence, name, VRAM, and driver version."""
    if not shutil.which("nvidia-smi"):
        return False, "CPU", 0, ""

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
            vram = int(float(parts[1])) if len(parts) > 1 and parts[1].isdigit() else 0
            driver = parts[2] if len(parts) > 2 else ""
            return True, name, vram, driver
    except Exception as e:
        logger.debug(f"nvidia-smi probe failed: {e}")

    return False, "CPU", 0, ""


def detect_hardware() -> HardwareProfile:
    """
    Detect system compute capabilities and determine the optimal acceleration backend.
    
    Returns:
        HardwareProfile with detected devices and recommended settings.
    """
    has_phys_gpu, gpu_name, vram_mb, driver_ver = _probe_nvidia_smi()

    # Check PyTorch CUDA availability if installed
    cuda_torch = False
    try:
        import torch
        cuda_torch = torch.cuda.is_available()
    except ImportError:
        pass

    # Check OpenCV CUDA support
    cuda_cv2 = False
    try:
        import cv2
        if hasattr(cv2, "cuda"):
            cuda_cv2 = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except ImportError:
        pass

    # Check ONNX Runtime execution providers
    onnx_providers: List[str] = []
    try:
        import onnxruntime as ort
        onnx_providers = ort.get_available_providers()
    except ImportError:
        pass

    # Determine recommended acceleration strategy
    recommended_device = "cpu"
    recommended_provider = "CPUExecutionProvider"

    if "CUDAExecutionProvider" in onnx_providers:
        recommended_device = "cuda"
        recommended_provider = "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in onnx_providers:
        recommended_device = "directml"
        recommended_provider = "DmlExecutionProvider"
    elif cuda_torch:
        recommended_device = "cuda"
        recommended_provider = "CPUExecutionProvider"
    elif has_phys_gpu:
        # Physical GPU present, but ONNX GPU runtime not yet bound to CUDA
        recommended_device = "cpu"
        recommended_provider = "CPUExecutionProvider"

    return HardwareProfile(
        has_gpu=has_phys_gpu,
        device_name=gpu_name,
        vram_mb=vram_mb,
        cuda_driver_version=driver_ver,
        cuda_available_torch=cuda_torch,
        cuda_available_cv2=cuda_cv2,
        onnx_providers=onnx_providers,
        recommended_device=recommended_device,
        recommended_provider=recommended_provider,
    )


def get_onnx_execution_providers(preferred: Optional[str] = None) -> List[str]:
    """
    Get ordered execution providers for ONNX Runtime with automatic CPU fallback.
    
    Args:
        preferred: Optional preferred provider (e.g. 'CUDAExecutionProvider').
        
    Returns:
        Ordered list of available providers, always ending with 'CPUExecutionProvider'.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    providers: List[str] = []
    if preferred and preferred in available:
        providers.append(preferred)

    # Standard priority order
    for candidate in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]:
        if candidate in available and candidate not in providers:
            providers.append(candidate)

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    return providers


def configure_opencv_dnn(net: Any, prefer_cuda: bool = True) -> bool:
    """
    Configure OpenCV DNN net for CUDA backend if available, with CPU fallback.
    
    Returns:
        True if CUDA backend was configured, False if falling back to CPU.
    """
    import cv2
    if prefer_cuda and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("OpenCV DNN configured with CUDA acceleration.")
            return True
        except Exception as e:
            logger.warning(f"Failed to set OpenCV DNN CUDA backend: {e}. Falling back to CPU.")

    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass
    return False
