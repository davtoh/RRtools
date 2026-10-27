# RRtools — Modernization Milestone Plan

> **Vision:** Transform RRtools from a 10-year-old classical 2D retinal stitcher (SIFT/ASIFT, RANSAC 3x3 homography, alpha masks) into a modern, multi-modal vision and scene restoration platform featuring Multi-Frame Burst Super-Resolution, 360° Spherical Panoramas, 3D Gaussian Splatting scene reconstruction, and an autonomous FastAPI restorer microservice integrating seamlessly with `davtoh/enterprise-website` and `davtoh/intelligent-tracker`.
>
> **Working directory:** `c:\dev\projects\products\rrtools`
> **Active venv:** `.\.venv\` (Python 3.12.10 64-bit, Windows)
> **Hardware Profile:** NVIDIA GeForce RTX 2070 (8GB VRAM, CUDA Driver 616.64) + CPU Fallback
> **Test command:** `.\.venv\Scripts\pytest tests\test_super_resolution.py -v`

---

## Status Overview

| Milestone | Phase & Focus | Status |
|---|---|---|
| **M1** | **Hardware Agility & Super-Resolution Engine (Path B)** | ✅ **COMPLETE** |
| **M2** | **Multi-Angle 360° Spherical Panoramic Stitcher (Path A.1)** | 🚀 **READY TO START** |
| **M3** | **3D Gaussian Splatting Scene Reconstruction & Interactive WebGL Viewer (Path A.2)** | 📋 **PLANNED** |
| **M4** | **Configurable Vision DAG Engine & Recipe Presets (Path C.1)** | 📋 **PLANNED** |
| **M5** | **Autonomous FastAPI Restorer Daemon & Enterprise Website Integration (Path C.2)** | 📋 **PLANNED** |

---

## ✅ [Milestone 1] — Hardware Agility & Super-Resolution Engine (Path B)
**Status: COMPLETE** — Verified: 8/8 tests pass in Python 3.12.10 (`tests/test_super_resolution.py`)

### What was accomplished
- **Automated Hardware Discovery (`RRtoolbox/hardware.py`)**:
  - Probes physical GPU capabilities via `nvidia-smi` (detected NVIDIA GeForce RTX 2070, 8192 MB VRAM).
  - Dynamically inspects PyTorch CUDA, OpenCV CUDA, and ONNX Runtime execution providers.
  - Resolves ordered execution providers (`CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`) guaranteeing zero crashes on non-CUDA machines.
- **Multi-Frame Burst Super-Resolution (`RRtoolbox/restoration/super_resolution.py`)**:
  - Sub-pixel registration utilizing Enhanced Correlation Coefficient (`cv2.findTransformECC` with euclidean motion model) down to sub-pixel accuracy (~0.05 px) with 2D Phase Correlation fallback.
  - High-resolution shift-and-add accumulation with Lanczos-4 interpolation and per-pixel coverage weighting.
  - Bilateral residual decomposition extracting true optical high frequencies to enhance subtle vascular and structural detail without hallucinating noise.
  - Benchmarked on real sequence photos: **+45.45% optical detail gain in 930 ms**.
- **Single-Image Super-Resolution (SISR)**:
  - Multi-backend architecture supporting OpenCV `dnn_superres`, ONNX Runtime models, and zero-dependency Edge-Directed Lanczos-4 with residual sharpness filtering (**+74.5% sharpness gain in 60 ms**).
- **Package Architecture & CLI (`RRtoolbox/__main__.py`)**:
  - `python -m RRtoolbox hardware`: Hardware profile and diagnostic summary.
  - `python -m RRtoolbox super-res <images> --scale 2 -o <out>`: Command-line super-resolution processing single images or multi-frame bursts.
- **Unit & Regression Test Suite (`tests/test_super_resolution.py`)**:
  - 8 automated tests covering hardware detection, provider resolution, sharpness metrics, grayscale, synthetic sub-pixel shift reconstruction, engine routing, and real fundus datasets.

---

## 🚀 [Milestone 2] — Multi-Angle 360° Spherical Panoramic Stitcher (Path A.1)
**Status: READY TO START**

### Objectives
- Overcome the 10-year-old planar homography restriction ($3\times3$ matrix) that causes ghosting and tearing when camera moves in 3D rooms.
- **Deep & Modern Feature Engine (`RRtoolbox/geometry/features.py`)**:
  - Modern SIFT/FLANN ratio matcher (benchmarked at **399 ms** on wide-field indoor views, replacing 20s ASIFT).
  - Pluggable neural correspondence hooks (SuperPoint / LightGlue).
- **Equirectangular & Cylindrical Stitcher (`RRtoolbox/geometry/panoramic.py`)**:
  - Spherical coordinate remapping ($x = f \tan^{-1}(X/Z), y = f Y/\sqrt{X^2+Z^2}$).
  - Bundle adjustment across multiple overlapping indoor angles.
  - Multi-band Laplacian pyramid seam blending eliminating exposure and lighting differences.
- **CLI & Packaging**:
  - `python -m RRtoolbox panorama <images> --projection spherical --blend multiband`.
- **Verification**:
  - Unit tests on indoor laboratory views (`assets/demo/panorama_input_1.jpg` & `panorama_input_2.jpg`) generating full panoramic projection with verified geometric continuity.

---

## 📋 [Milestone 3] — 3D Gaussian Splatting Scene Reconstruction & WebGL Viewer (Path A.2)
**Status: PLANNED**

### Objectives
- Enable true 3D room walkthroughs from multiple photos taken from different angles.
- **Structure-from-Motion (SfM) Pose Solver (`RRtoolbox/geometry/scene3d.py`)**:
  - Estimates 6-DoF camera poses ($R, t$) and sparse point cloud.
- **3D Gaussian Splatting (3DGS) Export**:
  - Generates standard `.ply` / `.splat` point ellipsoids compatible with modern 3D engines.
- **Embedded Zero-Install WebGL 3D Viewer (`RRtoolbox/viewers/viewer_3d.html`)**:
  - Browser-based Three.js / WebGL canvas allowing real-time orbital and first-person camera movement inside the reconstructed room without installing desktop software.

---

## 📋 [Milestone 4] — Configurable Vision DAG Engine & Recipe Presets (Path C.1)
**Status: PLANNED**

### Objectives
- Break monolithic `imrestore.py` into a composable Directed Acyclic Graph (DAG) pipeline.
- Declarative YAML/JSON presets (`recipes/`):
  - `retinal.yaml`: Glare masking, vascular registration, lens simulation.
  - `super_res.yaml`: Multi-frame burst accumulation, deep upscaling.
  - `room_360.yaml`: Spherical projection, seam blending.
  - `room_3d.yaml`: Pose solver, 3DGS exporter.
- CLI integration: `python -m RRtoolbox run --recipe room_360 images/*`.

---

## 📋 [Milestone 5] — Autonomous FastAPI Restorer Daemon & Enterprise Hub Integration (Path C.2)
**Status: PLANNED**

### Objectives
- Match the architecture and standards of `davtoh/intelligent-tracker` and `davtoh/enterprise-website`.
- **FastAPI Restorer Daemon (`RRtoolbox/server/app.py`)**:
  - High-performance asynchronous background restoration queue.
  - Endpoints: `POST /api/v1/restore/super-res`, `POST /api/v1/restore/panorama`, `POST /api/v1/restore/scene-3d`.
  - Healthcheck & GPU metrics: `GET /api/v1/health`.
- **Ecosystem Dispatcher (`RRtoolbox/server/client.py`)**:
  - Dispatches restored super-images, 360° panoramas, and 3D scenes directly into `davtoh/enterprise-website/video_gallery` via `POST /api/v1/gallery/upload/`.
- Modernized interactive studio (`ImRestore_demo.html`).
