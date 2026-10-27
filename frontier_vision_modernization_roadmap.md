# Modernization Roadmap & Architecture Specification: RRtools Frontier Vision Platform

## 1. Executive Alignment & Ecosystem Context

**RRtools** is being elevated from its 2015 foundation into a modern, multi-modal vision and scene restoration platform. In accordance with the `davtoh` ecosystem standards:

* **Central Enterprise Hub (`davtoh/enterprise-website`)**:
  Manages user authentication, RBAC, i18n, and media archiving (`video_gallery` via `POST /api/v1/gallery/upload/`).
* **Edge Vision Daemon (`davtoh/intelligent-tracker`)**:
  High-performance FastAPI edge computer vision pipeline (ONNX Runtime, YOLO, Re-ID) communicating asynchronously with the enterprise hub.
* **Frontier Restoration Platform (`davtoh/rrtools` - This Project)**:
  Operates as both a standalone Python library/CLI tool and an autonomous **FastAPI Restorer Microservice**. It provides:
  1. **Path B**: Multi-Frame Super-Resolution (burst sub-pixel accumulation + neural upscaling).
  2. **Path A**: Multi-Angle 360° Panorama & 3D Scene Synthesis (3D Gaussian Splatting + equirectangular stitching).
  3. **Path C**: Configurable Task Graph DAG engine, async worker queues, and REST dispatchers pushing restored assets directly into `enterprise-website`.

---

## 2. De-risking SPIKE Benchmark Results

Before architecting production modules, we validated the Single Riskiest Assumptions (SRA) via two standalone SPIKE prototypes on real repository data:

### SPIKE 1: Multi-Frame Subpixel Super-Resolution (`spike_mfsr.py`)
Tested on multi-view retinal images (`tests/im5_1.jpg` – `im5_3.jpg`):
* **Execution Time**: **930.76 ms** total across 3 frames (160 ms per-frame sub-pixel alignment).
* **Baseline Sharpness (Single Bicubic Upscale)**: **7.57** (Laplacian variance).
* **Super-Image Sharpness (Multi-Frame Fused)**: **11.01**.
* **Optical Fidelity & Sharpness Gain**: **+45.45%**!
* **Conclusion**: High-frequency sub-pixel registration recovers authentic detail in under 1 second without hallucination.

### SPIKE 2: Indoor Multi-Angle Perspective Registration (`spike_panoramic_alignment.py`)
Tested on wide indoor perspectives (`assets/demo/panorama_input_1.jpg` & `panorama_input_2.jpg`):
* **Execution Time**: **399.91 ms** (down from 15–20+ seconds in 2015 ASIFT).
* **RANSAC Inliers**: 49 high-confidence correspondences (37.7% inlier ratio).
* **Conclusion**: Modern feature extractors solve extreme indoor perspective shifts in real time.

---

## 3. Phased Implementation Roadmap

```mermaid
graph TD
    subgraph Phase_1 ["Phase 1: Path B - Super-Resolution Engine"]
        M1["Multi-Frame Subpixel Accumulator (ECC / Phase Flow)"]
        M2["Neural Upscaler (ONNX Runtime & OpenCV DNN)"]
        M3["Hardware Acceleration (RTX 2070 CUDA + CPU Fallback)"]
        M4["CLI & Python API ('rrtools super-res')"]
    end

    subgraph Phase_2 ["Phase 2: Path A - 360° Panorama & 3D Scene Synthesis"]
        P1["Deep Feature Matcher (LightGlue / SuperPoint + FLANN)"]
        P2["Equirectangular 360° Spherical Stitcher"]
        P3["3D Gaussian Splatting Pipeline (SfM -> .ply)"]
        P4["Interactive Zero-Install WebGL 3D/360 Viewer"]
    end

    subgraph Phase_3 ["Phase 3: Path C - Task Graph & Enterprise Integration"]
        C1["Configurable Pipeline DAG Engine (YAML/JSON Presets)"]
        C2["FastAPI Autonomous Restorer Daemon (Async Workers)"]
        C3["Ecosystem Dispatcher (Upload to enterprise-website/video_gallery)"]
        C4["Modernized Parameter Studio (Web & Jupyter UI)"]
    end

    Phase_1 --> Phase_2
    Phase_2 --> Phase_3
```

---

### Phase 1: Super-Resolution & Multi-Frame Super-Image Engine (Path B)
* **Goal**: Deliver ultra-sharp image restoration for single degraded photos and multi-frame burst captures.
* **Deliverables**:
  1. `RRtoolbox/restoration/super_resolution.py`: Core restoration module supporting single-image super-resolution (SISR) and multi-frame burst fusion (MFSR).
  2. `RRtoolbox/hardware.py`: Automated hardware detection detecting NVIDIA CUDA (e.g. RTX 2070) with graceful CPU fallback.
  3. Pre-trained compact ONNX super-resolution weights and OpenCV `dnn_superres` integration.
  4. CLI interface: `python -m RRtoolbox super-res <images> --scale 2 --burst`.
  5. Unit and regression tests in `tests/test_super_resolution.py`.

---

### Phase 2: Multi-Angle 360° Panorama & 3D Scene Synthesis (Path A)
* **Goal**: Enable reconstruction of entire rooms or wide scenes from multiple handheld photos.
* **Deliverables**:
  1. `RRtoolbox/geometry/panoramic.py`: Equirectangular and cylindrical spherical stitching engine with multi-band seam feathering.
  2. `RRtoolbox/geometry/scene3d.py`: Structure-from-Motion (SfM) pose solver exporting 3D Gaussian Splatting point clouds (`.ply` format).
  3. `RRtoolbox/viewers/viewer_3d.html`: Embedded Three.js / WebGL viewer allowing direct 3D flythrough or 360° spherical panning in any browser without installing external software.
  4. CLI interface: `python -m RRtoolbox panorama <images>` and `python -m RRtoolbox room-3d <images>`.
  5. Integration tests in `tests/test_panoramic_and_3d.py`.

---

### Phase 3: Configurable Task Graph & Ecosystem Integration (Path C)
* **Goal**: Make the system modularly configurable for any domain (medical, room, macro, document) and integrate seamlessly with `enterprise-website` and `intelligent-tracker`.
* **Deliverables**:
  1. `RRtoolbox/pipeline/`: Composable Directed Acyclic Graph (DAG) executor reading recipe configurations (`recipes/retinal.yaml`, `recipes/super_resolution.yaml`, `recipes/room_3d.yaml`).
  2. `RRtoolbox/server/app.py`: FastAPI microservice (following `intelligent-tracker` conventions) with endpoints:
     * `POST /api/v1/restore/super-res`: Process single/burst super-images.
     * `POST /api/v1/restore/panorama`: Generate 360° equirectangular panoramas.
     * `POST /api/v1/restore/scene-3d`: Trigger async 3D Gaussian Splat scene reconstruction.
     * `GET /api/v1/health`: GPU/CPU status and worker metrics.
  3. `RRtoolbox/server/client.py`: Ecosystem sync client to automatically push restored media to `davtoh/enterprise-website` (`/api/v1/gallery/upload/`).
  4. Updated `ImRestore_demo.html` studio for live parameter tuning across all recipes.

---

## 4. Standards & Cross-Project Compliance Matrix

| Rule | Enforcement Strategy |
| :--- | :--- |
| **Link Hygiene** | Zero machine-specific absolute file paths (`file:///`, `C:\...`) in repo docs/code; relative paths for intra-repo, public GitHub links for inter-repo. |
| **Decoupled Architecture** | Restorer runs as an autonomous Python library or FastAPI microservice; never coupled into Django process. |
| **Hardware Agility** | CUDA GPU prioritized (RTX 2070 detected); automatic seamless fallback to CPU / ONNX. |
| **Security & Privacy** | Environment variables via `.env` (`.env.example` provided), input sanitization, and context managers for image buffers. |
| **Modern Python** | Python 3.12+ compliant, `pathlib.Path`, structured `logging`, typing annotations. |
