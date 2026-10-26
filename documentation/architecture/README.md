# RRtoolbox Architecture & UML Diagrams

This directory contains the system architecture specifications, modern text-as-code class diagrams (Mermaid.js), and legacy diagram exports for **RRtools**.

---

## 1. High-Level System Architecture

The core pipeline in RRtools orchestrates feature detection, geometric matching, homography estimation, luminance segmentation, and layered alpha blending:

```mermaid
classDiagram
    direction TB

    class Restorer {
        +Image baseImage
        +list images
        +dict features
        +list homographies
        +restore()
        +step_detect_features()
        +step_match_features()
        +step_estimate_homography()
        +step_blend_canvas()
    }

    class RetinalRestore {
        +float glareThreshold
        +simulateLens()
        +get_bright_alpha()
        +retinal_mask()
        +get_layered_alpha()
    }

    class Feature {
        +string detector_type
        +bool asift
        +bool flann
        +detect(image)
        +match(desc1, desc2)
    }

    class MemoizedDict {
        +string path
        +dict cache
        +get(key)
        +set(key, value)
        +clear()
    }

    class MatchExplorer {
        +Image im1
        +Image im2
        +list inliers
        +plot_matches()
    }

    Restorer <|-- RetinalRestore : specializes for fundus
    Restorer o-- Feature : extracts keypoints
    Restorer o-- MemoizedDict : caches descriptors
    Restorer o-- MatchExplorer : visualizes inliers
```

---

## 2. Feature Detection & Affine Simulation Subsystem

RRtools provides robust invariant matching across varying tilts and angles using ASIFT (Affine SIFT) combined with the FLANN k-d tree search:

```mermaid
flowchart TD
    In["Input Images (Exposures 1 & 2)"] --> Pre["Color Conversion & CLAHE"]
    Pre --> Affine["ASIFT Tilt & Rotation Projections"]
    
    Affine --> Det{"Detector Strategy"}
    Det -->|SIFT| S["Scale-Invariant Feature Transform"]
    Det -->|SURF| U["Speeded-Up Robust Features"]
    Det -->|ORB| O["Oriented FAST and Rotated BRIEF"]
    Det -->|BRISK| B["Binary Robust Invariant Scalable Keypoints"]

    S --> Matcher["FLANN Matcher (k-d Tree)"]
    U --> Matcher
    O --> Matcher
    B --> Matcher

    Matcher --> Lowe["Lowe's Ratio Test (< 0.75)"]
    Lowe --> RANSAC["RANSAC Homography Matrix (3x3)"]
    RANSAC --> Rect["Quadrilateral Rectangularity Validation (> 0.50)"]
```

---

## 3. Retinal Glare Elimination & Layered Fusion

In clinical fundus photography, flash glare washes out critical diagnostic structures. The `RetinalRestore` engine dynamically isolates the reflection and substitutes valid anatomical structures from secondary perspectives:

```mermaid
flowchart LR
    A["Overexposed Fundus 1"] --> B["Luminance Analysis\n(get_bright_alpha)"]
    B --> C["Morphological Watershed\n(Glare Mask)"]

    D["Secondary Angle Fundus 2"] --> E["Perspective Warp\n(to Base Homography)"]

    C --> F["Layered Alpha Map\n(get_layered_alpha)"]
    E --> F
    A --> F

    F --> G["Seamless Vascular Composite"]
    G --> H["Optical Lens Aperture\n(simulateLens)"]
    H --> Out["Clinical Restored Presentation"]
```

---

## 4. Caching & Memory Management Subsystem

To optimize heavy computer vision routines and avoid redundant descriptor calculations:

```mermaid
classDiagram
    class ObjectGetter {
        +get_object(key)
    }

    class MemoizedDict {
        +string directory
        +serialize()
        +deserialize()
        +integrity_check()
    }

    class DirectoryManager {
        +string base_dir
        +get_path()
        +cleanup()
    }

    ObjectGetter <|-- MemoizedDict
    MemoizedDict o-- DirectoryManager
```

---

## 5. Legacy Diagrams Reference

This directory also retains the historical reference diagram files:

* [RRtoolbox.svg](file:///c:/dev/projects/products/rrtools/documentation/architecture/RRtoolbox.svg): Monolithic 25,871 × 6,974 px diagram generated via PyCharm's yFiles (ySVG) export engine.
* [RRtoolbox.uml](file:///c:/dev/projects/products/rrtools/documentation/architecture/RRtoolbox.uml): PyCharm's XML UML diagram file.
* [legacy_staruml_html/](file:///c:/dev/projects/products/rrtools/documentation/architecture/legacy_staruml_html): Historical StarUML HTML documentation export for the "Object counter" and "MOSSE" milestones.

### How to Regenerate Class Diagrams with `pyreverse`
Instead of manually drawing diagrams, generate fresh, code-synchronized UML using `pyreverse` (included with `pylint`):

```bash
# Generate PlantUML format
pyreverse -o puml -p RRtoolbox RRtoolbox/

# Generate Mermaid format
pyreverse -o mmd -p RRtoolbox RRtoolbox/
```
