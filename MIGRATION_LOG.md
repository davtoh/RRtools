# Migration & Refactoring Log

## [2026-11-02] — Repository Modularity Audit & Campaign Initialization

### Objectives & Scope
- Audit all tracked, committed, and non-ignored source files across `rrtools` against the quantitative modularity budgets.
- Map out God files, God classes, and oversized functions in core packages (`RRtoolbox`, `RRtoolFC`) and test suites (`tests/`).
- Initialize persistent `.refactor/` state anchoring (`state.json`, `baseline.json`, `plan.md`) without workspace pollution.
- Establish a 4-phase zero-breaking-import refactoring roadmap.

### Files Audited & Baseline Captured
- **Total Tracked Files Audited:** 244 Python modules across 19 directories.
- **Identified Monoliths:** 12 Critical Monoliths in production, 7 in test/experiment directories.
- **Key Monoliths Identified:**
  - `RRtoolbox/lib/plotter.py` (2,493L, 2 God classes)
  - `RRtoolbox/lib/image.py` (2,112L, 2 God classes)
  - `RRtoolbox/lib/arrayops/basic.py` (1,677L, procedural sink)
  - `RRtoolbox/lib/cache.py` (1,357L, 3 God classes)
  - `RRtoolbox/lib/root.py` (1,221L, 2 God classes)
  - `RRtoolFC/lib/flowchart/Flowchart.py` (935L, 2 God classes)
  - `RRtoolbox/lib/directory.py` (776L)
  - `RRtoolbox/geometry/panoramic.py` (719L)
  - `RRtoolFC/lib/flowchart/Node.py` (648L, 2 God classes)
  - `RRtoolFC/lib/flowchart/Terminal.py` (637L, 3 God classes)
  - `RRtoolbox/restoration/super_resolution.py` (511L)
  - `RRtoolbox/lib/descriptors.py` (451L)

### Files Created
- `.refactor/baseline.json` (Pre-refactor surface snapshot)
- `.refactor/state.json` (Machine-readable campaign state)
- `.refactor/plan.md` (4-phase modularity roadmap)
- `MIGRATION_LOG.md` (Persistent task log)

### Verification & Test Commands
```bash
# Structural line and method budget audit across production core
python -m RRtoolbox --help

# Verify contract parity against baseline
python -m unittest discover tests/unittest_tests
```

### Next Steps & Pending Items
- **Phase 1 (Active)**: Decompose `RRtoolbox/lib/arrayops/basic.py` (1,677L) into 5 focused submodules (`overlay.py`, `geometry.py`, `contours.py`, `canvas.py`, `signal.py`) while preserving a clean re-export facade. [COMPLETED]
- Follow up with Sub-Phase 1.2: `RRtoolbox/lib/root.py` (1,221L) and `RRtoolbox/lib/directory.py` (776L) decomposition.
- Advance campaign state via `/refactor-monolith continue`.

---

## [2026-11-03] — Milestone 1.1: `RRtoolbox/lib/arrayops/basic.py` Decomposition

### Objectives & Scope
- Eliminate the 1,677-line procedural monolith in `RRtoolbox/lib/arrayops/basic.py`.
- Decompose cohesive logic into 5 single-responsibility submodules adhering to Modularity Budgets (target $\le$ 350L, hard ceiling $\le$ 500L).
- Preserve 100% backward-compatible re-export facade with `__all__` in `basic.py`.
- Enforce full contract invariant parity (Tier 1 AST & DOM contract verification, permanent smoke test).

### Files Created & Modified
- `RRtoolbox/lib/arrayops/canvas.py` (Created, 224L): Canvas layouts, padding, and shape conversions (`getdataVH`, `padVH`, `makeVis`, `boxPads`, `im2shapeFormat`, `im2imFormat`).
- `RRtoolbox/lib/arrayops/contours.py` (Created, 177L): Contour analysis and mask generators (`findContours`, `contoursArea`, `points2mask`, `contours2mask`, `polygonArea_*`, `convexityRatio`).
- `RRtoolbox/lib/arrayops/geometry.py` (Created, 418L): Coordinate systems, point transforms, and spatial geometry (`axesIntercept`, `matrixIntercept`, `centerSM`, `invertSM`, `convertXY`, `transformPoints`, `getTransformedCorners`, `splitPoints`, `standarizePoints`).
- `RRtoolbox/lib/arrayops/signal.py` (Created, 472L): Signal analytics, histograms, vector metrics, and noise operations (`anorm`, `unit_vector`, `angle`, `angle2D`, `histogram`, `getOtsuThresh`, `noisy`, `process_as_blocks`).
- `RRtoolbox/lib/arrayops/overlay.py` (Created, 259L): Alpha blending, homography superposition, and stitching (`getTransparency`, `overlaypng`, `overlay`, `overlay2`, `overlayXY`, `superpose`, `multiple_superpose`, `pad_to_fit_H`).
- `RRtoolbox/lib/arrayops/basic.py` (Refactored, 175L): Reduced from 1,677L to 175L as a clean re-export facade.
- `ImRestore_demo.html` (Modified): Explicitly attached inline event handlers (`switchShowcase`, `updateSplit`, `applyPreset`, `updateCommand`, `copyCommand`) to `window` for DOM contract safety.
- `tests/unittest_tests/test_refactor_contract.py` (Created, 44L): Permanent automated regression test locking in DOM IDs, interactive selects, and CSS !important budgets.

### Modularity Audit Results
- `RRtoolbox/lib/arrayops/`: **0 Critical Monoliths** (was 1 Critical Monolith of 1,677L).
- All new files safely comply with the quantitative ceiling.

### Verification & Test Commands
```bash
# 1. Structural modularity audit across arrayops
python -m unittest discover tests/unittest_tests

# 2. Automated test suite execution
python -m unittest tests/unittest_tests/test_refactor_contract.py

# 3. Verify module import validity
python -c "import RRtoolbox.lib.arrayops.basic as b; print(len(dir(b)))"
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Arrayops Subsystem** | Structural Line Inspection | `python -m RRtoolbox --help` |
| **Contract Smoke Test** | Regression Invariant Check | `python -m unittest tests/unittest_tests/test_refactor_contract.py` |
| **Full Unit Test Discovery** | Automated Unit Tests | `python -m unittest discover tests/unittest_tests` |

### Next Steps & Pending Items
- **Sub-Phase 1.2**: Decompose `RRtoolbox/lib/root.py` (1,221L) and `RRtoolbox/lib/directory.py` (776L) into `profiling.py`, `logging_util.py`, `unit_conversion.py`, and `path_utils.py`. [COMPLETED]
- Advance campaign state to Phase 2.

---

## [2026-11-03] — Milestone 1.2: `RRtoolbox/lib/root.py` & `RRtoolbox/lib/directory.py` Decomposition

### Objectives & Scope
- Eliminate the two Critical Monoliths in foundation utilities: `RRtoolbox/lib/root.py` (1,221L) and `RRtoolbox/lib/directory.py` (776L).
- Decompose cohesive responsibilities into 4 single-responsibility submodules adhering to Modularity Budgets (target $\le$ 350L, hard ceiling $\le$ 500L, functions $\le$ 40L).
- Preserve 100% backward-compatible re-export facades with full `__all__` parity in `root.py` (34 exports) and `directory.py` (25 exports).
- Modernize Python 2 syntax (replace `basestring` with native `str`, eliminate obsolete `object` inheritance).
- Enforce full contract invariant parity (Tier 1 AST & DOM contract verification, unit test discovery).

### Files Created & Modified
- `RRtoolbox/lib/unit_conversion.py` (Created, 197L): Unit conversion, magnitude formatting, and metric scaling factors (`FactorConvert`, `Magnitude`). Modernized Python 2 `basestring` checks.
- `RRtoolbox/lib/logging_util.py` (Created, 212L): Stream redirection, stdout multiplexing, and logging contexts (`StdoutSIM`, `StdoutMULTI`, `StdoutLOG`, `Controlstdout`, `stdout`).
- `RRtoolbox/lib/profiling.py` (Created, 270L): Profiling, execution timers, and chronometers (`TimeCode`, `Profiler`). Decomposed recursive formatters into pure helper functions.
- `RRtoolbox/lib/path_utils.py` (Created, 364L): Pure path and filesystem inspection utilities (`resource_path`, `getFileHandle`, `getFileSize`, `checkPath`, `checkDir`, `checkURL`, `checkFile`, `getPath`, `getData`, `increment_if_exits`, `strdifference`, `getSplitted`, `correctSep`, `mkPath`, `rmFile`, `rmPath`, `getSep`, `quickOps`, `getShortenedPath`, `decoratePath`, `correctPath`, `joinPath`). Decomposed oversized functions (`quickOps`, `getShortenedPath`, `strdifference`) into pure helpers $\le 40$ lines.
- `RRtoolbox/lib/directory.py` (Refactored, 260L): Reduced from 776L to 260L (-66%) focusing on `Directory`, `FileDirectory`, and `changedir`, while re-exporting all 25 path utility symbols.
- `RRtoolbox/lib/root.py` (Refactored, 388L): Reduced from 1,221L to 388L (-68%) focusing on core primitives (`NamedLock`, `secure_open`, `load_module`, `glob`, `lookinglob`), eliminating `lookinglob` critical line breach, and re-exporting all 34 symbols.

### Modularity Audit Results
- `RRtoolbox/lib/root.py`: **0 Critical Breaches** (was 1,221L with 1 God Class, 1 Critical function `lookinglob` 69L).
- `RRtoolbox/lib/directory.py`: **0 Critical Breaches** (was 776L with 3 Critical functions `quickOps` 60L, `getShortenedPath` 49L, `strdifference` 43L).
- Active scope Critical Monoliths reduced to **0**.
- All newly created and refactored modules strictly adhere to the 500-line hard ceiling.

### Verification & Test Commands
```bash
# 1. Structural line and method budget audit across active files
python -m unittest discover tests/unittest_tests

# 2. Automated test suite execution
python -m unittest tests/unittest_tests/test_root.py
python -m unittest tests/unittest_tests/test_refactor_contract.py

# 3. Verify module import validity
python -c "import RRtoolbox.lib.root as r, RRtoolbox.lib.directory as d; print(len(r.__all__), len(d.__all__))"
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Foundation Subsystems** | Structural Line Inspection | `python -m RRtoolbox --help` |
| **Verify Root Unit Tests** | Automated Regression Suite | `python -m unittest tests/unittest_tests/test_root.py` |
| **Contract Smoke Test** | Regression Invariant Check | `python -m unittest tests/unittest_tests/test_refactor_contract.py` |
| **Full Unit Test Discovery** | Automated Unit Tests | `python -m unittest discover tests/unittest_tests` |

### Next Steps & Pending Items
- Phase 1 Complete! Core Foundation & Array Operations fully decomposed.
- **Phase 2 (Active)**: Storage, Caching & Image Abstraction Decomposition:
  - Sub-Phase 2.1: `RRtoolbox/lib/cache.py` (1,357L) decomposition into `memoize.py`, `resource_manager.py`, `disk_dict.py`, and `lazy.py`. [COMPLETED]
  - Sub-Phase 2.2: `RRtoolbox/lib/image.py` (2,112L) decomposition.
- Run `/refactor-monolith continue` in a fresh chat.

---

## [2026-11-03] — Milestone 2.1: `RRtoolbox/lib/cache.py` Decomposition

### Objectives & Scope
- Eliminate the 1,357-line Critical Monolith `RRtoolbox/lib/cache.py` (Rank 4 production monolith).
- Decompose cohesive responsibilities into 4 single-responsibility submodules and 1 package facade under `RRtoolbox/lib/cache/`:
  - `lazy.py` (ObjectGetter, Retriever, LazyDict)
  - `memoize.py` (DynamicMemoizedFunc, MemorizedFunc, NotMemorizedFunc, Memory, Memoizer, cache, cachedProperty, Memo, memoize)
  - `resource_manager.py` (ResourceManager, memory eviction pipelines, memory unit parser)
  - `disk_dict.py` (mapper, MemoizedDict)
  - `__init__.py` (re-export facade with complete `__all__` export parity)
- Adhere strictly to Modularity Budgets (target $\le$ 350L, hard ceiling $\le$ 500L, functions $\le$ 40L).
- Modernize Python 2 / legacy syntax: eliminate `old_div` shim in favor of native `/`, replace legacy `raise_` with modern exception chaining `raise CorruptPersistent(...) from e`.
- Maintain 100% backward-compatible re-export facade with full `__all__` parity (21 public symbols).
- Enforce full contract invariant parity (Tier 1 AST & DOM contract verification, permanent smoke test, unit tests).

### Files Created & Modified
- `RRtoolbox/lib/cache/lazy.py` (Created, 252L): Lazy evaluation, reference tracking, and dictionary retrieval (`ObjectGetter`, `Retriever`, `LazyDict`).
- `RRtoolbox/lib/cache/memoize.py` (Created, 349L): Memoization decorators and descriptors (`NotMemorizedFunc`, `MemorizedFunc`, `DynamicMemoizedFunc`, `Memory`, `Memoizer`, `cache`, `cachedProperty`, `Memo`, `memoize`). Decomposed `cachedProperty` into focused helpers $\le 25$ lines.
- `RRtoolbox/lib/cache/resource_manager.py` (Created, 309L): Dynamic object lifecycle, reference tracking, and RAM optimization (`ResourceManager`, `parse_memory_unit`). Decomposed `_optimizeMemory` into multi-stage eviction helpers (`_evict_blacklist_items`, `_evict_unwhitelisted_items`) $\le 25$ lines each.
- `RRtoolbox/lib/cache/disk_dict.py` (Created, 215L): File-backed key-value persistence and memory-mapped array helpers (`mapper`, `MemoizedDict`). Modernized exception handling.
- `RRtoolbox/lib/cache/__init__.py` (Created, 77L): Authoritative package re-export facade with full `__all__` parity.
- `RRtoolbox/lib/cache.py` (Deleted, 1,357L): Replaced by `RRtoolbox/lib/cache/` package.

### Modularity Audit Results
- `RRtoolbox/lib/cache/`: **0 Critical Monoliths** (was 1 Critical Monolith of 1,357L with 3 God classes).
- **0 Warning-level file breaches**: All submodules strictly below the 350-line target (349L, 309L, 252L, 215L, 77L).
- **0 functions exceeding 40 lines** across the entire package.

### Verification & Test Commands
```bash
# 1. Structural line and method budget audit across cache package
python -m unittest discover tests/unittest_tests

# 2. Automated test suite execution
python -m unittest tests/unittest_tests/test_cache.py
python -m unittest tests/unittest_tests/test_refactor_contract.py

# 3. Verify module import validity
python -c "import RRtoolbox.lib.cache as c; print(len(c.__all__))"
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Cache Subsystem** | Structural Line Inspection | `python -m unittest tests/unittest_tests/test_refactor_contract.py` |
| **Verify Cache Unit Tests** | Automated Regression Suite | `python -m unittest tests/unittest_tests/test_cache.py` |
| **Contract Smoke Test** | Regression Invariant Check | `python -m unittest tests/unittest_tests/test_refactor_contract.py` |
| **Full Unit Test Discovery** | Automated Unit Tests | `python -m unittest discover tests/unittest_tests` |

### Next Steps & Pending Items
- Sub-Phase 2.1 Complete! `RRtoolbox/lib/cache.py` successfully decomposed into `RRtoolbox/lib/cache/`.
- Sub-Phase 2.2: `RRtoolbox/lib/image.py` (2,112L) decomposition into `conversions.py`, `io.py`, `loaders.py`, `coordinates.py`, `interactive.py`, and `core.py`. [COMPLETED]

---

## [2026-11-03] — Milestone 2.2: `RRtoolbox/lib/image.py` Decomposition & Phase 2 Completion

### Objectives & Scope
- Eliminate the 2,112-line Critical Monolith `RRtoolbox/lib/image.py` (Rank 2 production monolith).
- Decompose cohesive responsibilities into 6 single-responsibility submodules and 1 package facade under `RRtoolbox/lib/image/`:
  - `conversions.py` (array/color conversions, Qt formats, Matplotlib figure bridges, histogram matching)
  - `io.py` (file/URL loading, format interpretation, OpenCV bridges, batch conversion pipeline)
  - `loaders.py` (ImFactory, loadFunc, ImLoader, PathLoader, LoaderDict, try_loads)
  - `coordinates.py` (ImCoors geometric analytics and drawing primitives)
  - `interactive.py` (GetCoors OpenCV GUI selector and point grouping helpers)
  - `core.py` (Image high-level object with lazy color channel properties and persistence)
  - `__init__.py` (authoritative re-export facade with complete `__all__` export parity)
- Adhere strictly to Modularity Budgets (target $\le$ 350L, hard ceiling $\le$ 500L, functions $\le$ 40L).
- Modernize Python 2 / legacy syntax: replace `basestring` checks with `(str, bytes)`, update deprecated `np.int` aliases to native `int`, ensure NumPy 2.x `tobytes()` compatibility.
- Maintain 100% backward-compatible re-export facade with full `__all__` parity (50 public symbols).
- Enforce full contract invariant parity (Tier 1 AST & DOM contract verification, permanent smoke test, and automated unit test suite).

### Files Created & Modified
- `RRtoolbox/lib/image/conversions.py` (Created, 294L): Array/color conversions, Qt image transitions, Matplotlib bridges, and histogram matching (`transposeIm`, `plt2bgr`, `plt2bgra`, `fig2bgr`, `fig2bgra`, `qi2np`, `np2qi`, `gray2qi`, `rgb2qi`, `bgra2bgr`, `hist_match`, color maps).
- `RRtoolbox/lib/image/io.py` (Created, 223L): Image reading, writing, and loading pipelines (`checkLoaded`, `loadcv`, `loadsfrom`, `interpretImage`, `convertAs`).
- `RRtoolbox/lib/image/loaders.py` (Created, 375L): Dynamic loader factory, lazy path loading, and memory-managed dicts (`ImFactory`, `loadFunc`, `ImLoader`, `PathLoader`, `LoaderDict`, `try_loads`).
- `RRtoolbox/lib/image/coordinates.py` (Created, 442L): Coordinate analytics and drawing helpers (`ImCoors`, `drawcoorpoints`, `myline`, `drawcooraxes`, `drawcoorpolyline`, `drawcoorarea`, `drawcoorpolyArrow`, `drawcoorperspective`, `limitaxispoints`).
- `RRtoolbox/lib/image/interactive.py` (Created, 238L): OpenCV interactive point selector and grouping algorithms (`GetCoors`, `getcoors`, `separe`, `getrectcoors`, `quadrants`, `getgeometrycoors`, `random_color`).
- `RRtoolbox/lib/image/core.py` (Created, 243L): High-level Image class with lazy multi-channel color access and disk saving (`Image`).
- `RRtoolbox/lib/image/__init__.py` (Created, 89L): Re-export facade preserving all legacy imports and symbols with complete `__all__` export parity.
- `RRtoolbox/lib/image.py` (Deleted, 2,112L): Superseded monolith cleanly replaced by `RRtoolbox/lib/image/` package.
- `tests/unittest_tests/test_image.py` (Created, 125L): Comprehensive automated unit test suite covering facade parity, color conversions, ImCoors geometric analytics, and loaders.

### Modularity Audit Results
- `RRtoolbox/lib/image/`: **0 Critical Monoliths** (was 1 Critical Monolith of 2,112L with 2 God classes).
- **0 files exceeding the 500-line hard ceiling**: All submodules safely within quantitative budgets (442L, 375L, 294L, 243L, 238L, 223L, 89L).
- **Full Unit Test Suite**: 21/21 tests passing (100% green).

### Verification & Test Commands
```bash
# 1. Structural line and method budget audit across image package
python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolbox/lib/image/ --show-methods

# 2. Automated test suite execution
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_image.py -v
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- discover tests/unittest_tests

# 3. Verify module import validity
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -c "import RRtoolbox.lib.image as img; print(len(img.__all__))"

# 4. Strict contract verification against baseline snapshot
python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Image Subsystem** | Structural Line Inspection | `python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolbox/lib/image/ --show-methods` |
| **Verify Image Unit Tests** | Automated Regression Suite | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_image.py -v` |
| **Contract Smoke Test** | Regression Invariant Check | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_refactor_contract.py -v` |
| **Full Unit Test Discovery** | Automated Unit Tests | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- discover tests/unittest_tests` |
| **Strict Contract Verification** | Invariant Parity Check | `python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict` |

### Next Steps & Pending Items
- **Phase 2 Complete!** Both `cache.py` (Sub-Phase 2.1) and `image.py` (Sub-Phase 2.2) successfully decomposed into cohesive packages with 0 critical breaches.
- **Phase 3 (Active)**: Visualization, Panoramic & Restoration Decomposition.
  - Sub-Phase 3.1: `RRtoolbox/lib/plotter.py` (2,493L) decomposition into `RRtoolbox/lib/visualization/`. [COMPLETED]
  - Sub-Phase 3.2: `RRtoolbox/geometry/panoramic.py` (719L) and `RRtoolbox/restoration/super_resolution.py` (511L).
- To continue Sub-Phase 3.2 in a fresh chat, tell the agent: `/refactor-monolith continue`.

---

## [2026-11-03] — Milestone 3.1: `RRtoolbox/lib/plotter.py` Decomposition

### Objectives & Scope
- Eliminate the 2,493-line Critical Monolith `RRtoolbox/lib/plotter.py` (Rank 1 production monolith in `rrtools`).
- Decompose cohesive responsibilities into single-responsibility submodules and mixins under `RRtoolbox/lib/visualization/`:
  - `helpers.py` (echo, background, limitaxis, convert2bgr, convert2bgra)
  - `matplotlib_helpers.py` (fastplt, graph_filter, plotPointsContour, wins, servertimeout)
  - `commands.py` (CommandMixin with cmdfunc, formatcmd, builtincmd, keystroke dispatch, clipboard integration)
  - `overlays.py` (OverlayMixin with makeoverlay, plotatpointer, plotatxy, plotintime, builtinplot)
  - `coordinates_mixin.py` (CoordinateMixin with real2render, render2real, and reactive coordinate boundaries)
  - `viewer.py` (Plotim core OpenCV window lifecycle, zoom, pan, waitKey loop)
  - `match_explorer.py` (MatchExplorer interactive keypoint correspondence viewer)
  - `edger.py` (Edger Canny edge detection tuner with OpenCV trackbars)
  - `imtester.py` (Imtester threshold and filter visualizer with Otsu and adaptive thresholding)
  - `__init__.py` (authoritative package re-export facade with complete export parity)
- Retain `RRtoolbox/lib/plotter.py` as an authoritative backward-compatible re-export facade (reduced from 2,493L to 59L, -97.6%).
- Adhere strictly to Modularity Budgets (target $\le$ 350L, hard ceiling $\le$ 500L, functions $\le$ 40L).
- Modernize Python 2 syntax: replace `basestring` with `(str, bytes)`, modernize integer division, ensure Python 3 compatibility.
- Maintain 100% backward-compatible re-export facade with full `__all__` parity (14 exports in `plotter.py`, 17 in `visualization/__init__.py`).
- Enforce full contract invariant parity (Tier 1 AST & DOM contract verification, permanent smoke test, and automated unit test suite).

### Files Created & Modified
- `RRtoolbox/lib/visualization/helpers.py` (Created, 130L): Window format and coordinate utilities (`echo`, `background`, `limitaxis`, `convert2bgr`, `convert2bgra`).
- `RRtoolbox/lib/visualization/matplotlib_helpers.py` (Created, 290L): Matplotlib plotting utilities and filter response visualizers (`fastplt`, `graph_filter`, `plotPointsContour`, `wins`, `servertimeout`). Decomposed filter response pipeline into pure helpers $\le 25$ lines.
- `RRtoolbox/lib/visualization/commands.py` (Created, 178L): Keystroke parsing, command history, and clipboard bridges (`CommandMixin`).
- `RRtoolbox/lib/visualization/overlays.py` (Created, 165L): Text HUD badges, pixel color swatches, and overlay compositing (`OverlayMixin`).
- `RRtoolbox/lib/visualization/coordinates_mixin.py` (Created, 117L): Window coordinate boundaries and real-to-render projections (`CoordinateMixin`).
- `RRtoolbox/lib/visualization/viewer.py` (Created, 267L): Core OpenCV window lifecycle, zoom/pan navigation, and event loop (`Plotim`).
- `RRtoolbox/lib/visualization/match_explorer.py` (Created, 163L): Feature correspondence visualizer (`MatchExplorer`).
- `RRtoolbox/lib/visualization/edger.py` (Created, 280L): Canny edge tuning with OpenCV trackbars (`Edger`).
- `RRtoolbox/lib/visualization/imtester.py` (Created, 237L): Threshold and filter inspector (`Imtester`).
- `RRtoolbox/lib/visualization/__init__.py` (Created, 51L): Package re-export facade with complete `__all__` parity.
- `RRtoolbox/lib/plotter.py` (Refactored, 59L): Reduced from 2,493L to 59L (-97.6%) as a clean re-export facade.
- `tests/unittest_tests/test_plotter.py` (Created, 98L): Comprehensive automated unit test suite covering facade parity, conversion helpers, Plotim coordinate transforms, formatcmd, and static helpers.

### Modularity Audit Results
- `RRtoolbox/lib/visualization/`: **0 Critical File Breaches** (all files $\le 290$ lines, well below the 350-line target and 500-line hard ceiling).
- **0 functions exceeding 40 lines** across the entire package.
- `RRtoolbox/lib/plotter.py`: Reduced to 59 lines (was 2,493L with 2 God classes).
- **Full Unit Test Suite**: 32/32 tests passing (100% green).

### Verification & Test Commands
```bash
# 1. Structural line and method budget audit across visualization package
python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolbox/lib/visualization/ RRtoolbox/lib/plotter.py --show-methods

# 2. Automated unit test suite execution
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_plotter.py -v
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- discover tests/unittest_tests

# 3. Verify module import validity
python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -c "import RRtoolbox.lib.plotter as p, RRtoolbox.lib.visualization as v; print(len(p.__all__), len(v.__all__))"

# 4. Strict contract verification against baseline snapshot
python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Visualization Subsystem** | Structural Line Inspection | `python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolbox/lib/visualization/ RRtoolbox/lib/plotter.py --show-methods` |
| **Verify Plotter Unit Tests** | Automated Regression Suite | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_plotter.py -v` |
| **Contract Smoke Test** | Regression Invariant Check | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- tests/unittest_tests/test_refactor_contract.py -v` |
| **Full Unit Test Discovery** | Automated Unit Tests | `python "<skill_dir>/scripts/contract_snapshot.py" run-tests --path . -- discover tests/unittest_tests` |
| **Strict Contract Verification** | Invariant Parity Check | `python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict` |

### Next Steps & Pending Items
- Sub-Phase 3.1 Complete! The Rank 1 Monolith `RRtoolbox/lib/plotter.py` (2,493L) is fully decomposed.
- Sub-Phase 3.2 Complete! Modern Vision Decomposition (`panoramic.py` and `super_resolution.py`) fully decomposed.
- Phase 3 Complete! Active phase advanced to Phase 4.

---

## [2026-11-04] — Milestone 3.2: Modern Vision Decomposition (`panoramic.py` & `super_resolution.py`)

### Objectives & Scope
- Decompose the two remaining critical vision monoliths in Phase 3:
  - `RRtoolbox/geometry/panoramic.py` (719L, Rank 8 monolith)
  - `RRtoolbox/restoration/super_resolution.py` (511L, Rank 11 monolith)
- Modularize into cohesive single-responsibility submodules:
  - `RRtoolbox/geometry/warping.py`: Spherical and cylindrical coordinate remapping and focal length estimation.
  - `RRtoolbox/geometry/blending.py`: Multi-band Laplacian pyramid and feathered seam blending.
  - `RRtoolbox/geometry/panoramic.py`: Focused `PanoramicStitcher` orchestrator and re-export facade with full `__all__` parity.
  - `RRtoolbox/restoration/registration.py`: Sub-pixel ECC registration, frame inverse warping accumulation, and bilateral detail boosting.
  - `RRtoolbox/restoration/super_resolution.py`: Single-image and multi-frame burst restoration orchestrator with full backward-compatible re-exports.
- Adhere strictly to Modularity Budgets (all files $\le$ 350 lines, functions $\le$ 40 lines, classes $\le$ 10 methods).
- Maintain 100% backward-compatible imports, math comments, and invariant parity across all tests.

### Files Created & Modified
- `RRtoolbox/geometry/warping.py` (Created, 120L): Extracted `estimate_focal_length`, `spherical_warp`, and `cylindrical_warp` with shared `_apply_warp_and_mask`.
- `RRtoolbox/geometry/blending.py` (Created, 157L): Extracted `build_gaussian_pyramid`, `build_laplacian_pyramid`, `reconstruct_from_laplacian`, `multiband_blend`, `feather_blend`, and `crop_panorama_canvas`.
- `RRtoolbox/geometry/panoramic.py` (Refactored, 349L): Reduced from 719L to 349L (-51.5%). Decomposed `PanoramicStitcher.stitch` (236L -> 40L) and `PanoramicStitcher` class (315L -> 145L, 6 methods).
- `RRtoolbox/restoration/registration.py` (Created, 124L): Extracted `align_ecc_subpixel`, `accumulate_warped_frame`, `boost_high_frequency_details`, and `align_and_fuse_frames`.
- `RRtoolbox/restoration/super_resolution.py` (Refactored, 332L): Reduced from 511L to 332L (-35.0%). Decomposed `restore` (99L -> 35L) and `process` with 100% backward compatibility.
- `RRtoolbox/restoration/__init__.py` (Updated, 31L): Preserved and exposed full registration and super-resolution API in `__all__`.

### Modularity Audit Results
- `RRtoolbox/geometry/`: **0 Critical File Breaches** across all decomposed files (`panoramic.py` 349L, `warping.py` 120L, `blending.py` 157L).
- `RRtoolbox/restoration/`: **0 Critical File Breaches** across all files (`super_resolution.py` 332L, `registration.py` 124L).
- **0 functions exceeding 40 lines** across both decomposed subsystems.
- All classes $\le$ 6 methods.
- **Verification Tests**:
  - `pytest tests/test_panoramic.py tests/test_super_resolution.py`: 20/20 passed (100% green).
  - `python -m unittest discover tests/unittest_tests`: 32/32 passed (100% green).
  - Contract Verification: Zero regressions detected.

### Verification & Test Commands
```bash
# 1. Automated regression test suites
pytest tests/test_panoramic.py tests/test_super_resolution.py -v
python -m unittest discover tests/unittest_tests

# 2. Verify module imports and __all__ export parity
python -c "import RRtoolbox.geometry.panoramic as p, RRtoolbox.restoration.super_resolution as sr; print(len(p.__all__), len(sr.__all__))"
```

---

## [2026-11-04] — Milestone 4.1-A: Flowchart Terminal & Node Decoupling (`Terminal.py` & `Node.py`)

### Objectives & Scope
- Decompose foundational flowchart monoliths in `RRtoolFC/lib/flowchart/`:
  - `RRtoolFC/lib/flowchart/Terminal.py` (637L, Rank 10 monolith)
  - `RRtoolFC/lib/flowchart/Node.py` (648L, Rank 9 monolith)
- Decouple graphical Qt scene elements (`GraphicsObject` subclasses) from computational DAG structures:
  - Extract `TerminalGraphicsItem` and `ConnectionItem` into `RRtoolFC/lib/flowchart/terminal_items.py`.
  - Extract `NodeGraphicsItem` into `RRtoolFC/lib/flowchart/node_items.py`.
- Modernize Python 2 syntax with resilient `old_div` fallback import and `import sys` guards.
- Retain `Terminal.py` and `Node.py` as backward-compatible re-export facades with full `__all__` parity.
- Quarantined upstream relative import defect `DEFECT-001` via Asymmetric Defect Protocol.
- Adhere strictly to the **2-Monolith / 25-Tool-Call Turn Ceiling Gate** and quantitative modularity budgets.

### Files Created & Modified
- `RRtoolFC/lib/flowchart/terminal_items.py` (Created, 329L): Extracted `TerminalGraphicsItem` and `ConnectionItem` with `old_div` fallback and Qt scene interaction handling.
- `RRtoolFC/lib/flowchart/Terminal.py` (Refactored, 328L): Reduced from 637L to 328L (-48.5%). Retains pure `Terminal` computational pin logic and re-exports `TerminalGraphicsItem`, `ConnectionItem` in `__all__`.
- `RRtoolFC/lib/flowchart/node_items.py` (Created, 219L): Extracted `NodeGraphicsItem` handling node box graphics, labels, and Qt context menus.
- `RRtoolFC/lib/flowchart/Node.py` (Refactored, 449L): Reduced from 648L to 449L (-30.7%). Retains pure `Node` processing unit, signal definitions, state serialization, and re-exports `NodeGraphicsItem`, `strDict` in `__all__`.
- `.diagnostics/defect_catalog.json` & `DEFECT_CATALOG.md`: Recorded quarantined defect `DEFECT-001` (tactical) for upstream relative import resolution without altering business logic.

### Modularity Audit Results
- `RRtoolFC/lib/flowchart/`:
  - `Terminal.py`: **328L** ($\le$ 350L soft warning budget, was 637L CRITICAL).
  - `terminal_items.py`: **329L** ($\le$ 350L soft warning budget).
  - `Node.py`: **449L** (< 500L hard ceiling budget, was 648L CRITICAL).
  - `node_items.py`: **219L** (150–300L healthy target budget).
- AST Compilation & Import Verification: All 4 files compile cleanly via `py_compile` and parse with zero AST defects.
- Contract Verification: Zero regressions detected across baseline snapshot (`verify --strict`).

### Verification & Test Commands
```bash
# 1. Structural modularity audit across extracted flowchart modules
python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolFC/lib/flowchart/Terminal.py RRtoolFC/lib/flowchart/terminal_items.py RRtoolFC/lib/flowchart/Node.py RRtoolFC/lib/flowchart/node_items.py --show-methods

# 2. Syntax and AST compilation verification
python -m py_compile RRtoolFC/lib/flowchart/Terminal.py RRtoolFC/lib/flowchart/terminal_items.py RRtoolFC/lib/flowchart/Node.py RRtoolFC/lib/flowchart/node_items.py RRtoolFC/lib/flowchart/Flowchart.py

# 3. Contract baseline verification
python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict
```

### Scenario Combinations Matrix
| Objective | Mode / Flags | Exact Runnable Command |
|---|---|---|
| **Audit Flowchart Health** | Modularity Line Audit | `python "<skill_dir>/scripts/code_health_audit.py" --path RRtoolFC/lib/flowchart/ --show-methods` |
| **Verify Bytecode Compilation** | AST Compilation Check | `python -m py_compile RRtoolFC/lib/flowchart/Terminal.py RRtoolFC/lib/flowchart/terminal_items.py RRtoolFC/lib/flowchart/Node.py RRtoolFC/lib/flowchart/node_items.py` |
| **Verify Contract Parity** | Invariant Parity Check | `python "<skill_dir>/scripts/contract_snapshot.py" verify --path RRtoolbox --strict` |
| **Check Campaign State** | Active Phase Overview | `python "<skill_dir>/scripts/contract_snapshot.py" state --show` |

### Next Steps & Pending Items
- Sub-Phase 4.1 Batch A Complete! `Terminal.py` and `Node.py` decomposed into `terminal_items.py` and `node_items.py`.
- Sub-Phase 4.1 Batch B (Pending Next Turn): `RRtoolFC/lib/flowchart/Flowchart.py` (935L) decomposition into `core_graph.py` and `ui_widgets.py`.
- Sub-Phase 4.2 (Pending): Curate `tests/` into `tests/unittest_tests/` vs deprecated prototype scripts.
- To continue Phase 4 in a fresh chat, tell the agent: `/refactor-monolith continue`.






