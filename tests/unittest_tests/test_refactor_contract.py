# Generated Contract Smoke Test - Enforces Zero-Regression Refactoring Parity
import pathlib
import re
import unittest

class TestCodebaseContractInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(__file__).resolve().parent.parent.parent
        ignored = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "_build", "build", "dist", ".refactor", "scratch"}
        cls.html_text = "".join(
            p.read_text(encoding="utf-8", errors="replace") + "\n"
            for p in cls.root.rglob("*.html")
            if not any(part in ignored for part in p.parts)
        )
        cls.js_text = "".join(
            p.read_text(encoding="utf-8", errors="replace") + "\n"
            for p in cls.root.rglob("*.js")
            if not any(part in ignored for part in p.parts)
        )

    def test_dom_interactive_selects_exist(self):
        """Verify all critical <select> dropdown IDs exist in HTML."""
        for sel in ['detector-select', 'preset-select', 'restorer-select']:
            found = f'id="{sel}"' in self.html_text or f"id='{sel}'" in self.html_text
            self.assertTrue(found, f"Missing select dropdown #{sel} in HTML!")

    def test_dom_element_ids_preserved(self):
        """Verify captured baseline HTML element IDs are preserved."""
        for el_id in ['A', 'ASIFT', 'ASIFT_iter', 'ASIFT_multiple', 'Asyncronous', 'B', 'Bandpass', 'Bandstop', 'BilateraParameter', 'BilateralParameters', 'C', 'Cache', 'ClassNotAllowed', 'Conection', 'ConfigTool', 'Controlstdout', 'CorruptPersistent', 'D', 'Directory', 'DirectoryManager', 'DynamicMemoizedFunc', 'E', 'Edger', 'EntropyPlot', 'F', 'FactorConvert', 'Feature', 'FileDirectory', 'FilterBase', 'G', 'GetCoors', 'GraphTrace', 'GraphTraceOutput', 'H', 'Highpass', 'I', 'ImCoors', 'ImFactory', 'ImLoader', 'Image', 'Imtester', 'InvertedBandpass', 'InvertedBandstop', 'J', 'K', 'L', 'LazyDict', 'LoaderDict', 'Logger', 'Lowpass', 'M', 'MATCH', 'MATCH_multiple', 'Magnitude', 'MatchExplorer', 'MemoizedDict', 'Memoizer', 'MemorizedFunc', 'Memory', 'N', 'NameSpace', 'NoParserFound', 'NotCallable', 'NotConvertibleToInt', 'NotCreatable', 'NotMemorizedFunc', 'O', 'ObjectGetter', 'P', 'PathLoader', 'Plotim', 'Profiler', 'Q', 'R', 'ResourceManager', 'Retriever', 'S', 'Shell', 'SimKeyPoint', 'StdoutLOG', 'StdoutMULTI', 'StdoutSIM', 'Syncronous', 'T', 'TimeCode', 'TimeOutException', 'TransferExeption', 'U', 'V', 'VariableNotAvailable', 'VariableNotDeletable', 'VariableNotGettable', 'VariableNotSettable', 'W', 'addto', 'affine_skew', 'angle', 'angle2D', 'angleXY', 'anorm', 'anorm2', 'app-logo', 'apply2kp_pairs', 'axesIntercept', 'background', 'badge-features', 'badge-resolution', 'bgra2bgr', 'biggestCnt', 'biggestCntData', 'bilateralFilter', 'boxPads', 'brightness', 'cachedProperty', 'cap-r', 'centerM', 'centerS', 'centerSM', 'changedir', 'checkDir', 'checkFile', 'checkFromSession', 'checkLoaded', 'checkPath', 'checkURL', 'cnt_hist', 'comp-img-after', 'comp-img-before', 'comp-label-left', 'comp-label-right', 'comp-meta', 'comp-overlay', 'comp-slider-box', 'contents', 'contour2points', 'contours2mask', 'contoursArea', 'conv3H4H', 'convert2bgr', 'convert2bgra', 'convertAs', 'convertXY', 'convexityRatio', 'copy-btn', 'correctPath', 'correctSep', 'decorateInstanceMethods', 'decoratePath', 'deleteFromSession', 'detector-select', 'dict2keyPoint', 'documentation', 'drawCircle', 'drawEllipse', 'drawcoorarea', 'drawcooraxes', 'drawcoorperspective', 'drawcoorpoints', 'drawcoorpolyArrow', 'drawcoorpolyline', 'echo', 'ensureList', 'entropy', 'entroyTest', 'f', 'fallback', 'fastplt', 'fig2bgr', 'fig2bgra', 'filterFactory', 'filter_matches', 'findContours', 'findModules', 'find_near', 'find_optic_disc_watershed', 'findmaxima', 'findminima', 'fitLens', 'flushSession', 'foreground', 'formatConsume', 'formatOnly', 'funcData', 'generateServer', 'getBilateralParameters', 'getData', 'getDocParamLines', 'getDocParameters', 'getEnviromentSession', 'getFileHandle', 'getFileSize', 'getModules', 'getOtsuThresh', 'getPackagePath', 'getPath', 'getSOpointRelation', 'getSep', 'getShortenedPath', 'getSplitted', 'getTransformedCorners', 'getTransparency', 'get_beta_params_Otsu', 'get_beta_params_hist', 'get_bright_alpha', 'get_layered_alpha', 'get_x_space', 'getcoors', 'getdataVH', 'getgeometrycoors', 'gethull', 'getrectcoors', 'github-link', 'glob', 'globFilter', 'graph_filter', 'gray2qi', 'hero-section', 'hist_cdf', 'hist_comp', 'hist_match', 'histogram', 'icons', 'id1', 'id2', 'id3', 'id4', 'im2imFormat', 'im2shapeFormat', 'increment_if_exits', 'index', 'indices-and-tables', 'initClient', 'initServer', 'init_feature', 'inlineRatio', 'instability_bf', 'installation', 'interpretImage', 'invertH', 'invertM', 'invertSM', 'isnumpy', 'joinPath', 'keyPoint2tuple', 'keywords', 'layeredfloods', 'limitaxis', 'limitaxispoints', 'load', 'loadFunc', 'loadcv', 'loadsfrom', 'lookinglob', 'makeVis', 'mapper', 'mask_watershed', 'matrixIntercept', 'meta-desc', 'meta-title', 'mkPath', 'module-RRtoolbox', 'module-contents', 'multiple_otsu', 'multiple_superpose', 'myline', 'name', 'navigation-tree', 'noisy', 'normalize', 'normalize2', 'normalizeCustom', 'normsigmoid', 'np2qi', 'np2str', 'outline', 'overlay', 'overlay2', 'overlayXY', 'overlaypng', 'overview', 'padVH', 'pad_to_fit_H', 'page-container', 'parseString', 'pf1', 'ping', 'pipeline-section', 'plotPointsContour', 'plt2bgr', 'plt2bgra', 'points2contour', 'points2mask', 'points2vectos', 'points_generator', 'polygonArea_calcule', 'polygonArea_contour', 'polygonArea_fill', 'preset-select', 'process_as_blocks', 'qi2np', 'quadrant', 'quadrants', 'quickOps', 'random_color', 'random_points', 'rcvPickle', 'readSession', 'recursiveMap', 'recv_into', 'relativeQuadrants', 'relativeVectors', 'releases', 'reloadFunc', 'reraise', 'rescale', 'resource_path', 'restorer-select', 'retina_markers_thresh', 'retinal_mask', 'retinal_mask_watershed', 'rgb2qi', 'rmFile', 'rmPath', 'rrbox', 'rrtoolbox-core-module', 'rrtoolbox-lib-arrayops-basic-module', 'rrtoolbox-lib-arrayops-convert-module', 'rrtoolbox-lib-arrayops-filters-module', 'rrtoolbox-lib-arrayops-mask-module', 'rrtoolbox-lib-arrayops-package', 'rrtoolbox-lib-cache-module', 'rrtoolbox-lib-config-module', 'rrtoolbox-lib-descriptors-module', 'rrtoolbox-lib-directory-module', 'rrtoolbox-lib-image-module', 'rrtoolbox-lib-inspector-module', 'rrtoolbox-lib-package', 'rrtoolbox-lib-plotter-module', 'rrtoolbox-lib-root-module', 'rrtoolbox-lib-serverservices-module', 'rrtoolbox-lib-session-module', 'rrtoolbox-package', 'rrtoolbox-run-module', 'rrtoolbox-shell-module', 'rrtoolbox-tools-lens-module', 'rrtoolbox-tools-package', 'rrtoolbox-tools-segmentation-module', 'rrtoolbox-tools-selectors-module', 'rrtoolbox-tools-sticher-module', 'rrtools-raster-restoration-tools-build-status-docs', 'saveSession', 'scan_ports', 'search-documentation', 'search-progress', 'search-results', 'searchbox', 'searchindexloader', 'sendPickle', 'send_from', 'separe', 'separePointsByAxis', 'sh2oh', 'shell_processor', 'shell_processor_parser', 'showcase-card', 'sidebar', 'sigmoid', 'simulateLens', 'slider-dist', 'slider-handle', 'slider-inline', 'smooth', 'spairs2opairs', 'split-range', 'splitPoints', 'spoint2opointfunc', 'standarizePoints', 'status-badge', 'stich', 'str2np', 'strdifference', 'string_interpreter', 'string_is_socket_address', 'studio-panel', 'submodules', 'subpackages', 'superpose', 'tab-panorama', 'tab-retina', 'terminal-command', 'testing-and-application', 'thresh_biggestCnt', 'thresh_hist', 'threshold_opening', 'toTupple', 'toggle-1', 'toggle-asift', 'toggle-flann', 'toggle-grow', 'toggle-lens', 'tools', 'tools2', 'tracer', 'transformPoint', 'transformPoints', 'translateQuadrants', 'transposeIm', 'try_loads', 'tuple2keyPoint', 'tuple_creator', 'unit_vector', 'updateSession', 'usage', 'val-dist', 'val-inline', 'vectorsAngles', 'vectorsQuadrants', 'vectos2points', 'vertexesAngles', 'view_as_blocks', 'view_as_windows', 'welcome-to-rrtoolbox-s-documentation']:
            found = f'id="{el_id}"' in self.html_text or f"id='{el_id}'" in self.html_text
            self.assertTrue(found, f"Critical DOM element id='{el_id}' was removed from HTML!")

    def test_zero_css_important_creep(self):
        """Verify scoped CSS does not exceed baseline !important threshold."""
        ignored = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "_build", "build", "dist", ".refactor", "scratch"}
        css_text = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in self.root.rglob("*.css")
            if not any(part in ignored for part in p.parts)
        )
        count = len(re.findall(r'!important\b', css_text, re.IGNORECASE))
        self.assertLessEqual(count, 11, f"!important count increased to {count}!")

if __name__ == "__main__":
    unittest.main()
