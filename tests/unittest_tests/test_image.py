import unittest
import numpy as np
import RRtoolbox.lib.image as img
from RRtoolbox.lib.image.conversions import (
    transposeIm, colors, bgra2bgr, np2str, str2np, hist_match
)
from RRtoolbox.lib.image.coordinates import (
    ImCoors, drawcoorpoints, limitaxispoints
)
from RRtoolbox.lib.image.loaders import (
    ImFactory, loadFunc, ImLoader, PathLoader
)
from RRtoolbox.lib.image.interactive import (
    separe, quadrants, getrectcoors, random_color
)
from RRtoolbox.lib.image.core import Image


class TestImageSubsystem(unittest.TestCase):

    def test_facade_reexports(self):
        """Verify that all core symbols are accessible via RRtoolbox.lib.image facade."""
        expected_symbols = [
            "transposeIm", "supported_formats", "green", "red", "white", "orange",
            "black", "blue", "colors", "bgra_dtype", "plt2bgr", "plt2bgra", "fig2bgr",
            "np2str", "str2np", "fig2bgra", "qi2np", "np2qi", "gray2qi", "rgb2qi",
            "bgra2bgr", "hist_match", "checkLoaded", "loadcv", "loadsfrom",
            "interpretImage", "convertAs", "ImFactory", "loadFunc", "ImLoader",
            "PathLoader", "LoaderDict", "try_loads", "ImCoors", "drawcoorpoints",
            "myline", "drawcooraxes", "drawcoorpolyline", "drawcoorarea",
            "drawcoorpolyArrow", "drawcoorperspective", "limitaxispoints",
            "GetCoors", "getcoors", "separe", "getrectcoors", "quadrants",
            "getgeometrycoors", "random_color", "Image",
        ]
        for sym in expected_symbols:
            self.assertTrue(hasattr(img, sym), "Missing symbol in facade: {}".format(sym))

    def test_conversions(self):
        """Test array and color conversions."""
        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        trans = transposeIm(arr)
        self.assertEqual(trans.shape, (30, 20, 3))

        s = np2str(arr)
        restored = str2np(s, arr.shape)
        np.testing.assert_array_equal(arr, restored)

        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        bgra[:, :, :3] = 100
        bgra[:, :, 3] = 255
        bgr = bgra2bgr(bgra)
        self.assertEqual(bgr.shape, (10, 10, 3))

    def test_hist_match(self):
        """Test histogram matching."""
        source = np.arange(256, dtype=np.uint8).reshape((16, 16))
        template = np.full((16, 16), 128, dtype=np.uint8)
        matched = hist_match(source, template)
        self.assertEqual(matched.shape, source.shape)

    def test_coordinates_imcoors(self):
        """Test ImCoors geometric properties and caching."""
        pts = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        coors = ImCoors(pts)
        self.assertEqual(len(coors), 4)
        self.assertEqual(coors.min, (0, 0))
        self.assertEqual(coors.max, (10, 10))
        self.assertEqual(coors.boxCenter, (5.0, 5.0))
        self.assertEqual(coors.mean, (5.0, 5.0))
        self.assertEqual(coors.area, 100.0)
        self.assertEqual(coors.rectangularArea, 100.0)
        self.assertAlmostEqual(coors.rectangularity, 1.0)

    def test_drawcoorpoints(self):
        """Test drawcoorpoints."""
        vis = np.zeros((50, 50, 3), dtype=np.uint8)
        pts = [(10, 10), (20, 20)]
        out = drawcoorpoints(vis, pts)
        self.assertEqual(out.shape, (50, 50, 3))

    def test_limitaxispoints(self):
        """Test point boundary clamping."""
        pt = (120, -5)
        clamped = limitaxispoints(pt, maxc=100, minc=0)
        self.assertEqual(clamped, (100, 0))

    def test_interactive_helpers(self):
        """Test geometric grouping and partition helpers."""
        vals = [(10, 20), (5, 30), (15, 10)]
        greater, lesser = separe(vals, sep=8, axis=0)
        self.assertEqual(len(greater), 2)
        self.assertEqual(len(lesser), 1)

        pts = [(0, 0), (10, 0), (0, 10), (10, 10)]
        rect = getrectcoors(pts)
        self.assertEqual(rect, [(0, 0), (10, 0), (0, 10), (10, 10)])

        quads = quadrants(pts)
        self.assertEqual(len(quads), 4)

        color = random_color(channels=3)
        self.assertEqual(len(color), 3)

    def test_loaders(self):
        """Test loader factories and PathLoader container."""
        factory = ImFactory()
        self.assertIsNotNone(factory)

        loader = loadFunc(flag=1)
        self.assertTrue(callable(loader))

        loader_class = ImLoader("dummy_path.png", flag=0)
        self.assertEqual(loader_class.path, "dummy_path.png")

        pl = PathLoader(["a.png", "b.png"])
        self.assertEqual(len(pl), 2)

    def test_image_core(self):
        """Test high-level Image wrapper."""
        img_obj = Image(name="test", ext="png", path="foo/bar")
        self.assertEqual(img_obj.name, "test")
        self.assertEqual(img_obj.ext, ".png")
        self.assertEqual(img_obj.path, "foo/bar/")

        img_obj.shape = (100, 200)
        self.assertEqual(img_obj.shape, (200, 100))


if __name__ == '__main__':
    unittest.main()

