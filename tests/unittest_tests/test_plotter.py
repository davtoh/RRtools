# -*- coding: utf-8 -*-
"""
Automated unit tests for RRtoolbox.lib.plotter facade and RRtoolbox.lib.visualization package.
"""

import unittest
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Enforce headless matplotlib for CI and unit tests

import RRtoolbox.lib.plotter as plotter
import RRtoolbox.lib.visualization as vis


class TestPlotterFacade(unittest.TestCase):
    """Verify facade re-exports match underlying visualization package."""

    def test_facade_symbols_present(self):
        for sym in plotter.__all__:
            self.assertTrue(hasattr(plotter, sym), "Missing symbol {} in plotter facade".format(sym))
            self.assertTrue(hasattr(vis, sym), "Missing symbol {} in visualization package".format(sym))

    def test_facade_and_pkg_identity(self):
        self.assertIs(plotter.Plotim, vis.Plotim)
        self.assertIs(plotter.MatchExplorer, vis.MatchExplorer)
        self.assertIs(plotter.Edger, vis.Edger)
        self.assertIs(plotter.Imtester, vis.Imtester)
        self.assertIs(plotter.fastplt, vis.fastplt)
        self.assertIs(plotter.graph_filter, vis.graph_filter)
        self.assertIs(plotter.plotPointsContour, vis.plotPointsContour)
        self.assertIs(plotter.limitaxis, vis.limitaxis)
        self.assertIs(plotter.background, vis.background)
        self.assertIs(plotter.convert2bgr, vis.convert2bgr)
        self.assertIs(plotter.convert2bgra, vis.convert2bgra)


class TestVisualizationHelpers(unittest.TestCase):
    """Verify helper and conversion functions in visualization."""

    def test_limitaxis(self):
        self.assertEqual(vis.limitaxis(50, 100, 0), 50)
        self.assertEqual(vis.limitaxis(150, 100, 0), 100)
        self.assertEqual(vis.limitaxis(-10, 100, 0), 0)

    def test_background(self):
        bg = vis.background((100, 150, 200), x=10, y=20)
        self.assertEqual(bg.shape, (20, 10, 3))
        self.assertEqual(tuple(bg[0, 0]), (100, 150, 200))

    def test_convert2bgr(self):
        gray = np.zeros((30, 40), dtype=np.uint8)
        gray[10, 10] = 200
        bgr = vis.convert2bgr(gray)
        self.assertEqual(bgr.shape, (30, 40, 3))
        self.assertEqual(tuple(bgr[10, 10]), (200, 200, 200))

        rgba = np.full((30, 40, 4), 255, dtype=np.uint8)
        bgr_from_rgba = vis.convert2bgr(rgba)
        self.assertEqual(bgr_from_rgba.shape, (30, 40, 3))

    def test_convert2bgra(self):
        bgr = np.zeros((30, 40, 3), dtype=np.uint8)
        bgra = vis.convert2bgra(bgr)
        self.assertEqual(bgra.shape, (30, 40, 4))


class TestPlotimAndComponents(unittest.TestCase):
    """Verify Plotim coordinate mapping, command formatting, and mixins."""

    def test_plotim_coordinates_and_transforms(self):
        test_img = np.zeros((200, 300, 3), dtype=np.uint8)
        plot = vis.Plotim("unit_test_win", test_img)
        self.assertEqual(plot.maxX, 300)
        self.assertEqual(plot.maxY, 200)

        real_x, real_y = plot.render2real(0, 0, astype=None)
        self.assertEqual((real_x, real_y), (0, 0))

        real_x2, real_y2 = plot.render2real(plot.rW, plot.rH, astype=None)
        self.assertAlmostEqual(real_x2, 300, delta=1.0)
        self.assertAlmostEqual(real_y2, 200, delta=1.0)

        rend_x, rend_y = plot.real2render(300, 200)
        self.assertAlmostEqual(rend_x, plot.rW, delta=1.0)
        self.assertAlmostEqual(rend_y, plot.rH, delta=1.0)
        plot.clean()

    def test_formatcmd(self):
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        plot = vis.Plotim("cmd_test_win", test_img)
        formatted = plot.formatcmd("showcoors = True")
        self.assertIn("self.showcoors", formatted)
        plot.clean()

    def test_imtester_static_helpers(self):
        info_lines = vis.Imtester.formatinfo("word1 word2 word3 word4 word5 word6", words=2)
        self.assertTrue(len(info_lines) >= 2)

        test_img = np.full((50, 50), 128, dtype=np.uint8)
        th, info, title = vis.Imtester.applythresh(
            test_img, cv2.THRESH_BINARY, adaptativetoggle=0, threshtoggle=1,
            th=100, blocksz=11, c=30, i="1", ti="1"
        )
        self.assertEqual(th.shape, (50, 50))
        self.assertIn("th1", info)


class TestMatplotlibHelpers(unittest.TestCase):
    """Verify Matplotlib plotting utilities."""

    def test_plot_points_contour(self):
        pts = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=float)
        ax = vis.plotPointsContour(pts, annotate=False)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()