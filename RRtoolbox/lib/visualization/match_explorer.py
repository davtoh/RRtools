# -*- coding: utf-8 -*-
"""
Feature correspondence visualizer for paired images and descriptor keypoints.
"""

import cv2
import numpy as np

from ..config import FLOAT
from ..arrayops.basic import anorm
from ..arrayops.convert import dict2keyPoint
from ..arrayops.filters import sigmoid
from .viewer import Plotim

__all__ = ["MatchExplorer"]


class MatchExplorer(Plotim):
    """
    Draws keypoint pairs obtained on descriptor matching across two images.
    """

    @staticmethod
    def randomColor():
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    def _init_vis_canvas(self, img1, img2):
        """Prepare horizontal side-by-side canvas for img1 and img2."""
        im1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) < 3 else img1
        im2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) < 3 else img2
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = im1
        vis[:h2, w1:w1 + w2] = im2
        return vis, w1, h1

    def _init_keypoints_and_corners(self, kp_pairs, status, H, w1, h1):
        """Standardize keypoint coordinates and projected bounding corners."""
        stat = np.ones(len(kp_pairs), np.bool_) if status is None else status
        kps = [(dict2keyPoint(i), dict2keyPoint(j)) for i, j in kp_pairs]
        p1 = FLOAT([kpp[0].pt for kpp in kps])
        p2 = FLOAT([kpp[1].pt for kpp in kps]) + (w1, 0)
        if H is not None:
            corners = FLOAT([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            self.corners = np.int32(cv2.perspectiveTransform(
                corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        return stat, kps, p1, p2

    def _init_colors(self):
        """Load default and palette colors from image module."""
        from ..image import colors
        for key, val in list(colors.items()):
            if not hasattr(self, key):
                setattr(self, key, val)
        def_color = (0, 255, 0)
        self.goodcolor = colors.get('green', def_color)
        self.badcolor = colors.get('red', def_color)
        self.kpcolor = colors.get('orange', def_color)
        self.framecolor = colors.get('blue', def_color)

    def __init__(self, win, img1, img2, kp_pairs=(), status=None, H=None,
                 show=True, block=True, daemon=True):
        vis, w1, h1 = self._init_vis_canvas(img1, img2)
        stat, kps, p1, p2 = self._init_keypoints_and_corners(kp_pairs, status, H, w1, h1)
        self.status = stat
        self.H = H
        self.p1, self.p2 = p1, p2
        self.kp_pairs = kps
        self.showgoods = True
        self.showbads = False
        self._init_colors()

        super(MatchExplorer, self).__init__(win, vis)
        self.cmdlist.extend(["showgoods", "showbads", "framecolor", "kpcolor", "badcolor", "goodcolor"])
        self.drawline()

        if show:
            self.show(block=block, daemon=daemon)

    def updaterenderer(self, img=None, zoom=True):
        """Update rendered image and re-render line overlays."""
        super(MatchExplorer, self).updaterenderer(img, zoom)
        self.drawline()

    def mousefunc(self):
        if self.builtincontrol():
            self.updaterenderer()
        self.drawrelation()

    def keyfunc(self):
        if self.builtincmd():
            self.drawrelation()

    def _draw_marker_points(self, vis, rp1, rp2, r):
        """Draw good keypoint circles and bad keypoint X-marks."""
        gcol, bcol = self.goodcolor, self.badcolor
        for i, inlier in enumerate(self.status):
            rx1, ry1 = rp1[i]
            rx2, ry2 = rp2[i]
            if self.showgoods and inlier:
                cv2.circle(vis, (rx1, ry1), r, gcol, -1)
                cv2.circle(vis, (rx2, ry2), r, gcol, -1)
            if self.showbads and not inlier:
                thick = r + 5
                cv2.line(vis, (rx1 - r, ry1 - r), (rx1 + r, ry1 + r), bcol, thick)
                cv2.line(vis, (rx1 - r, ry1 + r), (rx1 + r, ry1 - r), bcol, thick)
                cv2.line(vis, (rx2 - r, ry2 - r), (rx2 + r, ry2 + r), bcol, thick)
                cv2.line(vis, (rx2 - r, ry2 + r), (rx2 + r, ry2 - r), bcol, thick)

    def drawline(self):
        """Draw background visualization without interaction."""
        vis = self.rimg.copy()
        self.thick = int(sigmoid(vis.shape[0] * vis.shape[1], 1723567, 8080000, 5, 1))
        if self.H is not None:
            rcorners = np.array([self.real2render(c[0], c[1]) for c in self.corners], np.int32)
            cv2.polylines(vis, [rcorners], True, self.framecolor)

        rp1, rp2 = [], []
        for i, (x1, y1) in enumerate(self.p1):
            x2, y2 = self.p2[i]
            rx1, ry1 = self.real2render(x1, y1, np.int32)
            rx2, ry2 = self.real2render(x2, y2, np.int32)
            rp1.append((rx1, ry1))
            rp2.append((rx2, ry2))
            self.kp_pairs[i][0].pt = (rx1, ry1)
            self.kp_pairs[i][1].pt = (rx2, ry2)

        self._draw_marker_points(vis, rp1, rp2, self.thick)
        self.vis0 = vis.copy()
        self.rp1, self.rp2 = np.int32(rp1), np.int32(rp2)

        for (rx1, ry1), (rx2, ry2), inlier in zip(rp1, rp2, self.status):
            if inlier and self.showgoods:
                cv2.line(vis, (rx1, ry1), (rx2, ry2), self.goodcolor, self.thick)
        self.vis = vis

    def drawrelation(self):
        """Draw keypoint relations near pointer when left button is pressed."""
        if self.flags & cv2.EVENT_FLAG_LBUTTON:
            x, y = self.rx, self.ry
            cur_vis = self.vis0.copy()
            r = self.thick + 8
            m = (anorm(self.rp1 - (x, y)) < r) | (anorm(self.rp2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                rx1, ry1 = self.rp1[i]
                rx2, ry2 = self.rp2[i]
                col = self.goodcolor if self.status[i] else self.badcolor
                cv2.line(cur_vis, (rx1, ry1), (rx2, ry2), col, self.thick)
                kp1s.append(self.kp_pairs[i][0])
                kp2s.append(self.kp_pairs[i][1])
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=self.kpcolor)
            cur_vis = cv2.drawKeypoints(cur_vis, kp2s, flags=4, color=self.kpcolor)
            self.rimg = cur_vis
        else:
            self.rimg = self.vis

        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[int(self.y), int(self.x)])
        else:
            self.builtinplot()

