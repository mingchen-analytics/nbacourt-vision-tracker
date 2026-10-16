"""Detect static broadcast overlays (scoreboard, network logo) from temporal variance.

Overlay pixels stay constant while the camera moves. Candidate regions must also lie
near the image border, so a static camera's floor is not mistaken for an overlay.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_overlay_mask(frames_bgr: list[np.ndarray], std_thresh: float = 3.0,
                          border_px: int = 140, min_area: int = 400, pad: int = 6) -> np.ndarray:
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr])
    std = g.std(axis=0)
    cand = (std < std_thresh).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    h, w = cand.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        near_border = x < border_px or y < border_px or x + bw > w - border_px or y + bh > h - border_px
        if area >= min_area and near_border:
            out[y:y + bh, x:x + bw] = 255  # use the bounding box (overlays are rectangles)
    if pad:
        out = cv2.dilate(out, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    return out
