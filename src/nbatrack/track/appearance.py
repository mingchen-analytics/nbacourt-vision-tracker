"""Cheap hand-crafted person appearance descriptor for team / referee classification.

Three horizontal body regions of the box (upper body 15-45%, shorts 45-65%, legs 65-92%;
central 60% of the width) each give an HSV histogram (8 hue x 3 sat x 4 value bins).
Concatenated and square-rooted (Hellinger), so Euclidean distance between descriptors is
a Hellinger-type distance. Computed in a separate pass over the video from the cached
detections, so the detector does not need to be re-run.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from nbatrack.io.video import iter_frames

REGIONS = ((0.15, 0.45), (0.45, 0.65), (0.65, 0.92))
BINS = (8, 3, 4)
DIM = len(REGIONS) * BINS[0] * BINS[1] * BINS[2]


def descriptor(bgr: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 24:
        return None
    H, W = bgr.shape[:2]
    xa, xb = max(0, int(x1 + 0.2 * w)), min(W, int(x2 - 0.2 * w))
    parts = []
    for a, b in REGIONS:
        ya, yb = max(0, int(y1 + a * h)), min(H, int(y1 + b * h))
        if xb - xa < 2 or yb - ya < 2:
            return None
        hsv = cv2.cvtColor(bgr[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256]).ravel()
        parts.append(np.sqrt(hist / max(hist.sum(), 1e-6)))
    return np.concatenate(parts).astype(np.float32) / np.sqrt(len(REGIONS))


def extract_descriptors(video: str, det: pd.DataFrame, min_score: float = 0.3, out_path=None) -> np.ndarray:
    """Descriptors for person rows of the detection cache (NaN rows where skipped).
    With `out_path` the array is written straight into a .npy memory map (a long video has
    ~10^6 detection rows, ~1 GB as float32) instead of being held in memory."""
    if out_path is not None:
        out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(len(det), DIM))
        out[:] = np.nan
    else:
        out = np.full((len(det), DIM), np.nan, np.float32)
    rows = det[(det.cls == 1) & (det.score >= min_score)]
    by_f = {f: g for f, g in rows.groupby("frame_index")}
    pos = {idx: k for k, idx in enumerate(det.index)}
    for fr in iter_frames(video):
        g = by_f.get(fr.frame_index)
        if g is None:
            continue
        for idx, r in zip(g.index, g[["x1", "y1", "x2", "y2"]].to_numpy(float)):
            d = descriptor(fr.image, r)
            if d is not None:
                out[pos[idx]] = d
    if out_path is not None:
        out.flush()
    return out
