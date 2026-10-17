"""Shot (camera segment) boundaries, flash frames and a main-camera proxy.

Same rules as tools/inspect_video.py (which produced the reviewed clip-6 segments):
  hist_dist   total-variation distance of 16x4x4 HSV histograms of consecutive 320x180 frames
  flash       frame f differs from both neighbours but f-1 and f+1 match -> one over-exposed frame
  hard cut    a hist_dist peak above `cut_thr` that is not part of a flash
  floor_frac  share of hardwood / light-paint pixels in the lower 2/3: high for the main
              broadcast camera, low for close-ups, crowd shots and most graphics.
A camera cut is NOT a possession boundary; it only delimits views.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def small(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)


def hsv_hist(sm: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1)


def floor_fraction(sm: np.ndarray) -> float:
    hsv = cv2.cvtColor(sm[sm.shape[0] // 3:], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 5) & (h <= 30) & (s >= 30) & (s <= 200) & (v >= 120)).mean())


class ShotStats:
    """Feed every decoded frame in order; collect per-frame statistics."""

    def __init__(self):
        self.rows: list[dict] = []
        self._h1 = self._h2 = None

    def add(self, frame_index: int, ts: float, bgr: np.ndarray) -> None:
        sm = small(bgr)
        h = hsv_hist(sm)
        hd = 0.5 * float(np.abs(h - self._h1).sum()) if self._h1 is not None else 0.0
        hd2 = 0.5 * float(np.abs(h - self._h2).sum()) if self._h2 is not None else 0.0
        self.rows.append(dict(frame_index=frame_index, timestamp_sec=ts, hist_dist=hd, hist_dist2=hd2,
                              floor_frac=floor_fraction(sm)))
        self._h2, self._h1 = self._h1, h

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def detect_cuts(st: pd.DataFrame, cut_thr: float = 0.35) -> tuple[list[int], list[int]]:
    """(hard-cut frames = first frame of a new shot, flash frames)."""
    d = st.hist_dist.to_numpy()
    d2 = st.hist_dist2.to_numpy()
    f = st.frame_index.to_numpy()
    flash = {i for i in range(1, len(d) - 1)
             if d[i] > cut_thr and d[i + 1] > cut_thr and d2[i + 1] < cut_thr / 2}
    cuts = [int(f[i]) for i in np.flatnonzero(d > cut_thr) if i not in flash and (i - 1) not in flash]
    return cuts, [int(f[i]) for i in sorted(flash)]


def shot_segments(st: pd.DataFrame, cuts: list[int], main_floor_frac: float = 0.25) -> pd.DataFrame:
    """Segments between cuts with a view guess: `main_broadcast` when the median floor fraction
    is high, else `other` (close-up / crowd / graphic / replay-like). The guess is a proxy and is
    meant to be reviewed; a reviewed segments file overrides it."""
    starts = [int(st.frame_index.iloc[0])] + sorted(cuts)
    ends = [s - 1 for s in starts[1:]] + [int(st.frame_index.iloc[-1])]
    rows = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        g = st[(st.frame_index >= a) & (st.frame_index <= b)]
        ff = float(g.floor_frac.median())
        rows.append(dict(camera_segment_id=k, start_frame=a, end_frame=b,
                         start_sec=float(g.timestamp_sec.iloc[0]), end_sec=float(g.timestamp_sec.iloc[-1]),
                         median_floor_frac=ff, view_type="main_broadcast" if ff >= main_floor_frac else "other",
                         view_source="auto_floor_fraction"))
    return pd.DataFrame(rows)
