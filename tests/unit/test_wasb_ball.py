"""WASB ball detector: heatmap post-processing and streaming window bookkeeping
(fake model; the real ONNX model is exercised by tools/ball_detector_comparison.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbatrack.detect import wasb_ball
from nbatrack.detect.wasb_ball import WASBBallDetector, detect_video, to_detection_rows


class _Fake(WASBBallDetector):
    """No ONNX session: heatmap j of a window has a blob at x = 10 * frame index."""

    def __init__(self, thr=0.3):
        self.h, self.w, self.frames_in, self.thr = 36, 64, 3, thr
        self.calls = []

    def heatmaps(self, frames_bgr):
        idx = [int(f[0, 0, 0]) for f in frames_bgr]
        self.calls.append(idx)
        hm = np.zeros((3, self.h, self.w), np.float32)
        for j, i in enumerate(idx):
            hm[j, 10:13, 2 + i:5 + i] = 0.8
            hm[j, 11, 3 + i] = 0.9
        return hm


def _patch_frames(monkeypatch, n):
    class Fr:
        def __init__(self, i):
            self.frame_index = i
            self.image = np.full((72, 128, 3), i, np.uint8)
    monkeypatch.setattr("nbatrack.io.video.iter_frames", lambda path: (Fr(i) for i in range(n)))


def test_candidates_heat_weighted_centroid_in_source_pixels():
    d = _Fake()
    hm = np.zeros((36, 64), np.float32)
    hm[10:13, 20:23] = 0.5
    hm[11, 21] = 1.0
    hm[30, 60] = 0.2  # below threshold
    (x, y, peak, hs), = d.candidates(hm, (128, 72))
    assert x == pytest.approx(42.0) and y == pytest.approx(22.0) and peak == pytest.approx(1.0)
    assert d.candidates(np.zeros((36, 64), np.float32), (128, 72)) == []


@pytest.mark.parametrize("n,step", [(10, 1), (10, 3), (9, 3), (4, 3)])
def test_detect_video_covers_every_frame_once(monkeypatch, n, step):
    _patch_frames(monkeypatch, n)
    d = _Fake()
    c, ms = detect_video("x.mp4", d, step=step)
    assert sorted(c.frame_index) == list(range(n))  # one blob per frame, none duplicated
    assert np.allclose(c.x.to_numpy(), (c.frame_index.to_numpy() + 3) * 2)  # blob x follows the frame
    if step == 1:
        assert len(d.calls) == n - 2


def test_to_detection_rows_matches_track_ball_layout():
    c = pd.DataFrame(dict(frame_index=[3], x=[100.0], y=[50.0], peak=[0.7], heat_sum=[5.0]))
    r = to_detection_rows(c)
    assert list(r.columns) == ["frame_index", "cls", "score", "x1", "y1", "x2", "y2"]
    assert r.iloc[0].cls == 37 and (r.iloc[0].x1 + r.iloc[0].x2) / 2 == 100.0


@pytest.mark.skipif(not Path(wasb_ball.DEFAULT_ONNX).exists(), reason="WASB ONNX not exported")
def test_onnx_model_shapes():
    d = WASBBallDetector()
    hm = d.heatmaps([np.zeros((720, 1280, 3), np.uint8)] * 3)
    assert hm.shape == (3, 288, 512) and 0.0 <= hm.min() and hm.max() <= 1.0

def test_tiles_cover_the_frame_and_merge_keeps_highest_peak():
    from nbatrack.detect.wasb_ball import merge_close, tile_boxes
    t = tile_boxes(1280, 720, 2, 2, 0.25)
    assert len(t) == 4 and min(b[0] for b in t) == 0 and max(b[2] for b in t) == 1280
    assert min(b[1] for b in t) == 0 and max(b[3] for b in t) == 720
    w = t[0][2] - t[0][0]
    assert abs(w - 1280 / 1.75) < 2  # 2 tiles with 25% overlap: each ~731 px wide (shrunk 1.43x to 512)
    c = pd.DataFrame(dict(frame_index=[5, 5, 5], x=[100.0, 104.0, 300.0], y=[50.0, 52.0, 60.0],
                          peak=[0.5, 0.9, 0.6], heat_sum=1.0, tile=[-1, 0, 1]))
    m = merge_close(c)
    assert sorted(m.peak.tolist()) == [0.6, 0.9]  # the two within 8 px merged, higher peak kept
