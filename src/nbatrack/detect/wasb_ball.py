"""WASB (BMVC 2023) basketball ball detector -- experimental integration via ONNX.

Code: https://github.com/nttcom/WASB-SBDT (MIT). The authors' basketball checkpoint
(wasb_basketball_best.pth.tar, 5.8 MB, trained on the NBA dataset of Yan et al.; weight and
dataset licenses are NOT stated by the authors -> research use only) is exported once to
ONNX by tools/export_wasb_onnx.py in a torch-only process (running torch in the same
process as OpenCV/PyAV crashed with an access violation on this Windows setup, an OpenMP
runtime conflict). Runtime inference uses ONNX Runtime only.

Model: 3 consecutive RGB frames (ImageNet-normalized, 512x288, stacked to 9 channels)
-> 3 sigmoid heatmaps. Post-processing follows the repository: threshold, connected
components, heat-weighted centroid; mapped back to source pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_ONNX = Path("models/onnx/wasb_basketball_288x512.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class WASBBallDetector:
    def __init__(self, onnx_path: str | Path = DEFAULT_ONNX, provider: str = "cpu",
                 score_threshold: float = 0.3):
        prov = {"cpu": ["CPUExecutionProvider"], "dml": ["DmlExecutionProvider", "CPUExecutionProvider"]}[provider]
        so = ort.SessionOptions()
        if provider == "dml":
            so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        self.frames_in = int(inp.shape[1]) // 3
        self.thr = score_threshold

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((img - MEAN) / STD).transpose(2, 0, 1)

    def heatmaps(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """`frames_in` frames -> (frames_in, h, w) heatmaps in [0, 1]."""
        x = np.concatenate([self._prep(f) for f in frames_bgr], 0)[None].astype(np.float32)
        return self.sess.run(None, {self.name: x})[0][0]

    def candidates(self, hm: np.ndarray, src_wh: tuple[int, int]) -> list[tuple[float, float, float, float]]:
        """Blobs above threshold -> (x, y, peak, heat_sum) in source pixels."""
        sx, sy = src_wh[0] / self.w, src_wh[1] / self.h
        out = []
        if hm.max() <= self.thr:
            return out
        n, lab = cv2.connectedComponents((hm > self.thr).astype(np.uint8))
        for m in range(1, n):
            ys, xs = np.where(lab == m)
            wts = hm[ys, xs]
            out.append((float((xs * wts).sum() / wts.sum() * sx), float((ys * wts).sum() / wts.sum() * sy),
                        float(wts.max()), float(wts.sum())))
        return out


def detect_video(video: str, det: WASBBallDetector, step: int = 3, max_frames: int | None = None):
    """Stream a video through the model with non-overlapping (step=3) or sliding (step=1)
    windows of `frames_in` frames. A frame covered by several windows gets the average of
    their heatmaps. Only the current window is held in memory.
    Returns (candidates DataFrame [frame_index, x, y, peak, heat_sum], ms_per_frame), where
    ms_per_frame covers preprocessing, inference and post-processing (not decoding)."""
    import time
    from collections import deque

    import pandas as pd

    from nbatrack.io.video import iter_frames

    k = det.frames_in
    buf: deque = deque(maxlen=k)  # (frame_index, BGR image)
    pending: dict[int, list] = {}  # frame -> [heatmap sum, count]
    rows: list[dict] = []
    t_model = 0.0
    wh = None
    n = 0
    last_start = None

    def run(window):
        nonlocal t_model
        t0 = time.perf_counter()
        hm = det.heatmaps([im for _, im in window])
        t_model += time.perf_counter() - t0
        for (f, _), m in zip(window, hm):
            if f in pending:
                pending[f][0] += m
                pending[f][1] += 1
            elif f not in done:
                pending[f] = [m.copy(), 1]

    done: set[int] = set()

    def finalize(upto):
        nonlocal t_model
        t0 = time.perf_counter()
        for f in sorted(x for x in pending if x < upto):
            s, c = pending.pop(f)
            for x, y, peak, hs in det.candidates(s / c, wh):
                rows.append(dict(frame_index=f, x=x, y=y, peak=peak, heat_sum=hs))
            done.add(f)
        t_model += time.perf_counter() - t0

    for fr in iter_frames(video):
        if max_frames is not None and fr.frame_index >= max_frames:
            break
        wh = (fr.image.shape[1], fr.image.shape[0])
        buf.append((fr.frame_index, fr.image))
        n += 1
        if len(buf) == k:
            s = buf[0][0]
            if last_start is None or s - last_start >= step:
                run(list(buf))
                last_start = s
                finalize(s + step)
    if len(buf) == k and buf[-1][0] not in done and buf[-1][0] not in pending:
        run(list(buf))  # tail frames not covered by a regular window
    finalize(float("inf"))
    return pd.DataFrame(rows, columns=["frame_index", "x", "y", "peak", "heat_sum"]), 1000 * t_model / max(1, n)


def to_detection_rows(cands, half_size_px: float = 8.0, cls: int = 37):
    """WASB candidates -> rows in the detections-cache layout used by track_ball
    (cls 37 = COCO sports ball, score = heatmap peak, small box around the centroid)."""
    import pandas as pd

    return pd.DataFrame(dict(frame_index=cands.frame_index.astype(int), cls=cls, score=cands.peak,
                             x1=cands.x - half_size_px, y1=cands.y - half_size_px,
                             x2=cands.x + half_size_px, y2=cands.y + half_size_px))
