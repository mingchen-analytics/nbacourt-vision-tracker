"""Diagnostic side-by-side view: broadcast frame with overlays | top-down court."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np

from nbatrack.calib.models import CalibrationModel
from nbatrack.court.render import TopDownCanvas, draw_projected_court

TEAM_COLORS = {"A": (40, 40, 220), "B": (220, 200, 40), None: (160, 160, 160)}
QUALITY_COLORS = {"good": (0, 200, 0), "fair": (0, 200, 255), "poor": (0, 0, 255),
                  "failed": (0, 0, 160)}


def draw_frame_overlay(img: np.ndarray, players: list[dict], model: CalibrationModel | None,
                       quality: str, timestamp_sec: float, frame_index: int) -> np.ndarray:
    out = img.copy()
    if model is not None and quality in ("good", "fair"):
        draw_projected_court(out, model.court_to_image, (0, 255, 255), 1)
    for p in players:
        if p.get("bbox_x1") is None:
            continue
        x1, y1, x2, y2 = (int(round(p[k])) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        col = TEAM_COLORS.get(p.get("team_id"), TEAM_COLORS[None])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, p["track_id"].split("_t")[-1], (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 2)
        if p.get("image_x") is not None:
            cv2.circle(out, (int(p["image_x"]), int(p["image_y"])), 3, (255, 255, 255), -1)
    txt = f"t={timestamp_sec:8.3f}s  frame={frame_index}  calib={quality}"
    cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, QUALITY_COLORS.get(quality, (255, 255, 255)), 2)
    return out


def draw_topdown(canvas: TopDownCanvas, bg: np.ndarray, players: list[dict],
                 truth_xy: np.ndarray | None = None) -> np.ndarray:
    out = bg.copy()
    legend = "filled: estimate (ring = 2 sigma)"
    if truth_xy is not None:  # synthetic ground truth, hollow gray circles
        legend += " | hollow gray: synthetic truth (incl. off-camera)"
        for q in canvas.to_px(truth_xy):
            if np.all(np.isfinite(q)):
                cv2.circle(out, tuple(q.round().astype(int)), 9, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(out, legend, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1,
                cv2.LINE_AA)
    for p in players:
        if p.get("court_x_m") is None or p.get("position_status") == "unknown":
            continue
        q = canvas.to_px(np.array([[p["court_x_m"], p["court_y_m"]]]))[0].round().astype(int)
        col = TEAM_COLORS.get(p.get("team_id"), TEAM_COLORS[None])
        if p.get("court_sigma_m") is not None:
            r = int(max(2, min(60, 2 * p["court_sigma_m"] * canvas.s)))  # 2-sigma circle
            cv2.circle(out, tuple(q), r, col, 1, cv2.LINE_AA)
        cv2.circle(out, tuple(q), 6, col, -1, cv2.LINE_AA)
        cv2.putText(out, p["track_id"].split("_t")[-1], (q[0] + 7, q[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return out


def compose(left: np.ndarray, right: np.ndarray, height: int = 540) -> np.ndarray:
    def fit(img):
        s = height / img.shape[0]
        return cv2.resize(img, (int(round(img.shape[1] * s / 2) * 2), height))
    out = np.hstack([fit(left), fit(right)])
    if out.shape[1] % 2:
        out = out[:, :-1]
    return out


class VideoSink:
    """H.264 MP4 writer (browser-playable).

    Frames are piped as raw BGR into a separate `ffmpeg` process (libx264). The earlier
    in-process PyAV encoder intermittently produced corrupted NAL units when a PyAV
    decoder and two encoders ran in the same process (observed 2026-10-15; not
    reproducible in isolation), so encoding is isolated in its own process. Falls back to
    PyAV only if ffmpeg is not on PATH.

    Timing: output is constant-frame-rate at `fps`. Every write() checks the source
    timestamp against the CFR grid; deviations (> half a frame) are recorded in
    `<file>.timing.json` instead of being silently re-timed.
    Integrity: on close the file is fully decoded once; any decoder error is reported
    (`verify_errors`) and a `<file>.VERIFY_FAILED` marker is written.
    Writes go to a temporary file that atomically replaces the target; if the target is
    locked by a player, the new file is kept under a timestamped name.
    """

    def __init__(self, path: str | Path, fps: float = 30.0, verify: bool = True):
        import shutil
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path + ".partial.mp4"
        self.fps = fps
        self.verify = verify
        self.ffmpeg = shutil.which("ffmpeg")
        self.proc = None
        self.c = None
        self.s = None
        self.n = 0
        self.t0 = None
        self.timing_issues: list = []
        self.verify_errors = 0

    def _start(self, img: np.ndarray) -> None:
        import subprocess
        from fractions import Fraction
        h, w = img.shape[:2]
        if self.ffmpeg:
            rate = Fraction(self.fps).limit_denominator(1001)
            cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-threads", "4", "-x264-params", "rc-lookahead=10", "-movflags", "+faststart",
                   self.tmp]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:  # pragma: no cover - fallback
            self.c = av.open(self.tmp, "w")
            self.s = self.c.add_stream("libx264", rate=int(round(self.fps)))
            self.s.width, self.s.height = w, h
            self.s.pix_fmt = "yuv420p"
            self.s.options = {"crf": "23", "preset": "veryfast", "rc-lookahead": "10"}

    def write(self, img: np.ndarray, timestamp_sec: float) -> None:
        if self.proc is None and self.c is None:
            self._start(img)
            self.t0 = timestamp_sec
        expected = self.t0 + self.n / self.fps
        if abs(timestamp_sec - expected) > 0.5 / self.fps and len(self.timing_issues) < 1000:
            self.timing_issues.append({"frame": self.n, "source_ts": timestamp_sec, "output_ts": expected})
        img = np.ascontiguousarray(img, dtype=np.uint8)
        if self.proc is not None:
            self.proc.stdin.write(img.tobytes())
        else:  # pragma: no cover
            f = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in self.s.encode(f):
                self.c.mux(pkt)
        self.n += 1

    def close(self) -> None:
        import json
        import os
        import subprocess
        import time as _t
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.tmp}")
        elif self.c is not None:  # pragma: no cover
            for pkt in self.s.encode():
                self.c.mux(pkt)
            self.c.close()
        else:
            return
        if self.verify and self.ffmpeg:
            r = subprocess.run([self.ffmpeg, "-v", "error", "-i", self.tmp, "-f", "null", "-"],
                               capture_output=True, text=True)
            self.verify_errors = len([x for x in r.stderr.splitlines() if x.strip()])
        try:
            os.replace(self.tmp, self.path)
        except PermissionError:  # target open in a player: keep the new file under a new name
            alt = self.path[:-4] + _t.strftime("_%Y%m%d_%H%M%S") + ".mp4"
            os.replace(self.tmp, alt)
            print(f"WARNING: {self.path} is locked by another program; wrote {alt}")
            self.path = alt
        side = Path(self.path + ".timing.json")
        if self.timing_issues:
            side.write_text(json.dumps({"fps": self.fps, "non_cfr_frames": self.timing_issues}, indent=1))
        elif side.exists():
            side.unlink()
        marker = Path(self.path + ".VERIFY_FAILED")
        if self.verify_errors:
            marker.write_text(f"{self.verify_errors} decoder error lines")
            print(f"WARNING: {self.path} failed decode verification ({self.verify_errors} error lines)")
        elif marker.exists():
            marker.unlink()
