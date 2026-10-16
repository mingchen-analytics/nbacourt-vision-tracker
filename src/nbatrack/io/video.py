"""Video decoding with exact presentation timestamps (PyAV / FFmpeg).

Why not cv2.VideoCapture: its CAP_PROP_POS_MSEC is derived from a nominal frame rate and
is unreliable for variable-frame-rate files and files whose stream does not start at 0.

Timestamp definitions (all stored in outputs):
  pts            : raw integer presentation timestamp from the container, in `time_base`
  pts_sec        : pts * time_base (absolute stream time)
  timestamp_sec  : pts_sec - stream_start_sec, so the first presented frame is ~0.0.
                   This matches what HTML5 players report as `currentTime` for typical MP4
                   files whose first frame is at the stream start.
  frame_index    : 0-based index in presentation (decode output) order.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps_nominal: float  # average frame rate reported by the container
    time_base: str  # e.g. "1/15360"
    start_sec: float
    duration_sec: float | None
    n_frames_reported: int | None  # container metadata; may be missing / inaccurate
    codec: str
    sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    frame_index: int
    pts: int
    pts_sec: float
    timestamp_sec: float
    image: np.ndarray  # BGR uint8 (H, W, 3)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def probe(path: str | Path, with_hash: bool = False) -> VideoInfo:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = s.time_base
        start = float(s.start_time * tb) if s.start_time is not None else 0.0
        dur = float(s.duration * tb) if s.duration is not None else (
            c.duration / 1e6 if c.duration else None)
        rate = s.average_rate or s.guessed_rate
        return VideoInfo(
            path=str(path),
            width=s.codec_context.width,
            height=s.codec_context.height,
            fps_nominal=float(rate) if rate else float("nan"),
            time_base=f"{tb.numerator}/{tb.denominator}",
            start_sec=start,
            duration_sec=dur,
            n_frames_reported=int(s.frames) if s.frames else None,
            codec=s.codec_context.name,
            sha256=file_sha256(path) if with_hash else None,
        )


def iter_frames(path: str | Path, start_sec: float | None = None, end_sec: float | None = None,
                stride: int = 1, threads: int = 0) -> Iterator[Frame]:
    """Yield decoded frames in presentation order with exact timestamps.

    `frame_index` always counts every presented frame from the beginning of the file,
    even when `start_sec` / `stride` skip some, so indices are stable across runs.
    Seeking is done by decoding from the start (exact, simple); fine for clips, and
    Phase 4 will add keyframe seeking for full games.
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        if threads:
            s.thread_count = threads
        tb = s.time_base
        t0 = float(s.start_time * tb) if s.start_time is not None else None
        idx = -1
        for fr in c.decode(s):
            if fr.pts is None:
                continue  # cannot timestamp reliably; skipped frames are not indexed
            idx += 1
            pts_sec = float(Fraction(fr.pts) * tb)
            if t0 is None:
                t0 = pts_sec
            ts = pts_sec - t0
            if start_sec is not None and ts < start_sec - 1e-9:
                continue
            if end_sec is not None and ts > end_sec + 1e-9:
                break
            if idx % stride:
                continue
            yield Frame(idx, int(fr.pts), pts_sec, ts, fr.to_ndarray(format="bgr24"))
