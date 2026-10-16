"""Write synthetic test videos with known per-frame timestamps."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_video(path: str | Path, frames: list[np.ndarray], pts_sec: list[float],
                time_base: Fraction = Fraction(1, 90000), codec: str = "libx264",
                rate: int = 30) -> list[int]:
    """Encode BGR frames with explicit presentation times. Returns the integer pts used."""
    h, w = frames[0].shape[:2]
    pts = [int(round(t / time_base)) for t in pts_sec]
    with av.open(str(path), "w") as c:
        s = c.add_stream(codec, rate=rate)
        s.width, s.height = w, h
        s.pix_fmt = "yuv420p"
        s.time_base = time_base
        s.codec_context.time_base = time_base
        if codec == "libx264":
            s.options = {"crf": "18", "preset": "ultrafast", "bf": "0"}
        for img, p in zip(frames, pts):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")
            vf.pts = p
            vf.time_base = time_base
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return pts


def numbered_frames(n: int, size: tuple[int, int] = (320, 180)) -> list[np.ndarray]:
    """Frames whose mean brightness encodes the index (robust to compression)."""
    w, h = size
    out = []
    for i in range(n):
        img = np.full((h, w, 3), (i * 37) % 200 + 30, np.uint8)
        out.append(img)
    return out
