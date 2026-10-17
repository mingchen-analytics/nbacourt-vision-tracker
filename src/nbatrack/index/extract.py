"""Single decoding pass for the Video Indexing Engine (independent of the tracking pipeline).

Every frame: shot statistics (cuts, flashes, floor fraction).
Every `sample_every`-th frame: scoreboard presence score and the character glyphs of every
scoreboard field. Glyphs are stored raw (not yet read), so labelling templates or changing the
reader never requires decoding the video again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from nbatrack.index.scoreboard import ScoreboardConfig, crop, field_glyphs, ncc
from nbatrack.index.shots import ShotStats
from nbatrack.io.video import iter_frames, probe


@dataclass
class Extraction:
    frame_stats: pd.DataFrame  # every frame
    samples: pd.DataFrame  # sampled frames: sample_id, frame_index, timestamp_sec, presence
    glyph_meta: pd.DataFrame  # one row per glyph: sample_id, field, pos, n_in_field, aspect, x, y, w, h
    glyphs: np.ndarray  # (n_glyphs, H, W) float32
    info: dict

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.frame_stats.to_parquet(out / "frame_stats.parquet")
        self.samples.to_parquet(out / "samples.parquet")
        self.glyph_meta.to_parquet(out / "glyph_meta.parquet")
        np.save(out / "glyphs.npy", self.glyphs)
        pd.Series(self.info).to_json(out / "extract_info.json", indent=1)

    @classmethod
    def load(cls, out: Path) -> "Extraction":
        import json
        return cls(pd.read_parquet(out / "frame_stats.parquet"), pd.read_parquet(out / "samples.parquet"),
                   pd.read_parquet(out / "glyph_meta.parquet"), np.load(out / "glyphs.npy"),
                   json.loads((out / "extract_info.json").read_text()))


def extract(video: str | Path, cfg: ScoreboardConfig, sample_every: int = 6,
            max_frames: int | None = None) -> Extraction:
    t0 = time.time()
    proc = psutil.Process()
    peak = 0.0
    info = probe(video)
    shots = ShotStats()
    samples, meta, glyphs = [], [], []
    ref = None
    for fr in iter_frames(video):
        if max_frames is not None and fr.frame_index >= max_frames:
            break
        shots.add(fr.frame_index, fr.timestamp_sec, fr.image)
        if cfg.presence_roi is not None and ref is None and fr.frame_index == (cfg.presence_reference_frame or 0):
            ref = crop(fr.image, cfg.presence_roi).astype(np.float32).mean(2)
        if fr.frame_index % sample_every:
            continue
        sid = len(samples)
        pres = None
        if ref is not None:
            pres = ncc(crop(fr.image, cfg.presence_roi).astype(np.float32).mean(2), ref)
        samples.append(dict(sample_id=sid, frame_index=fr.frame_index, timestamp_sec=fr.timestamp_sec, presence=pres))
        for field, roi in cfg.fields.items():
            gl = field_glyphs(crop(fr.image, roi))
            for pos, d in enumerate(gl):
                x, y, w, h = d["box"]
                meta.append(dict(sample_id=sid, field=field, pos=pos, n_in_field=len(gl), aspect=d["aspect"],
                                 x=x, y=y, w=w, h=h))
                glyphs.append(d["glyph"])
        if sid % 50 == 0:
            peak = max(peak, proc.memory_info().private / 2**20)
    st = shots.frame()
    elapsed = time.time() - t0
    run = dict(video=str(video), n_frames=len(st), fps=info.fps_nominal, sample_every=sample_every,
               n_samples=len(samples), n_glyphs=len(glyphs), seconds=elapsed,
               ms_per_frame=1000 * elapsed / max(1, len(st)), peak_private_mb=peak)
    return Extraction(st, pd.DataFrame(samples), pd.DataFrame(meta),
                      np.asarray(glyphs, np.float32).reshape(-1, 24, 16), run)
