"""`index` step of tools/index_video.py: readings -> timeline -> events (JSON + Parquet).

Outputs in --out:
  readings.parquet        per sampled frame: parsed scoreboard values (None = unreadable)
  game_clock_map.parquet  stable game-clock runs: video t0/t1 <-> game clock (for play-by-play alignment)
  camera_segments.parquet camera segments (auto from cuts, or the reviewed segments file)
  events_auto.json        all automatic events (schema nbatrack.index.events/0.1); never edited by hand
  events_auto.parquet     the same, flat
  index_report.json       counts, clock consistency, runtime
Manual corrections are saved by the review UI to events_reviewed.json (events_auto.json stays
untouched, so re-indexing never destroys manual work).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from nbatrack.index.events import build_events
from nbatrack.index.extract import Extraction
from nbatrack.index.schema import events_frame, write_events
from nbatrack.index.scoreboard import GlyphReader, ScoreboardConfig
from nbatrack.index.shots import detect_cuts, shot_segments
from nbatrack.index.timeline import build_timeline, read_samples, stable_values
from nbatrack.io.video import file_sha256


def load_segments(ex: Extraction, segments_json: str | None) -> tuple[pd.DataFrame, list[int]]:
    st = ex.frame_stats
    cuts, flashes = detect_cuts(st)
    if not segments_json:
        return shot_segments(st, cuts), flashes
    d = json.loads(Path(segments_json).read_text(encoding="utf-8"))
    ts = dict(zip(st.frame_index, st.timestamp_sec))
    rows = []
    for s in d["segments"]:
        rows.append(dict(camera_segment_id=s["camera_segment_id"], start_frame=s["start_frame"], end_frame=s["end_frame"],
                         start_sec=float(ts[s["start_frame"]]), end_sec=float(ts[min(s["end_frame"], max(ts))]),
                         median_floor_frac=float(st[(st.frame_index >= s["start_frame"]) & (st.frame_index <= s["end_frame"])].floor_frac.median()),
                         view_type=s["view_type"], view_source="reviewed"))
    return pd.DataFrame(rows), sorted(set(flashes) | set(d.get("flash_frames", [])))


def cmd_index(a):
    t0 = time.time()
    out = Path(a.out)
    ex = Extraction.load(out)
    cfg = ScoreboardConfig.load(a.scoreboard)
    reader = GlyphReader.load(out / "glyph_clusters.npz", a.labels)
    rd = read_samples(ex.samples, ex.glyph_meta, ex.glyphs, reader)
    rd.to_parquet(out / "readings.parquet")
    tl = build_timeline(rd)
    pd.DataFrame(tl.clock_runs, columns=["video_t0", "video_t1", "game_clock_s"]).to_parquet(out / "game_clock_map.parquet")
    segs, flashes = load_segments(ex, a.segments)
    segs.to_parquet(out / "camera_segments.parquet")
    auto_cuts, auto_flashes = detect_cuts(ex.frame_stats)
    video = ex.info["video"]
    video_id = Path(video).stem
    tracking = None
    if a.tracking_run:
        tr = Path(a.tracking_run)
        vt = pd.read_parquet(tr / "tables" / "videos.parquet").iloc[0]
        tracking = dict(run_dir=str(tr), tables=["player_observations", "ball_observations"], video_sha256=vt.sha256,
                        note="time-aligned by frame_index / timestamp_sec; track IDs are local to the run")
    period_runs = stable_values(rd.timestamp_sec.to_numpy(), rd.period.tolist())
    events, bounds = build_events(tl, ex.frame_stats[["frame_index", "timestamp_sec"]], segs, cfg.teams, video_id,
                                  a.game_id, period_runs, tracking)
    sha = file_sha256(video)
    if tracking and tracking["video_sha256"] != sha:
        raise SystemExit("tracking run was made from a different video file (sha256 mismatch)")
    header = dict(game_id=a.game_id, video_id=video_id, video_path=str(video), video_sha256=sha,
                  fps=ex.info["fps"], n_frames=ex.info["n_frames"], duration_s=float(ex.frame_stats.timestamp_sec.iloc[-1]),
                  teams=cfg.teams, flash_frames=flashes,
                  auto_cuts=auto_cuts, auto_flashes=auto_flashes,
                  provenance=dict(scoreboard=str(a.scoreboard), glyph_labels=str(a.labels), segments=a.segments,
                                  sample_every=ex.info["sample_every"]))
    write_events(out / "events_auto.json", header, events)
    events_frame(events).to_parquet(out / "events_auto.parquet")
    cnt = pd.Series([e["event_type"] for e in events]).value_counts().to_dict()
    vis = rd.scoreboard_visible
    rep = dict(events_by_type=cnt, boundaries=[dict(t=round(b["t"], 3), conf=b["conf"], sources=b["sources"]) for b in bounds],
               needs_review=sum(e["review_status"] == "needs_review" for e in events),
               readable_frac={f: float(rd.loc[vis, f].notna().mean()) if vis.any() else None
                              for f in ("away_score", "home_score", "period", "game_clock_s", "shot_clock")},
               scoreboard_visible_frac=float(vis.mean()), clock_checks=tl.clock_checks,
               extract=ex.info, index_seconds=time.time() - t0)
    (out / "index_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("events_by_type", "boundaries", "needs_review", "readable_frac", "clock_checks")},
                     indent=1, default=float))
