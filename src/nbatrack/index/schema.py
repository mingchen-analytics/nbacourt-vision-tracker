"""Event schema of the Video Indexing Engine (version 0.1).

Every event is a video interval [video_start_seconds, video_end_seconds] on ONE source video
(point events have start == end). Times are presentation timestamps of the source file, the
same clock as the tracking tables' `timestamp_sec` and as HTML5 `video.currentTime`, so the
tracking engine, the indexer and the player align without re-processing.

Provenance is explicit:
  *_source fields   "ocr_observed" | "interpolated" | "inferred" | "manual" | None
  boundary_source   list of evidence kinds that produced the interval boundaries
  label_source      "auto" | "manual"
  review_status     "auto" | "needs_review" | "confirmed" | "edited" | "rejected"
Track IDs are never stored here; `tracking_data_reference` points to a tracking run by
time / frame range only (local track IDs are not global player identities).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "nbatrack.index.events/0.1"

EVENT_TYPES = {
    # automatic
    "camera_segment", "non_live", "score_change", "shot_clock_reset", "candidate_possession",
    # manual (user-added)
    "possession", "shot_attempt", "made_shot", "turnover", "foul", "free_throw", "rebound",
    "timeout", "replay", "other",
}
REVIEW_STATUS = {"auto", "needs_review", "confirmed", "edited", "rejected"}
VALUE_SOURCES = {"ocr_observed", "interpolated", "inferred", "manual", None}

EVENT_FIELDS = [
    "event_id", "game_id", "video_id", "event_type", "labels",
    "video_start_seconds", "video_end_seconds", "frame_start", "frame_end",
    "period", "period_source", "game_clock_start", "game_clock_end", "game_clock_source",
    "shot_clock_start", "shot_clock_source",
    "score_away_start", "score_home_start", "score_away_end", "score_home_end", "score_source",
    "offense_team_id", "defense_team_id", "team_source",
    "event_confidence", "boundary_source", "evidence", "review_status", "label_source",
    "camera_segment_id", "tracking_data_reference", "notes",
]


def new_event(**kw) -> dict:
    e = {k: None for k in EVENT_FIELDS}
    e.update(labels=[], boundary_source=[], evidence=[], review_status="auto", label_source="auto", notes="")
    e.update(kw)
    return e


def validate_event(e: dict) -> list[str]:
    errs = []
    for k in EVENT_FIELDS:
        if k not in e:
            errs.append(f"missing field {k}")
    if e.get("event_type") not in EVENT_TYPES:
        errs.append(f"unknown event_type {e.get('event_type')!r}")
    if e.get("review_status") not in REVIEW_STATUS:
        errs.append(f"bad review_status {e.get('review_status')!r}")
    if e.get("label_source") not in ("auto", "manual"):
        errs.append(f"bad label_source {e.get('label_source')!r}")
    for k in ("period_source", "game_clock_source", "shot_clock_source", "score_source", "team_source"):
        if e.get(k) not in VALUE_SOURCES:
            errs.append(f"bad {k} {e.get(k)!r}")
    s, t = e.get("video_start_seconds"), e.get("video_end_seconds")
    if s is None or t is None or t < s - 1e-9:
        errs.append(f"bad interval {s}..{t}")
    c = e.get("event_confidence")
    if c is not None and not 0.0 <= c <= 1.0:
        errs.append(f"event_confidence out of range {c}")
    return errs


def write_events(path: str | Path, header: dict, events: list[dict]) -> None:
    bad = {e.get("event_id"): validate_event(e) for e in events}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ValueError(f"invalid events: {bad}")
    head = {k: v for k, v in header.items() if k not in ("schema", "events")}
    doc = dict(schema=SCHEMA_VERSION, **head, events=events)
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_events(path: str | Path) -> tuple[dict, list[dict]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ev = d.pop("events")
    return d, ev


def events_frame(events: list[dict]) -> pd.DataFrame:
    """Flat table for Parquet / analysis (lists and dicts JSON-encoded)."""
    rows = []
    for e in events:
        r = dict(e)
        for k in ("labels", "boundary_source", "evidence", "tracking_data_reference"):
            r[k] = json.dumps(r.get(k), ensure_ascii=False)
        rows.append(r)
    return pd.DataFrame(rows, columns=EVENT_FIELDS)
