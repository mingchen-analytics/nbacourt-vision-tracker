"""Video Indexing Engine: glyph reading, timeline rules, possession boundaries, schema."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from nbatrack.index.events import build_events
from nbatrack.index.schema import EVENT_FIELDS, read_events, validate_event, write_events
from nbatrack.index.scoreboard import GlyphReader, field_glyphs, parse_clock, parse_int
from nbatrack.index.shots import detect_cuts
from nbatrack.index.timeline import build_timeline, clock_at, stable_values


def _text(s, w=110, h=34, fg=(20, 20, 20), bg=(235, 235, 235)):
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(img, s, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fg, 2, cv2.LINE_AA)
    return img


def test_glyphs_are_split_left_to_right_and_colon_is_dropped():
    g = field_glyphs(_text("9:36"))
    assert len(g) == 3  # 9, 3, 6 (colon dots too small)
    xs = [d["box"][0] for d in g]
    assert xs == sorted(xs)
    # polarity: bright text on dark works the same
    assert len(field_glyphs(_text("24", fg=(240, 240, 240), bg=(30, 30, 30)))) == 2


def test_reader_reads_known_digits_and_refuses_unknown_glyphs():
    tmpl = {}
    for d in "0123456789":
        gl = field_glyphs(_text(d))[0]
        tmpl.setdefault("f", []).append((d, gl["glyph"], gl["aspect"]))
    r = GlyphReader(tmpl)
    assert r.read_field("f", _text("907"))[0] == "907"
    only_some = GlyphReader({"f": [t for t in tmpl["f"] if t[0] in "01"]})
    assert only_some.read_field("f", _text("8"))[0] is None  # no template for 8 -> unreadable, never guessed


def test_parsers():
    assert parse_clock("936") == 576.0 and parse_clock("1200") == 720.0
    assert parse_clock("975") is None and parse_clock(None) is None and parse_clock("9") is None
    assert parse_int("24", 0, 24) == 24 and parse_int("25", 0, 24) is None


def _readings(T=40.0, dt=0.1, stop=(10.0, 13.0), score_at=25.0, reset_at=23.0):
    rows = []
    clock = 600.0
    shot = 24
    for k in range(int(T / dt)):
        t = k * dt
        if not (stop[0] <= t < stop[1]):
            clock_t = clock
        # clock: 1 tick per second, stopped during `stop`
        run_time = min(t, stop[0]) + max(0.0, t - stop[1])
        clock_t = 600.0 - np.floor(run_time)
        shot_t = 24 - np.floor(min(t, reset_at)) if t < reset_at else 24 - np.floor(t - reset_at)
        rows.append(dict(sample_id=k, frame_index=k * 6, timestamp_sec=t, presence=1.0, scoreboard_visible=True,
                         away_score=10 if t < score_at else 12, home_score=8, period=1, game_clock_s=float(clock_t),
                         shot_clock=int(max(shot_t, 0)), game_clock_text=None, shot_clock_text=None, min_conf=1.0))
    return pd.DataFrame(rows)


def test_timeline_detects_stop_score_change_and_reset():
    tl = build_timeline(_readings())
    assert tl.clock_checks["step_consistency"] == 1.0
    assert len(tl.stopped) == 1 and 10.5 <= tl.stopped[0][0] <= 11.5 and tl.stopped[0][1] >= 12.9
    assert [(s["team"], s["delta"]) for s in tl.score_changes] == [("away", 2)]
    assert len(tl.shot_resets) == 1 and tl.shot_resets[0]["after"] == 24
    g, src = clock_at(tl, 5.0)
    assert src == "ocr_observed" and g == 595.0


def test_stable_values_ignore_transient_garbage():
    ts = np.arange(10) * 0.1
    vals = [5, 5, 5, 9, 5, 6, 6, 6, 6, None]
    runs = stable_values(ts, vals, 3)
    assert [r[2] for r in runs] == [5, 6]


def _frames(T=40.0, fps=60.0):
    n = int(T * fps)
    return pd.DataFrame(dict(frame_index=np.arange(n), timestamp_sec=np.arange(n) / fps))


def _segs(cut_t=30.0, fps=60.0, T=40.0):
    c = int(cut_t * fps)
    return pd.DataFrame([dict(camera_segment_id=0, start_frame=0, end_frame=c - 1, start_sec=0.0, end_sec=(c - 1) / fps,
                              view_type="main_broadcast", view_source="reviewed", median_floor_frac=0.5),
                         dict(camera_segment_id=1, start_frame=c, end_frame=int(T * fps) - 1, start_sec=cut_t,
                              end_sec=T - 1 / fps, view_type="closeup", view_source="reviewed", median_floor_frac=0.0)])


def test_possession_boundaries_merge_reset_and_score_and_ignore_cuts():
    tl = build_timeline(_readings())
    teams = {"away": {"id": "LAL"}, "home": {"id": "DEN"}}
    ev, bounds = build_events(tl, _frames(), _segs(), teams, "v")
    assert len(bounds) == 1  # reset (23 s) + score shown at 25 s = ONE boundary at the reset
    b = bounds[0]
    assert abs(b["t"] - 23.0) < 0.2 and b["conf"] == 0.9 and b["scorer"] == "LAL"
    poss = [e for e in ev if e["event_type"] == "candidate_possession"]
    assert len(poss) == 2  # the camera cut at 30 s does NOT split a possession
    assert poss[0]["offense_team_id"] == "LAL" and poss[1]["offense_team_id"] == "DEN"
    assert all(e["team_source"] == "inferred" for e in poss)
    assert all(not validate_event(e) for e in ev)
    assert any(e["event_type"] == "non_live" for e in ev)  # the stopped clock


def test_reset_to_14_is_not_a_boundary():
    rd = _readings(score_at=1e9)
    rd.loc[rd.timestamp_sec >= 23.0, "shot_clock"] = rd.loc[rd.timestamp_sec >= 23.0, "shot_clock"].clip(upper=14)
    tl = build_timeline(rd)
    ev, bounds = build_events(tl, _frames(), _segs(), {"away": {"id": "A"}, "home": {"id": "B"}}, "v")
    assert tl.shot_resets and tl.shot_resets[0]["after"] == 14
    assert bounds == []


def test_cut_and_flash_detection():
    d = np.zeros(20)
    d[5], d[6] = 0.5, 0.5  # flash: frame 5 differs from 4 and 6, but 4 ~ 6
    d2 = np.full(20, 0.0)
    d2[6] = 0.05
    d[12] = 0.6  # real cut
    d2[12], d2[13] = 0.6, 0.6
    st = pd.DataFrame(dict(frame_index=np.arange(20), hist_dist=d, hist_dist2=d2))
    cuts, flashes = detect_cuts(st)
    assert flashes == [5] and cuts == [12]


def test_schema_roundtrip_and_validation(tmp_path):
    from nbatrack.index.schema import new_event
    e = new_event(event_id="x:1", video_id="v", event_type="shot_attempt", video_start_seconds=1.0,
                  video_end_seconds=2.0, label_source="manual", review_status="confirmed")
    p = tmp_path / "ev.json"
    write_events(p, {"video_id": "v", "schema": "old"}, [e])
    h, ev = read_events(p)
    assert h["schema"] == "nbatrack.index.events/0.1" and ev[0]["event_id"] == "x:1"
    assert set(ev[0]) == set(EVENT_FIELDS)
    bad = dict(e, video_end_seconds=0.5, review_status="maybe")
    with pytest.raises(ValueError):
        write_events(p, {}, [bad])
    assert json.loads(p.read_text())["events"][0]["review_status"] == "confirmed"  # failed write kept old file
