"""Offline tracklet stitching and short-gap interpolation."""

import numpy as np
import pandas as pd

from nbatrack.track.offline import StitchConfig, interpolate_gaps, stitch_tracklets


def _rows(tid, frames, x0, vx, team="A", fps=30.0):
    return [dict(video_id="v", frame_index=f, timestamp_sec=f / fps, camera_segment_id=0, track_id=tid,
                 role="player", team_id=team, team_confidence=0.9, team_vote=team,
                 position_status="observed_projected", court_x_m_raw=x0 + vx * f / fps, court_y_m_raw=0.0,
                 court_x_m=x0 + vx * f / fps, court_y_m=0.0, court_sigma_m=0.2,
                 bbox_x1=10.0, bbox_y1=10.0, bbox_x2=20.0, bbox_y2=40.0, _occ=0.0) for f in frames]


def _frames(n, fps=30.0):
    return pd.DataFrame(dict(frame_index=range(n), timestamp_sec=np.arange(n) / fps))


def test_stitch_links_physically_consistent_same_team_fragments():
    pl = pd.DataFrame(_rows("t1", range(0, 30), 0.0, 3.0) + _rows("t2", range(45, 90), 0.0, 3.0)
                      + _rows("t3", range(50, 90), 9.0, 0.0, team="B"))
    out, links = stitch_tracklets(pl, 30.0, None, "team_vote", StitchConfig())
    assert [(l["end_track"], l["start_track"]) for l in links] == [("t1", "t2")]
    assert set(out[out.source_track_id == "t2"].track_id) == {"t1"}
    assert set(out[out.source_track_id == "t3"].track_id) == {"t3"}  # other team: untouched
    # raw coordinates are never changed by stitching
    assert np.allclose(out.sort_index().court_x_m_raw, pl.court_x_m_raw)


def test_stitch_rejects_physically_impossible_jump():
    pl = pd.DataFrame(_rows("t1", range(0, 30), 0.0, 0.0) + _rows("t2", range(40, 60), 12.0, 0.0))
    _, links = stitch_tracklets(pl, 30.0, None, "team_vote", StitchConfig())
    assert links == []  # 12 m in 0.33 s


def test_stitch_requires_appearance_when_descriptors_exist():
    pl = pd.DataFrame(_rows("t1", range(0, 30), 0.0, 3.0) + _rows("t2", range(45, 90), 0.0, 3.0))
    _, links = stitch_tracklets(pl, 30.0, lambda i: None, "team_vote", StitchConfig())
    assert links == []  # identity cannot be verified -> no link


def test_interpolation_adds_separate_rows_only_for_short_gaps():
    pl = pd.DataFrame(_rows("t1", list(range(0, 10)) + list(range(20, 30)) + list(range(80, 90)), 0.0, 3.0))
    add = interpolate_gaps(pl, _frames(90), max_gap_s=1.0)
    assert sorted(add.frame_index) == list(range(10, 20))  # 0.37 s gap filled, 1.7 s gap not
    assert (add.position_status == "interpolated").all() and (~add.is_visible).all()
    assert "court_x_m_raw" not in add.columns or add.court_x_m_raw.isna().all()
    # constant-velocity motion is reproduced
    assert np.allclose(add.court_x_m, 3.0 * add.frame_index / 30.0, atol=1e-6)
    assert (add.court_sigma_m > 0.2).all()
