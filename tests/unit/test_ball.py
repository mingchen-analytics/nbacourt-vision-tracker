"""Offline ball tracking (Viterbi) and possession proxy."""

import numpy as np
import pandas as pd

from nbatrack.track.ball import track_ball, viterbi_ball


def _cands(rows):
    return pd.DataFrame(rows, columns=["frame_index", "cls", "score", "x1", "y1", "x2", "y2"])


def test_viterbi_prefers_consistent_weak_track_over_isolated_strong_false_positive():
    rows = []
    for f in range(20):
        x = 100 + 10 * f  # true ball moving right, weak detections
        rows.append((f, 37, 0.25, x - 10, 300, x + 10, 320))
        if f == 10:  # strong but far-away false positive
            rows.append((f, 37, 0.9, 1000, 100, 1020, 120))
    det = _cands(rows)
    out = viterbi_ball({f: g for f, g in det.groupby("frame_index")}, list(range(20)), {})
    chosen = det.loc[[out[f] for f in range(20)]]
    assert (chosen.x1 < 900).all()


def test_viterbi_allows_missing_frames_instead_of_jumping():
    rows = [(f, 37, 0.6, 100 + 5 * f, 300, 120 + 5 * f, 320) for f in range(10)]
    rows += [(f, 37, 0.6, 900, 300, 920, 320) for f in range(14, 20)]  # different object later
    det = _cands(rows)
    out = viterbi_ball({f: g for f, g in det.groupby("frame_index")}, list(range(20)), {})
    assert all(out[f] is None for f in range(10, 14))


def test_airborne_ball_projecting_beyond_the_court_is_kept_by_default():
    """Clip-6 regression: a shot arc projects > 12 m off court on the floor plane; the old
    'far projection' rule deleted the whole arc."""
    rows = [(f, 37, 0.8, 300 + 4 * f, 120 + 5 * abs(f - 10), 312 + 4 * f, 132 + 5 * abs(f - 10)) for f in range(20)]
    det = _cands(rows)
    frames = pd.DataFrame(dict(frame_index=range(20), timestamp_sec=np.arange(20) / 60))
    H = np.array([[10.0, 0, 640], [0, -10.0, 0], [0, 0, 1]])  # court y = -v / 10: every candidate at |y| > 12
    cal = pd.DataFrame(dict(frame_index=range(20), H_court_to_img=[H.ravel()] * 20, model=["ptz"] * 20))
    players = pd.DataFrame(columns=["frame_index", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "court_x_m", "court_y_m"])
    assert (track_ball(det, frames, cal, players, "v").image_source == "detected").sum() == 20
    assert len(track_ball(det, frames, cal, players, "v", max_abs_y_m=12.0)) == 0  # old behaviour


def test_track_ball_interpolates_short_gaps_and_uses_holder_position():
    rows = [(f, 37, 0.6, 200 + 2 * f, 150, 210 + 2 * f, 160) for f in range(10) if f not in (4, 5)]
    det = _cands(rows)
    frames = pd.DataFrame(dict(frame_index=range(10), timestamp_sec=np.arange(10) / 30))
    cal = pd.DataFrame(dict(frame_index=range(10), H_court_to_img=[None] * 10, model=["none"] * 10))
    players = pd.DataFrame([dict(frame_index=f, track_id="s000_t007", bbox_x1=180.0, bbox_y1=100.0,
                                 bbox_x2=260.0, bbox_y2=300.0, court_x_m=3.0, court_y_m=-1.0) for f in range(10)])
    b = track_ball(det, frames, cal, players, "v")
    assert list(b.frame_index) == list(range(10))
    assert set(b[b.frame_index.isin([4, 5])].image_source) == {"interpolated"}
    held = b[b.position_3d_status == "possession_proxy"]
    assert len(held) == 10 and (held.holder_track_id == "s000_t007").all()
    assert np.allclose(held.ball_x_m, 3.0) and held.ball_z_m.isna().all()
