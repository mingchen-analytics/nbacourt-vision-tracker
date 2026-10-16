"""Stage-B row bookkeeping: one row per (track, frame) after interpolation; box-height ratio."""

import numpy as np
import pandas as pd

from nbatrack.calib.camera import PinholeCamera, look_at_ptz, ptz_rotation
from nbatrack.pipeline.video_pipeline import _height_ratio, merge_interpolated
from nbatrack.track.offline import interpolate_gaps


def _rows():
    base = dict(video_id="v", camera_segment_id=0, track_id="s000_t001", role="player", team_id="A",
                team_confidence=1.0, court_sigma_m=0.3, bbox_x1=1.0, flags=None)
    rows = []
    for f in range(6):
        status = "unknown" if f in (2, 3) else "observed_projected"  # f2-3: flash / feet hidden
        rows.append(dict(base, frame_index=f, timestamp_sec=f / 30, position_status=status,
                         court_x_m=float(f) if status != "unknown" else np.nan, court_y_m=0.0 if status != "unknown" else np.nan,
                         court_x_m_raw=float(f) + 5.0 if f == 2 else float(f)))
    return pd.DataFrame(rows)


def test_interpolation_fills_unknown_rows_instead_of_duplicating():
    pl = _rows()
    frames = pd.DataFrame(dict(frame_index=range(6), timestamp_sec=np.arange(6) / 30))
    add = interpolate_gaps(pl, frames, 1.0)
    out, n = merge_interpolated(pl, add)
    assert not out.duplicated(["track_id", "frame_index"]).any()
    assert len(out) == 6 and n == 2
    r2 = out[out.frame_index == 2].iloc[0]
    assert r2.position_status == "interpolated" and "position_interpolated" in r2["flags"]
    assert abs(r2.court_x_m - 2.0) < 1e-6 and r2.court_x_m_raw == 7.0  # raw observation preserved
    assert r2.bbox_x1 == 1.0  # image observation preserved
    assert (out[out.frame_index.isin([0, 1, 4, 5])].position_status == "observed_projected").all()


def test_height_ratio_is_one_for_a_standing_person_and_small_for_a_truncated_box():
    C = np.array([0.0, -30.0, 12.0])
    pan, tilt = look_at_ptz(C, np.array([0.0, 0.0, 0.0]))
    cam = PinholeCamera(2500.0, 640, 360, ptz_rotation(pan, tilt, 0.0), C, image_size=(1280, 720))
    xy = np.array([[2.0, 1.0], [2.0, 1.0]])
    u0 = cam.project(np.array([[2.0, 1.0, 0.0]]))[0][0]
    u1 = cam.project(np.array([[2.0, 1.0, 1.95]]))[0][0]
    h = u0[1] - u1[1]
    boxes = np.array([[u0[0] - 20, u1[1], u0[0] + 20, u0[1]], [u0[0] - 20, u0[1] - 0.4 * h, u0[0] + 20, u0[1]]])
    r = _height_ratio(cam, xy, boxes, 1.95)
    assert abs(r[0] - 1.0) < 1e-6 and abs(r[1] - 0.4) < 1e-6
    assert np.isnan(_height_ratio(cam, np.array([[np.nan, np.nan]]), boxes[:1], 1.95)[0])
