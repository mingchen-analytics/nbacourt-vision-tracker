"""Output table schemas (v1). See docs/DATA_SCHEMA.md for field semantics.

Conventions
  * Units: meters, seconds, pixels in source-video resolution (origin top-left).
  * Missing values are NULL - never 0, -1 or NaN sentinels. (NaN floats are converted to
    NULL on write.)
  * Enumerations are stored as strings and validated on write.
"""

from __future__ import annotations

import pyarrow as pa

from nbatrack import SCHEMA_VERSION

F64, F32, I64, I32, STR, BOOL = pa.float64(), pa.float32(), pa.int64(), pa.int32(), pa.string(), pa.bool_()
MAT9 = pa.list_(pa.float64(), 9)


def _f(name, typ, nullable=True):
    return pa.field(name, typ, nullable=nullable)


ENUMS: dict[str, dict[str, set[str]]] = {
    "camera_segments": {"view_type": {"main_broadcast", "other", "unknown"}},
    "calibration": {
        "method": {"keypoint", "propagated", "manual", "smoothed", "held_flash", "none"},
        "model": {"homography", "camera_free", "ptz", "none"},
        "quality": {"good", "fair", "poor", "failed"},
    },
    "landmark_observations": {"source": {"model", "manual", "synthetic"}},
    "player_observations": {
        "position_status": {"observed_projected", "interpolated", "extrapolated", "unknown"},
        "ground_method": {"feet_keypoints", "bbox_bottom", "none"},
        "role": {"player", "referee", "other", "unknown"},
        "calibration_quality": {"good", "fair", "poor", "failed"},
    },
    "ball_observations": {"position_3d_status": {"unknown", "possession_proxy", "trajectory_fit", "floor_contact"},
                          "image_source": {"detected", "interpolated"}},
    "hoop_observations": {"hoop_side": {"left", "right", "unknown"}},
}

_FRAME_KEY = [_f("video_id", STR, False), _f("frame_index", I64, False),
              _f("timestamp_sec", F64, False)]

SCHEMAS: dict[str, pa.Schema] = {
    "videos": pa.schema([
        _f("video_id", STR, False), _f("source_path", STR), _f("sha256", STR),
        _f("width", I32, False), _f("height", I32, False), _f("fps_nominal", F64),
        _f("time_base", STR, False), _f("start_sec", F64), _f("duration_sec", F64),
        _f("n_frames", I64), _f("codec", STR), _f("pipeline_version", STR, False),
        _f("config_hash", STR), _f("created_at", STR, False),
    ]),
    "frames": pa.schema(_FRAME_KEY + [
        _f("pts", I64, False), _f("pts_sec", F64, False), _f("camera_segment_id", I32),
    ]),
    "camera_segments": pa.schema([
        _f("video_id", STR, False), _f("camera_segment_id", I32, False),
        _f("start_frame", I64, False), _f("end_frame", I64, False),
        _f("start_sec", F64, False), _f("end_sec", F64, False), _f("view_type", STR, False),
        _f("supported", BOOL, False), _f("boundary_method", STR),
        _f("boundary_confidence", F64),
    ]),
    "calibration": pa.schema(_FRAME_KEY + [
        _f("camera_segment_id", I32), _f("method", STR, False), _f("model", STR, False),
        _f("H_img_to_court", MAT9), _f("H_court_to_img", MAT9),
        _f("focal_px", F64), _f("cx", F64), _f("cy", F64), _f("k1", F64),
        _f("pan_deg", F64), _f("tilt_deg", F64), _f("roll_deg", F64),
        _f("cam_x_m", F64), _f("cam_y_m", F64), _f("cam_z_m", F64),
        _f("n_landmarks", I32), _f("n_inliers", I32), _f("reproj_rmse_px", F64),
        _f("line_alignment_px", F64), _f("quality", STR, False), _f("quality_score", F64),
    ]),
    "court_landmarks": pa.schema([
        _f("landmark_id", I32, False), _f("name", STR, False), _f("court_x_m", F64, False),
        _f("court_y_m", F64, False), _f("court_z_m", F64, False), _f("description", STR),
    ]),
    "landmark_observations": pa.schema(_FRAME_KEY + [
        _f("landmark_id", I32, False), _f("image_x", F64, False), _f("image_y", F64, False),
        _f("confidence", F64), _f("is_inlier", BOOL), _f("reproj_error_px", F64),
        _f("source", STR, False),
    ]),
    "player_observations": pa.schema(_FRAME_KEY + [
        _f("camera_segment_id", I32), _f("track_id", STR, False), _f("source_track_id", STR),
        _f("team_id", STR),
        _f("team_confidence", F64), _f("role", STR, False),
        _f("bbox_x1", F64), _f("bbox_y1", F64), _f("bbox_x2", F64), _f("bbox_y2", F64),
        _f("det_confidence", F64), _f("assoc_confidence", F64), _f("ground_method", STR, False),
        _f("image_x", F64), _f("image_y", F64),
        _f("court_x_m_raw", F64), _f("court_y_m_raw", F64),
        _f("court_x_m", F64), _f("court_y_m", F64), _f("court_sigma_m", F64),
        _f("position_status", STR, False), _f("flags", STR),
        _f("calibration_quality", STR), _f("is_visible", BOOL, False), _f("is_on_court", BOOL),
    ]),
    "ball_observations": pa.schema(_FRAME_KEY + [
        _f("camera_segment_id", I32), _f("ball_track_id", STR),
        _f("bbox_x1", F64), _f("bbox_y1", F64), _f("bbox_x2", F64), _f("bbox_y2", F64),
        _f("image_x", F64), _f("image_y", F64), _f("det_confidence", F64),
        _f("is_visible", BOOL, False), _f("image_source", STR), _f("position_3d_status", STR, False),
        _f("ball_x_m", F64), _f("ball_y_m", F64), _f("ball_z_m", F64), _f("ball_sigma_m", F64), _f("holder_track_id", STR),
    ]),
    "hoop_observations": pa.schema(_FRAME_KEY + [
        _f("hoop_side", STR, False),
        _f("bbox_x1", F64), _f("bbox_y1", F64), _f("bbox_x2", F64), _f("bbox_y2", F64),
        _f("image_x", F64), _f("image_y", F64), _f("confidence", F64),
        _f("predicted_image_x", F64), _f("predicted_image_y", F64), _f("residual_px", F64),
    ]),
}

UNITS = {
    "timestamp_sec": "s", "pts_sec": "s", "start_sec": "s", "end_sec": "s", "duration_sec": "s",
    "court_x_m": "m", "court_y_m": "m", "court_x_m_raw": "m", "court_y_m_raw": "m",
    "court_sigma_m": "m", "ball_x_m": "m", "ball_y_m": "m", "ball_z_m": "m",
    "cam_x_m": "m", "cam_y_m": "m", "cam_z_m": "m",
    "image_x": "px", "image_y": "px", "focal_px": "px", "reproj_rmse_px": "px",
}


COURT_SIGMA_STATUS = (b"UNCALIBRATED: first-order prior (bbox-bottom 10% box width / 2% box height "
                      b"+ 2 px calibration term). Not a validated confidence interval; measured "
                      b"bbox-bottom jitter on real footage is several times larger during occlusions.")


def schema_metadata(table: str) -> dict[bytes, bytes]:
    md = {
        b"nbatrack.schema_version": SCHEMA_VERSION.encode(),
        b"nbatrack.table": table.encode(),
        b"nbatrack.coordinate_system": b"court: origin center, +x right basket (main cam), "
                                       b"+y far sideline, +z up; meters",
    }
    if table == "player_observations":
        md[b"nbatrack.court_sigma_m.status"] = COURT_SIGMA_STATUS
    return md
