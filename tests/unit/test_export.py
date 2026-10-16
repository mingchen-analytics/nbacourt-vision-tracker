import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from nbatrack.court.geometry import court_landmarks
from nbatrack.export import SCHEMAS, SchemaError, build_table, write_rows


def _player_row(**kw):
    r = dict(video_id="v1", frame_index=3, timestamp_sec=0.1, track_id="s000_t001",
             role="player", ground_method="bbox_bottom", position_status="observed_projected",
             is_visible=True, court_x_m=1.5, court_y_m=float("nan"))
    r.update(kw)
    return r


def test_nan_becomes_null_and_roundtrip(tmp_path):
    paths = write_rows("player_observations", [_player_row()], tmp_path, ("parquet", "csv", "json"))
    t = pq.read_table(paths[0])
    assert t.schema.metadata[b"nbatrack.schema_version"] == b"1.3.0"
    row = t.to_pylist()[0]
    assert row["court_x_m"] == 1.5
    assert row["court_y_m"] is None  # NaN must not leak as a number
    assert row["team_id"] is None
    df = pd.read_csv(paths[1])
    assert math.isnan(df.loc[0, "court_y_m"]) and df.loc[0, "track_id"] == "s000_t001"


def test_required_fields_and_enums_enforced():
    with pytest.raises(SchemaError, match="must not be null"):
        build_table("player_observations", [_player_row(track_id=None)])
    with pytest.raises(SchemaError, match="position_status"):
        build_table("player_observations", [_player_row(position_status="guessed")])
    with pytest.raises(SchemaError, match="unknown fields"):
        build_table("player_observations", [_player_row(speed=3.0)])


def test_matrix_columns(tmp_path):
    H = np.arange(9, dtype=float).reshape(3, 3)
    row = dict(video_id="v", frame_index=0, timestamp_sec=0.0, method="manual", model="ptz",
               H_court_to_img=H.ravel(), quality="good")
    paths = write_rows("calibration", [row], tmp_path)
    got = pq.read_table(paths[0]).to_pylist()[0]
    assert got["H_court_to_img"] == list(H.ravel())
    assert got["H_img_to_court"] is None
    csv = pd.read_csv(paths[1])
    assert json.loads(csv.loc[0, "H_court_to_img"]) == list(H.ravel())
    assert pd.isna(csv.loc[0, "H_img_to_court"])


def test_court_landmarks_table():
    rows = [dict(landmark_id=lm.landmark_id, name=lm.name, court_x_m=lm.xyz[0],
                 court_y_m=lm.xyz[1], court_z_m=lm.xyz[2], description=lm.description)
            for lm in court_landmarks()]
    t = build_table("court_landmarks", rows)
    assert t.num_rows == 32


def test_all_schemas_have_frame_key_or_id():
    for name, s in SCHEMAS.items():
        assert "video_id" in s.names or name == "court_landmarks", name
