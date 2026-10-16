"""End-to-end plumbing test on a short synthetic broadcast."""

import json

import numpy as np
import pandas as pd
import pytest

from nbatrack.io.video import iter_frames, probe
from nbatrack.pipeline.synthetic_demo import run_synthetic_demo


@pytest.mark.parametrize("calib", ["ptz", "homography"])
def test_synthetic_pipeline(tmp_path, calib):
    m = run_synthetic_demo(tmp_path, n_frames=20, calib=calib, seed=3, render_video=(calib == "ptz"))
    assert json.loads((tmp_path / "metrics.json").read_text())["n_frames"] == 20
    pl = pd.read_parquet(tmp_path / "tables" / "player_observations.parquet")
    cal = pd.read_parquet(tmp_path / "tables" / "calibration.parquet")
    assert len(cal) == 20 and set(cal.quality) <= {"good", "fair", "poor", "failed"}
    # coordinates only where calibration permits; unknown rows carry NULLs, not numbers
    unk = pl[pl.position_status == "unknown"]
    assert unk.court_x_m.isna().all() and unk.court_sigma_m.isna().all()
    ok = pl[pl.position_status == "observed_projected"]
    assert ok.calibration_quality.isin(["good", "fair"]).all()
    assert ok.court_x_m.notna().all()
    # synthetic accuracy sanity (exact GT available): errors well below a meter
    assert m["player_court_err_m"]["median"] < 0.3
    if calib == "ptz":
        frames = list(iter_frames(tmp_path / "diagnostic_side_by_side.mp4"))
        assert len(frames) == 20
        assert np.allclose([f.timestamp_sec for f in frames], np.arange(20) / 30.0, atol=1e-3)
        assert probe(tmp_path / "synthetic_input.mp4").width == 1280
