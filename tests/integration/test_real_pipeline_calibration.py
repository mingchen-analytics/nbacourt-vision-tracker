"""Stage-A calibration tracking (keyframe -> KLT propagation -> line refinement) on a
rendered synthetic video with known per-frame cameras.

Requires the exported RF-DETR ONNX model (models/ is git-ignored); skipped otherwise.
The detector finds no people in the synthetic arena, which is fine: this test measures
calibration tracking only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbatrack.calib.manual import ManualCalibrationSession
from nbatrack.pipeline.video_pipeline import AnalyzeConfig, analyze_video, camera_from_row
from nbatrack.synthetic.render import SyntheticArena
from nbatrack.synthetic.scene import (
    BroadcastScenario,
    generate_sequence,
    observe_landmarks,
    sample_players,
    visible,
)
from nbatrack.synthetic.video import write_video

ONNX = Path("models/onnx/rfdetr_small_512x896/rfdetr-small.onnx")


@pytest.mark.skipif(not ONNX.exists(), reason="RF-DETR ONNX model not exported")
@pytest.mark.parametrize("kf_index", [0, 12])  # 12: mid-clip keyframe -> forward + backward
def test_stage_a_tracks_calibration_on_rendered_video(tmp_path, kf_index):
    k = 720 / 1080
    sc = BroadcastScenario(n_frames=24, seed=5, image_size=(1280, 720), f_range=(1800 * k, 2600 * k))
    frames = generate_sequence(sc)
    arena = SyntheticArena(4, floor_noise_amp=4, floor_seams=False)
    vid = tmp_path / "synthetic.mp4"
    write_video(vid, [arena.render(f.camera) for f in frames], [f.timestamp_sec for f in frames])
    obs, _ = observe_landmarks(frames[kf_index].camera, np.random.default_rng(0), noise_px=1.5,
                               include_3d=False)
    s = ManualCalibrationSession("synthetic", kf_index, frames[kf_index].timestamp_sec, (1280, 720),
                                 annotator="synthetic-gt+noise")
    for i, (u, v) in zip(obs.ids, obs.image):
        s.set_point(int(i), u, v)
    kf = tmp_path / "kf.json"
    s.save(kf)
    analyze_video(vid, tmp_path / "run", kf, "synthetic", AnalyzeConfig(provider="cpu"))
    cal = pd.read_parquet(tmp_path / "run" / "tables" / "calibration.parquet")
    assert len(cal) == 24 and (cal.quality == "good").mean() > 0.9
    assert list(cal.frame_index) == list(range(24))
    assert cal[cal.frame_index == kf_index].method.iloc[0] == "keypoint"
    probes = sample_players(np.random.default_rng(1), 300)
    errs = []
    for fr in frames:
        cam = camera_from_row(cal[cal.frame_index == fr.index].iloc[0], (1280, 720))
        uv, vis = visible(fr.camera, probes)
        xy = cam.backproject_to_plane(uv[vis])[0]
        errs.append(np.median(np.linalg.norm(xy - probes[vis], axis=1)))
    assert np.max(errs) < 0.15, errs  # meters, every frame
