"""Stage-A checkpoint / resume: an interrupted run resumed from its checkpoint must produce
the same calibration and detections as an uninterrupted run (synthetic rendered video).

Requires the exported RF-DETR ONNX model; skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbatrack.calib.manual import ManualCalibrationSession
from nbatrack.pipeline.video_pipeline import AnalyzeConfig, analyze_video
from nbatrack.synthetic.render import SyntheticArena
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks
from nbatrack.synthetic.video import write_video

ONNX = Path("models/onnx/rfdetr_small_512x896/rfdetr-small.onnx")


@pytest.mark.skipif(not ONNX.exists(), reason="RF-DETR ONNX model not exported")
def test_interrupted_stage_a_resumes_to_identical_output(tmp_path):
    k = 720 / 1080
    sc = BroadcastScenario(n_frames=20, seed=7, image_size=(1280, 720), f_range=(1800 * k, 2600 * k))
    frames = generate_sequence(sc)
    arena = SyntheticArena(4, floor_noise_amp=4, floor_seams=False)
    vid = tmp_path / "synthetic.mp4"
    write_video(vid, [arena.render(f.camera) for f in frames], [f.timestamp_sec for f in frames])
    obs, _ = observe_landmarks(frames[0].camera, np.random.default_rng(0), noise_px=1.5, include_3d=False)
    s = ManualCalibrationSession("synthetic", 0, frames[0].timestamp_sec, (1280, 720), annotator="synthetic")
    for i, (u, v) in zip(obs.ids, obs.image):
        s.set_point(int(i), u, v)
    kf = tmp_path / "kf.json"
    s.save(kf)

    full = tmp_path / "full"
    analyze_video(vid, full, kf, "synthetic", AnalyzeConfig(provider="cpu", checkpoint_every=0))
    part = tmp_path / "part"
    with pytest.raises(KeyboardInterrupt):
        analyze_video(vid, part, kf, "synthetic", AnalyzeConfig(provider="cpu", checkpoint_every=6, stop_after_frame=14))
    assert (part / "cache" / "stageA_ckpt.pkl").exists()
    run = analyze_video(vid, part, kf, "synthetic", AnalyzeConfig(provider="cpu", checkpoint_every=6))
    assert run["resumed_after_frame"] == 11
    assert not (part / "cache" / "stageA_ckpt.pkl").exists()
    a = pd.read_parquet(full / "tables" / "calibration.parquet")
    b = pd.read_parquet(part / "tables" / "calibration.parquet")
    assert list(a.frame_index) == list(b.frame_index) == list(range(20))
    for col in ("pan_deg", "tilt_deg", "focal_px"):
        assert np.allclose(a[col], b[col], atol=1e-9), col
    assert list(a.quality) == list(b.quality)
    da = pd.read_parquet(full / "cache" / "detections.parquet")
    db = pd.read_parquet(part / "cache" / "detections.parquet")
    assert len(da) == len(db)
