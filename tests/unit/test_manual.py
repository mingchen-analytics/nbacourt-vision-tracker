import numpy as np

from nbatrack.calib.manual import ManualCalibrationSession, overlay
from nbatrack.court.render import render_synthetic_view
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks


def _session():
    fr = generate_sequence(BroadcastScenario(n_frames=1, seed=4))[0]
    obs, _ = observe_landmarks(fr.camera, np.random.default_rng(0), noise_px=1.0)
    s = ManualCalibrationSession("synthetic", 0, 0.0, fr.camera.image_size, annotator="test")
    for i, (u, v) in zip(obs.ids, obs.image):
        s.set_point(int(i), u, v)
    return fr, s


def test_manual_fit_and_json_roundtrip(tmp_path):
    fr, s = _session()
    model, q = s.fit("camera")
    assert model is not None and q.quality in ("good", "fair")
    assert np.linalg.norm(model.camera.C - fr.camera.C) < 0.5
    hmodel, hq = s.fit("homography")
    assert hmodel is not None
    p = tmp_path / "cal.json"
    s.save(p)
    s2 = ManualCalibrationSession.load(p)
    assert s2.points == s.points and s2.annotator == "test"
    img = render_synthetic_view(fr.camera)
    out = overlay(img, s, model)
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_too_few_points_fails_cleanly():
    s = ManualCalibrationSession("v", 0, 0.0, (1920, 1080))
    s.set_point(0, 10, 10)
    s.set_point(1, 100, 10)
    m, q = s.fit("homography")
    assert m is None and q.quality == "failed"
    m, q = s.fit("camera")
    assert m is None and q.quality == "failed"
