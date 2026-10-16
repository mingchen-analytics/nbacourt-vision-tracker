"""Camera estimator tests on synthetic data with exact ground truth."""

import numpy as np
import pytest

from nbatrack.calib.camera import ptz_rotation
from nbatrack.calib.fit import fit_camera, fit_ptz_segment
from nbatrack.calib.models import Correspondences
from nbatrack.synthetic import BroadcastScenario, generate_sequence, observe_landmarks


def test_fit_camera_noise_free_exact():
    fr = generate_sequence(BroadcastScenario(n_frames=1, seed=3))[0]
    rng = np.random.default_rng(0)
    obs, _ = observe_landmarks(fr.camera, rng, noise_px=0.0)
    assert len(obs) >= 6
    fit = fit_camera(obs, fr.camera.image_size)
    assert fit.success
    assert fit.rmse_px < 1e-6
    assert np.allclose(fit.camera.C, fr.camera.C, atol=1e-5)
    assert fit.camera.f == pytest.approx(fr.camera.f, rel=1e-7)


def test_fit_camera_with_noise_and_outliers():
    fr = generate_sequence(BroadcastScenario(n_frames=1, seed=5, f_range=(1800, 1800)))[0]
    rng = np.random.default_rng(1)
    obs, _ = observe_landmarks(fr.camera, rng, noise_px=1.0, outlier_frac=0.15)
    fit = fit_camera(obs, fr.camera.image_size)
    assert fit.success
    assert fit.rmse_px < 2.0
    n_out = int(round(0.15 * len(obs)))
    assert (~fit.inliers).sum() >= n_out


def test_ptz_frame_fit_with_two_points():
    """With the rig center known, 2 landmarks (4 equations) determine pan, tilt, f."""
    frames = generate_sequence(BroadcastScenario(n_frames=1, seed=7))
    cam = frames[0].camera
    rng = np.random.default_rng(2)
    obs, _ = observe_landmarks(cam, rng, noise_px=0.0, include_3d=False)
    two = obs.subset(np.arange(len(obs)) < 2)
    init = cam.copy(f=cam.f * 1.1)
    init.R = ptz_rotation(cam.ptz[0] + 2, cam.ptz[1] - 1, cam.ptz[2])
    fit = fit_camera(two, cam.image_size, init=init, fixed_C=cam.C, fixed_roll=cam.ptz[2])
    assert fit.success
    assert fit.camera.f == pytest.approx(cam.f, rel=1e-6)
    assert fit.camera.ptz[:2] == pytest.approx(cam.ptz[:2], abs=1e-6)


def test_ptz_segment_recovers_fixed_center():
    sc = BroadcastScenario(n_frames=40, seed=11)
    frames = generate_sequence(sc)
    rng = np.random.default_rng(3)
    obs = [observe_landmarks(fr.camera, rng, noise_px=1.0, dropout=0.2)[0] for fr in frames]
    fit = fit_ptz_segment(obs, sc.image_size)
    assert fit.success
    err_C = np.linalg.norm(fit.rig.C - sc.C)
    assert err_C < 0.3, err_C
    f_err = [abs(fit.states[i].f - fr.state.f) / fr.state.f for i, fr in enumerate(frames)]
    assert np.median(f_err) < 0.01
    assert np.nanmedian(fit.frame_rmse_px) < 2.0


def test_ptz_segment_flags_moving_center():
    """If the camera center actually moves (e.g. a rail cam), the fixed-center model
    must show clearly larger residuals than with a truly fixed center."""
    rng = np.random.default_rng(4)

    def run(vel):
        sc = BroadcastScenario(n_frames=40, seed=12, center_velocity=np.array(vel))
        frames = generate_sequence(sc)
        obs = [observe_landmarks(fr.camera, rng, noise_px=0.5)[0] for fr in frames]
        fit = fit_ptz_segment(obs, sc.image_size)
        return np.nanmedian(fit.frame_rmse_px), np.nanmedian(fit.frame_inlier_frac)

    fixed_rmse, fixed_inl = run([0.0, 0.0, 0.0])
    mov_rmse, mov_inl = run([1.5, 0.0, 0.0])  # 1.5 m/s dolly along the sideline
    assert fixed_inl == 1.0 and fixed_rmse < 1.0
    # mismatch shows up as inflated residuals and/or rejected observations
    assert mov_rmse > 3 * fixed_rmse or mov_inl < 0.8, (fixed_rmse, mov_rmse, mov_inl)


def test_correspondences_validation():
    with pytest.raises(ValueError):
        Correspondences(np.zeros((3, 3)), np.zeros((2, 2)))
