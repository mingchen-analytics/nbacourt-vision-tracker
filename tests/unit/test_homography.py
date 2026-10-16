import numpy as np
import pytest

from nbatrack.calib.camera import PTZState
from nbatrack.calib.homography import (
    apply_h,
    camera_from_homography,
    dlt_homography,
    fit_homography,
    focal_from_homography,
)
from nbatrack.calib.models import CameraCalibration, HomographyCalibration
from nbatrack.synthetic import default_rig


def _cam(pan=10.0, tilt=21.0, f=2300.0, roll=0.3):
    return default_rig().camera(PTZState(pan, tilt, f, roll))


def _grid():
    gx, gy = np.meshgrid(np.linspace(-6, 6, 5), np.linspace(-6, 6, 4))
    return np.stack([gx.ravel(), gy.ravel()], 1)


def test_dlt_exact():
    cam = _cam()
    xy = _grid()
    H = dlt_homography(xy, cam.project(xy)[0])
    Hgt = cam.ground_homography()
    assert np.allclose(H / H[2, 2], Hgt / Hgt[2, 2], rtol=1e-7, atol=1e-7)


def test_fit_homography_with_outliers():
    rng = np.random.default_rng(0)
    cam = _cam()
    xy = _grid()
    uv = cam.project(xy)[0] + rng.normal(0, 0.5, (len(xy), 2))
    uv[[2, 7, 11]] += [[80, -40], [-60, 90], [120, 0]]
    fit = fit_homography(xy, uv, ransac_px=5.0)
    assert not fit.inliers[[2, 7, 11]].any()
    assert fit.n_inliers == len(xy) - 3
    assert fit.rmse_px < 1.0


@pytest.mark.parametrize("pan,tilt,f", [(0, 20, 1800), (25, 25, 3000), (-30, 15, 2200)])
def test_decompose_recovers_camera(pan, tilt, f):
    cam = _cam(pan, tilt, f)
    H = cam.ground_homography()
    assert focal_from_homography(H, 960, 540) == pytest.approx(f, rel=1e-8)
    rec = camera_from_homography(H, (1920, 1080), court_xy=_grid())
    assert rec is not None
    assert np.allclose(rec.C, cam.C, atol=1e-6)
    assert np.allclose(rec.R, cam.R, atol=1e-9)


def test_homography_calibration_floor_only():
    cam = _cam()
    hc = HomographyCalibration(cam.ground_homography())
    cc = CameraCalibration(cam)
    xy = _grid()
    uv = hc.court_to_image(xy)
    assert np.abs(hc.image_to_court(uv) - xy).max() < 1e-6
    assert np.abs(cc.image_to_court(uv) - xy).max() < 1e-6
    # homography must refuse off-floor points; full camera supports them
    rim = np.array([[12.725, 0.0, 3.048]])
    assert np.isnan(hc.court_to_image(rim)).all()
    assert np.isfinite(cc.court_to_image(rim)).all()
    assert np.isnan(hc.image_to_court(uv, z=1.0)).all()


def test_homography_rejects_beyond_horizon():
    cam = default_rig().camera(PTZState(0.0, 5.0, 2000.0))
    hc = HomographyCalibration(cam.ground_homography())
    assert np.isnan(hc.image_to_court(np.array([[960.0, 0.0]]))).all()
    assert np.isfinite(hc.image_to_court(np.array([[960.0, 1000.0]]))).all()


def test_apply_h_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_h(np.eye(3), pts), pts)
