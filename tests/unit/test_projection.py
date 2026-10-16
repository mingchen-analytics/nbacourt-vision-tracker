import numpy as np
import pytest

from nbatrack.calib.camera import PTZState
from nbatrack.calib.models import CameraCalibration, HomographyCalibration
from nbatrack.calib.quality import assess, emits_coordinates, hull_fraction
from nbatrack.project.projector import GroundPoints, bbox_bottom_points, project_ground_points
from nbatrack.synthetic import default_rig


def _model():
    return CameraCalibration(default_rig().camera(PTZState(0.0, 22.0, 2000.0)))


def test_bbox_bottom_point():
    gp = bbox_bottom_points(np.array([[100, 200, 140, 300]]))
    assert gp.uv[0] == pytest.approx([120, 300])
    assert gp.cov_uv[0, 0, 0] == pytest.approx((0.1 * 40) ** 2)


def test_projection_exact_and_sigma_grows_with_distance():
    m = _model()
    xy = np.array([[0.0, -6.0], [0.0, 6.0]])  # near and far sideline
    uv = m.court_to_image(xy)
    gp = GroundPoints(uv, np.tile(np.eye(2), (2, 1, 1)), "synthetic")
    cp = project_ground_points(m, gp)
    assert np.abs(cp.xy - xy).max() < 1e-6
    assert cp.sigma_m[1] > 1.5 * cp.sigma_m[0]  # far side: fewer px per meter
    assert cp.on_court.all()


def test_projection_homography_matches_camera():
    m = _model()
    h = HomographyCalibration(m.floor_homography())
    gp = bbox_bottom_points(np.array([[900, 500, 950, 620], [300, 700, 360, 860]]))
    a = project_ground_points(m, gp, 1.0)
    b = project_ground_points(h, gp, 1.0)
    assert np.allclose(a.xy, b.xy, atol=1e-6)
    assert np.allclose(a.sigma_m, b.sigma_m, rtol=1e-4)


def test_above_horizon_is_null():
    m = CameraCalibration(default_rig().camera(PTZState(0.0, 5.0, 2000.0)))
    gp = GroundPoints(np.array([[960.0, 0.0]]), np.eye(2)[None], "synthetic")
    cp = project_ground_points(m, gp)
    assert np.isnan(cp.xy).all() and np.isnan(cp.sigma_m).all() and not cp.on_court[0]


def test_quality_levels():
    size = (1920, 1080)
    spread = np.array([[100, 100], [1800, 120], [1700, 1000], [200, 950], [960, 540], [500, 500]])
    # 6 points / 8 params -> DOF factor sqrt(12 / 4) = 1.73 on the residual
    good = assess("homography", 6, 1.0, spread, size)
    assert good.quality == "good" and good.rmse_corrected_px == pytest.approx(3 ** 0.5)
    assert assess("homography", 6, 2.0, spread, size).quality == "fair"
    assert assess("homography", 6, 4.0, spread, size).quality == "poor"
    assert assess("homography", 6, 7.0, spread, size).quality == "failed"
    assert assess("homography", 3, 1.0, spread[:3], size).quality == "failed"
    # exactly-determined fit (4 points, 8 params) cannot be validated
    assert assess("homography", 4, 0.0, spread[:4], size).quality == "failed"
    clustered = np.array([[900, 500], [920, 510], [905, 530], [930, 520], [915, 505]])
    assert hull_fraction(clustered, size) < 0.02
    assert assess("homography", 5, 0.5, clustered, size).quality == "poor"
    # resolution scaling: same relative error at 720p is rated the same
    assert assess("homography", 6, 1.0 * 720 / 1080, spread * 2 / 3, (1280, 720)).quality == "good"
    cam = default_rig().camera(PTZState(0.0, 22.0, 2000.0))
    bad = cam.copy()
    bad.C = np.array([0.0, -20.0, -1.0])  # below the floor
    assert assess("camera", 6, 1.0, spread, size, bad).quality == "failed"
    assert emits_coordinates("fair") and not emits_coordinates("poor")
