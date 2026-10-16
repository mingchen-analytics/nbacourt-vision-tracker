import numpy as np
import pytest

from nbatrack.calib.camera import (
    PinholeCamera,
    PTZRig,
    PTZState,
    look_at_ptz,
    ptz_rotation,
    rotation_to_ptz,
)
from nbatrack.calib.homography import apply_h
from nbatrack.synthetic import default_rig


@pytest.mark.parametrize("pan,tilt,roll", [(0, 0, 0), (25, 20, 1.5), (-40, 35, -2), (5, 2, 0)])
def test_rotation_orthonormal_and_invertible(pan, tilt, roll):
    R = ptz_rotation(pan, tilt, roll)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert rotation_to_ptz(R) == pytest.approx((pan, tilt, roll), abs=1e-9)


def test_ptz_angle_semantics():
    # pan=tilt=0 -> optical axis world +Y; image x = world +X; image down = world -Z
    R = ptz_rotation(0, 0, 0)
    assert np.allclose(R[2], [0, 1, 0])
    assert np.allclose(R[0], [1, 0, 0])
    assert np.allclose(R[1], [0, 0, -1])
    # positive pan turns toward +X, positive tilt looks down
    assert ptz_rotation(30, 0)[2][0] > 0
    assert ptz_rotation(0, 30)[2][2] < 0


def test_look_at_centers_target():
    rig = default_rig()
    target = np.array([6.0, 2.0, 0.0])
    pan, tilt = look_at_ptz(rig.C, target)
    cam = rig.camera(PTZState(pan, tilt, 2500.0))
    uv, z = cam.project(target[None])
    assert uv[0] == pytest.approx([960.0, 540.0], abs=1e-6)
    assert z[0] > 0


@pytest.mark.parametrize("k1", [0.0, -0.08])
def test_project_backproject_roundtrip(k1):
    rig = PTZRig(np.array([1.0, -21.0, 9.0]), (1920, 1080), k1)
    cam = rig.camera(PTZState(12.0, 22.0, 2200.0, 0.5))
    rng = np.random.default_rng(1)
    xy = np.stack([rng.uniform(-14, 14, 200), rng.uniform(-7.6, 7.6, 200)], 1)
    uv, z = cam.project(xy)
    assert (z > 0).all()
    back, valid = cam.backproject_to_plane(uv, 0.0)
    assert valid.all()
    assert np.abs(back - xy).max() < 1e-6
    # plane at rim height
    X3 = np.hstack([xy[:20], np.full((20, 1), 3.048)])
    uv3, _ = cam.project(X3)
    back3, _ = cam.backproject_to_plane(uv3, 3.048)
    assert np.abs(back3 - xy[:20]).max() < 1e-6


def test_ground_homography_matches_projection():
    cam = default_rig().camera(PTZState(-15.0, 18.0, 1900.0, 0.4))
    xy = np.array([[0, 0], [10, 5], [-12, -7], [3, 7.6]], float)
    assert np.abs(apply_h(cam.ground_homography(), xy) - cam.project(xy)[0]).max() < 1e-8


def test_points_above_horizon_are_invalid():
    cam = default_rig().camera(PTZState(0.0, 5.0, 2000.0))
    # top row of the image looks above the horizon for a near-horizontal camera
    _, valid = cam.backproject_to_plane(np.array([[960.0, 0.0]]))
    assert not valid[0]


def test_point_behind_camera_is_nan():
    cam = PinholeCamera(2000, 960, 540, ptz_rotation(0, 20), np.array([0, -20, 9.0]))
    uv, z = cam.project(np.array([[0.0, -40.0, 0.0]]))
    assert z[0] < 0 and np.isnan(uv).all()
