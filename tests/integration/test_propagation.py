"""Camera-motion propagation on rendered synthetic frames (exact ground truth)."""

import cv2
import numpy as np
import pytest

from nbatrack.calib.models import HomographyCalibration
from nbatrack.calib.propagate import estimate_motion, floor_mask, mask_boxes, propagate
from nbatrack.synthetic.render import SyntheticArena
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, sample_players, visible


@pytest.fixture(scope="module")
def seq():
    fr = generate_sequence(BroadcastScenario(n_frames=31, seed=2, f_range=(1800, 3000)))
    ar = SyntheticArena(1)
    return fr, [cv2.cvtColor(ar.render(f.camera), cv2.COLOR_BGR2GRAY) for f in fr]


def _chain(fr, imgs, use_floor):
    H = fr[0].camera.ground_homography()
    for i in range(1, len(fr)):
        m = floor_mask(H, imgs[i - 1].shape) if use_floor else None
        H = propagate(H, estimate_motion(imgs[i - 1], imgs[i], m))
        assert H is not None, f"motion estimation failed at frame {i}"
    probes = sample_players(np.random.default_rng(0), 200)
    uv, vis = visible(fr[-1].camera, probes)
    return np.linalg.norm(HomographyCalibration(H).image_to_court(uv[vis]) - probes[vis], axis=1)


def test_floor_only_propagation_drift(seq):
    e = _chain(*seq, use_floor=True)
    assert np.median(e) < 0.05  # meters after 30 frames (synthetic texture)


def test_all_static_propagation_drift_for_fixed_center(seq):
    e = _chain(*seq, use_floor=False)
    assert np.median(e) < 0.01


def test_mask_boxes():
    m = np.full((100, 100), 255, np.uint8)
    out = mask_boxes(m, np.array([[10, 10, 20, 30]]), pad=2)
    assert out[20, 15] == 0 and out[50, 50] == 255 and m[20, 15] == 255
