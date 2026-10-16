"""The PTZ assumption must be accepted for a fixed-center camera and rejected for a
moving one (synthetic, exact ground truth)."""

import numpy as np
import pytest

from nbatrack.calib.select import select_model
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks


def _frames(velocity, seed):
    sc = BroadcastScenario(n_frames=40, seed=seed, center_velocity=np.array(velocity),
                           f_range=(1800, 3000))
    rng = np.random.default_rng(seed)
    return [observe_landmarks(f.camera, rng, 1.0, 0.2)[0] for f in generate_sequence(sc)], sc


@pytest.mark.parametrize("seed", [0, 1])
def test_fixed_center_selects_ptz(seed):
    frames, sc = _frames([0, 0, 0], seed)
    rep = select_model(frames, sc.image_size)
    assert rep.choice == "ptz", rep
    assert np.linalg.norm(rep.ptz_center - sc.C) < 0.3


@pytest.mark.parametrize("seed", [0, 1])
def test_moving_center_rejects_ptz(seed):
    frames, sc = _frames([1.0, 0, 0], seed)  # 1 m/s dolly
    rep = select_model(frames, sc.image_size)
    assert rep.choice == "camera_free", rep
