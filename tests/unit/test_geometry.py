import math

import numpy as np
import pytest

from nbatrack.court.geometry import (
    FT,
    NBA_COURT,
    court_landmarks,
    court_polylines,
    densify,
    hoop_positions,
    is_on_court,
    landmark_by_name,
)


def test_court_dimensions_meters():
    assert 2 * NBA_COURT.half_length == pytest.approx(28.6512, abs=1e-6)
    assert 2 * NBA_COURT.half_width == pytest.approx(15.24, abs=1e-6)
    assert NBA_COURT.hoop_x == pytest.approx(14.3256 - 1.6002, abs=1e-6)
    assert NBA_COURT.rim_height == pytest.approx(3.048, abs=1e-6)
    assert NBA_COURT.ft_line_x == pytest.approx(14.3256 - 19 * FT, abs=1e-9)
    # Free-throw line is 15 ft from the backboard face.
    assert NBA_COURT.backboard_x - NBA_COURT.ft_line_x == pytest.approx(15 * FT, abs=1e-9)


def test_three_point_line_is_continuous_and_correct_radius():
    lines = court_polylines()
    for side in ("L", "R"):
        pl = lines[f"{side}_3pt"]
        s = -1 if side == "L" else 1
        hoop = np.array([s * NBA_COURT.hoop_x, 0.0])
        arc = pl[2:-2]
        r = np.linalg.norm(arc - hoop, axis=1)
        assert np.allclose(r, NBA_COURT.three_pt_radius, atol=1e-9)
        # straight segment ends exactly on the arc (no gap)
        assert np.linalg.norm(pl[1] - arc[0]) < 1e-9
        assert np.linalg.norm(pl[-2] - arc[-1]) < 1e-9
        # corner lines 22 ft from basket center laterally
        assert abs(pl[0, 1]) == pytest.approx(22 * FT)
        # arc apex is toward midcourt
        assert abs(arc[:, 0]).min() == pytest.approx(NBA_COURT.three_pt_top_x, abs=1e-3)
    # corner straight-line length ~14.2 ft (rulebook rounds to 14 ft)
    straight = NBA_COURT.half_length - NBA_COURT.three_pt_break_x
    assert straight / FT == pytest.approx(14.2, abs=0.05)


def test_landmarks_unique_symmetric_and_inside():
    lms = court_landmarks()
    names = [lm.name for lm in lms]
    assert len(names) == len(set(names))
    assert [lm.landmark_id for lm in lms] == list(range(len(lms)))
    by = landmark_by_name()
    for lm in lms:
        if lm.name.startswith("L_"):
            twin = by["R_" + lm.name[2:]]
            assert twin.xyz[0] == pytest.approx(-lm.xyz[0])
            assert twin.xyz[1:] == pytest.approx(lm.xyz[1:])
        if lm.on_floor:
            assert is_on_court(np.array(lm.xy), margin_m=1e-9)[0]
    assert by["R_rim_center"].xyz == pytest.approx((NBA_COURT.hoop_x, 0.0, 3.048))
    assert not by["R_rim_center"].on_floor


def test_landmark_ids_are_frozen():
    # IDs are written to output files; changing them silently would corrupt old data.
    by = landmark_by_name()
    assert by["L_near_corner"].landmark_id == 0
    assert by["center_circle_R"].landmark_id == 9
    assert by["L_lane_baseline_near"].landmark_id == 10
    assert by["R_rim_center"].landmark_id == 31
    assert len(by) == 32


def test_hoops():
    h = hoop_positions()
    assert h["left"][0] == -h["right"][0]
    assert h["left"][2] == pytest.approx(10 * FT)


def test_polylines_within_court_and_densify():
    for name, pl in court_polylines().items():
        assert is_on_court(pl, margin_m=1e-9).all(), name
    d = densify(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.1)
    assert len(d) == 11
    assert np.allclose(np.diff(d[:, 0]), 0.1)


def test_is_on_court_margin():
    assert not is_on_court(np.array([[15.0, 0.0]]))[0]
    assert is_on_court(np.array([[15.0, 0.0]]), margin_m=1.0)[0]
