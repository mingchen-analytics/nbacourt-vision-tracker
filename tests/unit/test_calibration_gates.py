"""Regression tests from real calibration failures on clip 6 (2026-10-15).

Wrong poses produced by the (removed) automatic re-initialization drew the court onto the
stands and still got high line-alignment scores. The grading (line inliers AND floor IoU)
must reject them, and must accept the visually checked keyframe poses of the same video.
Fixtures: tests/fixtures/calib_clip6 (frames as JPEG + camera parameters).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from nbatrack.calib.camera import PinholeCamera, ptz_rotation
from nbatrack.calib.lines import alignment_stats, build_evidence, floor_agreement, floor_tone_mask, floor_valid_mask, template_points
from nbatrack.pipeline.video_pipeline import AnalyzeConfig, grade_calibration

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "calib_clip6"
POSES = json.loads((FIX / "poses.json").read_text())


def _camera(c) -> PinholeCamera:
    return PinholeCamera(c["f"], c["cx"], c["cy"], ptz_rotation(*c["ptz"]), np.array(c["C"]), c["k1"], (1280, 720))


def _scores(name):
    p = POSES[name]
    img = cv2.imread(str(FIX / f"frame_{p['frame']:05d}.jpg"))
    cam = _camera(p["camera"])
    ev = build_evidence(img, floor_valid_mask(cam, img.shape[:2], 1.0))
    n, inl, _ = alignment_stats(cam, ev, template_points())
    fa = floor_agreement(cam, floor_tone_mask(img))
    return n, inl, fa["iou"], fa["court_coverage"]


@pytest.mark.parametrize("name", sorted(POSES))
def test_real_poses_are_graded_as_expected(name):
    n, inl, iou, cov = _scores(name)
    q, why = grade_calibration(inl, n, iou, AnalyzeConfig(), court_coverage=cov)
    accepted = q in ("good", "fair")
    assert accepted == (POSES[name]["expect"] == "accept"), (name, q, why, round(inl, 3), round(iou, 3))


def test_line_inliers_alone_would_accept_a_court_on_the_stands():
    """Documents WHY floor IoU is required: the f2400 failure explains many 'lines' in the crowd."""
    n, inl, iou, cov = _scores("frame_02400_auto_init_v1")
    cfg = AnalyzeConfig()
    assert inl >= 0.55 and iou < cfg.floor_iou_ok and cov < cfg.court_coverage_ok


def test_grade_rules():
    cfg = AnalyzeConfig()
    assert grade_calibration(0.9, 1000, 0.7, cfg) == ("good", "ok")
    assert grade_calibration(0.72, 1000, 0.7, cfg) == ("fair", "ok")
    assert grade_calibration(0.9, 1000, 0.40, cfg)[0] == "poor"  # line fit fine, floor disagrees
    assert grade_calibration(0.9, 1000, 0.10, cfg)[0] == "failed"
    assert grade_calibration(0.6, 1000, 0.7, cfg)[0] == "poor"  # was "fair" before the gate study
    assert grade_calibration(0.9, 1000, 0.7, cfg, motion_implausible=True)[0] == "poor"
    assert grade_calibration(0.9, 10, 0.7, cfg)[0] == "failed"
    # wood-coloured apron: low IoU but the projected court is almost all floor tone
    assert grade_calibration(0.95, 1000, 0.34, cfg, court_coverage=0.99) == ("good", "ok")
    assert grade_calibration(0.95, 1000, 0.34, cfg, court_coverage=0.60)[0] == "poor"
    assert grade_calibration(0.95, 1000, 0.20, cfg, court_coverage=0.50)[0] == "failed"
    # natural failure on unseen clip-6 frames: tiny projected court inside the floor (IoU ~0, coverage ~1)
    assert grade_calibration(0.92, 1000, 0.004, cfg, court_coverage=1.0)[0] == "failed"
