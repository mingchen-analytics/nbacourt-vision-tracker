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
from nbatrack.calib.lines import alignment_stats, build_evidence, floor_iou, floor_tone_mask, floor_valid_mask, template_points
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
    return n, inl, floor_iou(cam, floor_tone_mask(img))


@pytest.mark.parametrize("name", sorted(POSES))
def test_real_poses_are_graded_as_expected(name):
    n, inl, iou = _scores(name)
    q, why = grade_calibration(inl, n, iou, AnalyzeConfig())
    accepted = q in ("good", "fair")
    assert accepted == (POSES[name]["expect"] == "accept"), (name, q, why, round(inl, 3), round(iou, 3))


def test_line_inliers_alone_would_accept_a_court_on_the_stands():
    """Documents WHY floor IoU is required: the f2400 failure explains many 'lines' in the crowd."""
    n, inl, iou = _scores("frame_02400_auto_init_v1")
    assert inl >= 0.55 and iou < AnalyzeConfig().floor_iou_ok


def test_grade_rules():
    cfg = AnalyzeConfig()
    assert grade_calibration(0.9, 1000, 0.7, cfg) == ("good", "ok")
    assert grade_calibration(0.72, 1000, 0.7, cfg) == ("fair", "ok")
    assert grade_calibration(0.9, 1000, 0.40, cfg)[0] == "poor"  # line fit fine, floor disagrees
    assert grade_calibration(0.9, 1000, 0.10, cfg)[0] == "failed"
    assert grade_calibration(0.6, 1000, 0.7, cfg)[0] == "poor"  # was "fair" before the gate study
    assert grade_calibration(0.9, 1000, 0.7, cfg, motion_implausible=True)[0] == "poor"
    assert grade_calibration(0.9, 10, 0.7, cfg)[0] == "failed"
