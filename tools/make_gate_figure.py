"""Render the calibration quality-gate comparison: the same frame with correct and wrong poses.

Uses the regression fixtures in tests/fixtures/calib_clip6 (real failures of the removed automatic
re-initialization). For each pose it draws the projected court lines and prints the three gate
signals (line inliers, floor IoU, court coverage) and the resulting grade.

Usage: python tools/make_gate_figure.py --out outputs/figures/gate
Writes one JPEG per pose plus gate_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nbatrack.calib.camera import PinholeCamera, ptz_rotation
from nbatrack.calib.lines import (alignment_stats, build_evidence, floor_agreement, floor_tone_mask,
                                  floor_valid_mask, template_points)
from nbatrack.court.render import draw_projected_court
from nbatrack.pipeline.video_pipeline import AnalyzeConfig, grade_calibration

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "calib_clip6"
POSES = json.loads((FIX / "poses.json").read_text())

# (pose name, output stem, overlay colour BGR)
PANELS = [
    ("frame_06100_keyframe", "correct", (80, 220, 80)),
    ("frame_06100_auto_init_v1", "wrong_stands", (60, 60, 240)),
    ("frame_06100_auto_init_v2", "wrong_diverged", (60, 60, 240)),
    ("frame_02400_auto_init_v1", "wrong_stands_f2400", (60, 60, 240)),
]


def camera(c) -> PinholeCamera:
    return PinholeCamera(c["f"], c["cx"], c["cy"], ptz_rotation(*c["ptz"]), np.array(c["C"]), c["k1"], (1280, 720))


def court_to_image(cam: PinholeCamera):
    def f(xy: np.ndarray) -> np.ndarray:
        X = np.column_stack([xy, np.zeros(len(xy))])
        uv, z = cam.project(X)
        uv = np.asarray(uv, float).copy()
        uv[z <= 0] = np.nan
        return uv
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/gate")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AnalyzeConfig()
    metrics = {}
    for name, stem, color in PANELS:
        p = POSES[name]
        img = cv2.imread(str(FIX / f"frame_{p['frame']:05d}.jpg"))
        cam = camera(p["camera"])
        ev = build_evidence(img, floor_valid_mask(cam, img.shape[:2], 1.0))
        n, inl, _ = alignment_stats(cam, ev, template_points())
        fa = floor_agreement(cam, floor_tone_mask(img))
        grade, why = grade_calibration(inl, n, fa["iou"], cfg, court_coverage=fa["court_coverage"])
        metrics[stem] = dict(pose=name, frame=p["frame"], line_inliers=round(float(inl), 3),
                             floor_iou=round(float(fa["iou"]), 3),
                             court_coverage=round(float(fa["court_coverage"]), 3),
                             grade=grade, reason=why, old_rule_accepts=bool(inl >= 0.55))
        vis = draw_projected_court(img.copy(), court_to_image(cam), color, 3)
        cv2.imwrite(str(out / f"{stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / "gate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
