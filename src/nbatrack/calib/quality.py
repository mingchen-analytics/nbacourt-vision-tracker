"""Per-frame calibration quality flags.

The flag gates whether court coordinates are emitted: frames rated 'poor' or 'failed'
produce NULL court coordinates (position_status = 'unknown').

All thresholds are initial engineering defaults expressed for 1080p and scaled by image
height. They MUST be re-tuned against annotated real footage; the synthetic study in
tools/quality_flag_study.py only checks that the flags are monotonic with true error.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nbatrack.calib.camera import PinholeCamera


@dataclass(frozen=True)
class QualityThresholds:
    good_rmse_px: float = 2.5
    fair_rmse_px: float = 5.0
    poor_rmse_px: float = 10.0
    min_points_homography: int = 4
    min_points_camera: int = 2
    good_min_points: int = 6
    good_min_hull_frac: float = 0.10  # landmark convex hull / image area
    fair_min_hull_frac: float = 0.02
    cam_height_m: tuple[float, float] = (2.0, 40.0)
    focal_px: tuple[float, float] = (300.0, 20000.0)  # at 1080p
    tilt_deg: tuple[float, float] = (-5.0, 70.0)


@dataclass
class QualityReport:
    quality: str  # good | fair | poor | failed
    score: float  # 0..1, higher is better
    reasons: list[str]
    rmse_corrected_px: float = float("nan")  # DOF-corrected residual scale (use for sigma)


MODEL_DOF = {"homography": 8, "camera": 7, "camera_free": 7, "ptz": 3}


def dof_corrected_rmse(rmse_px: float, n_points: int, n_params: int) -> float:
    """Unbiased residual scale: the fit RMSE underestimates the noise when the number of
    observations (2 per point) is close to the number of parameters."""
    m = 2 * n_points
    if m <= n_params:
        return float("inf")
    return float(rmse_px * np.sqrt(m / (m - n_params)))


def hull_fraction(uv: np.ndarray, image_size: tuple[int, int]) -> float:
    uv = np.asarray(uv, np.float32)
    if len(uv) < 3:
        return 0.0
    area = cv2.contourArea(cv2.convexHull(uv))
    return float(area / (image_size[0] * image_size[1]))


def assess(model_kind: str, n_inliers: int, rmse_px: float, inlier_uv: np.ndarray,
           image_size: tuple[int, int], camera: PinholeCamera | None = None,
           th: QualityThresholds = QualityThresholds(),
           n_params: int | None = None) -> QualityReport:
    """Rate one frame's calibration. Residual thresholds apply to the DOF-corrected
    RMSE, so a fit that merely interpolates its few points is not rated 'good'."""
    s = image_size[1] / 1080.0  # scale pixel thresholds with resolution
    reasons: list[str] = []
    min_pts = th.min_points_homography if model_kind == "homography" else th.min_points_camera
    if n_inliers < min_pts or not np.isfinite(rmse_px):
        return QualityReport("failed", 0.0, [f"inliers {n_inliers} < {min_pts}"])
    if camera is not None:
        pan, tilt, _ = camera.ptz
        f = camera.f / s
        if not th.cam_height_m[0] <= camera.C[2] <= th.cam_height_m[1]:
            reasons.append(f"camera height {camera.C[2]:.1f} m implausible")
        if not th.focal_px[0] <= f <= th.focal_px[1]:
            reasons.append(f"focal {camera.f:.0f}px implausible")
        if not th.tilt_deg[0] <= tilt <= th.tilt_deg[1]:
            reasons.append(f"tilt {tilt:.1f} deg implausible")
        if reasons:
            return QualityReport("failed", 0.0, reasons)
    hull = hull_fraction(inlier_uv, image_size)
    k = n_params if n_params is not None else MODEL_DOF.get(model_kind, 8)
    rc = dof_corrected_rmse(rmse_px, n_inliers, k)
    if not np.isfinite(rc):
        return QualityReport("failed", 0.0, [f"{n_inliers} points cannot validate {k} params"])
    r = rc / s
    # Continuous score: product of residual and geometry-support terms.
    score = float(np.clip(1.0 - r / th.poor_rmse_px, 0, 1)
                  * np.clip(hull / th.good_min_hull_frac, 0, 1) ** 0.5
                  * np.clip(n_inliers / th.good_min_points, 0, 1) ** 0.5)
    if r > th.poor_rmse_px:
        return QualityReport("failed", score, [f"rmse* {rc:.1f}px"], rc)
    if r > th.fair_rmse_px or hull < th.fair_min_hull_frac:
        reasons.append(f"rmse* {rc:.1f}px, hull {hull:.3f}")
        return QualityReport("poor", score, reasons, rc)
    if r > th.good_rmse_px or hull < th.good_min_hull_frac or n_inliers < th.good_min_points:
        reasons.append(f"rmse* {rc:.1f}px, hull {hull:.3f}, n {n_inliers}")
        return QualityReport("fair", score, reasons, rc)
    return QualityReport("good", score, reasons, rc)


def emits_coordinates(quality: str) -> bool:
    return quality in ("good", "fair")
