"""Image ground-contact points -> court coordinates with uncertainty.

Uncertainty model (first-order, documented approximation):
    Sigma_uv = Sigma_ground + (calib_rmse_px^2) * I
    Sigma_xy = J Sigma_uv J^T,   J = d(court_xy)/d(uv) at the point (numerical)
`court_sigma_m` = sqrt(largest eigenvalue of Sigma_xy), i.e. the 1-sigma length of the
major axis -- conservative, and it grows naturally toward the far sideline where a pixel
covers more floor.

Treating calibration error as isotropic image-space noise of size `calib_rmse_px` is a
simplification; tests/integration/test_projection_uncertainty.py measures how well it is
calibrated on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbatrack.calib.models import CalibrationModel
from nbatrack.court.geometry import is_on_court

# Prior image-space uncertainty of the bbox-bottom ground point, as a fraction of box size.
# These are UNCALIBRATED priors; replace with values measured on annotated footage.
BBOX_BOTTOM_SIGMA_FRAC_W = 0.10
BBOX_BOTTOM_SIGMA_FRAC_H = 0.02


@dataclass
class GroundPoints:
    uv: np.ndarray  # (N, 2) px
    cov_uv: np.ndarray  # (N, 2, 2) px^2
    method: str  # "bbox_bottom" | "feet_keypoints"


def bbox_bottom_points(bboxes: np.ndarray) -> GroundPoints:
    """Baseline ground-contact estimator: bottom-center of each (x1, y1, x2, y2) box."""
    b = np.atleast_2d(np.asarray(bboxes, float))
    uv = np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], 1)
    sx = BBOX_BOTTOM_SIGMA_FRAC_W * (b[:, 2] - b[:, 0])
    sy = BBOX_BOTTOM_SIGMA_FRAC_H * (b[:, 3] - b[:, 1])
    cov = np.zeros((len(b), 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = sx**2, sy**2
    return GroundPoints(uv, cov, "bbox_bottom")


@dataclass
class CourtPoints:
    xy: np.ndarray  # (N, 2) m, NaN where undefined
    sigma_m: np.ndarray  # (N,)
    cov_xy: np.ndarray  # (N, 2, 2)
    on_court: np.ndarray  # (N,) bool (within margin)


def image_jacobian(model: CalibrationModel, uv: np.ndarray, h: float = 0.5) -> np.ndarray:
    """d(court_xy)/d(uv) by central differences, (N, 2, 2)."""
    uv = np.atleast_2d(uv)
    J = np.empty((len(uv), 2, 2))
    for k in range(2):
        d = np.zeros(2)
        d[k] = h
        J[:, :, k] = (model.image_to_court(uv + d) - model.image_to_court(uv - d)) / (2 * h)
    return J


def project_ground_points(model: CalibrationModel, gp: GroundPoints, calib_rmse_px: float = 0.0,
                          on_court_margin_m: float = 1.0) -> CourtPoints:
    xy = model.image_to_court(gp.uv)
    J = image_jacobian(model, gp.uv)
    cov_uv = gp.cov_uv + (calib_rmse_px**2) * np.eye(2)[None]
    cov = J @ cov_uv @ np.transpose(J, (0, 2, 1))
    with np.errstate(invalid="ignore"):
        ev = np.linalg.eigvalsh(np.nan_to_num(cov, nan=np.inf))
    sigma = np.sqrt(np.clip(ev[:, -1], 0, None))
    bad = ~np.all(np.isfinite(xy), 1)
    sigma[bad] = np.nan
    on = np.zeros(len(xy), bool)
    on[~bad] = is_on_court(xy[~bad], on_court_margin_m)
    return CourtPoints(xy, sigma, cov, on)
