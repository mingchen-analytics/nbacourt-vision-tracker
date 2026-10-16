"""Frame-to-frame camera-motion estimation for propagating a calibration.

Given calibration at frame t-1 and the image motion between t-1 and t, the court->image
homography at t is  H_t = M_{t-1 -> t} @ H_{t-1}.

Which pixels may be used to estimate M:
  * floor pixels (default): for ANY camera motion, floor points obey a homography between
    two views, so this is valid even if the camera center moves.
  * all static scene pixels (stands, banners): obey one homography ONLY for a pure
    rotation (fixed-center PTZ). Using them adds many features but is an assumption,
    so it is opt-in (`use_all_static=True`).
Players move and must always be masked out.

Drift accumulates with each step; propagation must be re-anchored with landmark or manual
keyframes (see calib/track.py) and its accumulated error is part of the quality flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nbatrack.calib.homography import apply_h
from nbatrack.court.geometry import NBA_COURT


@dataclass
class MotionEstimate:
    M: np.ndarray | None  # 3x3, prev image -> cur image
    n_tracked: int
    n_inliers: int
    rmse_px: float
    ok: bool


def floor_mask(H_court_to_img: np.ndarray, image_shape: tuple[int, int],
               margin_m: float = 1.5) -> np.ndarray:
    """uint8 mask of image pixels on the court floor (+ margin), given a calibration."""
    h, w = image_shape
    hl, hw = NBA_COURT.half_length + margin_m, NBA_COURT.half_width + margin_m
    n = 40
    xs = np.linspace(-hl, hl, n)
    ys = np.linspace(-hw, hw, n // 2)
    border = np.vstack([np.stack([xs, np.full(n, -hw)], 1), np.stack([np.full(n // 2, hl), ys], 1),
                        np.stack([xs[::-1], np.full(n, hw)], 1),
                        np.stack([np.full(n // 2, -hl), ys[::-1]], 1)])
    uv = apply_h(H_court_to_img, border)
    m = np.zeros((h, w), np.uint8)
    ok = np.all(np.isfinite(uv), 1) & (np.abs(uv).max(1) < 1e5)
    if ok.sum() >= 3:
        cv2.fillPoly(m, [uv[ok].round().astype(np.int32)], 255)
    return m


def mask_boxes(mask: np.ndarray, boxes: np.ndarray | None, pad: int = 6) -> np.ndarray:
    """Zero out (dilated) bounding boxes, e.g. detected players, in a mask."""
    if boxes is None or len(boxes) == 0:
        return mask
    m = mask.copy()
    h, w = m.shape
    for x1, y1, x2, y2 in np.asarray(boxes, float):
        m[max(0, int(y1) - pad):min(h, int(y2) + pad), max(0, int(x1) - pad):min(w, int(x2) + pad)] = 0
    return m


def estimate_motion(prev_gray: np.ndarray, cur_gray: np.ndarray, mask: np.ndarray | None = None,
                    max_corners: int = 800, ransac_px: float = 1.5,
                    fb_max_px: float = 0.7, min_inliers: int = 30) -> MotionEstimate:
    """KLT feature tracking with forward-backward check + RANSAC homography."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.005, 8, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 8:
        return MotionEstimate(None, 0, 0, float("nan"), False)
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **lk)
    fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_max_px)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a) < max(8, min_inliers):
        return MotionEstimate(None, int(len(a)), 0, float("nan"), False)
    M, inl = cv2.findHomography(a, b, cv2.RANSAC, ransac_px, maxIters=3000, confidence=0.999)
    if M is None:
        return MotionEstimate(None, int(len(a)), 0, float("nan"), False)
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(apply_h(M, a[inl]) - b[inl], axis=1)
    return MotionEstimate(M / M[2, 2], int(len(a)), int(inl.sum()),
                          float(np.sqrt(np.mean(err**2))), bool(inl.sum() >= min_inliers))


def camera_from_motion(prev_cam, M: np.ndarray, fixed_C: np.ndarray | None = None,
                       grid_step_m: float = 1.0):
    """Predict the current camera from the previous one and the image motion M by fitting
    to virtual floor correspondences (court grid points moved by M)."""
    from nbatrack.calib.fit import fit_camera
    from nbatrack.calib.models import Correspondences

    w, h = prev_cam.image_size
    hl, hw = NBA_COURT.half_length + 1.0, NBA_COURT.half_width + 1.0
    gx, gy = np.meshgrid(np.arange(-hl, hl + 1e-9, grid_step_m), np.arange(-hw, hw + 1e-9, grid_step_m))
    xy = np.stack([gx.ravel(), gy.ravel()], 1)
    uv, z = prev_cam.project(xy)
    ok = (z > 0) & np.all(np.isfinite(uv), 1) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)
    if ok.sum() < 8:
        return None
    uv2 = apply_h(M, uv[ok])
    corr = Correspondences(np.hstack([xy[ok], np.zeros((ok.sum(), 1))]), uv2)
    fit = fit_camera(corr, (w, h), init=prev_cam, fixed_C=fixed_C, inlier_px=1e6, loss="linear")
    return fit.camera if fit.camera is not None else None


def propagate(H_prev: np.ndarray, motion: MotionEstimate) -> np.ndarray | None:
    if not motion.ok or motion.M is None:
        return None
    H = motion.M @ H_prev
    return H / H[2, 2]
