"""Planar homography estimation (court floor <-> image) and decomposition.

Homographies here map court floor (X, Y) in meters -> image (u, v) in pixels unless
named otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from nbatrack.calib.camera import PinholeCamera


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography to (N,2) points. Points mapped to infinity become NaN."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / ph[:, 2:3]
    out[np.abs(ph[:, 2]) < 1e-12] = np.nan
    return out


def _normalizer(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2.0) / max(d, 1e-12)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Hartley-normalized DLT. src, dst: (N>=4, 2). Returns H with dst ~ H src."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences")
    Ts, Td = _normalizer(src), _normalizer(dst)
    s = apply_h(Ts, src)
    d = apply_h(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def refine_homography(H: np.ndarray, src: np.ndarray, dst: np.ndarray, loss: str = "linear",
                      f_scale: float = 2.0) -> np.ndarray:
    """Levenberg-Marquardt / TRF refinement minimizing image-space reprojection error."""
    h0 = (H / H[2, 2]).ravel()[:8]

    def resid(h: np.ndarray) -> np.ndarray:
        Hh = np.append(h, 1.0).reshape(3, 3)
        return (apply_h(Hh, src) - dst).ravel()

    method = "lm" if loss == "linear" and len(src) * 2 >= 8 else "trf"
    r = least_squares(resid, h0, method=method, loss=loss, f_scale=f_scale)
    return np.append(r.x, 1.0).reshape(3, 3)


@dataclass
class HomographyFit:
    H: np.ndarray  # court -> image
    inliers: np.ndarray  # bool (N,)
    rmse_px: float  # over inliers
    n_points: int

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def fit_homography(court_xy: np.ndarray, image_uv: np.ndarray, ransac_px: float = 6.0,
                   refine: bool = True) -> HomographyFit:
    """Robust court->image homography from >= 4 floor correspondences."""
    court_xy = np.asarray(court_xy, float)
    image_uv = np.asarray(image_uv, float)
    n = len(court_xy)
    if n < 4:
        raise ValueError(f"need >= 4 floor correspondences, got {n}")
    if n == 4:
        H = dlt_homography(court_xy, image_uv)
        inl = np.ones(n, bool)
    else:
        # NOTE: cv2.USAC_* estimators return None on valid data in OpenCV 5.0.0 (verified
        # locally); classic RANSAC is used instead.
        H, mask = cv2.findHomography(court_xy, image_uv, cv2.RANSAC, ransac_px,
                                     maxIters=5000, confidence=0.9999)
        if H is None:
            raise RuntimeError("homography RANSAC failed")
        inl = mask.ravel().astype(bool)
        if inl.sum() < 4:
            raise RuntimeError("fewer than 4 RANSAC inliers")
        H = dlt_homography(court_xy[inl], image_uv[inl])
    if refine and inl.sum() > 4:
        H = refine_homography(H, court_xy[inl], image_uv[inl])
    err = np.linalg.norm(apply_h(H, court_xy[inl]) - image_uv[inl], axis=1)
    return HomographyFit(H=H / H[2, 2], inliers=inl, rmse_px=float(np.sqrt(np.mean(err**2))),
                         n_points=n)


def focal_from_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Focal length (px) from a court->image homography, assuming square pixels, zero
    skew and known principal point (cx, cy).

    Uses the two rotation constraints r1.r2 = 0 and |r1| = |r2| on K^-1 H. Each is linear
    in w = 1/f^2; solved jointly by least squares. Returns None when degenerate (e.g.
    the image plane is parallel to the court).
    """
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = T @ H
    G = G / np.linalg.norm(G)
    f0 = float(cx + cy)  # image-scale constant: solve for w' = f0^2 / f^2 ~ O(1)
    h1, h2 = G[:, 0], G[:, 1]
    # r1.r2 = 0:      (h1x h2x + h1y h2y) w + h1z h2z = 0
    # |r1|^2=|r2|^2:  (h1x^2 + h1y^2 - h2x^2 - h2y^2) w + (h1z^2 - h2z^2) = 0
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    a = a / f0**2
    # Weight equations by their natural magnitude; drop ones that are numerically
    # degenerate (e.g. r1.r2 = 0 is identically 0 = 0 for a pure-tilt camera).
    nrm = np.hypot(a, b)
    keep = nrm > 1e-8 * max(nrm.max(), 1e-300)
    a, b = a[keep], b[keep]
    denom = float(a @ a)
    if denom < 1e-30:
        return None
    w = -float(a @ b) / denom
    if not np.isfinite(w) or w <= 0:
        return None
    return f0 / math.sqrt(w)


def camera_from_homography(H: np.ndarray, image_size: tuple[int, int],
                           f: float | None = None,
                           court_xy: np.ndarray | None = None) -> PinholeCamera | None:
    """Decompose a court->image homography into a pinhole camera (k1 = 0).

    The principal point is assumed at the image center. `court_xy` (the points the
    homography was fit on) resolves the sign ambiguity: they must lie in front of the
    camera. Returns None if the focal length cannot be recovered.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    if f is None:
        f = focal_from_homography(H, cx, cy)
        if f is None:
            return None
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    # Observed court points must be in front of the camera (positive depth).
    pts = np.zeros((1, 2)) if court_xy is None else np.asarray(court_xy, float)
    depth = r1[2] * pts[:, 0] + r2[2] * pts[:, 1] + t[2]
    if np.median(depth) < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection -> flip
        R = U @ np.diag([1, 1, -1]) @ Vt
    C = -R.T @ t
    # Camera must be above the floor; if not, the solution is the mirrored one.
    if C[2] < 0:
        return None
    return PinholeCamera(f=f, cx=cx, cy=cy, R=R, C=C, image_size=image_size)
