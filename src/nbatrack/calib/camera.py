"""Pinhole camera and PTZ (pan-tilt-zoom) parameterization.

Conventions
-----------
World = court frame (meters): X along sidelines, Y toward far sideline, Z up.
Camera frame = OpenCV: x right, y down, z forward (optical axis).
`R` maps world -> camera, `C` is the camera center in world coordinates, so a world
point X maps to camera coordinates  Xc = R (X - C).

Pixel coordinates: origin at the top-left corner of the top-left pixel, x right, y down,
in the *source video resolution* (OpenCV convention, pixel centers at integer + 0.5 is
NOT assumed; we treat integer coordinates as pixel centers, as OpenCV does).

PTZ angles (degrees):
  pan  > 0 : camera turns toward +X (right basket)
  tilt > 0 : camera looks down
  roll > 0 : image rotates clockwise as seen by the viewer (rotation about +z_cam)
With pan = tilt = roll = 0 the optical axis is world +Y (horizontal, toward far side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# world -> camera for pan = tilt = roll = 0: x_cam = +X, y_cam = -Z, z_cam = +Y
R_BASE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ptz_rotation(pan_deg: float, tilt_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """World -> camera rotation for the given PTZ angles."""
    th, ph, ps = (math.radians(v) for v in (pan_deg, tilt_deg, roll_deg))
    return _rz(ps) @ _rx(ph) @ R_BASE @ _rz(th)


def rotation_to_ptz(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `ptz_rotation`. Returns (pan_deg, tilt_deg, roll_deg)."""
    d = R[2, :]  # optical axis in world coords = (sin th cos ph, cos th cos ph, -sin ph)
    pan = math.atan2(d[0], d[1])
    tilt = math.asin(float(np.clip(-d[2], -1.0, 1.0)))
    R0 = _rx(tilt) @ R_BASE @ _rz(pan)
    M = R @ R0.T
    roll = math.atan2(M[1, 0], M[0, 0])
    return math.degrees(pan), math.degrees(tilt), math.degrees(roll)


def _distort(xn: np.ndarray, k1: float) -> np.ndarray:
    if k1 == 0.0:
        return xn
    r2 = np.sum(xn**2, axis=1, keepdims=True)
    return xn * (1.0 + k1 * r2)


def _undistort(xd: np.ndarray, k1: float, iters: int = 20) -> np.ndarray:
    if k1 == 0.0:
        return xd
    xn = xd.copy()
    for _ in range(iters):
        r2 = np.sum(xn**2, axis=1, keepdims=True)
        xn = xd / (1.0 + k1 * r2)
    return xn


@dataclass
class PinholeCamera:
    """Square-pixel, zero-skew pinhole camera with optional 1-term radial distortion.

    k1 acts on normalized image coordinates (x/z, y/z), like OpenCV's k1.
    """

    f: float
    cx: float
    cy: float
    R: np.ndarray
    C: np.ndarray
    k1: float = 0.0
    image_size: tuple[int, int] = field(default=(1920, 1080))  # (width, height)

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.C = np.asarray(self.C, dtype=float).reshape(3)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    @property
    def ptz(self) -> tuple[float, float, float]:
        return rotation_to_ptz(self.R)

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] == 2:
            X = np.hstack([X, np.zeros((len(X), 1))])
        return (X - self.C) @ self.R.T

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (N,3) or floor (N,2) -> pixels (N,2), depth (N,). Pixels behind the
        camera are NaN."""
        Xc = self.to_camera(X)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, :2] / z[:, None]
        xd = _distort(xn, self.k1)
        uv = xd * self.f + np.array([self.cx, self.cy])
        uv[z <= 1e-9] = np.nan
        return uv, z

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit ray directions in world coordinates for pixels (N,2)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = (uv - np.array([self.cx, self.cy])) / self.f
        xn = _undistort(xd, self.k1)
        dc = np.hstack([xn, np.ones((len(xn), 1))])
        dw = dc @ self.R  # R^T applied to each row
        return dw / np.linalg.norm(dw, axis=1, keepdims=True)

    def backproject_to_plane(self, uv: np.ndarray, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the horizontal plane Z = z.

        Returns (xy (N,2), valid (N,)). Invalid where the ray is parallel to or points
        away from the plane (i.e. above the horizon for z below the camera).
        """
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[:, 2]
        valid = np.isfinite(s) & (s > 0)
        xy = self.C[:2] + s[:, None] * d[:, :2]
        xy[~valid] = np.nan
        return xy, valid

    def ground_homography(self) -> np.ndarray:
        """3x3 homography court (X, Y, 1) on Z=0 -> image (u, v, 1). Exact only if k1 == 0."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]

    def copy(self, **changes) -> "PinholeCamera":
        base = replace(self, R=self.R.copy(), C=self.C.copy())
        return replace(base, **changes) if changes else base


@dataclass
class PTZState:
    pan_deg: float
    tilt_deg: float
    f: float
    roll_deg: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.pan_deg, self.tilt_deg, self.f, self.roll_deg])


@dataclass
class PTZRig:
    """A camera whose optical center C (and lens distortion) are fixed; only pan, tilt,
    zoom (and optionally roll) change over time.

    This is a *hypothesis* about the broadcast main camera, not a guarantee; the
    calibration code measures whether it holds (see calib.fit / experiments).
    """

    C: np.ndarray
    image_size: tuple[int, int] = (1920, 1080)
    k1: float = 0.0

    def __post_init__(self) -> None:
        self.C = np.asarray(self.C, dtype=float).reshape(3)

    def camera(self, s: PTZState) -> PinholeCamera:
        w, h = self.image_size
        return PinholeCamera(
            f=s.f,
            cx=w / 2.0,
            cy=h / 2.0,
            R=ptz_rotation(s.pan_deg, s.tilt_deg, s.roll_deg),
            C=self.C.copy(),
            k1=self.k1,
            image_size=self.image_size,
        )


def look_at_ptz(C: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(pan_deg, tilt_deg) that point the optical axis from C at a world target."""
    d = np.asarray(target, float) - np.asarray(C, float)
    d = d / np.linalg.norm(d)
    return math.degrees(math.atan2(d[0], d[1])), math.degrees(math.asin(-d[2]))
