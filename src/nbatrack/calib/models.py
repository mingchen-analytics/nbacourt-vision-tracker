"""Calibration model interface shared by all estimators.

Every per-frame calibration result implements `CalibrationModel`, so downstream code
(projection of players, visualization, export) never needs to know whether it came from
a free planar homography, an unconstrained pinhole camera, or the PTZ rig model.

Only models with `supports_3d = True` may project off-floor points (rim, airborne ball).
A planar homography is a valid mapping for z = 0 ONLY.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from nbatrack.calib.camera import PinholeCamera
from nbatrack.calib.homography import apply_h


@dataclass
class Correspondences:
    """Known world points and their observed image positions for one frame."""

    world: np.ndarray  # (N, 3) meters
    image: np.ndarray  # (N, 2) pixels
    ids: np.ndarray | None = None  # (N,) landmark ids
    sigma_px: np.ndarray | None = None  # (N,) observation std, optional

    def __post_init__(self) -> None:
        self.world = np.atleast_2d(np.asarray(self.world, float)).reshape(-1, 3)
        self.image = np.atleast_2d(np.asarray(self.image, float)).reshape(-1, 2)
        if len(self.world) != len(self.image):
            raise ValueError("world/image length mismatch")
        if self.ids is not None:
            self.ids = np.asarray(self.ids)

    def __len__(self) -> int:
        return len(self.world)

    @property
    def floor_mask(self) -> np.ndarray:
        return np.abs(self.world[:, 2]) < 1e-9

    def subset(self, mask: np.ndarray) -> "Correspondences":
        return Correspondences(
            self.world[mask],
            self.image[mask],
            None if self.ids is None else self.ids[mask],
            None if self.sigma_px is None else self.sigma_px[mask],
        )


class CalibrationModel(ABC):
    kind: str = "abstract"
    supports_3d: bool = False

    @abstractmethod
    def court_to_image(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) world or (N,2) floor points -> (N,2) pixels (NaN when undefined)."""

    @abstractmethod
    def image_to_court(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """(N,2) pixels -> (N,2) court XY on the plane Z = z (NaN when undefined)."""

    @abstractmethod
    def floor_homography(self) -> np.ndarray:
        """3x3 court floor -> image homography (approximate if lens distortion != 0)."""


class HomographyCalibration(CalibrationModel):
    kind = "homography"
    supports_3d = False

    def __init__(self, H_court_to_image: np.ndarray):
        H = np.asarray(H_court_to_image, float)
        self.H = H / H[2, 2]
        self.H_inv = np.linalg.inv(self.H)
        self.H_inv /= self.H_inv[2, 2]

    def court_to_image(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.atleast_2d(np.asarray(xyz, float))
        out = apply_h(self.H, xyz[:, :2])
        if xyz.shape[1] == 3:
            out[np.abs(xyz[:, 2]) > 1e-9] = np.nan  # undefined off the floor plane
        return out

    def image_to_court(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        if z != 0.0:
            return np.full((len(np.atleast_2d(uv)), 2), np.nan)
        # Reject points beyond the horizon: they map to the "other side" of the plane.
        uv = np.atleast_2d(np.asarray(uv, float))
        ph = np.hstack([uv, np.ones((len(uv), 1))]) @ self.H_inv.T
        out = apply_h(self.H_inv, uv)
        # sign of the homogeneous coordinate must match a known in-front point
        ref = self.H_inv @ np.append(self.court_to_image(np.zeros((1, 2)))[0], 1.0)
        out[np.sign(ph[:, 2]) != np.sign(ref[2])] = np.nan
        return out

    def floor_homography(self) -> np.ndarray:
        return self.H


class CameraCalibration(CalibrationModel):
    """Full 3D pinhole camera (from either an unconstrained fit or a PTZ rig)."""

    supports_3d = True

    def __init__(self, camera: PinholeCamera, kind: str = "camera"):
        self.camera = camera
        self.kind = kind

    def court_to_image(self, xyz: np.ndarray) -> np.ndarray:
        return self.camera.project(xyz)[0]

    def image_to_court(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        return self.camera.backproject_to_plane(uv, z)[0]

    def floor_homography(self) -> np.ndarray:
        return self.camera.ground_homography()
