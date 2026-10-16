"""Textured synthetic broadcast frames (floor + stands) for testing image-based modules
such as camera-motion propagation. The rendering is crude but geometrically exact."""

from __future__ import annotations

import cv2
import numpy as np

from nbatrack.calib.camera import PinholeCamera
from nbatrack.court.geometry import NBA_COURT, court_polylines, densify

PX_PER_M = 40.0
FLOOR_MARGIN_M = 3.0
STAND_HEIGHT_M = 14.0


def _noise_texture(h: int, w: int, rng: np.random.Generator, base, scale: int, amp: float) -> np.ndarray:
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale), 3))
    n = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 0.35, (h, w, 3))
    img = np.asarray(base, float)[None, None] + amp * (n + fine)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_floor_texture(rng: np.random.Generator, noise_amp: float = 14.0,
                       seams: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Wood-like floor with court lines. Returns (BGR image, H_court_to_tex)."""
    hl, hw = NBA_COURT.half_length + FLOOR_MARGIN_M, NBA_COURT.half_width + FLOOR_MARGIN_M
    w, h = int(2 * hl * PX_PER_M), int(2 * hw * PX_PER_M)
    tex = _noise_texture(h, w, rng, (95, 150, 200), 6, noise_amp)
    if seams:  # plank seams along x
        for y in range(0, h, int(0.12 * PX_PER_M)):
            tex[y:y + 1] = (tex[y:y + 1] * 0.8).astype(np.uint8)
    H = np.array([[PX_PER_M, 0, hl * PX_PER_M], [0, -PX_PER_M, hw * PX_PER_M], [0, 0, 1.0]])
    for pl in court_polylines().values():
        p = densify(pl, 0.05)
        q = (np.hstack([p, np.ones((len(p), 1))]) @ H.T)[:, :2]
        cv2.polylines(tex, [q.round().astype(np.int32)], False, (245, 245, 245), 2, cv2.LINE_AA)
    return tex, H


def make_stand_texture(rng: np.random.Generator, length_m: float) -> np.ndarray:
    w, h = int(length_m * PX_PER_M / 2), int(STAND_HEIGHT_M * PX_PER_M / 2)
    tex = _noise_texture(h, w, rng, (60, 60, 70), 3, 40)
    for _ in range(int(w * h / 400)):  # "spectators"
        c = rng.integers(20, 235, 3).tolist()
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(tex, (x, y), int(rng.integers(2, 5)), c, -1)
    return tex


class SyntheticArena:
    def __init__(self, seed: int = 0, floor_noise_amp: float = 14.0, floor_seams: bool = True):
        rng = np.random.default_rng(seed)
        self.floor, self.H_court_to_tex = make_floor_texture(rng, floor_noise_amp, floor_seams)
        self.far_len = 2 * (NBA_COURT.half_length + 8.0)
        self.side_len = 2 * (NBA_COURT.half_width + FLOOR_MARGIN_M) + 6
        self.far_tex = make_stand_texture(rng, self.far_len)
        self.side_tex = [make_stand_texture(rng, self.side_len) for _ in range(2)]

    def _plane_H(self, cam: PinholeCamera, origin, u_axis, v_axis, tex_shape, len_u, len_v):
        """Homography texture px -> image for a planar rectangle in the world."""
        th, tw = tex_shape[:2]
        o, u, v = (np.asarray(a, float) for a in (origin, u_axis, v_axis))
        # world point = o + (s / tw * len_u) u + (1 - t / th) * len_v * v  (t = row, from top)
        A = np.column_stack([u * len_u / tw, -v * len_v / th, o + v * len_v])
        Hw = cam.K @ np.column_stack([cam.R @ A[:, 0], cam.R @ A[:, 1], cam.R @ (A[:, 2] - cam.C)])
        return Hw / Hw[2, 2]

    def render(self, cam: PinholeCamera) -> np.ndarray:
        w, h = cam.image_size
        img = np.full((h, w, 3), 30, np.uint8)
        hl = NBA_COURT.half_length + FLOOR_MARGIN_M
        hw = NBA_COURT.half_width + FLOOR_MARGIN_M
        planes = [
            (self.far_tex, (-self.far_len / 2, hw, 0.0), (1, 0, 0), (0, 0, 1), self.far_len),
            (self.side_tex[0], (-hl, -hw - 6, 0.0), (0, 1, 0), (0, 0, 1), self.side_len),
            (self.side_tex[1], (hl, hw, 0.0), (0, -1, 0), (0, 0, 1), self.side_len),
        ]
        for tex, o, u, v, L in planes:
            H = self._plane_H(cam, o, u, v, tex.shape, L, STAND_HEIGHT_M)
            # skip planes behind the camera
            corners = np.array([o, np.add(o, np.multiply(u, L)), np.add(o, np.multiply(v, STAND_HEIGHT_M))])
            if (cam.to_camera(corners)[:, 2] <= 0.1).any():
                continue
            warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_TRANSPARENT, dst=img)
            img = warped
        Hf = cam.ground_homography() @ np.linalg.inv(self.H_court_to_tex)
        fl = cv2.warpPerspective(self.floor, Hf, (w, h), flags=cv2.INTER_LINEAR)
        fm = cv2.warpPerspective(np.full(self.floor.shape[:2], 255, np.uint8), Hf, (w, h),
                                 flags=cv2.INTER_NEAREST)
        img[fm > 0] = fl[fm > 0]
        return img
