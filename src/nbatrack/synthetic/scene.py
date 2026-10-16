"""Synthetic sideline-broadcast camera sequences.

Assumed main-camera placement (an assumption for simulation only; real values are
estimated from footage): centered on the midcourt line, ~14 m behind the near sideline,
~9 m above the floor. Focal lengths 1500-3500 px at 1920x1080 roughly span a full-court
wide shot to a half-court shot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nbatrack.calib.camera import PinholeCamera, PTZRig, PTZState, look_at_ptz
from nbatrack.calib.models import Correspondences
from nbatrack.court.geometry import NBA_COURT, court_landmarks

DEFAULT_C = np.array([0.0, -(NBA_COURT.half_width + 14.0), 9.0])


def default_rig(image_size: tuple[int, int] = (1920, 1080), k1: float = 0.0) -> PTZRig:
    return PTZRig(DEFAULT_C.copy(), image_size, k1)


@dataclass
class BroadcastScenario:
    n_frames: int = 120
    fps: float = 30.0
    image_size: tuple[int, int] = (1920, 1080)
    C: np.ndarray = field(default_factory=lambda: DEFAULT_C.copy())
    roll_deg: float = 0.3
    k1: float = 0.0
    # camera center drift (m/s) to simulate a camera that is NOT a fixed-center PTZ
    center_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # target follows the action between these x limits
    target_x_range: tuple[float, float] = (-11.0, 11.0)
    f_range: tuple[float, float] = (1800.0, 3200.0)
    seed: int = 0


@dataclass
class SyntheticFrame:
    index: int
    timestamp_sec: float
    camera: PinholeCamera  # ground-truth camera
    state: PTZState


def generate_sequence(sc: BroadcastScenario) -> list[SyntheticFrame]:
    """Smooth pan / zoom trajectory following a moving point of interest."""
    rng = np.random.default_rng(sc.seed)
    t = np.arange(sc.n_frames) / sc.fps
    T = max(t[-1], 1e-6)
    lo, hi = sc.target_x_range
    phase = rng.uniform(0, 2 * math.pi)
    tx = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2 * math.pi * t / max(T, 4.0) + phase)
    ty = 1.0 * np.sin(2 * math.pi * t / 3.0 + rng.uniform(0, 6))
    f_lo, f_hi = sc.f_range
    f = f_lo + (f_hi - f_lo) * 0.5 * (1 + np.sin(2 * math.pi * t / max(T, 5.0) + phase / 2))
    frames = []
    for i in range(sc.n_frames):
        C = sc.C + sc.center_velocity * t[i]
        pan, tilt = look_at_ptz(C, np.array([tx[i], ty[i], 0.0]))
        st = PTZState(pan, tilt, float(f[i]), sc.roll_deg)
        cam = PTZRig(C, sc.image_size, sc.k1).camera(st)
        frames.append(SyntheticFrame(i, float(t[i]), cam, st))
    return frames


def visible(camera: PinholeCamera, X: np.ndarray, border_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    uv, z = camera.project(X)
    w, h = camera.image_size
    m = (z > 0) & np.all(np.isfinite(uv), axis=1)
    m &= (uv[:, 0] >= border_px) & (uv[:, 0] <= w - 1 - border_px)
    m &= (uv[:, 1] >= border_px) & (uv[:, 1] <= h - 1 - border_px)
    return uv, m


def observe_landmarks(camera: PinholeCamera, rng: np.random.Generator, noise_px: float = 1.0,
                      dropout: float = 0.0, outlier_frac: float = 0.0,
                      outlier_px: float = 60.0, include_3d: bool = True,
                      ids: list[int] | None = None) -> tuple[Correspondences, Correspondences]:
    """Simulate a landmark detector.

    Returns (observed, ground_truth) correspondences for the visible landmarks; the
    observed ones have Gaussian pixel noise, random dropout and gross outliers.
    """
    lms = court_landmarks()
    if not include_3d:
        lms = [lm for lm in lms if lm.on_floor]
    if ids is not None:
        lms = [lm for lm in lms if lm.landmark_id in set(ids)]
    X = np.array([lm.xyz for lm in lms])
    lid = np.array([lm.landmark_id for lm in lms])
    uv, m = visible(camera, X)
    keep = m & (rng.random(len(X)) >= dropout)
    gt = Correspondences(X[keep], uv[keep], lid[keep])
    obs_uv = uv[keep] + rng.normal(0, noise_px, size=(int(keep.sum()), 2))
    n_out = int(round(outlier_frac * len(obs_uv)))
    if n_out:
        idx = rng.choice(len(obs_uv), n_out, replace=False)
        ang = rng.uniform(0, 2 * math.pi, n_out)
        obs_uv[idx] += outlier_px * np.stack([np.cos(ang), np.sin(ang)], 1)
    obs = Correspondences(X[keep], obs_uv, lid[keep], np.full(len(obs_uv), noise_px))
    return obs, gt


def sample_players(rng: np.random.Generator, n: int = 10, margin_m: float = 0.5) -> np.ndarray:
    """Uniformly random floor positions inside the court (meters, (n, 2))."""
    hl, hw = NBA_COURT.half_length - margin_m, NBA_COURT.half_width - margin_m
    return np.stack([rng.uniform(-hl, hl, n), rng.uniform(-hw, hw, n)], 1)
