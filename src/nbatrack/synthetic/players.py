"""Simulated player trajectories and their rendering as simple upright boxes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nbatrack.calib.camera import PinholeCamera
from nbatrack.court.geometry import NBA_COURT

TEAM_BGR = {"A": (40, 40, 220), "B": (220, 200, 40), "REF": (30, 30, 30)}


@dataclass
class PlayerSim:
    pos: np.ndarray  # (T, N, 2) meters
    team: list[str]  # (N,) "A" | "B" | "REF"
    height_m: np.ndarray  # (N,)


def simulate_players(n_frames: int, fps: float, rng: np.random.Generator, n_per_team: int = 5,
                     n_refs: int = 3, max_speed: float = 7.0) -> PlayerSim:
    team = ["A"] * n_per_team + ["B"] * n_per_team + ["REF"] * n_refs
    n = len(team)
    hl, hw = NBA_COURT.half_length - 0.3, NBA_COURT.half_width - 0.3
    p = np.stack([rng.uniform(-hl * 0.8, hl * 0.8, n), rng.uniform(-hw * 0.8, hw * 0.8, n)], 1)
    v = np.zeros((n, 2))
    dt = 1.0 / fps
    out = np.empty((n_frames, n, 2))
    for t in range(n_frames):
        v = 0.97 * v + rng.normal(0, 0.6, (n, 2))  # OU-like velocity
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(sp > max_speed, v * max_speed / np.maximum(sp, 1e-9), v)
        p = p + v * dt
        for k, lim in ((0, hl), (1, hw)):
            over = np.abs(p[:, k]) > lim
            p[over, k] = np.sign(p[over, k]) * lim
            v[over, k] *= -1
        out[t] = p
    heights = np.where(np.array(team) == "REF", 1.8, rng.uniform(1.9, 2.15, n))
    return PlayerSim(out, team, heights)


def player_boxes(cam: PinholeCamera, xy: np.ndarray, height_m: np.ndarray,
                 width_m: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Image boxes (N, 4) for upright players standing at xy. Returns (boxes, depth)."""
    n = len(xy)
    foot = np.hstack([xy, np.zeros((n, 1))])
    head = np.hstack([xy, height_m[:, None]])
    uf, zf = cam.project(foot)
    uh, _ = cam.project(head)
    h = uf[:, 1] - uh[:, 1]
    # width from projecting a horizontal offset along the camera x direction
    side = foot + np.outer(np.ones(n), cam.R[0]) * (width_m / 2)
    us, _ = cam.project(side)
    half_w = np.abs(us[:, 0] - uf[:, 0])
    boxes = np.stack([uf[:, 0] - half_w, uh[:, 1], uf[:, 0] + half_w, uf[:, 1]], 1)
    boxes[~np.isfinite(h) | (zf <= 0)] = np.nan
    return boxes, zf


def draw_players(img: np.ndarray, boxes: np.ndarray, depth: np.ndarray, team: list[str]) -> np.ndarray:
    order = np.argsort(-np.nan_to_num(depth, nan=-1))  # far first
    for i in order:
        b = boxes[i]
        if not np.all(np.isfinite(b)):
            continue
        x1, y1, x2, y2 = b.round().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), TEAM_BGR[team[i]], -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return img
