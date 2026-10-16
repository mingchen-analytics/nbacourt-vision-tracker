"""Constant-velocity Kalman filters used by the tracker."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKF:
    """State [p (k), v (k)] with white-acceleration process noise.

    q: acceleration noise std (units of p per s^2); r: measurement noise std (units of p)
    may be scalar or per-dimension.
    """

    def __init__(self, p0: np.ndarray, q: float | np.ndarray, r: float | np.ndarray,
                 v0_std: float | np.ndarray):
        p0 = np.asarray(p0, float)
        self.k = len(p0)
        self.x = np.concatenate([p0, np.zeros(self.k)])
        q = np.broadcast_to(np.asarray(q, float), (self.k,))
        self.q = q
        self.r = np.broadcast_to(np.asarray(r, float), (self.k,))
        v0 = np.broadcast_to(np.asarray(v0_std, float), (self.k,))
        self.P = np.diag(np.concatenate([self.r**2, v0**2]))

    def predict(self, dt: float) -> None:
        k = self.k
        F = np.eye(2 * k)
        F[:k, k:] = np.eye(k) * dt
        G = np.vstack([np.eye(k) * 0.5 * dt**2, np.eye(k) * dt])
        Q = G @ np.diag(self.q**2) @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    @property
    def pos(self) -> np.ndarray:
        return self.x[: self.k]

    @property
    def pos_cov(self) -> np.ndarray:
        return self.P[: self.k, : self.k]

    def innovation_cov(self, r: np.ndarray | None = None) -> np.ndarray:
        r = self.r if r is None else np.broadcast_to(np.asarray(r, float), (self.k,))
        return self.pos_cov + np.diag(r**2)

    def update(self, z: np.ndarray, r: np.ndarray | float | None = None) -> None:
        k = self.k
        H = np.hstack([np.eye(k), np.zeros((k, k))])
        S = self.innovation_cov(r)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - self.pos)
        self.P = (np.eye(2 * k) - K @ H) @ self.P

    def mahalanobis2(self, z: np.ndarray, r: np.ndarray | float | None = None) -> float:
        d = np.asarray(z, float) - self.pos
        return float(d @ np.linalg.solve(self.innovation_cov(r), d))
