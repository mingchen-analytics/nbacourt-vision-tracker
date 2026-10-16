"""Physically reasonable temporal filtering of court trajectories.

Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother per track, using each
observation's own sigma. Raw coordinates are kept separately by the caller; this only
produces the filtered estimate. Large innovations are NOT smoothed away silently: they
are flagged (`speed_violation` when the raw step implies > max_speed).

Gaps are not filled: a frame without an observation has no filtered position (no
interpolation across occlusions at this stage).
"""

from __future__ import annotations

import numpy as np

MAX_PLAYER_SPEED = 10.0  # m/s; elite sprint ~9.5 m/s


def rts_smooth(t: np.ndarray, xy: np.ndarray, sigma: np.ndarray, accel_std: float = 6.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Smooth one track. t: (N,) seconds (increasing), xy: (N,2), sigma: (N,) meters.
    Returns (xy_smoothed (N,2), speed (N,) m/s)."""
    n = len(t)
    if n == 0:
        return xy.copy(), np.zeros(0)
    x = np.zeros((n, 4))
    P = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4))
    Pp = np.zeros((n, 4, 4))
    Fs = np.zeros((n, 4, 4))
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    xc = np.array([xy[0, 0], xy[0, 1], 0.0, 0.0])
    Pc = np.diag([sigma[0] ** 2, sigma[0] ** 2, 16.0, 16.0])
    for i in range(n):
        if i > 0:
            dt = max(t[i] - t[i - 1], 1e-3)
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
            Q = G @ G.T * accel_std**2
            xc = F @ xc
            Pc = F @ Pc @ F.T + Q
            Fs[i] = F
        xp[i], Pp[i] = xc, Pc
        R = np.eye(2) * max(sigma[i], 0.02) ** 2
        S = H @ Pc @ H.T + R
        K = Pc @ H.T @ np.linalg.inv(S)
        xc = xc + K @ (xy[i] - H @ xc)
        Pc = (np.eye(4) - K @ H) @ Pc
        x[i], P[i] = xc, Pc
    xs, Ps = x.copy(), P.copy()
    for i in range(n - 2, -1, -1):
        C = P[i] @ Fs[i + 1].T @ np.linalg.inv(Pp[i + 1])
        xs[i] = x[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = P[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    return xs[:, :2], np.linalg.norm(xs[:, 2:], axis=1)


def raw_speed_violations(t: np.ndarray, xy: np.ndarray, sigma: np.ndarray,
                         max_speed: float = MAX_PLAYER_SPEED) -> np.ndarray:
    """Flag observations whose step from the previous one is faster than max_speed even
    after allowing 2-sigma of measurement error on both ends."""
    flags = np.zeros(len(t), bool)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        d = np.linalg.norm(xy[i] - xy[i - 1]) - 2 * (sigma[i] + sigma[i - 1])
        flags[i] = d / dt > max_speed
    return flags
