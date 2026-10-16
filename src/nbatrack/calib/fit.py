"""Camera estimators: unconstrained pinhole per frame, PTZ per frame (known center),
and joint PTZ bundle adjustment over a camera segment.

All residuals are image-space reprojection errors in pixels. Robust losses are used so
that a few wrong landmark detections do not dominate.

Parameterization (shared by all estimators): rotation as PTZ angles (pan, tilt, roll)
in degrees, focal length in pixels, camera center C in meters. Euler angles are safe here
because broadcast cameras never approach tilt = +/-90 deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from nbatrack.calib.camera import R_BASE, PinholeCamera, PTZRig, PTZState, ptz_rotation
from nbatrack.calib.homography import camera_from_homography, fit_homography
from nbatrack.calib.models import Correspondences

# Focal length is optimized in kilo-pixels so all parameters have comparable magnitude
# (otherwise relative step tolerances terminate early on the angles).
F_UNIT = 1000.0
TOL = dict(xtol=1e-10, ftol=1e-10, gtol=1e-10)


def _ptz_rotations(pan: np.ndarray, tilt: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Vectorized `ptz_rotation` -> (F, 3, 3)."""
    th, ph, ps = (np.radians(np.asarray(a, float)) for a in (pan, tilt, roll))
    n = len(th)
    Rz_w = np.zeros((n, 3, 3))
    Rz_w[:, 0, 0] = Rz_w[:, 1, 1] = np.cos(th)
    Rz_w[:, 0, 1], Rz_w[:, 1, 0] = -np.sin(th), np.sin(th)
    Rz_w[:, 2, 2] = 1.0
    Rx = np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = Rx[:, 2, 2] = np.cos(ph)
    Rx[:, 1, 2], Rx[:, 2, 1] = -np.sin(ph), np.sin(ph)
    Rr = np.zeros((n, 3, 3))
    Rr[:, 0, 0] = Rr[:, 1, 1] = np.cos(ps)
    Rr[:, 0, 1], Rr[:, 1, 0] = -np.sin(ps), np.sin(ps)
    Rr[:, 2, 2] = 1.0
    return Rr @ Rx @ R_BASE @ Rz_w


def _project(world: np.ndarray, pan: float, tilt: float, roll: float, f: float,
             C: np.ndarray, cx: float, cy: float, k1: float) -> np.ndarray:
    R = ptz_rotation(pan, tilt, roll)
    Xc = (world - C) @ R.T
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn = Xc[:, :2] / z
    if k1 != 0.0:
        xn = xn * (1.0 + k1 * np.sum(xn**2, axis=1, keepdims=True))
    return xn * f + np.array([cx, cy])


@dataclass
class CameraFit:
    camera: PinholeCamera | None
    success: bool
    rmse_px: float = float("nan")
    residuals_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    message: str = ""
    param_cov: np.ndarray | None = None  # covariance of the free parameters
    param_names: tuple[str, ...] = ()


def _residual_stats(cam: PinholeCamera, corr: Correspondences) -> np.ndarray:
    uv, _ = cam.project(corr.world)
    return np.linalg.norm(uv - corr.image, axis=1)


def _covariance(res, n_obs: int) -> np.ndarray | None:
    J = res.jac
    dof = max(1, n_obs - J.shape[1])
    s2 = 2.0 * res.cost / dof
    try:
        return np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        return None


def prior_candidates(corr: Correspondences, image_size: tuple[int, int],
                     behind_m=(6.0, 10.0, 14.0, 18.0, 24.0, 30.0),
                     height_m=(4.0, 7.0, 10.0, 13.0, 16.0),
                     hfov_deg=(30.0, 50.0), max_rmse_px: float = 40.0) -> list[CameraFit]:
    """All plausible fixed-center, zero-roll fits (pan, tilt, f free) over a grid of
    sideline camera positions. Roll is fixed to 0 because with few, partly collinear
    landmarks a free roll compensates for a wrong camera position."""
    from nbatrack.calib.camera import look_at_ptz
    from nbatrack.court.geometry import NBA_COURT

    w, h = image_size
    target = corr.world.mean(axis=0)
    out = []
    for d in behind_m:
        for hh in height_m:
            for fov in hfov_deg:
                C = np.array([0.0, -(NBA_COURT.half_width + d), hh])
                pan, tilt = look_at_ptz(C, target)
                f = (w / 2) / np.tan(np.radians(fov / 2))
                init = PinholeCamera(f, w / 2, h / 2, ptz_rotation(pan, tilt, 0.0), C, image_size=image_size)
                # keep every point: with so few landmarks, "outlier" rejection discards real data
                fit = fit_camera(corr, image_size, init=init, fixed_C=C, fixed_roll=0.0, inlier_px=1e6)
                if fit.camera is not None and np.isfinite(fit.rmse_px) and fit.rmse_px < max_rmse_px \
                        and 0.0 < fit.camera.ptz[1] < 60.0:
                    out.append(fit)
    return out


def initial_camera_from_prior(corr: Correspondences, image_size: tuple[int, int],
                              behind_m=(10.0, 16.0, 24.0), height_m=(6.0, 10.0, 14.0),
                              hfov_deg=(25.0, 45.0, 70.0), side_x=(0.0,)) -> CameraFit:
    """Initialize from a sideline-broadcast prior when correspondences are few or nearly
    degenerate (closed-form homography decomposition is then unreliable).

    Grid over camera distance behind the near sideline, height and horizontal field of
    view; each candidate looks at the centroid of the observed world points and is
    refined with `fit_camera`. The best plausible fit (lowest RMSE, camera above the
    floor and on the near side) is returned.
    """
    from nbatrack.calib.camera import look_at_ptz
    from nbatrack.court.geometry import NBA_COURT

    w, h = image_size
    target = corr.world.mean(axis=0)
    best = CameraFit(None, False, message="no plausible prior fit")
    for bx in side_x:
        for d in behind_m:
            for hh in height_m:
                for fov in hfov_deg:
                    C = np.array([bx, -(NBA_COURT.half_width + d), hh])
                    pan, tilt = look_at_ptz(C, target)
                    f = (w / 2) / np.tan(np.radians(fov / 2))
                    init = PinholeCamera(f, w / 2, h / 2, ptz_rotation(pan, tilt, 0.0), C,
                                         image_size=image_size)
                    # Center fixed at the grid value: with few points, a free center is
                    # unidentifiable (zoom / dolly ambiguity) and runs off to infinity.
                    fit = fit_camera(corr, image_size, init=init, fixed_C=C, inlier_px=25.0)
                    if fit.camera is None or not np.isfinite(fit.rmse_px):
                        continue
                    c = fit.camera
                    plausible = c.C[2] > 2.0 and c.C[1] < -NBA_COURT.half_width and abs(c.ptz[2]) < 10
                    if plausible and (best.camera is None or fit.rmse_px < best.rmse_px):
                        best = fit
    return best


def initial_camera(corr: Correspondences, image_size: tuple[int, int]) -> PinholeCamera | None:
    """Closed-form initialization from >= 4 floor correspondences (homography decomposition)."""
    fm = corr.floor_mask
    if fm.sum() < 4:
        return None
    try:
        hf = fit_homography(corr.world[fm, :2], corr.image[fm])
    except (ValueError, RuntimeError):
        return None
    return camera_from_homography(hf.H, image_size, court_xy=corr.world[fm, :2][hf.inliers])


def fit_camera(corr: Correspondences, image_size: tuple[int, int],
               init: PinholeCamera | None = None, fixed_C: np.ndarray | None = None,
               fixed_roll: float | None = None, k1: float = 0.0, loss: str = "huber",
               f_scale_px: float = 3.0, inlier_px: float = 8.0) -> CameraFit:
    """Fit a pinhole camera to one frame.

    * fixed_C is None  -> unconstrained 6-7 DOF camera (pan, tilt, roll, f, C).
    * fixed_C given    -> PTZ frame fit (pan, tilt, f [, roll]) with the rig center fixed.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    if init is None:
        init = initial_camera(corr, image_size)
        if init is None:
            return CameraFit(None, False, message="no initialization (need >= 4 floor points)")
    pan0, tilt0, roll0 = init.ptz
    names = ["pan", "tilt"] + ([] if fixed_roll is not None else ["roll"]) + ["f_kpx"]
    x0 = [pan0, tilt0] + ([] if fixed_roll is not None else [roll0]) + [init.f / F_UNIT]
    if fixed_C is None:
        names += ["Cx", "Cy", "Cz"]
        x0 += list(init.C)
    n_free = len(x0)
    if 2 * len(corr) < n_free:
        return CameraFit(None, False, message=f"{len(corr)} points too few for {n_free} DOF")

    def unpack(x):
        d = dict(zip(names, x))
        roll = fixed_roll if fixed_roll is not None else d["roll"]
        C = fixed_C if fixed_C is not None else np.array([d["Cx"], d["Cy"], d["Cz"]])
        return d["pan"], d["tilt"], roll, d["f_kpx"] * F_UNIT, np.asarray(C, float)

    def solve(mask):
        W = corr.world[mask]
        U = corr.image[mask]

        def resid(x):
            pan, tilt, roll, f, C = unpack(x)
            return (_project(W, pan, tilt, roll, f, C, cx, cy, k1) - U).ravel()

        return least_squares(resid, np.asarray(x0, float), loss=loss, f_scale=f_scale_px,
                             x_scale="jac", method="trf", **TOL)

    mask = np.ones(len(corr), bool)
    res = solve(mask)
    for _ in range(2):  # reject gross outliers, refit
        pan, tilt, roll, f, C = unpack(res.x)
        cam = PinholeCamera(f, cx, cy, ptz_rotation(pan, tilt, roll), C, k1, image_size)
        err = _residual_stats(cam, corr)
        new_mask = err < inlier_px
        if new_mask.sum() * 2 < n_free or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        res = solve(mask)
    pan, tilt, roll, f, C = unpack(res.x)
    cam = PinholeCamera(f, cx, cy, ptz_rotation(pan, tilt, roll), C, k1, image_size)
    err = _residual_stats(cam, corr)
    inl = err < inlier_px
    ok = bool(res.success and f > 0 and C[2] > 0 and inl.sum() * 2 >= n_free)
    rmse = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan")
    return CameraFit(cam, ok, rmse, err, inl, res.message, _covariance(res, 2 * int(mask.sum())),
                     tuple(names))


# --------------------------------------------------------------------------------------
# Joint PTZ bundle adjustment over a segment
# --------------------------------------------------------------------------------------


@dataclass
class PTZSegmentFit:
    rig: PTZRig
    states: list[PTZState | None]
    frame_rmse_px: np.ndarray  # (F,) NaN for frames without enough data
    success: bool
    message: str = ""
    C_std_m: np.ndarray | None = None  # 1-sigma of the shared center
    frame_inlier_frac: np.ndarray | None = None

    def camera(self, i: int) -> PinholeCamera | None:
        s = self.states[i]
        return None if s is None else self.rig.camera(s)


def fit_ptz_segment(frames: list[Correspondences], image_size: tuple[int, int],
                    init_C: np.ndarray | None = None, k1: float = 0.0,
                    share_roll: bool = True, loss: str = "huber",
                    f_scale_px: float = 3.0, min_points: int = 2,
                    inlier_px: float = 8.0) -> PTZSegmentFit:
    """Jointly estimate one fixed camera center (and shared roll) plus per-frame
    pan / tilt / focal over a camera segment.

    Initialization: unconstrained per-frame fits on frames with >= 4 floor points;
    C0 = per-coordinate median of their centers. Frames with fewer points are
    initialized from the nearest initialized frame.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    F = len(frames)
    inits: list[PinholeCamera | None] = []
    for corr in frames:
        fit = fit_camera(corr, image_size) if len(corr) >= 4 else None
        inits.append(fit.camera if fit is not None and fit.success else None)
    good = [i for i, c in enumerate(inits) if c is not None]
    if init_C is None:
        if not good:
            return PTZSegmentFit(PTZRig(np.zeros(3), image_size, k1), [None] * F,
                                 np.full(F, np.nan), False, "no frame could be initialized")
        init_C = np.median(np.array([inits[i].C for i in good]), axis=0)
    rolls = [inits[i].ptz[2] for i in good]
    roll0 = float(np.median(rolls)) if rolls else 0.0

    use = [i for i in range(F) if len(frames[i]) >= min_points]
    if not use:
        return PTZSegmentFit(PTZRig(init_C, image_size, k1), [None] * F, np.full(F, np.nan),
                             False, "no frame has enough points")
    st0 = []
    for i in use:
        src = inits[i]
        if src is None and good:
            src = inits[min(good, key=lambda g: abs(g - i))]
        if src is None:
            return PTZSegmentFit(PTZRig(init_C, image_size, k1), [None] * F,
                                 np.full(F, np.nan), False, "cannot initialize frame states")
        p, t, r = src.ptz
        st0.append([p, t, src.f / F_UNIT] + ([] if share_roll else [r]))
    per = 3 if share_roll else 4
    n_glob = 3 + (1 if share_roll else 0)
    x0 = np.concatenate([init_C, [roll0] if share_roll else [], np.ravel(st0)])

    obs_frame = np.concatenate([np.full(len(frames[i]), k) for k, i in enumerate(use)])
    W_all = np.vstack([frames[i].world for i in use])
    U_all = np.vstack([frames[i].image for i in use])

    n_use = len(use)

    def resid(x):
        C = x[:3]
        P = x[n_glob:].reshape(n_use, per)
        roll = np.full(n_use, x[3]) if share_roll else P[:, 3]
        R = _ptz_rotations(P[:, 0], P[:, 1], roll)[obs_frame]  # (N, 3, 3)
        Xc = np.einsum("nij,nj->ni", R, W_all - C)
        z = Xc[:, 2:3]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        xn = Xc[:, :2] / z
        if k1 != 0.0:
            xn = xn * (1.0 + k1 * np.sum(xn**2, axis=1, keepdims=True))
        uv = xn * (P[obs_frame, 2:3] * F_UNIT) + np.array([cx, cy])
        return (uv - U_all).ravel()

    # Sparsity: each observation depends on globals + its own frame block.
    S = lil_matrix((2 * len(W_all), len(x0)), dtype=int)
    for j, k in enumerate(obs_frame):
        S[2 * j: 2 * j + 2, :n_glob] = 1
        S[2 * j: 2 * j + 2, n_glob + per * k: n_glob + per * (k + 1)] = 1
    # Stage 1: robust loss to get past gross outliers. (scipy's TRF + sparse Jacobian +
    # non-linear loss does not reliably meet its termination tolerances, so this stage is
    # capped and only used as an initializer.)
    res = least_squares(resid, x0, jac_sparsity=S, loss=loss, f_scale=f_scale_px,
                        x_scale="jac", method="trf", max_nfev=100, **TOL)
    # Stage 2: iterative outlier rejection + plain least squares. The threshold starts
    # adaptive (3x the median residual) because stage 1 may stop before convergence;
    # rejecting with a fixed tight threshold on unconverged residuals discarded most
    # valid points (found by the cross-validation study). It tightens to inlier_px.
    x = res.x
    for _ in range(4):
        err = np.linalg.norm(resid(x).reshape(-1, 2), axis=1)
        thr = max(inlier_px, 3.0 * float(np.median(err)))
        inl = err < thr
        wts = np.repeat(inl.astype(float), 2)
        res = least_squares(lambda x_: resid(x_) * wts, x, jac_sparsity=S, loss="linear",
                            x_scale="jac", method="trf", max_nfev=200, **TOL)
        x = res.x
        if thr == inlier_px:
            break
    err = np.linalg.norm(resid(x).reshape(-1, 2), axis=1)
    inl = err < inlier_px
    C = res.x[:3]
    rig = PTZRig(C, image_size, k1)
    states: list[PTZState | None] = [None] * F
    rmse = np.full(F, np.nan)
    inlier_frac = np.full(F, np.nan)
    for k, i in enumerate(use):
        p = res.x[n_glob + per * k: n_glob + per * (k + 1)]
        roll = res.x[3] if share_roll else p[3]
        states[i] = PTZState(pan_deg=p[0], tilt_deg=p[1], f=p[2] * F_UNIT, roll_deg=roll)
        sel = obs_frame == k
        e = err[sel][inl[sel]]
        inlier_frac[i] = float(inl[sel].mean())
        rmse[i] = float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")
    C_std = None
    if len(x0) <= 3000:  # dense covariance is O(n^3); skip for very long segments
        try:
            J = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
            dof = max(1, 2 * int(inl.sum()) - J.shape[1])
            cov = np.linalg.pinv(J.T @ J) * (2 * res.cost / dof)
            C_std = np.sqrt(np.clip(np.diag(cov)[:3], 0, None))
        except (np.linalg.LinAlgError, MemoryError):
            pass
    return PTZSegmentFit(rig, states, rmse, bool(res.success and C[2] > 0), res.message, C_std,
                         inlier_frac)
