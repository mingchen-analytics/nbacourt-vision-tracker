"""Data-driven choice between the fixed-center PTZ model and per-frame models.

The PTZ assumption is TESTED per camera segment by landmark cross-validation: landmarks
are randomly held out of each frame, every model is fit on the rest, and the held-out
reprojection errors are compared on the frames where all candidates could be fit.
PTZ is selected only if it is not worse than the unconstrained per-frame camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nbatrack.calib.fit import fit_camera, fit_ptz_segment
from nbatrack.calib.homography import fit_homography
from nbatrack.calib.models import Correspondences, HomographyCalibration


@dataclass
class SelectionReport:
    choice: str  # "ptz" | "camera_free" | "homography" | "none"
    heldout_median_px: dict[str, float] = field(default_factory=dict)
    heldout_p90_px: dict[str, float] = field(default_factory=dict)
    n_frames_compared: int = 0
    ptz_center: np.ndarray | None = None
    reason: str = ""


def _split(corr: Correspondences, rng: np.random.Generator, frac: float, min_fit: int):
    n = len(corr)
    k = min(int(round(frac * n)), n - min_fit)
    if k <= 0:
        return corr, None
    ho = np.zeros(n, bool)
    ho[rng.choice(n, k, replace=False)] = True
    return corr.subset(~ho), corr.subset(ho & corr.floor_mask)


def crossval_errors(frames: list[Correspondences], image_size: tuple[int, int],
                    repeats: int = 3, holdout_frac: float = 0.3, seed: int = 0) -> dict[str, list]:
    """Per-frame median held-out error per model; NaN where a model could not be fit."""
    rng = np.random.default_rng(seed)
    F = len(frames)
    errs = {m: [[] for _ in range(F)] for m in ("ptz", "camera_free", "homography")}
    for _ in range(repeats):
        splits = [_split(c, rng, holdout_frac, 4) for c in frames]
        fit_sets = [s[0] for s in splits]
        seg = fit_ptz_segment(fit_sets, image_size)
        for i, (fit, ho) in enumerate(splits):
            if ho is None or len(ho) == 0:
                continue
            if seg.success and seg.states[i] is not None:
                e = np.linalg.norm(seg.camera(i).project(ho.world)[0] - ho.image, axis=1)
                errs["ptz"][i].extend(e)
            if len(fit) >= 4:
                cf = fit_camera(fit, image_size)
                if cf.success:
                    errs["camera_free"][i].extend(np.linalg.norm(cf.camera.project(ho.world)[0] - ho.image, axis=1))
            fm = fit.floor_mask
            if fm.sum() >= 5:  # >= 5 so the homography is over-determined
                try:
                    hf = fit_homography(fit.world[fm, :2], fit.image[fm])
                    errs["homography"][i].extend(np.linalg.norm(
                        HomographyCalibration(hf.H).court_to_image(ho.world) - ho.image, axis=1))
                except (ValueError, RuntimeError):
                    pass
    return {m: [float(np.median(v)) if v else np.nan for v in per] for m, per in errs.items()}


def select_model(frames: list[Correspondences], image_size: tuple[int, int],
                 tolerance: float = 1.15, seed: int = 0) -> SelectionReport:
    cv = crossval_errors(frames, image_size, seed=seed)
    A = np.array([cv["ptz"], cv["camera_free"], cv["homography"]], float)
    common = np.all(np.isfinite(A[:2]), axis=0)  # frames where ptz and free both work
    rep = SelectionReport("none", n_frames_compared=int(common.sum()))
    for k, m in enumerate(("ptz", "camera_free", "homography")):
        v = A[k][common] if m != "homography" else A[k][np.isfinite(A[k])]
        rep.heldout_median_px[m] = float(np.median(v)) if len(v) else float("nan")
        rep.heldout_p90_px[m] = float(np.percentile(v, 90)) if len(v) else float("nan")
    if common.sum() < 5:
        ok = [m for m in ("camera_free", "homography") if np.isfinite(rep.heldout_median_px[m])]
        rep.choice = ok[0] if ok else "none"
        rep.reason = "too few frames to test the PTZ assumption"
        return rep
    mp, mf = rep.heldout_median_px["ptz"], rep.heldout_median_px["camera_free"]
    pp, pf = rep.heldout_p90_px["ptz"], rep.heldout_p90_px["camera_free"]
    if mp <= tolerance * mf and pp <= 1.5 * tolerance * pf:
        rep.choice = "ptz"
        rep.reason = f"PTZ held-out median {mp:.2f}px <= {tolerance} x free {mf:.2f}px"
        rep.ptz_center = fit_ptz_segment(frames, image_size).rig.C
    else:
        rep.choice = "camera_free"
        rep.reason = (f"fixed-center assumption rejected: PTZ held-out median {mp:.2f}px / "
                      f"p90 {pp:.2f}px vs free {mf:.2f}px / {pf:.2f}px")
    return rep
