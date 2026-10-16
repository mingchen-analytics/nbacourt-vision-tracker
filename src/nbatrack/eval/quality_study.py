"""Synthetic study: do quality flags and court_sigma_m track TRUE error?

For each synthetic frame and each calibration model we record the assigned quality flag
and, for random floor probe points, the true back-projection error plus the predicted
sigma. Ground truth comes from the simulator, independent of the estimated model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbatrack.calib.fit import fit_camera, fit_ptz_segment
from nbatrack.calib.homography import fit_homography
from nbatrack.calib.models import CameraCalibration, HomographyCalibration
from nbatrack.calib.quality import assess
from nbatrack.project.projector import GroundPoints, project_ground_points
from nbatrack.synthetic.scene import (
    BroadcastScenario,
    generate_sequence,
    observe_landmarks,
    sample_players,
    visible,
)


def run(sc: BroadcastScenario, noise_px: float = 1.0, dropout: float = 0.3,
        outlier_frac: float = 0.05, foot_noise_px: float = 0.0, n_probe: int = 20,
        seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = generate_sequence(sc)
    obs = [observe_landmarks(fr.camera, rng, noise_px, dropout, outlier_frac)[0] for fr in frames]
    seg = fit_ptz_segment(obs, sc.image_size)
    rows = []
    for i, (fr, corr) in enumerate(zip(frames, obs)):
        probes = sample_players(rng, n_probe)
        uv_true, vis = visible(fr.camera, probes)
        if not vis.any():
            continue
        uv_obs = uv_true[vis] + rng.normal(0, foot_noise_px, (int(vis.sum()), 2))
        gp = GroundPoints(uv_obs, np.tile(np.eye(2) * max(foot_noise_px, 1e-6) ** 2,
                                          (len(uv_obs), 1, 1)), "synthetic")
        cands = []
        fm = corr.floor_mask
        if fm.sum() >= 4:
            try:
                hf = fit_homography(corr.world[fm, :2], corr.image[fm])
                q = assess("homography", hf.n_inliers, hf.rmse_px, corr.image[fm][hf.inliers],
                           sc.image_size)
                cands.append(("homography", HomographyCalibration(hf.H), q, hf.rmse_px))
            except (ValueError, RuntimeError):
                pass
        if len(corr) >= 4:
            cf = fit_camera(corr, sc.image_size)
            if cf.success:
                q = assess("camera", int(cf.inliers.sum()), cf.rmse_px,
                           corr.image[cf.inliers], sc.image_size, cf.camera)
                cands.append(("camera_free", CameraCalibration(cf.camera), q, cf.rmse_px))
        cam = seg.camera(i) if seg.success else None
        if cam is not None:
            err = np.linalg.norm(cam.project(corr.world)[0] - corr.image, axis=1)
            inl = err < 8.0
            q = assess("ptz", int(inl.sum()), float(seg.frame_rmse_px[i]), corr.image[inl],
                       sc.image_size, cam)
            cands.append(("ptz_segment", CameraCalibration(cam, "ptz"), q,
                          float(seg.frame_rmse_px[i])))
        for name, model, q, _rmse in cands:
            cp = project_ground_points(model, gp, calib_rmse_px=q.rmse_corrected_px)
            e = np.linalg.norm(cp.xy - probes[vis], axis=1)
            for k in range(len(e)):
                rows.append(dict(frame=i, model=name, quality=q.quality, score=q.score,
                                 n_landmarks=len(corr), err_m=e[k], sigma_m=cp.sigma_m[k],
                                 probe_y=probes[vis][k, 1]))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "quality"])
    out = g["err_m"].agg(n="size", median="median", p95=lambda x: np.nanpercentile(x, 95),
                         frac_gt_0_5m=lambda x: float(np.mean(x > 0.5)))
    # Coverage: fraction of true errors within 2 * predicted sigma (major axis).
    out["cover_2sigma"] = g.apply(lambda d: float(np.mean(d.err_m <= 2 * d.sigma_m)),
                                  include_groups=False)
    return out
