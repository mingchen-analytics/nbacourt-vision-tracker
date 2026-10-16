"""Empirical comparison of calibration models on synthetic sequences.

Models compared (all fed the same simulated landmark detections):
  * homography      : per-frame planar homography (floor landmarks only, needs >= 4)
  * camera_free     : per-frame unconstrained pinhole camera (center free)
  * ptz_segment     : joint fixed-center PTZ bundle adjustment over the segment

Metrics are computed against exact synthetic ground truth that is INDEPENDENT of the
estimated model (true world coordinates / true camera), so meter errors are meaningful
here -- but only for the simulated conditions. They say nothing about real footage.

  heldout_reproj_px : landmarks withheld from fitting, reprojected (floor only, so all
                      models are comparable)
  court_err_m       : random floor points (e.g. player feet) at their TRUE pixel location,
                      back-projected by the estimated model, vs true world position
  rim_err_px        : 3D rim center reprojection (3D models only)
  static_jitter_m   : for static floor points, the std over frames of their position error
                      (truth: zero motion), median over points
  fail_rate         : frames for which the model could not be estimated
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from nbatrack.calib.fit import fit_camera, fit_ptz_segment
from nbatrack.calib.homography import fit_homography
from nbatrack.calib.models import CalibrationModel, CameraCalibration, Correspondences, HomographyCalibration
from nbatrack.court.geometry import hoop_positions
from nbatrack.synthetic.scene import (
    BroadcastScenario,
    generate_sequence,
    observe_landmarks,
    sample_players,
    visible,
)

MODELS = ("homography", "camera_free", "ptz_segment")


@dataclass
class ModelStats:
    model: str
    frames: int
    fail_rate: float
    heldout_reproj_px_median: float
    heldout_reproj_px_p95: float
    court_err_m_median: float
    court_err_m_p95: float
    rim_err_px_median: float
    static_jitter_m: float


def _split(obs: Correspondences, gt: Correspondences, rng, holdout_frac: float):
    """Split by landmark into fit / held-out sets, never holding out so many that a
    frame drops below 4 fit points if it had them."""
    n = len(obs)
    k = int(np.floor(holdout_frac * n))
    k = min(k, max(0, n - 4))
    idx = rng.permutation(n)
    ho = np.zeros(n, bool)
    ho[idx[:k]] = True
    return obs.subset(~ho), gt.subset(ho & gt.floor_mask)


def run_scenario(sc: BroadcastScenario, noise_px: float = 1.0, dropout: float = 0.2,
                 outlier_frac: float = 0.0, holdout_frac: float = 0.25,
                 n_probe_points: int = 30, seed: int = 0) -> list[ModelStats]:
    rng = np.random.default_rng(seed)
    frames = generate_sequence(sc)
    fit_sets, holdouts = [], []
    for fr in frames:
        obs, gt = observe_landmarks(fr.camera, rng, noise_px, dropout, outlier_frac)
        a, b = _split(obs, gt, rng, holdout_frac)
        fit_sets.append(a)
        holdouts.append(b)
    probes = sample_players(rng, n_probe_points)
    rim = np.vstack(list(hoop_positions().values()))

    est: dict[str, list[CalibrationModel | None]] = {m: [] for m in MODELS}
    for corr in fit_sets:
        fm = corr.floor_mask
        try:
            hf = fit_homography(corr.world[fm, :2], corr.image[fm]) if fm.sum() >= 4 else None
        except (ValueError, RuntimeError):
            hf = None
        est["homography"].append(None if hf is None else HomographyCalibration(hf.H))
        cf = fit_camera(corr, sc.image_size) if len(corr) >= 4 else None
        est["camera_free"].append(CameraCalibration(cf.camera, "camera_free")
                                  if cf is not None and cf.success else None)
    seg = fit_ptz_segment(fit_sets, sc.image_size)
    for i in range(len(frames)):
        cam = seg.camera(i) if seg.success else None
        est["ptz_segment"].append(None if cam is None else CameraCalibration(cam, "ptz"))

    out = []
    for m in MODELS:
        ho_err, court_err, rim_err = [], [], []
        # per-frame error vectors of each (static) probe point; NaN when not visible
        probe_err = np.full((len(frames), len(probes), 2), np.nan)
        fails = 0
        for fi, (fr, model, ho) in enumerate(zip(frames, est[m], holdouts)):
            if model is None:
                fails += 1
                continue
            if len(ho):
                ho_err.extend(np.linalg.norm(model.court_to_image(ho.world) - ho.image, axis=1))
            uv, vis = visible(fr.camera, probes)
            if vis.any():
                xy = model.image_to_court(uv[vis])
                court_err.extend(np.linalg.norm(xy - probes[vis], axis=1))
                probe_err[fi, vis] = xy - probes[vis]
            if model.supports_3d:
                ruv, rvis = visible(fr.camera, rim)
                if rvis.any():
                    rim_err.extend(np.linalg.norm(model.court_to_image(rim[rvis]) - ruv[rvis], axis=1))

        def q(a, p):
            a = np.asarray(a, float)
            a = a[np.isfinite(a)]
            return float(np.percentile(a, p)) if len(a) else float("nan")

        # temporal jitter: frame-to-frame variation of the error of a static point
        jit = []
        for j in range(len(probes)):
            e = probe_err[:, j]
            e = e[np.all(np.isfinite(e), 1)]
            if len(e) >= 5:
                jit.append(float(np.sqrt(np.sum(np.var(e, axis=0)))))
        jitter = float(np.median(jit)) if jit else float("nan")
        out.append(ModelStats(m, len(frames), fails / len(frames), q(ho_err, 50), q(ho_err, 95),
                              q(court_err, 50), q(court_err, 95), q(rim_err, 50), jitter))
    return out


def stats_to_rows(scenario: str, stats: list[ModelStats]) -> list[dict]:
    return [{"scenario": scenario, **asdict(s)} for s in stats]
