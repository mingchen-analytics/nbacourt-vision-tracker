"""Synthetic tracking benchmark.

The detector is simulated from ground truth with:
  * occlusion: a box covered > `occl_miss` by nearer boxes is missed; partially covered
    boxes get low scores and noisier boxes (ByteTrack's low-score stage handles these);
  * random misses and box noise;
  * false positives: bench / sideline people standing just off the court;
  * a broadcast camera that pans and zooms (calibration from simulated landmarks).

Limits: players are rigid boxes with smooth random motion and no appearance, so this
measures association logic only. It is NOT a substitute for MOT evaluation on real,
annotated footage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbatrack.calib.fit import fit_ptz_segment
from nbatrack.calib.models import CameraCalibration
from nbatrack.calib.quality import assess, emits_coordinates
from nbatrack.court.geometry import NBA_COURT
from nbatrack.eval.mot_metrics import MOTResult, evaluate
from nbatrack.project.projector import bbox_bottom_points, project_ground_points
from nbatrack.synthetic.players import player_boxes, simulate_players
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks
from nbatrack.track.tracker import Detection, Tracker, TrackerConfig

CONFIGS = {
    "bytetrack_iou": dict(use_cmc=False, use_court=False),
    "+cmc": dict(use_cmc=True, use_court=False),
    "+cmc+court": dict(use_cmc=True, use_court=True),
}


@dataclass
class SimFrame:
    gt: list[tuple[int, np.ndarray]]
    dets: list[Detection]
    M: np.ndarray | None  # estimated image motion prev -> cur (from calibrations)


def _coverage(boxes: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Fraction of each box covered by boxes nearer to the camera (approximate:
    max single-occluder coverage)."""
    n = len(boxes)
    cov = np.zeros(n)
    for i in range(n):
        a = boxes[i]
        area = (a[2] - a[0]) * (a[3] - a[1])
        for j in range(n):
            if j == i or not depth[j] < depth[i]:
                continue
            b = boxes[j]
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            if w > 0 and h > 0:
                cov[i] = max(cov[i], w * h / area)
    return cov


def simulate(n_frames: int = 300, seed: int = 0, half_court: bool = True, n_bench: int = 4,
             occl_miss: float = 0.7, rand_miss: float = 0.03, box_noise_px: float = 2.0,
             image_size=(1280, 720)) -> list[SimFrame]:
    rng = np.random.default_rng(seed)
    k = image_size[1] / 1080
    sc = BroadcastScenario(n_frames=n_frames, seed=seed, image_size=image_size,
                           f_range=(2000 * k, 3600 * k),
                           target_x_range=(4.0, 11.0) if half_court else (-11.0, 11.0))
    cams = generate_sequence(sc)
    sim = simulate_players(n_frames, sc.fps, rng)
    pos = sim.pos.copy()
    if half_court:  # crowd everyone into one half -> more crossings / occlusions
        pos[..., 0] = 7.0 + 0.45 * pos[..., 0]
    # off-court people (bench / staff) near the camera-side sideline, nearly static
    bench = np.stack([rng.uniform(-6, 12, n_bench), -NBA_COURT.half_width - rng.uniform(1.0, 2.5, n_bench)], 1)
    obs = [observe_landmarks(c.camera, rng, 1.0, 0.3, 0.05)[0] for c in cams]
    seg = fit_ptz_segment(obs, image_size)
    cals = []
    for i, corr in enumerate(obs):
        cam = seg.camera(i)
        err = np.linalg.norm(cam.project(corr.world)[0] - corr.image, axis=1)
        inl = err < 8
        q = assess("ptz", int(inl.sum()), float(seg.frame_rmse_px[i]), corr.image[inl], image_size, cam)
        cals.append((CameraCalibration(cam, "ptz"), q))
    frames = []
    H_prev = None
    for t, c in enumerate(cams):
        allxy = np.vstack([pos[t], bench + rng.normal(0, 0.02, bench.shape)])
        heights = np.concatenate([sim.height_m, np.full(n_bench, 1.8)])
        boxes, depth = player_boxes(c.camera, allxy, heights)
        w, h = image_size
        vis = (np.all(np.isfinite(boxes), 1) & (boxes[:, 0] > 0) & (boxes[:, 2] < w)
               & (boxes[:, 1] > 0) & (boxes[:, 3] < h))
        cov = _coverage(np.where(vis[:, None], boxes, -1e6), np.where(vis, depth, np.inf))
        gt = [(i, boxes[i]) for i in range(len(pos[t])) if vis[i]]  # bench people are not GT
        model, q = cals[t]
        dets = []
        for i in np.flatnonzero(vis):
            if cov[i] > occl_miss or rng.random() < rand_miss:
                continue
            partial = cov[i] > 0.3
            noise = box_noise_px * (3 if partial else 1)
            b = boxes[i] + rng.normal(0, noise, 4)
            score = rng.uniform(0.2, 0.5) if partial else rng.uniform(0.6, 0.95)
            d = Detection(b, float(score))
            if emits_coordinates(q.quality):
                cp = project_ground_points(model, bbox_bottom_points(b[None]), q.rmse_corrected_px)
                if np.all(np.isfinite(cp.xy[0])):
                    d.court_xy, d.court_sigma_m = cp.xy[0], float(cp.sigma_m[0])
            dets.append(d)
        H = model.floor_homography()
        M = None if H_prev is None else H @ np.linalg.inv(H_prev)
        H_prev = H
        frames.append(SimFrame(gt, dets, M))
    return frames


def run_tracker(frames: list[SimFrame], cfg: TrackerConfig, court_filter_m: float | None = None,
                fps: float = 30.0) -> MOTResult:
    tr = Tracker(cfg, fps)
    gt_d, pred_d = {}, {}
    for f, fr in enumerate(frames):
        dets = fr.dets
        if court_filter_m is not None:  # drop detections whose ground point is off the court
            hl, hw = NBA_COURT.half_length + court_filter_m, NBA_COURT.half_width + court_filter_m
            dets = [d for d in dets if d.court_xy is None
                    or (abs(d.court_xy[0]) <= hl and abs(d.court_xy[1]) <= hw)]
        gt_d[f] = fr.gt
        pred_d[f] = [(a.track_id, dets[a.det_index].bbox) for a in tr.update(dets, fr.M)]
    return evaluate(gt_d, pred_d)
