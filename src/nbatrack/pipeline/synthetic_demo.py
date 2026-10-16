"""End-to-end run on a synthetic broadcast with exact ground truth.

What this exercises: calibration (chosen model) -> quality flags -> bbox-bottom ground
points -> court projection with sigma -> schema-validated export -> side-by-side video.

What it does NOT validate: the player detector, the tracker, team classification, or any
behavior on real footage. Detections here are ground-truth boxes with added noise and
random misses; track IDs are the simulator's true identities.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nbatrack import SCHEMA_VERSION, __version__
from nbatrack.calib.fit import fit_camera, fit_ptz_segment
from nbatrack.calib.homography import fit_homography
from nbatrack.calib.models import CalibrationModel, CameraCalibration, HomographyCalibration
from nbatrack.calib.quality import QualityReport, assess, emits_coordinates
from nbatrack.court.geometry import court_landmarks
from nbatrack.court.render import TopDownCanvas
from nbatrack.export.writers import write_rows
from nbatrack.project.projector import bbox_bottom_points, project_ground_points
from nbatrack.synthetic.players import draw_players, player_boxes, simulate_players
from nbatrack.synthetic.render import SyntheticArena
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks
from nbatrack.viz.side_by_side import VideoSink, compose, draw_frame_overlay, draw_topdown


def _calibrate(mode: str, obs_list, image_size) -> list[tuple[CalibrationModel | None, QualityReport, dict]]:
    out = []
    seg = fit_ptz_segment(obs_list, image_size) if mode == "ptz" else None
    for i, corr in enumerate(obs_list):
        info = dict(n_landmarks=len(corr))
        if mode == "homography":
            fm = corr.floor_mask
            try:
                hf = fit_homography(corr.world[fm, :2], corr.image[fm])
            except (ValueError, RuntimeError):
                out.append((None, QualityReport("failed", 0.0, ["fit failed"]), info))
                continue
            q = assess("homography", hf.n_inliers, hf.rmse_px, corr.image[fm][hf.inliers], image_size)
            info.update(n_inliers=hf.n_inliers, rmse=hf.rmse_px)
            out.append((HomographyCalibration(hf.H), q, info))
        elif mode == "camera":
            cf = fit_camera(corr, image_size) if len(corr) >= 4 else None
            if cf is None or not cf.success:
                out.append((None, QualityReport("failed", 0.0, ["fit failed"]), info))
                continue
            q = assess("camera", int(cf.inliers.sum()), cf.rmse_px, corr.image[cf.inliers],
                       image_size, cf.camera)
            info.update(n_inliers=int(cf.inliers.sum()), rmse=cf.rmse_px)
            out.append((CameraCalibration(cf.camera, "camera_free"), q, info))
        elif mode == "ptz":
            cam = seg.camera(i) if seg is not None and seg.success else None
            if cam is None:
                out.append((None, QualityReport("failed", 0.0, ["segment fit failed"]), info))
                continue
            err = np.linalg.norm(cam.project(corr.world)[0] - corr.image, axis=1)
            inl = err < 8.0
            rmse = float(seg.frame_rmse_px[i])
            q = assess("ptz", int(inl.sum()), rmse, corr.image[inl], image_size, cam)
            info.update(n_inliers=int(inl.sum()), rmse=rmse)
            out.append((CameraCalibration(cam, "ptz"), q, info))
        else:
            raise ValueError(mode)
    return out


def _calib_row(video_id, fi, ts, model, q, info, mode) -> dict:
    r = dict(video_id=video_id, frame_index=fi, timestamp_sec=ts, camera_segment_id=0,
             method="keypoint" if model is not None else "none",
             model=("camera_free" if mode == "camera" else mode) if model is not None else "none",
             n_landmarks=info.get("n_landmarks"), n_inliers=info.get("n_inliers"),
             reproj_rmse_px=info.get("rmse"), quality=q.quality, quality_score=q.score)
    if model is not None:
        H = model.floor_homography()
        r["H_court_to_img"] = (H / H[2, 2]).ravel()
        Hi = np.linalg.inv(H)
        r["H_img_to_court"] = (Hi / Hi[2, 2]).ravel()
        if isinstance(model, CameraCalibration):
            c = model.camera
            pan, tilt, roll = c.ptz
            r.update(focal_px=c.f, cx=c.cx, cy=c.cy, k1=c.k1, pan_deg=pan, tilt_deg=tilt,
                     roll_deg=roll, cam_x_m=c.C[0], cam_y_m=c.C[1], cam_z_m=c.C[2])
    return r


def run_synthetic_demo(out_dir: str | Path, n_frames: int = 150, calib: str = "ptz",
                       seed: int = 0, landmark_noise_px: float = 1.5, dropout: float = 0.4,
                       outlier_frac: float = 0.05, box_noise_px: float = 2.0,
                       miss_rate: float = 0.05, render_video: bool = True,
                       image_size: tuple[int, int] = (1280, 720)) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    k = image_size[1] / 1080.0  # same fields of view as the 1080p scenarios
    sc = BroadcastScenario(n_frames=n_frames, seed=seed, image_size=image_size,
                           f_range=(1700 * k, 3200 * k))
    frames = generate_sequence(sc)
    sim = simulate_players(n_frames, sc.fps, rng)
    video_id = f"synthetic_seed{seed}"
    obs_list = [observe_landmarks(fr.camera, rng, landmark_noise_px, dropout, outlier_frac)[0]
                for fr in frames]
    t_cal = time.time()
    calibs = _calibrate(calib, obs_list, sc.image_size)
    t_cal = time.time() - t_cal

    arena = SyntheticArena(seed) if render_video else None
    canvas = TopDownCanvas(px_per_m=22)
    bg = canvas.background()
    sink_in = VideoSink(out / "synthetic_input.mp4", sc.fps) if render_video else None
    sink_viz = VideoSink(out / "diagnostic_side_by_side.mp4", sc.fps) if render_video else None

    rows_frames, rows_cal, rows_lm, rows_pl = [], [], [], []
    errs = []  # (quality, err_m, sigma_m)
    n_visible = n_valid = 0
    for fr, corr, (model, q, info) in zip(frames, obs_list, calibs):
        fi, ts = fr.index, fr.timestamp_sec
        rows_frames.append(dict(video_id=video_id, frame_index=fi, timestamp_sec=ts,
                                pts=int(round(ts * 90000)), pts_sec=ts, camera_segment_id=0))
        rows_cal.append(_calib_row(video_id, fi, ts, model, q, info, calib))
        for lid, (u, v) in zip(corr.ids, corr.image):
            rows_lm.append(dict(video_id=video_id, frame_index=fi, timestamp_sec=ts,
                                landmark_id=int(lid), image_x=u, image_y=v, source="synthetic"))
        xy_true = sim.pos[fi]
        boxes, depth = player_boxes(fr.camera, xy_true, sim.height_m)
        w, h = sc.image_size
        in_img = (np.all(np.isfinite(boxes), 1) & (boxes[:, 2] > 0) & (boxes[:, 0] < w)
                  & (boxes[:, 3] > 0) & (boxes[:, 3] < h))
        det = in_img & (rng.random(len(boxes)) >= miss_rate)
        noisy = boxes + rng.normal(0, box_noise_px, boxes.shape)
        frame_players = []
        if det.any():
            gp = bbox_bottom_points(noisy[det])
            ok_model = model is not None and q.quality != "failed"
            if ok_model:
                cp = project_ground_points(model, gp, calib_rmse_px=q.rmse_corrected_px)
            for k, i in enumerate(np.flatnonzero(det)):
                team = sim.team[i]
                emitted = ok_model and emits_coordinates(q.quality) and np.all(np.isfinite(cp.xy[k]))
                r = dict(video_id=video_id, frame_index=fi, timestamp_sec=ts, camera_segment_id=0,
                         track_id=f"s000_t{i:03d}", team_id=None if team == "REF" else team,
                         role="referee" if team == "REF" else "player",
                         bbox_x1=noisy[i, 0], bbox_y1=noisy[i, 1], bbox_x2=noisy[i, 2],
                         bbox_y2=noisy[i, 3], det_confidence=1.0, ground_method="bbox_bottom",
                         image_x=gp.uv[k, 0], image_y=gp.uv[k, 1],
                         court_x_m_raw=cp.xy[k, 0] if ok_model else None,
                         court_y_m_raw=cp.xy[k, 1] if ok_model else None,
                         court_x_m=cp.xy[k, 0] if emitted else None,
                         court_y_m=cp.xy[k, 1] if emitted else None,
                         court_sigma_m=cp.sigma_m[k] if emitted else None,
                         position_status="observed_projected" if emitted else "unknown",
                         flags="synthetic_gt_identity",
                         calibration_quality=q.quality, is_visible=True,
                         is_on_court=bool(cp.on_court[k]) if ok_model else None)
                rows_pl.append(r)
                frame_players.append(r)
                if team != "REF":
                    n_visible += 1
                    if emitted:
                        n_valid += 1
                if emitted:
                    errs.append((q.quality, float(np.linalg.norm(cp.xy[k] - xy_true[i])),
                                 float(cp.sigma_m[k])))
        if render_video:
            img = arena.render(fr.camera)
            draw_players(img, boxes, depth, sim.team)
            sink_in.write(img, ts)
            left = draw_frame_overlay(img, frame_players, model, q.quality, ts, fi)
            right = draw_topdown(canvas, bg, [p for p in frame_players if p["role"] == "player"],
                                 truth_xy=xy_true[np.array(sim.team) != "REF"])
            sink_viz.write(compose(left, right), ts)
    if render_video:
        sink_in.close()
        sink_viz.close()

    tables = out / "tables"
    write_rows("videos", [dict(video_id=video_id, source_path="synthetic", width=sc.image_size[0],
                               height=sc.image_size[1], fps_nominal=sc.fps, time_base="1/90000",
                               start_sec=0.0, duration_sec=n_frames / sc.fps, n_frames=n_frames,
                               codec="synthetic", pipeline_version=__version__,
                               created_at=datetime.now(timezone.utc).isoformat())], tables)
    write_rows("frames", rows_frames, tables)
    write_rows("camera_segments", [dict(video_id=video_id, camera_segment_id=0, start_frame=0,
                                        end_frame=n_frames - 1, start_sec=0.0,
                                        end_sec=frames[-1].timestamp_sec, view_type="main_broadcast",
                                        supported=True, boundary_method="synthetic")], tables)
    write_rows("calibration", rows_cal, tables)
    write_rows("court_landmarks", [dict(landmark_id=lm.landmark_id, name=lm.name, court_x_m=lm.xyz[0],
                                        court_y_m=lm.xyz[1], court_z_m=lm.xyz[2],
                                        description=lm.description) for lm in court_landmarks()], tables)
    write_rows("landmark_observations", rows_lm, tables, ("parquet",))
    write_rows("player_observations", rows_pl, tables)

    e = np.array([x[1] for x in errs]) if errs else np.array([np.nan])
    s = np.array([x[2] for x in errs]) if errs else np.array([np.nan])
    qual = [c[1].quality for c in calibs]
    metrics = {
        "NOTE": "SYNTHETIC data with exact ground truth; detections are noisy GT boxes. "
                "Not evidence of real-footage accuracy.",
        "schema_version": SCHEMA_VERSION, "calibration_model": calib, "n_frames": n_frames,
        "frame_quality_counts": {k: qual.count(k) for k in ("good", "fair", "poor", "failed")},
        "player_court_err_m": {"median": float(np.nanmedian(e)), "p95": float(np.nanpercentile(e, 95)),
                               "max": float(np.nanmax(e)), "frac_gt_0_5m": float(np.nanmean(e > 0.5))},
        "coverage_2sigma": float(np.nanmean(e <= 2 * s)),
        "visible_players_with_valid_coords": n_valid / max(1, n_visible),
        "calibration_seconds": t_cal, "total_seconds": time.time() - t_start,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
