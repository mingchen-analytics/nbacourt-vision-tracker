"""Render diagnostic videos from exported tables (separate pass from analysis).

Outputs:
  annotated.mp4         source frames + boxes, track IDs, team colors, projected court lines
  side_by_side.mp4      annotated frame | top-down court with filtered positions (2-sigma rings)
Both use the source PTS, so they stay synchronized with the original video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nbatrack.calib.homography import apply_h
from nbatrack.court.render import TopDownCanvas, draw_projected_court
from nbatrack.io.video import iter_frames
from nbatrack.viz.side_by_side import QUALITY_COLORS, VideoSink, compose

COLORS = {"A": (40, 40, 230), "B": (230, 200, 40), "REF": (40, 40, 40), None: (170, 170, 170)}


def _color(row) -> tuple:
    if row.role == "referee":
        return COLORS["REF"]
    return COLORS.get(row.team_id if isinstance(row.team_id, str) else None, COLORS[None])


def render_videos(video_path: str | Path, run_dir: str | Path, max_frames: int | None = None,
                  trail_s: float = 1.0, retries: int = 1) -> list[Path]:
    """Render annotated + side-by-side videos. An output that fails decode verification
    (rare, intermittent zeroed NAL sizes on this machine; not reproducible in isolation) is
    re-rendered up to `retries` times; a remaining failure keeps its .VERIFY_FAILED marker."""
    for attempt in range(retries + 1):
        paths, errors = _render_once(video_path, run_dir, max_frames, trail_s)
        if not any(errors):
            break
        if attempt < retries:
            print(f"[render] decode verification failed {errors}; re-rendering")
    return paths


def _render_once(video_path, run_dir, max_frames, trail_s) -> tuple[list[Path], list[int]]:
    run = Path(run_dir)
    tables = run / "tables"
    pl = pd.read_parquet(tables / "player_observations.parquet")
    cal = pd.read_parquet(tables / "calibration.parquet").set_index("frame_index")
    ball = pd.read_parquet(tables / "ball_observations.parquet")
    by_frame = dict(tuple(pl.groupby("frame_index")))
    balls = dict(tuple(ball.groupby("frame_index"))) if len(ball) else {}
    canvas = TopDownCanvas(px_per_m=24)
    bg = canvas.background()
    fps = float(pd.read_parquet(tables / "videos.parquet").fps_nominal.iloc[0])
    sink_a = VideoSink(run / "annotated.mp4", fps)
    sink_s = VideoSink(run / "side_by_side.mp4", fps)
    hist: list[pd.DataFrame] = []
    for fr in iter_frames(video_path):
        fi = fr.frame_index
        if max_frames is not None and fi >= max_frames:
            break
        img = fr.image.copy()
        q = cal.quality.get(fi, "failed")
        H = cal.H_court_to_img.get(fi)
        if H is not None and q in ("good", "fair"):
            Hm = np.asarray(H, float).reshape(3, 3)
            draw_projected_court(img, lambda p: apply_h(Hm, p), (0, 255, 255), 1)
        rows = by_frame.get(fi)
        if rows is not None:
            for r in rows.itertuples():
                if not np.isfinite(r.bbox_x1):  # interpolated row: no image observation
                    continue
                col = _color(r)
                x1, y1, x2, y2 = int(r.bbox_x1), int(r.bbox_y1), int(r.bbox_x2), int(r.bbox_y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                label = r.track_id.split("_t")[-1] + ("R" if r.role == "referee" else "")
                cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(img, (int(r.image_x), int(r.image_y)), 3, (0, 255, 0), -1)
        bf = balls.get(fi)
        ORANGE = (0, 140, 255)
        ball_tr = ball[(ball.frame_index <= fi) & (ball.frame_index > fi - int(0.5 * fps))] if len(ball) else ball
        if len(ball_tr) >= 2:  # 0.5 s image trail
            cv2.polylines(img, [ball_tr[["image_x", "image_y"]].to_numpy(float).round().astype(np.int32)], False,
                          ORANGE, 2, cv2.LINE_AA)
        ball_state = "ball: not found"
        if bf is not None:
            b = bf.iloc[0]
            c = (int(b.image_x), int(b.image_y))
            cv2.circle(img, c, 11, ORANGE, -1 if b.image_source == "detected" else 2, cv2.LINE_AA)
            cv2.circle(img, c, 12, (255, 255, 255), 1, cv2.LINE_AA)
            ball_state = ("ball: held by " + str(b.holder_track_id).split("_t")[-1].split("_")[0]
                          if b.position_3d_status == "possession_proxy" else "ball: in air / 3D unknown")
            if b.image_source == "interpolated":
                ball_state += " (interp.)"
        cv2.putText(img, ball_state, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, ball_state, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)
        txt = f"t={fr.timestamp_sec:7.3f}s f={fi} calib={q}"
        cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, QUALITY_COLORS.get(q, (255, 255, 255)), 2)
        sink_a.write(img, fr.timestamp_sec)

        top = bg.copy()
        if rows is not None:
            hist.append(rows)
        hist = [h for h in hist if h.timestamp_sec.iloc[0] >= fr.timestamp_sec - trail_s]
        if hist:
            trail = pd.concat(hist)
            for tid, g in trail.groupby("track_id"):
                g = g.dropna(subset=["court_x_m"]).sort_values("timestamp_sec")
                if len(g) >= 2:
                    pts = canvas.to_px(g[["court_x_m", "court_y_m"]].to_numpy(float)).round().astype(np.int32)
                    cv2.polylines(top, [pts], False, _color(g.iloc[-1]), 1, cv2.LINE_AA)
        if rows is not None:
            for r in rows.itertuples():
                if r.position_status not in ("observed_projected", "interpolated") or not np.isfinite(r.court_x_m):
                    continue
                p = canvas.to_px(np.array([[r.court_x_m, r.court_y_m]]))[0].round().astype(int)
                col = _color(r)
                rad = int(max(3, min(60, 2 * r.court_sigma_m * canvas.s)))
                cv2.circle(top, tuple(p), rad, col, 1, cv2.LINE_AA)
                filled = -1 if r.position_status == "observed_projected" else 2  # hollow = interpolated
                cv2.circle(top, tuple(p), 7, col, filled, cv2.LINE_AA)
                cv2.putText(top, r.track_id.split("_t")[-1], (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        if bf is not None and bf.iloc[0].position_3d_status == "possession_proxy":
            b = bf.iloc[0]
            p = canvas.to_px(np.array([[b.ball_x_m, b.ball_y_m]]))[0].round().astype(int) + np.array([9, 9])
            cv2.circle(top, tuple(p), 6, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(top, tuple(p), 7, (20, 20, 20), 1, cv2.LINE_AA)
        if H is not None and q in ("good", "fair"):  # camera footprint on the court
            Hi = np.linalg.inv(np.asarray(H, float).reshape(3, 3))
            h, w = img.shape[:2]
            corners = apply_h(Hi, np.array([[0, h * 0.35], [w, h * 0.35], [w, h], [0, h]], float))
            if np.all(np.isfinite(corners)):
                cv2.polylines(top, [canvas.to_px(corners).round().astype(np.int32)], True, (90, 90, 90), 1)
        cv2.putText(top, "filled: observed, hollow: interpolated, ring: 2 sigma (UNCALIBRATED), orange: ball at holder (only when held)",
                    (8, top.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        sink_s.write(compose(img, top, height=540), fr.timestamp_sec)
    sink_a.close()
    sink_s.close()
    return [Path(sink_a.path), Path(sink_s.path)], [sink_a.verify_errors, sink_s.verify_errors]
