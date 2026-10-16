"""Compute measured diagnostics for a real-video run directory and write report files.

Everything here is MEASURED from the run outputs; none of it is ground-truth accuracy:
  * calibration: self-consistency (fraction of projected court template on markings),
    motion-estimation success, camera parameter ranges, optional reprojection error of
    independently annotated landmarks on a later frame;
  * detection: counts only (no precision/recall without annotation);
  * tracking: proxies (on-court IDs, interior births, track lengths, flagged switches);
  * coordinates: raw-vs-smoothed residual, speed distribution, speed violations,
    coverage of tracked on-court players with valid coordinates;
  * runtime and memory from run_analyze.json.

Usage: python tools/real_video_report.py outputs/real/clip2_ptz [--check data/annotations/clip2/check_0300.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbatrack.calib.manual import ManualCalibrationSession  # noqa: E402
from nbatrack.pipeline.video_pipeline import camera_from_row  # noqa: E402


def report(run_dir: Path, check_json: list[Path]) -> dict:
    t = run_dir / "tables"
    ra = json.loads((run_dir / "run_analyze.json").read_text(encoding="utf-8"))
    rt = json.loads((run_dir / "run_track.json").read_text(encoding="utf-8"))
    cal = pd.read_parquet(t / "calibration.parquet")
    diag = pd.read_csv(run_dir / "calibration_diagnostics.csv")
    det = pd.read_parquet(run_dir / "cache" / "detections.parquet")
    pl = pd.read_parquet(t / "player_observations.parquet")
    ball = pd.read_parquet(t / "ball_observations.parquet")
    n = ra["n_frames"]
    size = (ra["width"], ra["height"])
    out: dict = {"video_id": ra["video_id"], "n_frames": n, "fps": ra["fps"],
                 "duration_s": n / ra["fps"]}

    # ---- calibration (self-consistency)
    out["calibration"] = {
        "mode": ra["config"]["calib_mode"],
        "quality_counts": cal.quality.value_counts().to_dict(),
        "alignment_inlier_frac": {"median": float(diag.inlier.median()), "p05": float(diag.inlier.quantile(0.05)),
                                  "min": float(diag.inlier.min())},
        "template_points_median": float(diag.n.median()),
        "motion_estimation_success": float(diag.motion_ok.mean()),
        "pan_deg_range": [float(diag.pan.min()), float(diag.pan.max())],
        "tilt_deg_range": [float(diag.tilt.min()), float(diag.tilt.max())],
        "focal_px_range": [float(diag.f.min()), float(diag.f.max())],
        "camera_center_m_first": [float(diag.Cx.iloc[0]), float(diag.Cy.iloc[0]), float(diag.Cz.iloc[0])],
        "camera_center_m_spread_std": [float(diag.Cx.std()), float(diag.Cy.std()), float(diag.Cz.std())],
        "frame_to_frame_param_jumps_p95": {
            "pan_deg": float(diag.pan.diff().abs().quantile(0.95)),
            "tilt_deg": float(diag.tilt.diff().abs().quantile(0.95)),
            "focal_rel": float((diag.f.pct_change().abs()).quantile(0.95)),
        },
    }
    checks = []
    for cj in check_json:
        s = ManualCalibrationSession.load(cj)
        row = cal[cal.frame_index == s.frame_index].iloc[0]
        cam = camera_from_row(row, size)
        c = s.correspondences()
        e = np.linalg.norm(cam.project(c.world)[0] - c.image, axis=1)
        checks.append({"frame": s.frame_index, "annotator": s.annotator, "n_landmarks": len(c),
                       "reproj_err_px": [round(float(x), 1) for x in e],
                       "median_px": float(np.median(e)), "max_px": float(e.max())})
    out["calibration"]["independent_landmark_checks"] = checks

    # ---- detection (counts only)
    per = det[det.cls == 1].groupby("frame_index")
    out["detection"] = {
        "persons_score_ge_0.5_per_frame_median": float(per.apply(lambda g: (g.score >= 0.5).sum()).median()),
        "persons_all_scores_per_frame_median": float(per.size().median()),
        "ball_detected_frames_frac(score>=0.3)": float(
            det[(det.cls == 37) & (det.score >= 0.3)].frame_index.nunique() / n),
    }
    # ---- tracking proxies
    onc_all = pl[pl.role.isin(["player", "referee", "unknown"])]
    onc = onc_all[onc_all.bbox_x1.notna()]  # observed rows
    first = onc.sort_values("frame_index").groupby("track_id").first()
    w = ra["width"]
    interior = first[(first.frame_index > 5) & (first.bbox_x1 > 30) & (first.bbox_x2 < w - 30)]
    lens = onc.groupby("track_id").size()
    out["tracking"] = {
        "track_ids_total": int(pl.track_id.nunique()),
        "on_court_track_ids": int(onc.track_id.nunique()),
        "roles_track_count": pl.groupby("role").track_id.nunique().to_dict(),
        "interior_births_after_frame5": int(len(interior)),
        "track_length_frames_median": float(lens.median()),
        "on_court_tracked_per_frame_median": float(onc.groupby("frame_index").size().median()),
        "suspected_cross_class_switches": rt.get("suspected_cross_class_switches"),
        "stitch_enabled": rt.get("stitch_enabled"), "stitch_links": len(rt.get("stitch_links") or []),
        "interpolated_rows": int((onc_all.position_status == "interpolated").sum()),
        "rows_flagged_suspected_id_switch": rt.get("rows_flagged_suspected_id_switch"),
    }
    # ---- coordinates
    ok = onc[onc.position_status == "observed_projected"].dropna(subset=["court_x_m"])
    resid = np.hypot(ok.court_x_m_raw - ok.court_x_m, ok.court_y_m_raw - ok.court_y_m)
    speeds = []
    for _, g in ok.sort_values("timestamp_sec").groupby("track_id"):
        if len(g) > 2:
            dt = np.diff(g.timestamp_sec.to_numpy())
            dd = np.hypot(np.diff(g.court_x_m.to_numpy()), np.diff(g.court_y_m.to_numpy()))
            speeds.extend(dd[dt > 0] / dt[dt > 0])
    speeds = np.asarray(speeds)
    out["coordinates"] = {
        "on_court_rows": int(len(onc)),
        "with_valid_court_xy_frac": float(len(ok) / max(1, len(onc))),
        "court_sigma_m_median": float(ok.court_sigma_m.median()),
        "raw_minus_smoothed_m": {"median": float(resid.median()), "p95": float(resid.quantile(0.95))},
        "smoothed_speed_mps": {"median": float(np.median(speeds)) if len(speeds) else None,
                               "p95": float(np.percentile(speeds, 95)) if len(speeds) else None,
                               "max": float(speeds.max()) if len(speeds) else None},
        "raw_speed_violation_rows": int(onc["flags"].fillna("").str.contains("speed_violation").sum()),
        "positions_outside_court_frac(margin 0.5m)": float(
            ((ok.court_x_m.abs() > 14.33 + 0.5) | (ok.court_y_m.abs() > 7.62 + 0.5)).mean()),
    }
    out["ball"] = {"rows": int(len(ball)), "position_3d": "unknown for all rows (not estimated)"}
    out["runtime"] = {"processing_fps_stage_A": ra["processing_fps"], "ms_per_frame": ra["ms_per_frame"],
                      "peak_private_memory_mb": ra["peak_private_memory_mb"],
                      "detector_provider": ra["config"]["provider"], "stage_B_seconds": rt["seconds"]}

    # ---- plots
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(diag.t, diag.inlier)
    ax[0].axhline(0.75, ls="--", c="g")
    ax[0].axhline(0.55, ls="--", c="orange")
    ax[0].set_ylabel("alignment inlier frac")
    ax[1].plot(diag.t, diag.pan, label="pan")
    ax[1].plot(diag.t, diag.tilt, label="tilt")
    ax[1].legend()
    ax[1].set_ylabel("deg")
    ax[2].plot(diag.t, diag.f)
    ax[2].set_ylabel("focal px")
    ax[3].plot(per.size().index / ra["fps"], per.size().values, label="persons (all)")
    ax[3].plot(onc.groupby("frame_index").size().index / ra["fps"], onc.groupby("frame_index").size().values,
               label="tracked on-court")
    ax[3].legend()
    ax[3].set_xlabel("video time (s)")
    fig.suptitle(f"{ra['video_id']} - calibration and detection diagnostics (self-consistency, not GT)")
    fig.tight_layout()
    fig.savefig(run_dir / "diagnostics.png", dpi=110)
    plt.close(fig)
    (run_dir / "report_metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()
    r = report(Path(a.run_dir), [Path(c) for c in a.check])
    print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
