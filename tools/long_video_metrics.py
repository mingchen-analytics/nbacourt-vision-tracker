"""Proxy metrics for a long-video run (no ground truth): calibration stability, identity
continuity, role classification, ball coverage, speeds. Optionally compares several runs of
the same video (e.g. stitching off / on).

  python tools/long_video_metrics.py outputs/real/clip6_long_115s [other_run ...] --out metrics.json

All values are PROXIES computed from the system's own output; none is an accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else None


def run_metrics(run: Path) -> dict:
    t = run / "tables"
    pl = pd.read_parquet(t / "player_observations.parquet")
    ball = pd.read_parquet(t / "ball_observations.parquet")
    fr = pd.read_parquet(t / "frames.parquet")
    segs = pd.read_parquet(t / "camera_segments.parquet")
    diag = pd.read_csv(run / "calibration_diagnostics.csv")
    fps = float(pd.read_parquet(t / "videos.parquet").fps_nominal.iloc[0])
    out = {"n_frames": int(len(fr)), "duration_s": float(fr.timestamp_sec.iloc[-1]), "segments": []}
    obs = pl[pl.bbox_x1.notna()]
    for s in segs.itertuples():
        f = fr[fr.camera_segment_id == s.camera_segment_id]
        n = len(f)
        m = {"camera_segment_id": int(s.camera_segment_id), "view_type": s.view_type, "supported": bool(s.supported),
             "frames": n, "duration_s": round(n / fps, 2)}
        if s.supported:
            d = diag[diag.segment == s.camera_segment_id]
            m["calibration_quality"] = d.quality.value_counts().to_dict()
            m["flash_frames"] = int((d.method == "held_flash").sum())
            nd = d[d.method != "held_flash"]
            m["inlier_frac_p05_p50"] = [q(nd.inlier, 5), q(nd.inlier, 50)]
            m["floor_iou_p05_p50"] = [q(nd.floor_iou, 5), q(nd.floor_iou, 50)]
            m["floor_iou_below_0.45_frames"] = int((nd.floor_iou < 0.45).sum())
            m["pan_range_deg"] = [q(nd.pan, 0), q(nd.pan, 100)]
            m["focal_range_px"] = [q(nd.f, 0), q(nd.f, 100)]
            dp = nd.pan.diff().abs()
            m["max_abs_dpan_per_frame"] = q(dp, 100)
            # players
            so = obs[obs.camera_segment_id == s.camera_segment_id]
            role_tracks = so.groupby("track_id").role.agg(lambda v: v.mode().iloc[0]).value_counts().to_dict()
            m["tracks_by_role"] = role_tracks
            life = so.groupby("track_id").agg(f0=("frame_index", "min"), f1=("frame_index", "max"),
                                              role=("role", lambda v: v.mode().iloc[0]), n=("frame_index", "size"))
            life["dur_s"] = (life.f1 - life.f0 + 1) / fps
            pt = life[life.role == "player"]
            m["player_track_duration_s_p50_p90_max"] = [q(pt.dur_s, 50), q(pt.dur_s, 90), q(pt.dur_s, 100)]
            m["player_tracks_per_minute"] = round(len(pt) / (n / fps / 60), 1)
            m["player_tracks_ge_10s"] = int((pt.dur_s >= 10).sum())
            per_f = so[so.role == "player"].groupby("frame_index").size().reindex(f.frame_index, fill_value=0)
            m["players_per_frame"] = {"p10": q(per_f, 10), "p50": q(per_f, 50), "p90": q(per_f, 90),
                                      "frac_eq_10": float((per_f == 10).mean()), "frac_gt_10": float((per_f > 10).mean()),
                                      "frac_lt_8": float((per_f < 8).mean())}
            refs = so[so.role == "referee"].groupby("frame_index").size().reindex(f.frame_index, fill_value=0)
            m["referees_per_frame_p50_max"] = [q(refs, 50), q(refs, 100)]
            fl = so["flags"].fillna("")
            m["rows_speed_violation_per_1000"] = round(1000 * fl.str.contains("speed_violation").mean(), 2)
            m["tracks_with_suspected_id_switch"] = int(so[fl.str.contains("suspected_id_switch")].track_id.nunique())
            sp = pl[(pl.camera_segment_id == s.camera_segment_id) & (pl.role == "player") & pl.court_x_m.notna()] \
                .sort_values(["track_id", "frame_index"])
            v = []
            for _, g in sp.groupby("track_id"):
                if len(g) > 2:
                    dxy = np.hypot(np.diff(g.court_x_m), np.diff(g.court_y_m))
                    dt = np.diff(g.timestamp_sec)
                    v.append(dxy[dt > 0] / dt[dt > 0])
            v = np.concatenate(v) if v else np.zeros(0)
            m["player_speed_smoothed_m_s_p50_p95_p99_max"] = [q(v, 50), q(v, 95), q(v, 99), q(v, 100)]
            m["interpolated_player_rows"] = int((pl[(pl.camera_segment_id == s.camera_segment_id)].position_status == "interpolated").sum())
            # ball
            b = ball[ball.camera_segment_id == s.camera_segment_id]
            have = f.frame_index.isin(b.frame_index).to_numpy()
            gaps, cur = [], 0
            for h in have:
                if h:
                    if cur:
                        gaps.append(cur)
                    cur = 0
                else:
                    cur += 1
            if cur:
                gaps.append(cur)
            gaps = np.array(gaps) / fps
            m["ball"] = {"detected": float((b.image_source == "detected").sum() / n),
                         "interpolated": float((b.image_source == "interpolated").sum() / n),
                         "missing": float(1 - have.mean()),
                         "held_proxy": float((b.position_3d_status == "possession_proxy").sum() / n),
                         "gaps_ge_1s": int((gaps >= 1).sum()), "gaps_ge_3s": int((gaps >= 3).sum()),
                         "longest_gap_s": float(gaps.max()) if len(gaps) else 0.0}
        out["segments"].append(m)
    rt = run / "run_track.json"
    if rt.exists():
        j = json.loads(rt.read_text())
        out["stitch_enabled"] = j.get("stitch_enabled")
        out["stitch_links"] = len(j.get("stitch_links", []))
        out["n_tracks_total"] = j.get("n_tracks")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {Path(r).name: run_metrics(Path(r)) for r in a.runs}
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
