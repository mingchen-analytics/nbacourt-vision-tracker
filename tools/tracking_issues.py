"""Automatic candidate-error intervals for review (NOT verified errors).

Reads a run directory (tables/, calibration_diagnostics.csv, annotated.mp4) and writes
  tracking_issues.csv         one row per candidate interval (sorted by time)
  tracking_issues_summary.json counts per category / severity
  issues/<issue_id>.mp4       short clips cut from annotated.mp4 (+-1 s) for the top issues

Categories (all heuristics; thresholds in the code next to each rule):
  calibration_failure      supported-view frames with quality poor/failed or floor IoU < 0.45
                           (floor IoU is independent of the line alignment that sets quality)
  calibration_jump         per-frame pan/tilt/focal change far above normal camera motion
  camera_cut / flash       informational: segment boundaries and over-exposed single frames
  possible_id_swap         (a) team-colour vote changes inside a track (suspected_id_switch);
                           (b) close crossings (box IoU >= 0.4 for >= 3 frames) between two
                           tracks; severity raised if a speed violation / colour change follows
  possible_fragmentation   a track ends and another starts <= 3 s later within reach
                           (8 m/s * gap + 1.5 m) in the same camera segment
  short_track              player/unknown tracks shorter than 1 s
  implausible_speed        rows flagged speed_violation (sigma-aware test in filter/temporal.py)
  coordinate_jump          raw court step > 1.5 m between observations <= 0.1 s apart; the
                           likely cause is classified as calibration / occlusion / ground-contact
  team_inconsistency       player tracks with team vote fraction < 0.6
  unclassified_on_court    tracks with role "unknown" on court for >= 2 s
  too_many_players         intervals with > 10 simultaneous "player" rows (at least one false
                           player or duplicate); > 3 referees likewise
  possible_sideline_person "player" tracks whose median position is within 1 m of a sideline /
                           baseline (inside or outside) and which move < 1.5 m in total
  long_ball_gap            >= 1 s without any ball row inside a supported segment
  ball_jump                consecutive detected ball points > 120 px apart after camera-motion
                           compensation (possible wrong Viterbi link)
  ball_static_unheld       unheld detected ball staying at one floor point (< 0.3 m) >= 0.5 s

Usage: python tools/tracking_issues.py outputs/real/clip6_long_115s [--clips 60]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.calib.homography import apply_h  # noqa: E402
from nbatrack.court.geometry import NBA_COURT  # noqa: E402
from nbatrack.track.tracker import iou_matrix  # noqa: E402

SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def runs(mask: np.ndarray, frames: np.ndarray, max_gap: int = 0):
    """Contiguous runs of True (allowing gaps <= max_gap frames) -> list of (f0, f1)."""
    fs = frames[mask]
    if not len(fs):
        return []
    out, s, p = [], fs[0], fs[0]
    for f in fs[1:]:
        if f - p > max_gap + 1:
            out.append((s, p))
            s = f
        p = f
    out.append((s, p))
    return out


class Issues:
    def __init__(self, ts_of, seg_of):
        self.rows = []
        self.ts_of, self.seg_of = ts_of, seg_of

    def add(self, cat, f0, f1, tids, severity, evidence, cause=None, score=None):
        self.rows.append(dict(category=cat, severity=severity, score=score, start_frame=int(f0), end_frame=int(f1),
                              start_sec=round(self.ts_of(f0), 3), end_sec=round(self.ts_of(f1), 3),
                              camera_segment_id=int(self.seg_of(f0)),
                              track_ids=";".join(sorted(set(map(str, tids)))) if tids else "",
                              likely_cause=cause, evidence=evidence))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--clips", type=int, default=60, help="cut review clips for the top-N high/medium issues")
    a = ap.parse_args()
    run = Path(a.run)
    t = run / "tables"
    pl = pd.read_parquet(t / "player_observations.parquet")
    ball = pd.read_parquet(t / "ball_observations.parquet")
    frames = pd.read_parquet(t / "frames.parquet")
    segs = pd.read_parquet(t / "camera_segments.parquet")
    cal = pd.read_parquet(t / "calibration.parquet").set_index("frame_index")
    diag = pd.read_csv(run / "calibration_diagnostics.csv").set_index("frame")
    fps = float(pd.read_parquet(t / "videos.parquet").fps_nominal.iloc[0])
    ts = dict(zip(frames.frame_index, frames.timestamp_sec))
    segf = dict(zip(frames.frame_index, frames.camera_segment_id))
    sup = dict(zip(segs.camera_segment_id, segs.supported))
    I = Issues(lambda f: ts.get(int(f), np.nan), lambda f: segf.get(int(f), -1))
    F = frames.frame_index.to_numpy()
    seg_arr = frames.camera_segment_id.to_numpy()
    sup_arr = np.array([bool(sup.get(s, False)) for s in seg_arr])

    # ---------------------------------------------------------------- camera / calibration
    for r in segs.itertuples():
        if r.start_frame > 0:
            I.add("camera_cut", r.start_frame, r.start_frame, [], "info",
                  f"segment {r.camera_segment_id} starts: view={r.view_type} supported={r.supported}")
        if not r.supported:
            I.add("unsupported_view", r.start_frame, r.end_frame, [], "info",
                  f"view={r.view_type}: no calibration, no tracking, no ball (by design)")
    d = diag.reindex(F)
    is_flash = (d["method"] == "held_flash").to_numpy()
    for f in F[is_flash]:
        I.add("flash_frame", f, f, [], "info", "over-exposed frame: camera held from previous frame, positions unknown")
    q = d["quality"].astype(str).to_numpy()
    fiou = d["floor_iou"].to_numpy(float) if "floor_iou" in d else np.full(len(F), np.nan)
    bad = sup_arr & ~is_flash & (np.isin(q, ["poor", "failed"]) | (fiou < 0.45))
    for f0, f1 in runs(bad, F, max_gap=5):
        w = (F >= f0) & (F <= f1)
        n = int(bad[w].sum())
        sev = "high" if n >= fps else "medium" if n >= 10 else "low"
        I.add("calibration_failure", f0, f1, [], sev,
              f"{n} bad frames; quality counts {pd.Series(q[w]).value_counts().to_dict()}; "
              f"min inlier {np.nanmin(d.inlier.to_numpy(float)[w]):.2f}; min floor_iou {np.nanmin(fiou[w]) if np.isfinite(fiou[w]).any() else float('nan'):.2f}",
              cause="calibration", score=n)
    dp = np.abs(np.diff(d.pan.to_numpy(float), prepend=np.nan))
    dt_ = np.abs(np.diff(d.tilt.to_numpy(float), prepend=np.nan))
    dfr = np.abs(np.diff(np.log(d.f.to_numpy(float)), prepend=np.nan))
    same_seg = np.r_[False, seg_arr[1:] == seg_arr[:-1]]
    prev_flash = np.r_[False, is_flash[:-1]]
    jump = sup_arr & same_seg & ~is_flash & ~prev_flash & ((dp > 1.0) | (dt_ > 0.5) | (dfr > 0.03))
    for f0, f1 in runs(jump, F, max_gap=3):
        w = (F >= f0) & (F <= f1)
        I.add("calibration_jump", f0, f1, [], "medium",
              f"max |dpan| {np.nanmax(dp[w]):.2f} deg, |dtilt| {np.nanmax(dt_[w]):.2f} deg, |df/f| {np.nanmax(dfr[w]):.3f} per frame",
              cause="calibration")
    calib_bad_frames = set(F[bad | jump])

    # ---------------------------------------------------------------- players
    obs = pl[pl.position_status == "observed_projected"].sort_values(["track_id", "frame_index"])
    flags = pl["flags"].fillna("")
    by_f = {f: g for f, g in pl[pl.bbox_x1.notna()].groupby("frame_index")}
    box_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    def overlap_near(tid, f, win=5):
        ov = 0.0
        for k in range(f - win, f + win + 1):
            g = by_f.get(k)
            if g is None:
                continue
            me, ot = g[g.track_id == tid], g[g.track_id != tid]
            if len(me) and len(ot):
                ov = max(ov, float(iou_matrix(me[box_cols].to_numpy(float), ot[box_cols].to_numpy(float)).max()))
        return ov

    # implausible speed + coordinate jumps
    for tid, g in pl.assign(fl=flags).sort_values("frame_index").groupby("track_id"):
        sv = g.fl.str.contains("speed_violation").to_numpy()
        for f0, f1 in runs(sv, g.frame_index.to_numpy(), max_gap=int(0.5 * fps)):
            cal_hit = any(f in calib_bad_frames for f in range(f0 - 2, f1 + 3))
            ov = overlap_near(tid, int(f0))
            cause = "calibration" if cal_hit else "occlusion/crossing" if ov > 0.3 else "ground_contact/detection"
            I.add("implausible_speed", f0, f1, [tid], "medium" if f1 - f0 >= 3 else "low",
                  f"speed_violation rows {int(sv[(g.frame_index >= f0).to_numpy() & (g.frame_index <= f1).to_numpy()].sum())}; "
                  f"max box IoU with others +-5f {ov:.2f}", cause=cause)
    for tid, g in obs.groupby("track_id"):
        f = g.frame_index.to_numpy()
        xy = g[["court_x_m_raw", "court_y_m_raw"]].to_numpy(float)
        h = (g.bbox_y2 - g.bbox_y1).to_numpy(float)
        tt = g.timestamp_sec.to_numpy(float)
        for i in range(1, len(g)):
            step = float(np.hypot(*(xy[i] - xy[i - 1])))
            if tt[i] - tt[i - 1] <= 0.1 and step > 1.5:
                hj = abs(h[i] - h[i - 1]) / max(h[i - 1], 1)
                ov = overlap_near(tid, int(f[i]), 2)
                cal_hit = any(k in calib_bad_frames for k in range(int(f[i]) - 2, int(f[i]) + 3))
                cause = ("calibration" if cal_hit else "occlusion/crossing" if ov > 0.3 else
                         "ground_contact (box height change)" if hj > 0.25 else "detection/unknown")
                I.add("coordinate_jump", f[i - 1], f[i], [tid], "high" if step > 3 else "medium",
                      f"raw court step {step:.2f} m in {tt[i] - tt[i - 1]:.3f} s; box height change {hj:.0%}; IoU {ov:.2f}",
                      cause=cause, score=round(step, 2))

    # colour-vote changes (suspected switches)
    for tid, g in pl.assign(fl=flags).sort_values("frame_index").groupby("track_id"):
        s = g.fl.str.contains("suspected_id_switch").to_numpy()
        for f0, f1 in runs(s, g.frame_index.to_numpy(), max_gap=int(fps)):
            I.add("possible_id_swap", f0, f1, [tid], "high",
                  "team-colour vote changes inside the track (suspected_id_switch flag)", cause="colour_change")

    # close crossings
    pair_frames: dict = {}
    pp = pl[pl.bbox_x1.notna() & pl.role.isin(["player", "unknown", "referee"])]
    for f, g in pp.groupby("frame_index"):
        if len(g) < 2:
            continue
        b = g[box_cols].to_numpy(float)
        M = iou_matrix(b, b)
        np.fill_diagonal(M, 0)
        ii, jj = np.where(np.triu(M) >= 0.4)
        tids = g.track_id.to_numpy()
        for i, j in zip(ii, jj):
            pair_frames.setdefault(tuple(sorted((tids[i], tids[j]))), []).append(int(f))
    sw_rows = pl.assign(fl=flags)
    sw_rows = sw_rows[sw_rows.fl.str.contains("speed_violation|suspected_id_switch")]
    team_of = pl.dropna(subset=["team_id"]).groupby("track_id").team_id.agg(lambda v: v.mode().iloc[0]).to_dict()
    for (ta, tb), fl_ in pair_frames.items():
        fl_ = np.array(sorted(fl_))
        for f0, f1 in runs(np.ones(len(fl_), bool), fl_, max_gap=2):
            if f1 - f0 + 1 < 3:
                continue
            after = sw_rows[sw_rows.track_id.isin([ta, tb]) & sw_rows.frame_index.between(f0, f1 + int(fps))]
            same = team_of.get(ta) is not None and team_of.get(ta) == team_of.get(tb)
            sev = "high" if len(after) else "medium"
            I.add("possible_id_swap", f0, f1, [ta, tb], sev,
                  f"boxes overlap (IoU>=0.4) for {f1 - f0 + 1} frames; teams {team_of.get(ta)}/{team_of.get(tb)}"
                  f"{' (same team: colour cannot separate them)' if same else ''}; "
                  f"{len(after)} speed/colour flags on these tracks within 1 s after",
                  cause="crossing")

    # fragmentation + short tracks
    life = pl[pl.bbox_x1.notna()].groupby("track_id").agg(f0=("frame_index", "min"), f1=("frame_index", "max"),
                                                          seg=("camera_segment_id", "first"),
                                                          role=("role", lambda v: v.mode().iloc[0]))
    seg_end = dict(zip(segs.camera_segment_id, segs.end_frame))
    seg_start = dict(zip(segs.camera_segment_id, segs.start_frame))
    endxy = obs.groupby("track_id").last()[["court_x_m_raw", "court_y_m_raw", "frame_index"]]
    startxy = obs.groupby("track_id").first()[["court_x_m_raw", "court_y_m_raw", "frame_index"]]
    for tid, r in life.iterrows():
        dur = (r.f1 - r.f0 + 1) / fps
        if r.role in ("player", "unknown") and dur < 1.0:
            I.add("short_track", r.f0, r.f1, [tid], "low", f"{r.role} track lasting {dur:.2f} s", cause="fragmentation")
        if r.role == "other" or tid not in endxy.index or r.f1 >= seg_end[r.seg] - int(0.5 * fps):
            continue
        cands = []
        for t2, r2 in life.iterrows():
            if t2 == tid or r2.seg != r.seg or r2.role == "other" or t2 not in startxy.index:
                continue
            gap = (r2.f0 - r.f1) / fps
            if not 0 < gap <= 3.0:
                continue
            dist = float(np.hypot(startxy.loc[t2, "court_x_m_raw"] - endxy.loc[tid, "court_x_m_raw"],
                                  startxy.loc[t2, "court_y_m_raw"] - endxy.loc[tid, "court_y_m_raw"]))
            if dist <= 8.0 * gap + 1.5:
                cands.append((dist, gap, t2))
        if cands:
            cands.sort()
            dist, gap, t2 = cands[0]
            I.add("possible_fragmentation", r.f1, life.loc[t2, "f0"], [tid, t2], "medium" if gap < 1.5 else "low",
                  f"{tid} ends, {t2} starts {gap:.2f} s later {dist:.2f} m away"
                  f"{f' (+{len(cands) - 1} other candidates)' if len(cands) > 1 else ''}; teams {team_of.get(tid)}/{team_of.get(t2)}",
                  cause="lost_track")

    # roles / on-court classification
    tconf = pl[pl.role == "player"].groupby("track_id").team_confidence.median()
    for tid, c in tconf.items():
        if c is not None and np.isfinite(c) and c < 0.6:
            r = life.loc[tid]
            I.add("team_inconsistency", r.f0, r.f1, [tid], "medium", f"team vote fraction {c:.2f}", cause="team_classification")
    for tid, g in obs.groupby("track_id"):
        role = g.role.mode().iloc[0]
        xy = g[["court_x_m_raw", "court_y_m_raw"]].to_numpy(float)
        dur = (g.frame_index.max() - g.frame_index.min() + 1) / fps
        if role == "unknown" and dur >= 2.0:
            I.add("unclassified_on_court", g.frame_index.min(), g.frame_index.max(), [tid], "low",
                  f"role unknown for {dur:.1f} s", cause="team_classification")
        if role == "player" and len(xy) >= fps:
            med = np.median(xy, 0)
            edge = min(NBA_COURT.half_width - abs(med[1]), NBA_COURT.half_length - abs(med[0]))
            travel = float(np.hypot(*(np.percentile(xy, 95, 0) - np.percentile(xy, 5, 0))))
            if edge < 1.0 and travel < 1.5:
                I.add("possible_sideline_person", g.frame_index.min(), g.frame_index.max(), [tid], "medium",
                      f"median position ({med[0]:.1f}, {med[1]:.1f}) m, {edge:.2f} m from the boundary; moves only {travel:.2f} m in {dur:.1f} s",
                      cause="on_court_classification")
    cnt = pl[pl.bbox_x1.notna()].groupby(["frame_index", "role"]).size().unstack(fill_value=0).reindex(F, fill_value=0)
    for role, lim, cat in (("player", 10, "too_many_players"), ("referee", 3, "too_many_referees")):
        if role in cnt:
            m = (cnt[role] > lim).to_numpy()
            for f0, f1 in runs(m, F, max_gap=int(0.25 * fps)):
                if f1 - f0 + 1 < int(0.25 * fps):
                    continue
                g = pl[pl.frame_index.between(f0, f1) & (pl.role == role)]
                I.add(cat, f0, f1, list(g.track_id.unique())[:20], "medium",
                      f"up to {int(cnt[role][(F >= f0) & (F <= f1)].max())} simultaneous {role} rows (limit {lim})",
                      cause="on_court_classification/duplicate")

    # ---------------------------------------------------------------- ball
    have = np.isin(F, ball.frame_index.to_numpy())
    for sid in segs.camera_segment_id[segs.supported]:
        m = (seg_arr == sid) & ~have
        for f0, f1 in runs(m, F):
            dur = (f1 - f0 + 1) / fps
            if dur >= 1.0:
                I.add("long_ball_gap", f0, f1, [], "medium" if dur >= 2 else "low", f"no ball row for {dur:.2f} s",
                      cause="ball_detection", score=round(dur, 2))
    det = ball[ball.image_source == "detected"].sort_values("frame_index")
    Hc = {}
    for f, r in cal.iterrows():
        if r.model != "none" and r.H_court_to_img is not None:
            Hc[f] = np.asarray(r.H_court_to_img, float).reshape(3, 3)
    fb = det.frame_index.to_numpy()
    pts = det[["image_x", "image_y"]].to_numpy(float)
    for i in range(1, len(det)):
        f0, f1 = int(fb[i - 1]), int(fb[i])
        if f1 - f0 > 3 or segf.get(f0) != segf.get(f1):
            continue
        p = pts[i - 1]
        if f0 in Hc and f1 in Hc:
            p = apply_h(Hc[f1] @ np.linalg.inv(Hc[f0]), p[None])[0]
        step = float(np.hypot(*(pts[i] - p))) / (f1 - f0)
        if step > 120 * 60 / fps:
            I.add("ball_jump", f0, f1, [], "medium", f"ball image step {step:.0f} px/frame after camera compensation",
                  cause="ball_association")
    unheld = det[det.position_3d_status != "possession_proxy"]
    fl_ = {}
    for r in unheld.itertuples():
        H = Hc.get(r.frame_index)
        if H is not None:
            fl_[r.frame_index] = apply_h(np.linalg.inv(H), np.array([[r.image_x, r.image_y]]))[0]
    if fl_:
        ff = np.array(sorted(fl_))
        P = np.array([fl_[f] for f in ff])
        static = np.zeros(len(ff), bool)
        w = int(0.5 * fps)
        for i in range(len(ff)):
            j = (ff >= ff[i] - w) & (ff <= ff[i] + w)
            if j.sum() >= w and np.hypot(*(P[j] - P[i]).T).max() < 0.3:
                static[i] = True
        for f0, f1 in runs(static, ff, max_gap=3):
            I.add("ball_static_unheld", f0, f1, [], "medium",
                  f"unheld ball stays within 0.3 m on the floor for {(f1 - f0 + 1) / fps:.2f} s (possible shoe/floor-mark/logo)",
                  cause="ball_false_positive")

    # ---------------------------------------------------------------- write
    df = pd.DataFrame(I.rows)
    df["_o"] = df.severity.map(SEV_ORDER)
    df = df.sort_values(["start_frame", "_o"]).drop(columns="_o").reset_index(drop=True)
    df.insert(0, "issue_id", [f"I{k:04d}" for k in range(len(df))])
    df["review_clip"] = ""
    ann = run / "annotated.mp4"
    ff = shutil.which("ffmpeg")
    if a.clips and ann.exists() and ff:
        (run / "issues").mkdir(exist_ok=True)
        top = df[df.severity.isin(["high", "medium"])].copy()
        top["_o"] = top.severity.map(SEV_ORDER)
        top = top.sort_values(["_o", "score"], ascending=[True, False]).head(a.clips)
        for i, r in top.iterrows():
            clip = run / "issues" / f"{r.issue_id}_{r.category}.mp4"
            s = max(0.0, r.start_sec - 1.0)
            dur = (r.end_sec - r.start_sec) + 2.0
            subprocess.run([ff, "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", str(ann), "-t", f"{min(dur, 12):.3f}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an", str(clip)], check=False)
            df.loc[i, "review_clip"] = str(clip.relative_to(run)) if clip.exists() else ""
    df.to_csv(run / "tracking_issues.csv", index=False)
    summ = dict(n_issues=len(df), by_category=df.groupby("category").size().to_dict(),
                by_severity=df.groupby("severity").size().to_dict(),
                by_category_severity={f"{c}|{s}": int(n) for (c, s), n in df.groupby(["category", "severity"]).size().items()},
                note="automatic heuristic flags, NOT verified errors")
    (run / "tracking_issues_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
