"""Root-cause diagnostics for suspected identity switches and implausible speeds.

For every event (start of a `suspected_id_switch` run, or a `speed_violation` row) the
following evidence is computed from the run outputs, and a crop strip is saved:

  overlap        max IoU between the track's box and any other tracked box within +-5 frames
  gap_before     frames the track was unmatched just before the event
  calib_jump     |delta pan| or |delta tilt| at the event vs. the clip's 99th percentile
  box_h_jump     relative change of box height between consecutive observations
  court_jump_m   raw court displacement between consecutive observations

Heuristic cause (first matching rule): calibration > crossing (overlap > 0.3) >
re-acquisition after gap (gap >= 3) > ground-point jump (box height change > 25%) > other.
Heuristics guide inspection; the crop strips are the evidence.

Usage: python tools/identity_diagnostics.py outputs/real/clip2_ptz --video data/raw/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nbatrack.io.video import iter_frames
from nbatrack.track.tracker import iou_matrix


def events(pl: pd.DataFrame) -> pd.DataFrame:
    ev = []
    pl = pl.sort_values(["track_id", "frame_index"])
    fl = pl["flags"].fillna("")
    for tid, g in pl.assign(fl=fl).groupby("track_id"):
        s = g.fl.str.contains("suspected_id_switch").to_numpy()
        starts = np.flatnonzero(s & ~np.r_[False, s[:-1]])
        for k in starts:
            ev.append(dict(kind="suspected_id_switch", track_id=tid, frame=int(g.frame_index.iloc[k])))
        for f in g[g.fl.str.contains("speed_violation")].frame_index:
            ev.append(dict(kind="speed_violation", track_id=tid, frame=int(f)))
    return pd.DataFrame(ev, columns=["kind", "track_id", "frame"])


def evidence(pl: pd.DataFrame, diag: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    by_f = dict(tuple(pl.groupby("frame_index")))
    dpan = diag.pan.diff().abs()
    dtilt = diag.tilt.diff().abs()
    p99p, p99t = dpan.quantile(0.99), dtilt.quantile(0.99)
    rows = []
    for e in ev.itertuples():
        g = pl[pl.track_id == e.track_id].sort_values("frame_index")
        i = int(np.flatnonzero(g.frame_index.to_numpy() == e.frame)[0])
        ov = 0.0
        for f in range(e.frame - 5, e.frame + 6):
            fr = by_f.get(f)
            if fr is None:
                continue
            me = fr[fr.track_id == e.track_id]
            ot = fr[fr.track_id != e.track_id]
            if len(me) and len(ot):
                ov = max(ov, float(iou_matrix(me[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float),
                                              ot[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)).max()))
        gap = int(e.frame - g.frame_index.iloc[i - 1] - 1) if i > 0 else 0
        h = (g.bbox_y2 - g.bbox_y1).to_numpy()
        hj = float(abs(h[i] - h[i - 1]) / max(h[i - 1], 1)) if i > 0 else 0.0
        cj = float(np.hypot(g.court_x_m_raw.iloc[i] - g.court_x_m_raw.iloc[i - 1],
                            g.court_y_m_raw.iloc[i] - g.court_y_m_raw.iloc[i - 1])) if i > 0 else 0.0
        cal = bool(dpan.iloc[e.frame] > max(p99p, 0.5) or dtilt.iloc[e.frame] > max(p99t, 0.3)) if e.frame > 0 else False
        cause = ("calibration_jump" if cal else "crossing_overlap" if ov > 0.3 else
                 "reacquired_after_gap" if gap >= 3 else "ground_point_jump" if hj > 0.25 else "other")
        rows.append(dict(kind=e.kind, track_id=e.track_id, frame=e.frame, overlap=round(ov, 2), gap_before=gap,
                         box_h_jump=round(hj, 2), court_jump_m=round(cj, 2), calib_jump=cal, cause=cause))
    return pd.DataFrame(rows)


def crop_strips(video: str, pl: pd.DataFrame, ev: pd.DataFrame, out: Path, max_events: int = 30) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sel = ev.drop_duplicates(["track_id", "frame"]).head(max_events)
    need = {f for e in sel.itertuples() for f in range(e.frame - 10, e.frame + 11, 5)}
    frames = {}
    for fr in iter_frames(video):
        if fr.frame_index in need:
            frames[fr.frame_index] = fr.image
    by_f = dict(tuple(pl.groupby("frame_index")))
    for e in sel.itertuples():
        tiles = []
        g = pl[pl.track_id == e.track_id]
        c = g.iloc[(g.frame_index - e.frame).abs().argmin()]
        cx, cy = (c.bbox_x1 + c.bbox_x2) / 2, (c.bbox_y1 + c.bbox_y2) / 2
        for f in range(e.frame - 10, e.frame + 11, 5):
            if f not in frames:
                continue
            img = frames[f].copy()
            for r in by_f.get(f, pd.DataFrame()).itertuples():
                col = (0, 0, 255) if r.track_id == e.track_id else (200, 200, 200)
                cv2.rectangle(img, (int(r.bbox_x1), int(r.bbox_y1)), (int(r.bbox_x2), int(r.bbox_y2)), col, 2)
                cv2.putText(img, r.track_id.split("_t")[-1], (int(r.bbox_x1), int(r.bbox_y1) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            x0 = int(np.clip(cx - 200, 0, img.shape[1] - 400))
            y0 = int(np.clip(cy - 150, 0, img.shape[0] - 300))
            t = img[y0:y0 + 300, x0:x0 + 400].copy()
            cv2.putText(t, f"f{f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(t)
        if tiles:
            cv2.imwrite(str(out / f"{e.kind}_{e.track_id}_f{e.frame:04d}.jpg"), np.hstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--video")
    a = ap.parse_args()
    run = Path(a.run)
    pl = pd.read_parquet(run / "tables" / "player_observations.parquet")
    pl = pl[(pl.role != "other") & pl.bbox_x1.notna()]
    diag = pd.read_csv(run / "calibration_diagnostics.csv")
    ev = events(pl)
    df = evidence(pl, diag, ev)
    out = run / "diagnostics_identity"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "events.csv", index=False)
    summary = {k: df[df.kind == k].cause.value_counts().to_dict() for k in df.kind.unique()} if len(df) else {}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if a.video:
        crop_strips(a.video, pl, df, out / "strips")


if __name__ == "__main__":
    main()
