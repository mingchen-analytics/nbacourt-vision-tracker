"""Offline (bidirectional) trajectory reconstruction on recorded video.

Three distinct problems (see docs/OFFLINE_RECONSTRUCTION.md):

1. Tracklet stitching   (implemented)  re-link fragments of the same person across
   occlusions / crossings / missed detections, using BOTH the end of the earlier and the
   start of the later fragment.
2. Short-gap interpolation (implemented) fill missing frames between two reliable
   observations of the same (stitched) track. Output rows are new rows with
   position_status = "interpolated"; observed rows and raw coordinates are never changed.
3. Off-screen backward inference (NOT implemented; design only) -- a person first seen
   at t1 was somewhere before t1; future observations alone cannot pin that down. At most
   a feasible region (reachable set, outside the camera footprint) can be stated.

All thresholds are engineering defaults to be validated against reviewed ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class StitchConfig:
    max_gap_s: float = 2.0  # largest temporal gap a link may bridge
    v_max: float = 7.0  # m/s, upper bound on average player speed over the gap
    margin_m: float = 1.0  # slack for ground-point noise at both ends (bbox-bottom jitter)
    max_app_dist: float = 0.45  # appearance descriptor (Hellinger-type) distance limit
    w_app: float = 0.5  # weight of appearance in the link cost
    end_window: int = 10  # observations used for the end / start position
    app_window: int = 30  # observations searched for appearance (fragments often end occluded)
    require_appearance: bool = True  # no descriptor at either end -> no link
    require_known_labels: bool = True  # with class labels available, both ends must be known and equal
    max_cost: float = 1.0


def _end_summary(g: pd.DataFrame, desc_of_row, window: int, first: bool, app_window: int = 30) -> dict:
    h = g.head(window) if first else g.tail(window)
    obs = h[h.position_status == "observed_projected"]
    xy = obs[["court_x_m_raw", "court_y_m_raw"]].dropna().to_numpy(float)
    d = None
    if desc_of_row is not None:
        w = g.head(app_window) if first else g.tail(app_window)
        occ = w.get("_occ", pd.Series(0.0, index=w.index)).fillna(0)
        # prefer unoccluded rows; fragments typically end in an occlusion
        order = occ.sort_values(kind="stable").index
        vecs = []
        for i in order:
            v = desc_of_row(i)
            if v is not None:
                vecs.append(v)
            if len(vecs) >= 8 or (len(vecs) >= 3 and occ[i] >= 0.1):
                break
        if vecs:
            d = np.mean(vecs, axis=0)
    return {"frame": int(h.frame_index.iloc[0] if first else h.frame_index.iloc[-1]),
            "xy": np.median(xy, axis=0) if len(xy) else None, "desc": d}


def stitch_tracklets(pl: pd.DataFrame, fps: float, desc_of_row=None, label_col: str | None = None,
                     cfg: StitchConfig | None = None, exclude: set | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (pl with merged track_id and `source_track_id` set, list of accepted links).

    A link (end of fragment i -> start of fragment j) requires: j starts after i ends,
    gap <= max_gap_s, no overlap; court distance <= v_max * gap + margin; compatible class
    label (if `label_col` gives a known label for both); appearance distance <=
    max_app_dist (if descriptors available). One-to-one links are chosen by Hungarian
    assignment on the combined cost; chains are followed transitively.
    """
    cfg = cfg or StitchConfig()
    exclude = exclude or set()
    pl = pl.sort_values(["track_id", "frame_index"]).copy()
    if "source_track_id" not in pl.columns:
        pl["source_track_id"] = pl.track_id
    tids = [t for t in pl.track_id.unique() if t not in exclude]
    groups = {t: g for t, g in pl.groupby("track_id") if t in tids}
    label = {}
    if label_col and label_col in pl.columns:
        for t, g in groups.items():
            v = g[label_col][g[label_col].astype(str).isin(["A", "B", "REF"])]
            label[t] = v.mode().iloc[0] if len(v) and v.value_counts(normalize=True).iloc[0] >= 0.7 else None
    ends = {t: _end_summary(g, desc_of_row, cfg.end_window, False, cfg.app_window) for t, g in groups.items()}
    starts = {t: _end_summary(g, desc_of_row, cfg.end_window, True, cfg.app_window) for t, g in groups.items()}
    use_labels = bool(label)
    E, S = list(groups), list(groups)
    C = np.full((len(E), len(S)), np.inf)
    detail = {}
    for i, a in enumerate(E):
        for j, b in enumerate(S):
            if a == b:
                continue
            if "camera_segment_id" in pl.columns and \
                    groups[a].camera_segment_id.iloc[-1] != groups[b].camera_segment_id.iloc[0]:
                continue  # never link across a camera cut
            gap = (starts[b]["frame"] - ends[a]["frame"]) / fps
            if gap <= 0 or gap > cfg.max_gap_s:
                continue
            if use_labels and cfg.require_known_labels:
                if not label.get(a) or not label.get(b) or label[a] != label[b]:
                    continue
            elif label.get(a) and label.get(b) and label[a] != label[b]:
                continue
            pa, pb = ends[a]["xy"], starts[b]["xy"]
            if pa is None or pb is None:
                continue
            reach = cfg.v_max * gap + cfg.margin_m
            dist = float(np.linalg.norm(pb - pa))
            if dist > reach:
                continue
            cost = dist / reach
            app = None
            if desc_of_row is not None and cfg.require_appearance and (
                    ends[a]["desc"] is None or starts[b]["desc"] is None):
                continue  # identity cannot be checked
            if ends[a]["desc"] is not None and starts[b]["desc"] is not None:
                app = float(np.linalg.norm(ends[a]["desc"] - starts[b]["desc"]))
                if app > cfg.max_app_dist:
                    continue
                cost = (1 - cfg.w_app) * cost + cfg.w_app * app / cfg.max_app_dist
            if cost <= cfg.max_cost:
                C[i, j] = cost
                detail[(a, b)] = dict(gap_s=gap, dist_m=dist, reach_m=reach, app_dist=app, cost=cost)
    links = []
    if np.isfinite(C).any():
        r, c = linear_sum_assignment(np.where(np.isfinite(C), C, 1e6))
        for i, j in zip(r, c):
            if np.isfinite(C[i, j]):
                links.append(dict(end_track=E[i], start_track=S[j], **detail[(E[i], S[j])]))
    nxt = {l["end_track"]: l["start_track"] for l in links}
    has_prev = {l["start_track"] for l in links}
    new_id = {}
    for t in tids:
        if t in has_prev:
            continue
        chain, cur = [t], t
        while cur in nxt:
            cur = nxt[cur]
            chain.append(cur)
        for c_ in chain:
            new_id[c_] = t
    pl["track_id"] = pl.track_id.map(lambda t: new_id.get(t, t))
    return pl.sort_values(["frame_index", "track_id"]), links


def interpolate_gaps(pl: pd.DataFrame, frames: pd.DataFrame, max_gap_s: float = 1.0,
                     sigma_growth_mps: float = 2.0) -> pd.DataFrame:
    """New rows for missing frames inside each track between two observed rows with court
    coordinates, gap <= max_gap_s. Positions: cubic Hermite between the smoothed endpoint
    positions using finite-difference velocities (linear if unavailable). Sigma grows with
    the distance in time to the nearest observation (uncalibrated heuristic)."""
    ts_of = dict(zip(frames.frame_index, frames.timestamp_sec))
    new = []
    for tid, g in pl.sort_values("frame_index").groupby("track_id"):
        o = g[(g.position_status == "observed_projected") & g.court_x_m.notna()]
        if len(o) < 2:
            continue
        f = o.frame_index.to_numpy()
        xy = o[["court_x_m", "court_y_m"]].to_numpy(float)
        t = np.array([ts_of[x] for x in f])
        sg = o.court_sigma_m.to_numpy(float)
        for k in range(len(f) - 1):
            if f[k + 1] - f[k] < 2 or t[k + 1] - t[k] > max_gap_s:
                continue
            dt = t[k + 1] - t[k]
            v0 = (xy[k] - xy[k - 1]) / (t[k] - t[k - 1]) if k > 0 and t[k] > t[k - 1] else (xy[k + 1] - xy[k]) / dt
            v1 = (xy[k + 2] - xy[k + 1]) / (t[k + 2] - t[k + 1]) if k + 2 < len(f) and t[k + 2] > t[k + 1] else (xy[k + 1] - xy[k]) / dt
            same_lbl = o.role.iloc[k] == o.role.iloc[k + 1] and o.team_id.iloc[k] == o.team_id.iloc[k + 1]
            for fi in range(f[k] + 1, f[k + 1]):
                if fi not in ts_of:
                    continue
                s = (ts_of[fi] - t[k]) / dt
                h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
                p = h00 * xy[k] + h10 * dt * v0 + h01 * xy[k + 1] + h11 * dt * v1
                near = min(ts_of[fi] - t[k], t[k + 1] - ts_of[fi])
                new.append(dict(
                    video_id=o.video_id.iloc[k], frame_index=int(fi), timestamp_sec=float(ts_of[fi]),
                    camera_segment_id=o.camera_segment_id.iloc[k], track_id=tid,
                    source_track_id=None,
                    team_id=o.team_id.iloc[k] if same_lbl else None,
                    team_confidence=o.team_confidence.iloc[k] if same_lbl else None,
                    role=o.role.iloc[k] if same_lbl else "unknown",
                    ground_method="none", court_x_m=float(p[0]), court_y_m=float(p[1]),
                    court_sigma_m=float(max(sg[k], sg[k + 1]) + sigma_growth_mps * near),
                    position_status="interpolated", flags="interpolated",
                    calibration_quality=None, is_visible=False, is_on_court=None))
    return pd.DataFrame(new)
