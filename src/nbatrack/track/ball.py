"""Offline single-ball tracking in image space + possession-based court proxy.

Image track: Viterbi over per-frame ball candidates (RF-DETR `sports ball`, score >= 0.1)
plus a "not visible" state. Costs:
  emission     -log(score) for a candidate; a fixed penalty for "not visible"
  transition   image distance between candidates after camera-motion compensation
               (frame-to-frame floor homography), relative to a per-frame motion
               allowance; switching between visible / not visible has a penalty.
Using the whole clip (forward and backward) lets a weak but consistent detection beat a
strong isolated false positive. Short invisible gaps (<= max_interp_frames) between two
selected detections are filled by linear image interpolation (source = "interpolated").

Court position: the ball's true 3D position is NOT observable from one view in flight.
When the ball image point lies inside a tracked player's box (between 15% and 90% of its
height), the ball is assumed held / dribbled by that player and the player's court XY is
reported as a proxy (position_3d_status = "possession_proxy", z unknown). Otherwise the
court position stays unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbatrack.calib.homography import apply_h


def viterbi_ball(cands: dict[int, pd.DataFrame], frames: list[int], H_by_frame: dict[int, np.ndarray],
                 max_step_px: float = 60.0, miss_cost: float = 2.3, switch_cost: float = 1.5,
                 motion_weight: float = 1.0) -> dict[int, int | None]:
    """Returns frame -> index label of the chosen candidate row (or None = not visible)."""
    states: list[list] = []  # per frame: list of (label, x, y, emission)
    for f in frames:
        c = cands.get(f)
        st = [(None, np.nan, np.nan, miss_cost)]
        if c is not None:
            for idx, r in c.iterrows():
                st.append((idx, (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, -np.log(max(r.score, 1e-3))))
        states.append(st)
    cost = [np.array([s[3] for s in states[0]])]
    back = []
    for t in range(1, len(frames)):
        prev, cur = states[t - 1], states[t]
        H0, H1 = H_by_frame.get(frames[t - 1]), H_by_frame.get(frames[t])
        M = H1 @ np.linalg.inv(H0) if H0 is not None and H1 is not None else np.eye(3)
        pxy = np.array([[s[1], s[2]] for s in prev], float)
        vis = np.isfinite(pxy[:, 0])
        pw = np.full_like(pxy, np.nan)
        if vis.any():
            pw[vis] = apply_h(M, pxy[vis])  # previous positions warped into the current frame
        dt = frames[t] - frames[t - 1]
        c_new = np.empty(len(cur))
        b_new = np.empty(len(cur), int)
        for j, s in enumerate(cur):
            if s[0] is None:
                trans = np.where(vis, switch_cost, 0.0)
            else:
                d = np.hypot(pw[:, 0] - s[1], pw[:, 1] - s[2]) / (max_step_px * dt)
                trans = np.where(vis, motion_weight * d ** 2, switch_cost)
                trans = np.where(vis & (d > 3.0), np.inf, trans)  # impossible jump
            tot = cost[-1] + trans
            k = int(np.argmin(tot))
            c_new[j] = tot[k] + s[3]
            b_new[j] = k
        cost.append(c_new)
        back.append(b_new)
    k = int(np.argmin(cost[-1]))
    out = {}
    for t in range(len(frames) - 1, -1, -1):
        out[frames[t]] = states[t][k][0]
        if t > 0:
            k = int(back[t - 1][k])
    return out


def remove_static_clutter(balls: pd.DataFrame, H: dict, players: pd.DataFrame | None = None,
                          window: int = 30, persist_frac: float = 0.6, bin_m: float = 0.35,
                          max_abs_y_m: float | None = 12.0) -> pd.DataFrame:
    """Drop ball candidates at world-static locations (rim, net, floor graphics, objects in
    the stands).

    With a fixed-center camera any static object maps to a constant floor point through
    the floor homography, whatever its height. A candidate is clutter if, within +-`window`
    frames, a candidate sits at the same floor location (same or neighbouring bin) in at
    least `persist_frac` of the frames AND it is not inside any tracked player's box (a
    held ball is inside the holder's box). Optionally, candidates projecting deep into the
    stands (|y| > max_abs_y_m) are dropped too; NOTE that a ball high in the air (a shot or a
    long pass) also projects far beyond the court on the floor plane, so this rule deletes
    airborne balls (clip 6, f937-1019: a whole shot arc). A loose ball lying still on the floor
    would also be removed (documented limitation).
    """
    if not len(balls) or not H:
        return balls
    pts = []
    for r in balls.itertuples():
        Hf = H.get(r.frame_index)
        xy = apply_h(np.linalg.inv(Hf), np.array([[(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2]]))[0] \
            if Hf is not None else np.array([np.nan, np.nan])
        pts.append(xy)
    pts = np.array(pts)
    b = balls.assign(cfx=pts[:, 0], cfy=pts[:, 1],
                     cbx=np.floor(pts[:, 0] / bin_m), cby=np.floor(pts[:, 1] / bin_m))
    from collections import defaultdict
    frames_at = defaultdict(set)
    for f, bx, by in zip(b.frame_index, b.cbx, b.cby):
        if np.isfinite(bx):
            frames_at[(int(bx), int(by))].add(int(f))
    inside = np.zeros(len(b), bool)
    if players is not None and len(players):
        pby = {f: g for f, g in players[players.bbox_x1.notna()].groupby("frame_index")}
        for k, r in enumerate(b.itertuples()):
            g = pby.get(r.frame_index)
            if g is not None:
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                inside[k] = bool(((g.bbox_x1 <= cx) & (cx <= g.bbox_x2) & (g.bbox_y1 <= cy) & (cy <= g.bbox_y2)).any())
    keep = np.ones(len(b), bool)
    for k, r in enumerate(b.itertuples()):
        if not np.isfinite(r.cfx):
            continue
        if max_abs_y_m is not None and abs(r.cfy) > max_abs_y_m:
            keep[k] = False
            continue
        if inside[k]:
            continue
        near = set()
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                near |= frames_at.get((int(r.cbx) + i, int(r.cby) + j), set())
        lo, hi = r.frame_index - window, r.frame_index + window
        n_win = sum(1 for f in near if lo <= f <= hi)
        if n_win >= persist_frac * (2 * window + 1):
            keep[k] = False
    return balls[keep]


def track_ball(det: pd.DataFrame, frames_tbl: pd.DataFrame, cal: pd.DataFrame, players: pd.DataFrame,
               video_id: str, min_score: float = 0.1, max_interp_frames: int = 8,
               max_abs_y_m: float | None = None) -> pd.DataFrame:
    # max_abs_y_m off by default (2026-10-16): with WASB candidates the 'far floor projection'
    # rule only removed airborne balls; 48/48 sampled frames it had deleted on clips 2/4/5/6
    # were the real ball in flight (tools/ball_filter_experiment.py), 0 frames lost without it.
    balls = det[(det.cls == 37) & (det.score >= min_score)].copy()
    frames = list(frames_tbl.frame_index)
    ts = dict(zip(frames_tbl.frame_index, frames_tbl.timestamp_sec))
    H = {}
    for f, r in cal.set_index("frame_index").iterrows():
        if r.H_court_to_img is not None and r.model != "none":
            H[f] = np.asarray(r.H_court_to_img, float).reshape(3, 3)
    balls = remove_static_clutter(balls, H, players, max_abs_y_m=max_abs_y_m)
    cands = {f: g for f, g in balls.groupby("frame_index")}
    choice = viterbi_ball(cands, frames, H)
    rows = []
    for f in frames:
        idx = choice.get(f)
        if idx is None:
            continue
        r = balls.loc[idx]
        rows.append(dict(frame_index=f, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, score=r.score, source="detected"))
    sel = pd.DataFrame(rows)
    if len(sel) >= 2:  # short-gap image interpolation
        add = []
        fr = sel.frame_index.to_numpy()
        for a, b in zip(range(len(fr) - 1), range(1, len(fr))):
            gap = fr[b] - fr[a]
            if 1 < gap <= max_interp_frames + 1:
                for f in range(fr[a] + 1, fr[b]):
                    w = (f - fr[a]) / gap
                    ra, rb = sel.iloc[a], sel.iloc[b]
                    add.append(dict(frame_index=f, **{k: (1 - w) * ra[k] + w * rb[k] for k in ("x1", "y1", "x2", "y2")},
                                    score=None, source="interpolated"))
        if add:
            sel = pd.concat([sel, pd.DataFrame(add)]).sort_values("frame_index")
    pby = {f: g for f, g in players[players.bbox_x1.notna() & players.court_x_m.notna()].groupby("frame_index")}
    out = []
    for r in sel.itertuples():
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        status, bx, by, holder = "unknown", None, None, None
        g = pby.get(r.frame_index)
        if g is not None:
            h = g.bbox_y2 - g.bbox_y1
            inside = (g.bbox_x1 <= cx) & (cx <= g.bbox_x2) & (g.bbox_y1 + 0.15 * h <= cy) & (cy <= g.bbox_y1 + 0.9 * h)
            if inside.any():
                c = g[inside]
                k = ((c.bbox_x1 + c.bbox_x2) / 2 - cx).abs().idxmin()
                status, bx, by, holder = "possession_proxy", float(c.loc[k, "court_x_m"]), float(c.loc[k, "court_y_m"]), c.loc[k, "track_id"]
        out.append(dict(video_id=video_id, frame_index=int(r.frame_index), timestamp_sec=float(ts[r.frame_index]),
                        camera_segment_id=0, ball_track_id="ball_0",
                        bbox_x1=r.x1, bbox_y1=r.y1, bbox_x2=r.x2, bbox_y2=r.y2, image_x=cx, image_y=cy,
                        det_confidence=r.score if r.source == "detected" else None,
                        is_visible=r.source == "detected", image_source=r.source,
                        position_3d_status=status, ball_x_m=bx, ball_y_m=by, ball_z_m=None,
                        ball_sigma_m=None, holder_track_id=holder))
    return pd.DataFrame(out)
