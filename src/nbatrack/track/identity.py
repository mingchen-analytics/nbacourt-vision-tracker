"""Post-hoc identity consistency check using jersey color.

A track whose per-frame jersey-color class (team A / team B, nearest k-means center of
torso Lab; referee-class frames ignored) changes and STAYS changed for at least
`min_run` frames (default 1 s at 30 fps) almost
certainly contains a cross-class identity switch (e.g. it jumped from a black-jersey
player to a white-jersey player at a crossing). Such tracks are split at the change
point. Same-class switches (two teammates) are NOT detectable this way.

The number of splits is reported as `suspected_cross_class_switches` -- a lower bound on
identity errors, not a measured ID-switch count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_mode(labels: np.ndarray, window: int) -> np.ndarray:
    out = labels.copy()
    h = window // 2
    for i in range(len(labels)):
        seg = labels[max(0, i - h): i + h + 1]
        seg = seg[seg >= 0]
        if len(seg):
            out[i] = np.bincount(seg).argmax()
    return out


def split_on_color_change(df: pd.DataFrame, centers: np.ndarray, min_run: int = 30,
                          window: int = 31, ignore: tuple[int, ...] = ()) -> tuple[pd.DataFrame, int]:
    """df: rows with track_id, frame_index, torso_L/a/b. `ignore`: center indices whose
    frames are treated as uninformative (the referee class, whose gray lies between
    dark and light jerseys and absorbs noisy crops). Returns (df, number of splits)."""
    df = df.sort_values(["track_id", "frame_index"]).copy()
    n_splits = 0
    new_ids = df.track_id.to_numpy(object).copy()
    for tid, g in df.groupby("track_id", sort=False):
        X = g[["torso_L", "torso_a", "torso_b"]].to_numpy(float)
        ok = np.all(np.isfinite(X), 1)
        lab = np.full(len(g), -1)
        if ok.any():
            d = np.linalg.norm(X[ok, None, :] - centers[None], axis=2)
            lab[ok] = d.argmin(1)
            lab[np.isin(lab, ignore)] = -1
        sm = _rolling_mode(lab, window)
        # runs of constant smoothed label
        starts = [0] + [i for i in range(1, len(sm)) if sm[i] != sm[i - 1]]
        runs = [(s, (starts[k + 1] if k + 1 < len(starts) else len(sm))) for k, s in enumerate(starts)]
        # only informative (team-class) runs can trigger a split
        runs = [r for r in runs if r[1] - r[0] >= min_run and sm[r[0]] >= 0]
        # merge consecutive long runs with the same label
        merged = []
        for r in runs:
            if merged and sm[merged[-1][0]] == sm[r[0]]:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        if len(merged) <= 1:
            continue
        idx = g.index.to_numpy()
        for k, (s, e) in enumerate(merged[1:], start=1):
            cut = s  # first frame of the new long run
            end = merged[k + 1][0] if k + 1 < len(merged) else len(g)
            pos = df.index.get_indexer(idx[cut:end])
            new_ids[pos] = f"{tid}_{k + 1}"
            n_splits += 1
    df["track_id"] = new_ids
    return df, n_splits


def label_change_rows(df: pd.DataFrame, label_col: str, min_run: int = 30, window: int = 31) -> pd.Series:
    """Boolean per row: True for rows after a persistent change of a per-frame class label
    (empty string = abstain) inside the same track. Same run logic as
    `split_on_color_change`, for labels coming from any classifier."""
    names = sorted({v for v in df[label_col].unique() if v})
    code = {n: i for i, n in enumerate(names)}
    out = pd.Series(False, index=df.index)
    for _, g in df.sort_values("frame_index").groupby("track_id", sort=False):
        lab = np.array([code.get(v, -1) for v in g[label_col]])
        sm = _rolling_mode(lab, window)
        starts = [0] + [i for i in range(1, len(sm)) if sm[i] != sm[i - 1]]
        runs = [(s, (starts[k + 1] if k + 1 < len(starts) else len(sm))) for k, s in enumerate(starts)]
        runs = [r for r in runs if r[1] - r[0] >= min_run and sm[r[0]] >= 0]
        merged = []
        for r in runs:
            if merged and sm[merged[-1][0]] == sm[r[0]]:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        if len(merged) > 1:
            out.loc[g.index[merged[1][0]:]] = True
    return out


def segment_ids(df: pd.DataFrame, label_col: str, min_run: int = 30, window: int = 31) -> pd.Series:
    """Per-row segment index inside each track: a new segment starts where the smoothed
    per-frame class changes persistently (>= min_run frames). Lets labels change at a
    suspected identity switch while staying constant between such points."""
    names = sorted({v for v in df[label_col].unique() if v})
    code = {n: i for i, n in enumerate(names)}
    seg = pd.Series(0, index=df.index)
    for _, g in df.sort_values("frame_index").groupby("track_id", sort=False):
        lab = np.array([code.get(v, -1) for v in g[label_col]])
        sm = _rolling_mode(lab, window)
        starts = [0] + [i for i in range(1, len(sm)) if sm[i] != sm[i - 1]]
        runs = [(s, (starts[k + 1] if k + 1 < len(starts) else len(sm))) for k, s in enumerate(starts)]
        runs = [r for r in runs if r[1] - r[0] >= min_run and sm[r[0]] >= 0]
        merged = []
        for r in runs:
            if merged and sm[merged[-1][0]] == sm[r[0]]:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        ids = np.zeros(len(g), int)
        for k, (st, _) in enumerate(merged[1:], start=1):
            ids[st:] = k
        seg.loc[g.index] = ids
    return seg
