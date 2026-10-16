"""Multi-object tracking metrics: CLEAR-MOT (MOTA, ID switches) and IDF1.

Implemented directly (small, auditable) following Bernardin & Stiefelhagen 2008 and
Ristani et al. 2016. HOTA is not implemented here; use TrackEval for it on real data.

Input: dicts frame -> list of (id, bbox x1y1x2y2) for ground truth and predictions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from nbatrack.track.tracker import iou_matrix


@dataclass
class MOTResult:
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    id_switches: int
    mota: float
    idtp: int
    idf1: float
    precision: float
    recall: float
    frag: int  # GT trajectory interruptions (tracked -> untracked -> tracked)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate(gt: dict[int, list[tuple]], pred: dict[int, list[tuple]], iou_thresh: float = 0.5) -> MOTResult:
    frames = sorted(set(gt) | set(pred))
    tp = fp = fn = ids = frag = 0
    last_match: dict = {}  # gt id -> pred id (last matched)
    tracked_prev: dict = {}  # gt id -> was matched in its previous appearance
    pair_counts: dict = defaultdict(int)
    gt_counts: dict = defaultdict(int)
    pred_counts: dict = defaultdict(int)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        for gid, _ in g:
            gt_counts[gid] += 1
        for pid, _ in p:
            pred_counts[pid] += 1
        if g and p:
            iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in p]))
            # IDF1 co-occurrence: every (gt, pred) pair overlapping in this frame
            for i, j in zip(*np.nonzero(iou >= iou_thresh)):
                pair_counts[(g[i][0], p[j][0])] += 1
            # CLEAR: keep previous correspondences if still valid, then Hungarian
            cost = np.where(iou >= iou_thresh, 1 - iou, 1e6)
            for i, (gid, _) in enumerate(g):
                if gid in last_match:
                    for j, (pid, _) in enumerate(p):
                        if pid == last_match[gid] and iou[i, j] >= iou_thresh:
                            cost[i, j] = -1.0  # prefer continuing correspondence
            r, c = linear_sum_assignment(cost)
            matched = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thresh]
        else:
            matched = []
        mg = {i for i, _ in matched}
        mp = {j for _, j in matched}
        tp += len(matched)
        fn += len(g) - len(mg)
        fp += len(p) - len(mp)
        for i, j in matched:
            gid, pid = g[i][0], p[j][0]
            if gid in last_match and last_match[gid] != pid:
                ids += 1
            last_match[gid] = pid
        for i, (gid, _) in enumerate(g):
            now = i in mg
            if now and tracked_prev.get(gid) is False and gid in last_match:
                frag += 1
            tracked_prev[gid] = now
    n_gt = sum(gt_counts.values())
    n_pred = sum(pred_counts.values())
    # IDF1: one-to-one mapping between GT and predicted identities maximizing IDTP
    gids = sorted(gt_counts)
    pids = sorted(pred_counts)
    idtp = 0
    if gids and pids:
        W = np.zeros((len(gids), len(pids)))
        gi = {g: k for k, g in enumerate(gids)}
        pi = {p: k for k, p in enumerate(pids)}
        for (g, p), n in pair_counts.items():
            W[gi[g], pi[p]] = n
        r, c = linear_sum_assignment(-W)
        idtp = int(W[r, c].sum())
    idf1 = 2 * idtp / max(1, n_gt + n_pred)
    mota = 1 - (fn + fp + ids) / max(1, n_gt)
    return MOTResult(n_gt, n_pred, tp, fp, fn, ids, mota, idtp, idf1,
                     tp / max(1, tp + fp), tp / max(1, tp + fn), frag)
