"""Thin bridge to the reference TrackEval implementation (HOTA, CLEAR, Identity).

TrackEval (MIT, github.com/JonathonLuiten/TrackEval) still uses numpy aliases removed in
numpy 2 (np.float, np.int, np.bool). The aliases are restored on the numpy module before
import; TrackEval's code itself is not modified. Sequence data is built directly in
TrackEval's preprocessed format (contiguous 0-based ids, IoU similarity per timestep), so
none of its dataset readers are used.
"""

from __future__ import annotations

import numpy as np

for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

from nbatrack.track.tracker import iou_matrix  # noqa: E402


def build_sequence(gt: dict[int, list[tuple]], pred: dict[int, list[tuple]], n_frames: int) -> dict:
    """gt / pred: frame -> list of (id, (x1, y1, x2, y2)). Frames are 0..n_frames-1."""
    gmap, pmap = {}, {}
    data = {"num_timesteps": n_frames, "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    n_gd = n_pd = 0
    for t in range(n_frames):
        g = gt.get(t, [])
        p = pred.get(t, [])
        gi = np.array([gmap.setdefault(i, len(gmap)) for i, _ in g], int)
        pi = np.array([pmap.setdefault(i, len(pmap)) for i, _ in p], int)
        sim = iou_matrix(np.array([b for _, b in g]).reshape(-1, 4), np.array([b for _, b in p]).reshape(-1, 4))
        data["gt_ids"].append(gi)
        data["tracker_ids"].append(pi)
        data["similarity_scores"].append(sim)
        n_gd += len(gi)
        n_pd += len(pi)
    data.update(num_gt_ids=len(gmap), num_tracker_ids=len(pmap), num_gt_dets=n_gd, num_tracker_dets=n_pd)
    return data


def evaluate(gt: dict, pred: dict, n_frames: int) -> dict:
    data = build_sequence(gt, pred, n_frames)
    h = HOTA({"PRINT_CONFIG": False}).eval_sequence(data)
    c = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    i = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(h["HOTA"])), "DetA": float(np.mean(h["DetA"])),
        "AssA": float(np.mean(h["AssA"])), "LocA": float(np.mean(h["LocA"])),
        "MOTA": float(c["MOTA"]), "MOTP": float(c["MOTP"]), "IDSW": int(c["IDSW"]),
        "Frag": int(c["Frag"]), "MT": int(c["MT"]), "ML": int(c["ML"]),
        "CLR_TP": int(c["CLR_TP"]), "CLR_FN": int(c["CLR_FN"]), "CLR_FP": int(c["CLR_FP"]),
        "IDF1": float(i["IDF1"]), "IDP": float(i["IDP"]), "IDR": float(i["IDR"]),
        "num_gt_ids": data["num_gt_ids"], "num_pred_ids": data["num_tracker_ids"],
        "num_gt_dets": data["num_gt_dets"], "num_pred_dets": data["num_tracker_dets"],
    }
