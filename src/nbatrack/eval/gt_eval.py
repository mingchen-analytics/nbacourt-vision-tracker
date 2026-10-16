"""Evaluate a pipeline run against reviewed ground truth (CVAT export).

Conventions (documented in docs/ANNOTATION_GUIDE.md and docs/EVALUATION.md):
  * Frames: CVAT task frame == our frame_index (0-based, presentation order).
  * Evaluated objects: GT `player` and `referee` boxes that are not `outside`.
    Occluded boxes are included (they are visible people) and also reported separately.
  * `other_person` GT boxes are ignore regions: predictions overlapping them (IoU >= 0.5)
    are removed before matching, so bench / staff are neither TP nor FP.
  * Matching: IoU >= 0.5 (CLEAR / Identity); HOTA integrates over IoU 0.05..0.95.
  * Predictions: pipeline output rows whose role != "other".

Ground-truth status: unless every annotated track has review == "reviewed", results are
labeled NOT GROUND TRUTH and must not be reported as accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nbatrack.annot.cvat import GroundTruth, check_frame_count
from nbatrack.court.geometry import court_polylines, densify, landmark_by_name
from nbatrack.eval import trackeval_bridge
from nbatrack.pipeline.video_pipeline import camera_from_row
from nbatrack.track.tracker import iou_matrix

ARC_LANDMARK_CURVES = {"ft_circle_top": "ft_circle", "3pt_top": "3pt", "center_circle": "center_circle"}


def _frames_dict(df, id_col, cols=("x1", "y1", "x2", "y2")) -> dict:
    out: dict = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r.frame), []).append((getattr(r, id_col), tuple(getattr(r, c) for c in cols)))
    return out


def _remove_ignored(pred: pd.DataFrame, ignore: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    if not len(ignore) or not len(pred):
        return pred
    keep = np.ones(len(pred), bool)
    ig = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in ignore.groupby("frame")}
    for k, r in enumerate(pred.itertuples()):
        if r.frame in ig and iou_matrix(np.array([[r.x1, r.y1, r.x2, r.y2]]), ig[r.frame]).max() >= thr:
            keep[k] = False
    return pred[keep]


def _per_frame_matches(gt: pd.DataFrame, pr: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Hungarian IoU matching per frame. Returns rows (frame, gt_index, pred_index, iou)."""
    rows = []
    pf = dict(tuple(pr.groupby("frame")))
    for f, g in gt.groupby("frame"):
        p = pf.get(f)
        if p is None:
            continue
        m = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(float), p[["x1", "y1", "x2", "y2"]].to_numpy(float))
        r, c = linear_sum_assignment(-m)
        for i, j in zip(r, c):
            if m[i, j] >= thr:
                rows.append((f, g.index[i], p.index[j], m[i, j]))
    return pd.DataFrame(rows, columns=["frame", "gi", "pi", "iou"])


def detection_pr(gt: pd.DataFrame, det: pd.DataFrame, thr_iou: float = 0.5) -> dict:
    m = _per_frame_matches(gt, det, thr_iou)
    tp = len(m)
    return {"precision": tp / max(1, len(det)), "recall": tp / max(1, len(gt)), "tp": tp,
            "fp": len(det) - tp, "fn": len(gt) - tp}


def _best_team_mapping(pairs: list[tuple[str, str]]) -> tuple[float, dict, int]:
    """Accuracy of predicted team labels vs GT under the best A/B permutation."""
    valid = [(g, p) for g, p in pairs if g in ("A", "B") and p in ("A", "B")]
    if not valid:
        return float("nan"), {}, 0
    same = sum(g == p for g, p in valid)
    if same >= len(valid) - same:
        return same / len(valid), {"A": "A", "B": "B"}, len(valid)
    return (len(valid) - same) / len(valid), {"A": "B", "B": "A"}, len(valid)


def _landmark_errors(gt: GroundTruth, cal: pd.DataFrame, size) -> pd.DataFrame:
    by = landmark_by_name()
    lines = court_polylines()
    rows = []
    for r in gt.landmarks.itertuples():
        if r.name not in by or r.frame not in set(cal.frame_index):
            continue
        cam = camera_from_row(cal[cal.frame_index == r.frame].iloc[0], size)
        lm = by[r.name]
        curve = next((v for k, v in ARC_LANDMARK_CURVES.items() if k in r.name), None)
        if curve is not None:
            key = (r.name[:2] + curve) if r.name[:2] in ("L_", "R_") else curve
            q = cam.project(densify(lines[key], 0.01))[0]
            err = float(np.nanmin(np.linalg.norm(q - [r.x, r.y], axis=1)))
            kind = "point_to_curve"
        else:
            err = float(np.linalg.norm(cam.project(np.array([lm.xyz]))[0][0] - [r.x, r.y]))
            kind = "point"
        rows.append(dict(frame=r.frame, name=r.name, kind=kind, err_px=err))
    return pd.DataFrame(rows, columns=["frame", "name", "kind", "err_px"])


def _foot_eval(gt: GroundTruth, gtb: pd.DataFrame, pr: pd.DataFrame, cal: pd.DataFrame, size) -> dict:
    """Image-space bbox-bottom error vs annotated foot contact (independent of camera),
    plus its effect in court meters through the SAME estimated camera (a component, not
    absolute accuracy)."""
    rows = []
    for r in gt.feet.itertuples():
        if r.state != "planted":
            continue
        g = gtb[gtb.frame == r.frame]
        if not len(g):
            continue
        inside = (g.x1 - 0.1 * (g.x2 - g.x1) <= r.x) & (r.x <= g.x2 + 0.1 * (g.x2 - g.x1)) & \
                 (g.y1 + 0.5 * (g.y2 - g.y1) <= r.y) & (r.y <= g.y2 + 0.15 * (g.y2 - g.y1))
        g = g[inside]
        if not len(g):
            continue
        gi = (np.hypot((g.x1 + g.x2) / 2 - r.x, g.y2 - r.y)).idxmin()
        gb = gtb.loc[gi]
        p = pr[pr.frame == r.frame]
        pb = None
        if len(p):
            m = iou_matrix(np.array([[gb.x1, gb.y1, gb.x2, gb.y2]]), p[["x1", "y1", "x2", "y2"]].to_numpy(float))[0]
            if m.max() >= 0.5:
                pb = p.iloc[int(m.argmax())]
        h = gb.y2 - gb.y1
        row = dict(frame=r.frame, foot_x=r.x, foot_y=r.y, box_h=h, occluded=bool(gb.occluded))
        if pb is not None:
            bx, byy = (pb.x1 + pb.x2) / 2, pb.y2
            row.update(err_px=float(np.hypot(bx - r.x, byy - r.y)), err_y_px=float(byy - r.y),
                       err_x_px=float(bx - r.x))
            c = cal[cal.frame_index == r.frame]
            if len(c) and c.iloc[0].quality in ("good", "fair"):
                cam = camera_from_row(c.iloc[0], size)
                a = cam.backproject_to_plane(np.array([[bx, byy], [r.x, r.y]]))[0]
                row["court_component_m"] = float(np.linalg.norm(a[0] - a[1]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df) or "err_px" not in df:
        return {"n": int(len(df)), "status": "no matched foot annotations"}
    e = df.err_px.dropna()
    out = {"n_annotated_planted": int(len(df)), "n_matched": int(len(e)),
           "bbox_bottom_err_px": {"median": float(e.median()), "p90": float(e.quantile(0.9))},
           "bbox_bottom_err_rel_box_height": {"median": float((df.err_px / df.box_h).median())},
           "vertical_bias_px_median": float(df.err_y_px.median()),
           "occluded_vs_not_median_px": {
               "occluded": float(df[df.occluded].err_px.median()) if df.occluded.any() else None,
               "not_occluded": float(df[~df.occluded].err_px.median()) if (~df.occluded).any() else None}}
    if "court_component_m" in df:
        cc = df.court_component_m.dropna()
        out["court_component_m_through_estimated_camera"] = {
            "median": float(cc.median()), "p90": float(cc.quantile(0.9)),
            "note": "ground-contact error only, projected through the estimated camera; NOT absolute accuracy"}
    return out


def evaluate_run(gt: GroundTruth, run_dir: str | Path, allow_unreviewed: bool = False,
                 interval: dict | None = None) -> dict:
    """`interval` (interval.json written by make_cvat_package --start-frame): the GT covers
    source frames start_frame..end_frame, annotated as CVAT frames 0..n-1; run outputs are
    restricted to that interval and renumbered before matching."""
    run = Path(run_dir)
    ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
    size = (ra["width"], ra["height"])
    n = ra["n_frames"]
    off = 0
    if interval is not None:
        off, n = int(interval["start_frame"]), int(interval["n_frames"])

    def window(df, col):
        if interval is None:
            return df
        df = df[df[col].between(off, off + n - 1)].copy()
        df[col] -= off
        return df
    status = "reviewed ground truth" if gt.fully_reviewed else \
        f"NOT GROUND TRUTH ({gt.n_tracks_unreviewed} tracks unreviewed)"
    if not gt.fully_reviewed and not allow_unreviewed:
        raise ValueError(f"annotations are not fully reviewed: {gt.n_tracks_unreviewed} tracks still "
                         "have review=unreviewed (use allow_unreviewed only for tooling checks)")
    issues = check_frame_count(gt, n)
    gtb = gt.boxes[gt.boxes.label.isin(["player", "referee"])].copy()
    ign = gt.boxes[gt.boxes.label == "other_person"]
    pl = window(pd.read_parquet(run / "tables" / "player_observations.parquet"), "frame_index")
    pr = pl[(pl.role != "other") & pl.bbox_x1.notna()].rename(columns={"frame_index": "frame", "bbox_x1": "x1", "bbox_y1": "y1",
                                                "bbox_x2": "x2", "bbox_y2": "y2"})
    pr = _remove_ignored(pr.reset_index(drop=True), ign)
    det = window(pd.read_parquet(run / "cache" / "detections.parquet"), "frame_index").rename(columns={"frame_index": "frame"})
    det = _remove_ignored(det[det.cls == 1].reset_index(drop=True), ign)
    cal = window(pd.read_parquet(run / "tables" / "calibration.parquet"), "frame_index")

    out: dict = {"ground_truth_status": status, "frame_alignment_issues": issues,
                 "n_frames": n, "n_gt_boxes": int(len(gtb)), "n_gt_ids": int(gtb.gt_id.nunique())}
    # ---- tracking (reference implementation)
    out["tracking_trackeval"] = trackeval_bridge.evaluate(_frames_dict(gtb, "gt_id"),
                                                          _frames_dict(pr, "track_id"), n)
    # ---- detection P/R (raw detector, score >= 0.5) and recall at the low threshold
    hi = det[det.score >= 0.5]
    out["detection_raw_score_ge_0.5"] = detection_pr(gtb, hi)
    out["detection_recall_any_score_ge_0.1"] = detection_pr(gtb, det)["recall"]
    out["detection_note"] = ("raw-detector precision counts unannotated spectators in the stands as FP "
                             "unless they were labeled other_person; tracked-output precision is in "
                             "tracking_trackeval (CLR_FP)")
    # ---- failure decomposition per GT box
    m_out = _per_frame_matches(gtb, pr)
    m_det = _per_frame_matches(gtb, det)
    matched_out = set(m_out.gi)
    matched_det = set(m_det.gi)
    out["miss_decomposition"] = {
        "gt_boxes": int(len(gtb)),
        "covered_by_output": int(len(matched_out)),
        "missed_detector_never_fired": int(len(set(gtb.index) - matched_det)),
        "missed_detected_but_dropped_by_tracking_or_filters": int(len(matched_det - matched_out)),
        "occluded_gt_boxes": int(gtb.occluded.sum()),
        "occluded_recall_of_output": float(len(matched_out & set(gtb[gtb.occluded].index)) / max(1, gtb.occluded.sum())),
    }
    # ---- classification on matched boxes
    j = m_out.merge(gtb[["label", "team"]], left_on="gi", right_index=True).merge(
        pr[["role", "team_id", "court_x_m"]], left_on="pi", right_index=True)
    role_ok = ((j.label == "player") & (j.role == "player")) | ((j.label == "referee") & (j.role == "referee"))
    known = j.role.isin(["player", "referee"])
    acc, mapping, nt = _best_team_mapping(list(zip(j[j.label == "player"].team, j[j.label == "player"].team_id)))
    out["classification"] = {
        "player_vs_referee_accuracy_on_matched_known": float(role_ok[known].mean()) if known.any() else None,
        "role_unknown_frac": float((~known).mean()) if len(j) else None,
        "team_accuracy_best_permutation": acc, "team_mapping_pred_to_gt": mapping,
        "team_evaluated_boxes": nt,
        "team_unknown_frac_on_gt_players": float(j[j.label == "player"].team_id.isna().mean()) if len(j) else None,
        "note": "computed per matched box-frame (IoU >= 0.5)",
    }
    out["coverage"] = {
        "gt_boxes_with_matched_output_and_court_xy": float(j.court_x_m.notna().sum() / max(1, len(gtb))),
        "gt_boxes_with_matched_output": float(len(matched_out) / max(1, len(gtb))),
    }
    out["fragmentation"] = {
        "trackeval_Frag": out["tracking_trackeval"]["Frag"],
        "mean_pred_ids_per_gt_id": float(m_out.merge(pr[["track_id"]], left_on="pi", right_index=True)
                                         .merge(gtb[["gt_id"]], left_on="gi", right_index=True)
                                         .groupby("gt_id").track_id.nunique().mean()) if len(m_out) else None,
    }
    out["ground_contact"] = _foot_eval(gt, gtb, pr, cal, size) if len(gt.feet) else {"status": "no foot annotations"}
    le = _landmark_errors(gt, cal, size)
    out["calibration_landmarks"] = ({"n": int(len(le)), "median_px": float(le.err_px.median()),
                                     "p90_px": float(le.err_px.quantile(0.9)), "max_px": float(le.err_px.max()),
                                     "by_frame": le.groupby("frame").err_px.median().round(2).to_dict()}
                                    if len(le) else {"status": "no landmark annotations"})
    return out
