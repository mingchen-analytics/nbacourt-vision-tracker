"""Evaluate one or more pipeline runs against a reviewed CVAT ground-truth export.

Usage:
  python tools/evaluate_against_gt.py --gt data/annotations/clip2/gt_cvat_export.zip \
      --runs outputs/real/clip2_ptz outputs/real/clip2_variantX --out outputs/eval/clip2

Refuses to run on annotations with unreviewed tracks unless --allow-unreviewed is given,
in which case every output is stamped NOT GROUND TRUTH (for tooling checks only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nbatrack.annot.cvat import parse_cvat_xml
from nbatrack.eval.gt_eval import evaluate_run

ROWS = [("tracking_trackeval", k) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag", "num_pred_ids", "num_gt_ids")] + [
    ("detection_raw_score_ge_0.5", "precision"), ("detection_raw_score_ge_0.5", "recall"),
    ("detection_recall_any_score_ge_0.1", None),
    ("miss_decomposition", "missed_detector_never_fired"),
    ("miss_decomposition", "missed_detected_but_dropped_by_tracking_or_filters"),
    ("classification", "player_vs_referee_accuracy_on_matched_known"),
    ("classification", "team_accuracy_best_permutation"), ("classification", "role_unknown_frac"),
    ("coverage", "gt_boxes_with_matched_output_and_court_xy")]


def _get(d, a, b):
    v = d.get(a)
    return v if b is None else (v or {}).get(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unreviewed", action="store_true")
    ap.add_argument("--interval", help="interval.json of an interval CVAT package (long videos)")
    a = ap.parse_args()
    gt = parse_cvat_xml(a.gt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    interval = json.loads(Path(a.interval).read_text()) if a.interval else None
    results = {}
    for r in a.runs:
        res = evaluate_run(gt, r, allow_unreviewed=a.allow_unreviewed, interval=interval)
        results[Path(r).name] = res
        (out / f"eval_{Path(r).name}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    status = next(iter(results.values()))["ground_truth_status"]
    md = [f"# Evaluation vs {Path(a.gt).name}", "", f"**Ground-truth status: {status}**", "",
          "| metric | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for sec, key in ROWS:
        vals = []
        for res in results.values():
            v = _get(res, sec, key)
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        md.append(f"| {sec}.{key or ''} | " + " | ".join(vals) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
