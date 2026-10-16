"""Benchmark tracker variants on cached detections, one change at a time.

Each variant re-runs stage B (tracking + export) into its own copy of the run directory
(`<run>_ablation/<variant>/`), so the baseline is never overwritten. Metrics:
  * if a reviewed GT export exists for the clip: full GT evaluation (TrackEval etc.)
  * always: proxies (on-court IDs, interior births, speed-violation rows, flagged
    cross-class switch rows) -- proxies are NOT accuracy.

Usage: python tools/tracker_ablation.py --runs outputs/real/clip2_ptz outputs/real/clip5_ptz
       [--gt clip2=data/annotations/clip2/gt_cvat_export.zip]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nbatrack.pipeline.video_pipeline import DEFAULT_TRACKER, TrackConfig, track_and_export

VARIANTS = {  # name: (tracker config, offline stitching)
    "V0_baseline": (DEFAULT_TRACKER, False),
    "V1_occlusion_aware_gate": (replace(DEFAULT_TRACKER, court_min_sigma_m=0.3, court_min_sigma_occluded_m=1.0), False),
    "V2_team_aware": (replace(DEFAULT_TRACKER, w_team=0.3, team_gate=35.0), False),
    "V3_longer_recovery": (replace(DEFAULT_TRACKER, max_time_lost_s=2.0), False),
    "S1_offline_stitch": (DEFAULT_TRACKER, True),
    "V1+S1": (replace(DEFAULT_TRACKER, court_min_sigma_m=0.3, court_min_sigma_occluded_m=1.0), True),
}


def proxies(run: Path) -> dict:
    p = pd.read_parquet(run / "tables" / "player_observations.parquet")
    onc = p[(p.role != "other") & p.bbox_x1.notna()]
    first = onc.sort_values("frame_index").groupby("track_id").first()
    w = json.loads((run / "run_analyze.json").read_text())["width"]
    interior = first[(first.frame_index > 5) & (first.bbox_x1 > 30) & (first.bbox_x2 < w - 30)]
    fl = onc["flags"].fillna("")
    return {"on_court_ids": int(onc.track_id.nunique()), "interior_births": int(len(interior)),
            "speed_violation_rows": int(fl.str.contains("speed_violation").sum()),
            "suspect_switch_rows": int(fl.str.contains("suspected_id_switch").sum()),
            "rows": int(len(onc))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gt", nargs="*", default=[], help="clipname=path pairs")
    ap.add_argument("--out", default="outputs/eval/tracker_ablation.csv")
    a = ap.parse_args()
    gts = dict(x.split("=", 1) for x in a.gt)
    rows = []
    for r in a.runs:
        base = Path(r)
        clip = base.name.split("_")[0]
        teams = Path("data/annotations") / clip / "teams.json"
        for name, (tc, stitch) in VARIANTS.items():
            dst = base.parent / f"{base.name}_ablation" / name
            if dst.exists():
                shutil.rmtree(dst)
            (dst / "tables").mkdir(parents=True)
            shutil.copytree(base / "cache", dst / "cache")
            for f in ("run_analyze.json",):
                shutil.copy(base / f, dst / f)
            for t in ("calibration", "frames", "videos", "camera_segments", "court_landmarks", "landmark_observations"):
                shutil.copy(base / "tables" / f"{t}.parquet", dst / "tables" / f"{t}.parquet")
            res = track_and_export(dst, TrackConfig(tracker=tc, stitch=stitch,
                                                    team_init=str(teams) if teams.exists() else None,
                                                    ball_detector="rfdetr"))  # player ablation; ball not compared
            row = {"clip": clip, "variant": name, **proxies(dst), "stitch_links": len(res["stitch_links"]),
                   "interpolated_rows": res["interpolated_rows"]}
            if clip in gts:
                from nbatrack.annot.cvat import parse_cvat_xml
                from nbatrack.eval.gt_eval import evaluate_run
                ev = evaluate_run(parse_cvat_xml(gts[clip]), dst)
                t = ev["tracking_trackeval"]
                row.update(gt_status=ev["ground_truth_status"], HOTA=t["HOTA"], AssA=t["AssA"], IDF1=t["IDF1"],
                           IDSW=t["IDSW"], MOTA=t["MOTA"],
                           team_acc=ev["classification"]["team_accuracy_best_permutation"])
            else:
                row.update(gt_status="no ground truth (proxies only)")
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
