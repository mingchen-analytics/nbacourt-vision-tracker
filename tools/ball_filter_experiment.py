"""Experiment: does the 'far floor projection' rule of remove_static_clutter (|y| > 12 m ->
dropped) delete airborne balls? Same WASB candidates, same players and calibration; only
max_abs_y_m differs (12.0 = baseline, None = rule off).

Per run: detected / interpolated frames for both variants, frames gained / lost, and a
contact sheet of up to 12 frames that only the variant has (for visual review).

  python tools/ball_filter_experiment.py outputs/real/clip1_ptz ... outputs/real/clip6_long_115s_diag
  -> outputs/experiments/ball_far_rule/<run>/sheet_gained.jpg, summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.detect.wasb_ball import to_detection_rows  # noqa: E402
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.track.ball import track_ball  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="outputs/experiments/ball_far_rule")
    a = ap.parse_args()
    summ = {}
    for r in a.runs:
        run = Path(r)
        t = run / "tables"
        ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
        frames = pd.read_parquet(t / "frames.parquet")
        cal = pd.read_parquet(t / "calibration.parquet")
        pl = pd.read_parquet(t / "player_observations.parquet")
        segs = pd.read_parquet(t / "camera_segments.parquet") if (t / "camera_segments.parquet").exists() else None
        cands = to_detection_rows(pd.read_parquet(run / "cache" / "wasb_ball_candidates.parquet"))
        if "camera_segment_id" not in frames:
            frames["camera_segment_id"] = 0
        sup = dict(zip(segs.camera_segment_id, segs.supported)) if segs is not None else {}
        res = {}
        for name, mx in (("baseline_far_rule", 12.0), ("no_far_rule", None)):
            parts = []
            for sid, fg in frames.groupby("camera_segment_id"):
                if not sup.get(sid, True):
                    continue
                fs = set(fg.frame_index)
                b = track_ball(cands[cands.frame_index.isin(fs)].reset_index(drop=True), fg, cal[cal.frame_index.isin(fs)].reset_index(drop=True),
                               pl[pl.frame_index.isin(fs)], ra["video_id"], max_abs_y_m=mx)
                parts.append(b)
            res[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        b0, b1 = res["baseline_far_rule"], res["no_far_rule"]
        d0 = set(b0[b0.image_source == "detected"].frame_index)
        d1 = set(b1[b1.image_source == "detected"].frame_index)
        n = len(frames)
        gained, lost = sorted(d1 - d0), sorted(d0 - d1)
        moved = 0
        both = sorted(d0 & d1)
        if both:
            x0 = b0.set_index("frame_index").loc[both, ["image_x", "image_y"]].to_numpy(float)
            x1 = b1.set_index("frame_index").loc[both, ["image_x", "image_y"]].to_numpy(float)
            moved = int((np.hypot(*(x0 - x1).T) > 15).sum())
        summ[run.name] = dict(n_frames=n, detected_baseline=len(d0) / n, detected_no_far_rule=len(d1) / n,
                              frames_gained=len(gained), frames_lost=len(lost), frames_moved_gt_15px=moved)
        print(run.name, summ[run.name], flush=True)
        od = ROOT / a.out / run.name
        od.mkdir(parents=True, exist_ok=True)
        if gained:
            pick = set(np.array(gained)[np.linspace(0, len(gained) - 1, min(12, len(gained))).round().astype(int)].tolist())
            B = b1.set_index("frame_index")
            tiles = []
            for fr in iter_frames(ra["source"]):
                if fr.frame_index > max(pick):
                    break
                if fr.frame_index not in pick:
                    continue
                x, y = int(B.at[fr.frame_index, "image_x"]), int(B.at[fr.frame_index, "image_y"])
                im = fr.image.copy()
                cv2.circle(im, (x, y), 18, (0, 255, 0), 2)
                pad = cv2.copyMakeBorder(im, 120, 120, 120, 120, cv2.BORDER_CONSTANT)
                crop = cv2.resize(pad[y:y + 240, x:x + 240], (240, 240))
                small = cv2.resize(im, (427, 240))
                tile = np.hstack([small, crop])
                cv2.putText(tile, f"f{fr.frame_index}", (6, 20), 0, 0.6, (255, 255, 255), 2)
                tiles.append(tile)
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(od / "sheet_gained.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    out = ROOT / a.out / "summary.json"
    out.write_text(json.dumps(summ, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
