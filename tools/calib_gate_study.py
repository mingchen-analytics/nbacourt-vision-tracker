"""How well do independent calibration checks separate good poses from wrong ones?

For every run: sample frames, compute for the STORED (accepted) camera
  floor_iou   projected court rectangle vs floor-tone pixels (calib/lines.floor_iou)
  inlier      line-alignment inlier fraction (the quality score)
and for PERTURBED copies of the same camera (wrong poses: tilt +-3/6 deg, pan +-8/15 deg,
focal x0.7 / x1.4), the same two numbers. A useful gate must keep the stored poses and
reject the perturbed ones; the perturbations include the kind of failure seen on clip 6 (court
drawn onto the stands: tilt too small / pan off).

Stored poses of clips 1-5 were checked visually (overlay sheets), not against ground truth.

Usage: python tools/calib_gate_study.py outputs/real/clip1_ptz ... --every 30 --out outputs/experiments/calib_gates
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

from nbatrack.calib.camera import PinholeCamera, ptz_rotation  # noqa: E402
from nbatrack.calib.lines import (  # noqa: E402
    alignment_stats,
    build_evidence,
    floor_iou,
    floor_tone_mask,
    floor_valid_mask,
    template_points,
)
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.pipeline.video_pipeline import camera_from_row  # noqa: E402

PERTURB = {"tilt-6": (0, -6, 1.0), "tilt-3": (0, -3, 1.0), "tilt+3": (0, 3, 1.0), "tilt+6": (0, 6, 1.0),
           "pan-15": (-15, 0, 1.0), "pan-8": (-8, 0, 1.0), "pan+8": (8, 0, 1.0), "pan+15": (15, 0, 1.0),
           "f0.7": (0, 0, 0.7), "f1.4": (0, 0, 1.4)}


def perturbed(cam: PinholeCamera, dp, dt, fs) -> PinholeCamera:
    p, t, r = cam.ptz
    return PinholeCamera(cam.f * fs, cam.cx, cam.cy, ptz_rotation(p + dp, t + dt, r), cam.C.copy(), cam.k1, cam.image_size)


def score(cam, img, tone, overlay, tpl):
    h, w = img.shape[:2]
    ev = build_evidence(img, floor_valid_mask(cam, (h, w), 1.0, None, overlay))
    n, inl, _ = alignment_stats(cam, ev, tpl)
    return floor_iou(cam, tone, overlay), inl, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--out", default="outputs/experiments/calib_gates")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tpl = template_points()
    rows = []
    for r in a.runs:
        run = Path(r)
        ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
        cal = pd.read_parquet(run / "tables" / "calibration.parquet").set_index("frame_index")
        ov_p = run / "overlay_mask.png"
        overlay = cv2.imread(str(ov_p), 0) if ov_p.exists() else None
        size = (ra["width"], ra["height"])
        for fr in iter_frames(ra["source"]):
            if fr.frame_index % a.every or fr.frame_index not in cal.index:
                continue
            c = cal.loc[fr.frame_index]
            if c.model == "none" or c.method == "held_flash":
                continue
            cam = camera_from_row(c, size)
            tone = floor_tone_mask(fr.image)
            iou, inl, n = score(cam, fr.image, tone, overlay, tpl)
            rows.append(dict(run=run.name, frame=fr.frame_index, pose="stored", quality=c.quality, floor_iou=iou, inlier=inl, n=n))
            for name, (dp, dt, fs) in PERTURB.items():
                iou, inl, n = score(perturbed(cam, dp, dt, fs), fr.image, tone, overlay, tpl)
                rows.append(dict(run=run.name, frame=fr.frame_index, pose=name, quality=None, floor_iou=iou, inlier=inl, n=n))
        print(run.name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gate_scores.csv", index=False)
    st = df[df.pose == "stored"]
    wr = df[df.pose != "stored"]
    summ = {"stored_floor_iou_by_run_p01_p05_p50": st.groupby("run").floor_iou.quantile([0.01, 0.05, 0.5]).unstack().round(3).to_dict("index"),
            "stored_inlier_by_run_p05_p50": st.groupby("run").inlier.quantile([0.05, 0.5]).unstack().round(3).to_dict("index"),
            "wrong_floor_iou_by_pose_p50_p95_max": wr.groupby("pose").floor_iou.quantile([0.5, 0.95, 1.0]).unstack().round(3).to_dict("index"),
            "wrong_inlier_by_pose_p50_p95_max": wr.groupby("pose").inlier.quantile([0.5, 0.95, 1.0]).unstack().round(3).to_dict("index")}
    for thr in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
        summ[f"iou_thr_{thr}"] = {"stored_rejected": float((st.floor_iou < thr).mean()),
                                  "wrong_accepted": float((wr.floor_iou >= thr).mean()),
                                  "wrong_accepted_by_pose": wr.assign(acc=wr.floor_iou >= thr).groupby("pose").acc.mean().round(3).to_dict()}
    # relative test: is the stored pose's IoU higher than every perturbation of the same frame?
    piv = df.pivot_table(index=["run", "frame"], columns="pose", values="floor_iou")
    best_wrong = piv.drop(columns="stored").max(axis=1)
    summ["stored_iou_minus_best_wrong_p05_p50"] = [float((piv.stored - best_wrong).quantile(0.05)),
                                                   float((piv.stored - best_wrong).quantile(0.5))]
    (out / "gate_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
