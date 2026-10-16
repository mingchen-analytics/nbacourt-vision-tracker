"""Validation of the calibration grading on NATURALLY occurring wrong solutions.

Wrong poses are produced the way they occurred on 2026-10-15: an automatic re-initialization
(same camera centre as a known pose; exhaustive pan/tilt/focal grid; top candidates refined
by line alignment). Each refined candidate is labelled by comparing it with the run's own
calibration of that frame (the reference; its correctness is checked separately on overlay
sheets): median distance of the visible court template between the two cameras
  < 8 px -> correct,  > 30 px -> wrong,  otherwise ambiguous (excluded).
For every candidate the grading inputs (line inliers, floor IoU, court coverage) are
computed exactly as in stage A, and grading rules are evaluated on them.

Frames: every --step-th frame of the supported segments, EXCLUDING the development frames
(0, 2400, 4000, 6091-6100, used to design the grading / tests) and frames within 3 of a flash.
The frame list is fixed by --step before any result is seen.

  python tools/calib_natural_failures.py outputs/real/clip6_long_115s --step 400
  -> outputs/experiments/calib_natural/<run>/candidates.csv, summary.json
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
    floor_agreement,
    floor_tone_mask,
    floor_valid_mask,
    refine_camera,
    template_points,
)
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.pipeline.video_pipeline import camera_from_row  # noqa: E402

DEV_FRAMES = {0, 2400, 4000} | set(range(6091, 6101))


def candidates(img, prior: PinholeCamera, overlay, tpl, scoring: str, top_k: int = 8):
    """Grid over pan/tilt/focal (roll 0) around `prior` with prior.C fixed, scored by
    `scoring` ("lines": inlier count on markings anywhere; "floor": floor IoU), then the top_k
    are refined by line alignment with C fixed. Returns refined cameras."""
    h, w = img.shape[:2]
    tone = floor_tone_mask(img)
    if scoring == "lines":
        valid = np.full((h, w), 255, np.uint8)
        if overlay is not None:
            valid[overlay > 0] = 0
        ev_all = build_evidence(img, valid)
    pan0, tilt0, _ = prior.ptz
    grid = []
    for dp in np.arange(-45.0, 45.1, 3.0):
        for dt in (-4.0, -2.0, 0.0, 2.0, 4.0):
            for fs in (0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0):
                c = PinholeCamera(prior.f * fs, prior.cx, prior.cy, ptz_rotation(pan0 + dp, tilt0 + dt, 0.0),
                                  prior.C.copy(), prior.k1, prior.image_size)
                if scoring == "lines":
                    n, frac, _ = alignment_stats(c, ev_all, tpl)
                    s = n * frac
                else:
                    s = floor_agreement(c, tone, overlay)["iou"]
                grid.append((s, c))
    grid.sort(key=lambda g: -g[0])
    out = []
    for _, c in grid[:top_k]:
        ev = build_evidence(img, floor_valid_mask(c, (h, w), 1.5, None, overlay))
        r = refine_camera(c, ev, tpl, fixed_C=prior.C, scales=(12.0, 5.0, 2.0))
        out.append(r.camera)
    return out


def grading_inputs(cam, img, overlay, tpl, tone):
    h, w = img.shape[:2]
    ev = build_evidence(img, floor_valid_mask(cam, (h, w), 1.0, None, overlay))
    n, inl, _ = alignment_stats(cam, ev, tpl)
    fa = floor_agreement(cam, tone, overlay)
    return n, inl, fa["iou"], fa["court_coverage"]


def pose_distance_px(a: PinholeCamera, b: PinholeCamera, tpl, size) -> float:
    w, h = size
    ua, za = a.project(tpl)
    ub, zb = b.project(tpl)
    ok = (za > 0) & (zb > 0) & np.all(np.isfinite(ua), 1) & np.all(np.isfinite(ub), 1)
    inb = ok & (((ub[:, 0] >= 0) & (ub[:, 0] < w) & (ub[:, 1] >= 0) & (ub[:, 1] < h))
                | ((ua[:, 0] >= 0) & (ua[:, 0] < w) & (ua[:, 1] >= 0) & (ua[:, 1] < h)))
    if inb.sum() < 50:
        return float("inf")
    return float(np.median(np.linalg.norm(np.clip(ua[inb], -5 * w, 5 * w) - np.clip(ub[inb], -5 * w, 5 * w), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=400)
    a = ap.parse_args()
    run = Path(a.run)
    ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
    size = (ra["width"], ra["height"])
    cal = pd.read_parquet(run / "tables" / "calibration.parquet").set_index("frame_index")
    segs = pd.read_parquet(run / "tables" / "camera_segments.parquet")
    flashes = set(ra.get("flash_frames", []))
    overlay = cv2.imread(str(run / "overlay_mask.png"), 0)
    tpl = template_points()
    frames = []
    for s in segs[segs.supported].itertuples():
        for f in range(s.start_frame + a.step // 2, s.end_frame + 1, a.step):
            if f in DEV_FRAMES or any(abs(f - x) <= 3 for x in flashes):
                continue
            frames.append(f)
    print("frames", frames, flush=True)
    out_dir = ROOT / "outputs" / "experiments" / "calib_natural" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    want = set(frames)
    for fr in iter_frames(ra["source"]):
        if fr.frame_index > max(want):
            break
        if fr.frame_index not in want:
            continue
        ref = camera_from_row(cal.loc[fr.frame_index], size)
        tone = floor_tone_mask(fr.image)
        n, inl, iou, cov = grading_inputs(ref, fr.image, overlay, tpl, tone)
        rows.append(dict(frame=fr.frame_index, source="reference", dist_px=0.0, label="reference",
                         n=n, inlier=inl, floor_iou=iou, court_coverage=cov, stored_quality=cal.loc[fr.frame_index, "quality"]))
        # the prior must not BE the answer: perturb it (another moment of the same camera)
        prior = PinholeCamera(ref.f * 1.25, ref.cx, ref.cy, ptz_rotation(ref.ptz[0] + 20.0, ref.ptz[1] + 1.0, 0.0),
                              ref.C.copy(), ref.k1, ref.image_size)
        for scoring in ("lines", "floor"):
            for k, c in enumerate(candidates(fr.image, prior, overlay, tpl, scoring)):
                d = pose_distance_px(c, ref, tpl, size)
                lab = "correct" if d < 8 else "wrong" if d > 30 else "ambiguous"
                n, inl, iou, cov = grading_inputs(c, fr.image, overlay, tpl, tone)
                rows.append(dict(frame=fr.frame_index, source=f"auto_{scoring}_{k}", dist_px=d, label=lab,
                                 n=n, inlier=inl, floor_iou=iou, court_coverage=cov, stored_quality=None,
                                 pan=c.ptz[0], tilt=c.ptz[1], roll=c.ptz[2], f=c.f))
        print(fr.frame_index, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidates.csv", index=False)
    lab = df[df.label.isin(["correct", "wrong", "reference"])]
    rules = {
        "old: inlier>=0.55": lambda x: (x.inlier >= 0.55) & (x.n >= 150),
        "inlier>=0.70": lambda x: (x.inlier >= 0.70) & (x.n >= 150),
        "inlier>=0.70 & iou>=0.45 (current)": lambda x: (x.inlier >= 0.70) & (x.floor_iou >= 0.45) & (x.n >= 150),
        "inlier>=0.70 & coverage>=0.6": lambda x: (x.inlier >= 0.70) & (x.court_coverage >= 0.6) & (x.n >= 150),
        "inlier>=0.70 & (iou>=0.45 | coverage>=0.75)": lambda x: (x.inlier >= 0.70) & ((x.floor_iou >= 0.45) | (x.court_coverage >= 0.75)) & (x.n >= 150),
    }
    summ = {"frames": frames, "n_candidates": int((df.source != "reference").sum()),
            "labels": df.label.value_counts().to_dict(),
            "note": "unseen frames (development frames excluded); wrong solutions produced by an automatic re-init search"}
    for name, rule in rules.items():
        acc = rule(lab)
        summ[name] = {"wrong_accepted": f"{int(acc[lab.label == 'wrong'].sum())}/{int((lab.label == 'wrong').sum())}",
                      "correct_candidates_accepted": f"{int(acc[lab.label == 'correct'].sum())}/{int((lab.label == 'correct').sum())}",
                      "references_accepted": f"{int(acc[lab.label == 'reference'].sum())}/{int((lab.label == 'reference').sum())}"}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
