"""Experiment: ball candidates from RF-DETR `sports ball` vs WASB (nttcom/WASB-SBDT), each
fed into the SAME offline tracker (nbatrack.track.ball.track_ball: static-clutter filter,
Viterbi, short-gap interpolation, possession proxy).

Isolated: reads the baseline run in outputs/real/<clip>_ptz (calibration, players, cached
RF-DETR detections) and writes only to --out. Baseline tables are not modified.

No ball ground truth exists yet, so every number is a PROXY:
  detected / interpolated / missing  share of frames by image_source
  held_proxy        share of frames with position_3d_status = possession_proxy
  static_not_held   detected, not held, and other selected unheld frames (>= 5 frames
                    away, anywhere in the clip) put the ball at the same floor point
                    (< 0.3 m) for >= 0.5 s in total -> a world-static object (floor
                    paint, logo, rim, object in the stands); a real ball rarely revisits
                    the same unheld spot that long
  longest_missing_s longest run without a detected or interpolated ball
  agreement         frames where two variants both detect: share within 15 px
Visual contact sheets of disagreement frames are written for manual review.

Usage:
  python tools/ball_detector_comparison.py --clips clip1 clip2 clip3 clip4 clip5 --provider dml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.calib.homography import apply_h  # noqa: E402
from nbatrack.detect.wasb_ball import WASBBallDetector, detect_video, to_detection_rows  # noqa: E402
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.track.ball import track_ball  # noqa: E402


def load_run(clip: str):
    d = ROOT / "outputs" / "real" / f"{clip}_ptz"
    t = d / "tables"
    return dict(dir=d,
                video=pd.read_parquet(t / "videos.parquet").iloc[0],
                frames=pd.read_parquet(t / "frames.parquet"),
                cal=pd.read_parquet(t / "calibration.parquet"),
                players=pd.read_parquet(t / "player_observations.parquet"),
                dets=pd.read_parquet(d / "cache" / "detections.parquet"),
                ball_shipped=pd.read_parquet(t / "ball_observations.parquet"))


def wasb_candidates(run, out: Path, provider: str, step: int, redo: bool):
    f = out / f"wasb_candidates_step{step}.parquet"
    meta = out / f"wasb_candidates_step{step}.json"
    if f.exists() and meta.exists() and not redo:
        return pd.read_parquet(f), json.loads(meta.read_text())["ms_per_frame"]
    det = WASBBallDetector(provider=provider)
    c, ms = detect_video(run["video"].source_path, det, step=step)
    c.to_parquet(f)
    meta.write_text(json.dumps(dict(ms_per_frame=ms, provider=provider, step=step,
                                    threshold=det.thr, n_candidates=len(c))))
    return c, ms


def metrics(ball: pd.DataFrame, run, fps: float) -> dict:
    n = len(run["frames"])
    det = ball[ball.image_source == "detected"]
    held = ball.position_3d_status == "possession_proxy"
    Hinv = {}
    for r in run["cal"].itertuples():
        if r.model != "none" and r.H_img_to_court is not None:
            Hinv[r.frame_index] = np.asarray(r.H_img_to_court, float).reshape(3, 3)
    fl = {}
    for r in det.itertuples():
        if r.frame_index in Hinv and r.position_3d_status != "possession_proxy":
            fl[r.frame_index] = apply_h(Hinv[r.frame_index], np.array([[r.image_x, r.image_y]]))[0]
    # static revisit: other selected (unheld) frames, >= 5 frames away, anywhere in the clip,
    # at the same floor point (< 0.3 m) for >= 0.5 s in total -> world-static object
    fr_ = np.array(list(fl.keys()), int)
    P = np.array(list(fl.values()), float).reshape(-1, 2)
    static = 0
    for i in range(len(fr_)):
        d = np.hypot(*(P - P[i]).T)
        if ((d < 0.3) & (np.abs(fr_ - fr_[i]) >= 5)).sum() >= 0.5 * fps:
            static += 1
    have = np.zeros(n, bool)
    have[ball.frame_index.to_numpy(int)] = True
    longest, cur = 0, 0
    for h in have:
        cur = 0 if h else cur + 1
        longest = max(longest, cur)
    return dict(detected=len(det) / n, interpolated=(ball.image_source == "interpolated").sum() / n,
                missing=1 - have.mean(), held_proxy=held.sum() / n,
                static_not_held_frames=static, longest_missing_s=longest / fps)


def agreement(a: pd.DataFrame, b: pd.DataFrame, tol: float = 15.0) -> dict:
    a = a[a.image_source == "detected"].set_index("frame_index")
    b = b[b.image_source == "detected"].set_index("frame_index")
    common = a.index.intersection(b.index)
    if not len(common):
        return dict(both_detected=0, within_15px=None, median_px=None)
    d = np.hypot(a.loc[common, "image_x"] - b.loc[common, "image_x"], a.loc[common, "image_y"] - b.loc[common, "image_y"])
    return dict(both_detected=len(common), within_15px=float((d <= tol).mean()), median_px=float(d.median()))


def contact_sheet(run, variants: dict[str, pd.DataFrame], a: str, b: str, path: Path, n_tiles: int = 12,
                  audit_seed: int | None = None):
    """Default: frames where `a` and `b` disagree (one detects and the other not, or > 30 px
    apart), evenly sampled. With `audit_seed`: a uniform random sample of all frames where
    either variant has a DETECTED ball (unbiased spot check of precision).
    Circles: a = red, b = green; thick = detected, thin = interpolated."""
    A = variants[a].set_index("frame_index")
    B = variants[b].set_index("frame_index")
    frames = sorted(set(A.index) | set(B.index))
    dis = []
    for f in frames:
        ia, ib = f in A.index, f in B.index
        if audit_seed is not None:
            if (ia and A.at[f, "image_source"] == "detected") or (ib and B.at[f, "image_source"] == "detected"):
                dis.append(f)
        elif ia != ib:
            dis.append(f)
        elif np.hypot(A.at[f, "image_x"] - B.at[f, "image_x"], A.at[f, "image_y"] - B.at[f, "image_y"]) > 30:
            dis.append(f)
    if not dis:
        return 0
    if audit_seed is not None:
        pick = set(np.random.default_rng(audit_seed).choice(dis, min(n_tiles, len(dis)), replace=False).tolist())
    else:
        pick = set(np.array(dis)[np.linspace(0, len(dis) - 1, min(n_tiles, len(dis))).round().astype(int)].tolist())
    tiles = []
    for fr in iter_frames(run["video"].source_path):
        if fr.frame_index > max(pick):
            break
        if fr.frame_index not in pick:
            continue
        im = fr.image.copy()
        H, W = im.shape[:2]
        insets = []
        for V, col in ((A, (0, 0, 255)), (B, (0, 200, 0))):
            if fr.frame_index in V.index:
                r = V.loc[fr.frame_index]
                x, y = int(r.image_x), int(r.image_y)
                x0, y0 = min(max(0, x - 50), W - 100), min(max(0, y - 50), H - 100)
                crop = cv2.resize(fr.image[y0:y0 + 100, x0:x0 + 100], (240, 240), interpolation=cv2.INTER_CUBIC)
                cv2.circle(crop, ((x - x0) * 12 // 5, (y - y0) * 12 // 5), 34, col, 2)
                cv2.rectangle(crop, (0, 0), (239, 239), col, 6)
                insets.append(crop)
                th = -1 if r.image_source == "detected" else 2
                cv2.circle(im, (x, y), 22, col, 3 if th < 0 else 1)
        tile = cv2.resize(im, (640, 360))
        cv2.putText(tile, f"f{fr.frame_index} red={a} green={b}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        side = np.zeros((360, 180, 3), np.uint8)  # zoomed 100x100 px crops, red above green
        for i, c in enumerate(insets[:2]):
            side[i * 180:(i + 1) * 180] = cv2.resize(c, (180, 180))
        tiles.append(np.hstack([tile, side]))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(dis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["clip1", "clip2", "clip3", "clip4", "clip5"])
    ap.add_argument("--provider", default="dml", choices=["cpu", "dml"])
    ap.add_argument("--out", default="outputs/experiments/ball_detector_comparison")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    out_root = ROOT / args.out
    summary = []
    for clip in args.clips:
        run = load_run(clip)
        out = out_root / clip
        out.mkdir(parents=True, exist_ok=True)
        fps = float(run["video"].fps_nominal)
        vid = run["video"].video_id
        c1, ms1 = wasb_candidates(run, out, args.provider, 1, args.redo)
        c3, ms3 = wasb_candidates(run, out, args.provider, 3, args.redo)
        rf = run["dets"]
        cand = {
            "rfdetr": rf,
            "wasb_s1": to_detection_rows(c1),
            "wasb_s1_p50": to_detection_rows(c1[c1.peak >= 0.5]),
            "wasb_s3": to_detection_rows(c3),
        }
        cand["fused"] = pd.concat([rf[rf.cls == 37][["frame_index", "cls", "score", "x1", "y1", "x2", "y2"]],
                                   cand["wasb_s1"]], ignore_index=True)
        variants = {}
        n = len(run["frames"])
        for name, d in cand.items():
            t0 = time.perf_counter()
            b = track_ball(d.reset_index(drop=True), run["frames"], run["cal"].reset_index(drop=True), run["players"], vid)
            if not len(b):
                b = pd.DataFrame(columns=run["ball_shipped"].columns)
            b.to_parquet(out / f"ball_{name}.parquet")
            variants[name] = b
            raw = d[(d.cls == 37) & (d.score >= 0.1)]
            m = dict(clip=clip, variant=name, n_frames=n, fps=fps,
                     raw_frames_with_candidate=raw.frame_index.nunique() / n,
                     raw_candidates_per_frame=len(raw) / n,
                     tracker_s=time.perf_counter() - t0, **metrics(b, run, fps))
            m["wasb_ms_per_frame"] = {"wasb_s1": ms1, "wasb_s1_p50": ms1, "wasb_s3": ms3, "fused": ms1}.get(name)
            summary.append(m)
        for other in ("wasb_s1", "wasb_s3", "fused"):
            ag = agreement(variants["rfdetr"], variants[other])
            summary.append(dict(clip=clip, variant=f"agree:rfdetr~{other}", **ag))
        ndis = contact_sheet(run, variants, "rfdetr", "wasb_s1", out / "disagreement_rfdetr_vs_wasb_s1.jpg")
        contact_sheet(run, variants, "rfdetr", "wasb_s1", out / "audit_random_rfdetr_vs_wasb_s1.jpg",
                      n_tiles=15, audit_seed=0)
        print(clip, "disagreement frames rfdetr vs wasb_s1:", ndis, flush=True)
    df = pd.DataFrame(summary)
    df.to_csv(out_root / "summary.csv", index=False)
    (out_root / "summary.json").write_text(df.to_json(orient="records", indent=1))
    cols = ["clip", "variant", "raw_frames_with_candidate", "detected", "interpolated", "missing", "held_proxy",
            "static_not_held_frames", "longest_missing_s", "wasb_ms_per_frame"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[~df.variant.str.startswith("agree")][cols].round(3).to_string(index=False))
        print(df[df.variant.str.startswith("agree")][["clip", "variant", "both_detected", "within_15px", "median_px"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
