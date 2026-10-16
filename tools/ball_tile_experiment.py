"""Experiment: WASB on overlapping image tiles (higher effective resolution) vs the whole frame.

WASB sees the frame at 512x288 (1280x720 shrunk 2.5x). A distant-view ball of 6-8 px becomes
~3 px. Here the frame is also cut into an RxC grid of overlapping tiles; each tile is resized
to 512x288 (a 2x2 grid with 25% overlap shrinks only ~1.4x), WASB runs per tile on the same
sliding 3-frame windows, and tile candidates are mapped back and merged (within 8 px the
higher peak is kept).

Variants, all through the SAME tracker (nbatrack.track.ball.track_ball):
  full         production candidates (cache/wasb_ball_candidates.parquet of the run)
  tiles        tile candidates only
  full+tiles   union (duplicates within 8 px merged)
Outputs per run: summary (detected / interpolated / missing, gained / lost frames, ms/frame),
a contact sheet of frames gained by full+tiles, and a random sheet of its detected frames for a
precision spot check. No ground truth: coverage is a proxy; sheets are for visual review.

  python tools/ball_tile_experiment.py outputs/real/clip3_ptz outputs/real/clip2_ptz \
      --interval clip6=outputs/real/clip6_long_115s:3420:4019
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.detect.wasb_ball import WASBBallDetector, to_detection_rows  # noqa: E402
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.track.ball import track_ball  # noqa: E402


def tile_boxes(w: int, h: int, rows: int, cols: int, overlap: float):
    tw, th = w / (cols - (cols - 1) * overlap), h / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = int(round(c * tw * (1 - overlap))), int(round(r * th * (1 - overlap)))
            out.append((x0, y0, min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))))
    return out


def tiled_candidates(video: str, det: WASBBallDetector, f0: int, f1: int, rows=2, cols=2, overlap=0.25):
    """Sliding 3-frame windows (step 1) per tile; per-frame tile heatmaps are averaged over the
    windows covering the frame, thresholded, and mapped back to source pixels."""
    k = det.frames_in
    buf: deque = deque(maxlen=k)
    acc: dict = {}
    rows_out = []
    tiles = None
    t_model = 0.0
    n = 0

    def finalize(upto):
        for (f, ti) in sorted(x for x in acc if x[0] < upto):
            s, cnt = acc.pop((f, ti))
            x0, y0, x1, y1 = tiles[ti]
            for x, y, peak, hs in det.candidates(s / cnt, (x1 - x0, y1 - y0)):
                rows_out.append(dict(frame_index=f, x=x + x0, y=y + y0, peak=peak, heat_sum=hs, tile=ti))

    for fr in iter_frames(video):
        if fr.frame_index > f1:
            break
        if fr.frame_index < f0:
            continue
        if tiles is None:
            tiles = tile_boxes(fr.image.shape[1], fr.image.shape[0], rows, cols, overlap)
        buf.append((fr.frame_index, fr.image))
        n += 1
        if len(buf) == k:
            t0 = time.perf_counter()
            for ti, (x0, y0, x1, y1) in enumerate(tiles):
                hm = det.heatmaps([im[y0:y1, x0:x1] for _, im in buf])
                for (f, _), m in zip(buf, hm):
                    if (f, ti) in acc:
                        acc[(f, ti)][0] += m
                        acc[(f, ti)][1] += 1
                    else:
                        acc[(f, ti)] = [m.copy(), 1]
            t_model += time.perf_counter() - t0
            finalize(buf[0][0] + 1)
    finalize(10**12)
    return pd.DataFrame(rows_out, columns=["frame_index", "x", "y", "peak", "heat_sum", "tile"]), 1000 * t_model / max(1, n)


def merge_close(c: pd.DataFrame, px: float = 8.0) -> pd.DataFrame:
    keep = []
    for f, g in c.sort_values("peak", ascending=False).groupby("frame_index", sort=False):
        taken = []
        for r in g.itertuples():
            if all(np.hypot(r.x - x, r.y - y) > px for x, y in taken):
                taken.append((r.x, r.y))
                keep.append(r.Index)
    return c.loc[keep].sort_values(["frame_index", "peak"], ascending=[True, False])


def sheet(video, ball: pd.DataFrame, frames: list[int], path: Path, title: str):
    if not frames:
        return
    B = ball.set_index("frame_index")
    want = set(frames)
    tiles = []
    for fr in iter_frames(video):
        if fr.frame_index > max(want):
            break
        if fr.frame_index not in want:
            continue
        x, y = int(B.at[fr.frame_index, "image_x"]), int(B.at[fr.frame_index, "image_y"])
        im = fr.image.copy()
        cv2.circle(im, (x, y), 16, (0, 255, 0), 2)
        pad = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
        crop = cv2.resize(pad[y:y + 200, x:x + 200], (240, 240), interpolation=cv2.INTER_CUBIC)
        tile = np.hstack([cv2.resize(im, (427, 240)), crop])
        cv2.putText(tile, f"{title} f{fr.frame_index}", (6, 20), 0, 0.55, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def run_case(name, run: Path, f0: int, f1: int, det, out_root: Path):
    t = run / "tables"
    ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
    frames = pd.read_parquet(t / "frames.parquet")
    frames = frames[frames.frame_index.between(f0, f1)]
    if "camera_segment_id" in frames:
        segs = pd.read_parquet(t / "camera_segments.parquet")
        sup = dict(zip(segs.camera_segment_id, segs.supported))
        frames = frames[frames.camera_segment_id.map(lambda s: bool(sup.get(s, True)))]
    cal = pd.read_parquet(t / "calibration.parquet")
    cal = cal[cal.frame_index.isin(frames.frame_index)].reset_index(drop=True)
    pl = pd.read_parquet(t / "player_observations.parquet")
    pl = pl[pl.frame_index.isin(frames.frame_index)]
    full = pd.read_parquet(run / "cache" / "wasb_ball_candidates.parquet")
    full = full[full.frame_index.isin(frames.frame_index)]
    od = out_root / name
    od.mkdir(parents=True, exist_ok=True)
    cache = od / "tile_candidates.parquet"
    if cache.exists():
        tiles = pd.read_parquet(cache)
        ms = json.loads((od / "tile_meta.json").read_text())["ms_per_frame"]
    else:
        tiles, ms = tiled_candidates(ra["source"], det, int(frames.frame_index.min()), int(frames.frame_index.max()))
        tiles.to_parquet(cache)
        (od / "tile_meta.json").write_text(json.dumps(dict(ms_per_frame=ms)))
    tiles = merge_close(tiles)
    union = merge_close(pd.concat([full.assign(tile=-1), tiles], ignore_index=True))
    res, summ = {}, dict(n_frames=len(frames), tile_ms_per_frame=ms)
    for v, c in (("full", full), ("tiles", tiles), ("full+tiles", union)):
        b = track_ball(to_detection_rows(c).reset_index(drop=True), frames, cal, pl, ra["video_id"])
        res[v] = b
        d = b[b.image_source == "detected"] if len(b) else b
        summ[v] = dict(detected=round(len(d) / len(frames), 3),
                       interpolated=round(int((b.image_source == "interpolated").sum()) / len(frames), 3) if len(b) else 0,
                       candidates_per_frame=round(len(c) / len(frames), 2))
    d0 = set(res["full"][res["full"].image_source == "detected"].frame_index)
    d1 = set(res["full+tiles"][res["full+tiles"].image_source == "detected"].frame_index)
    gained, lost = sorted(d1 - d0), sorted(d0 - d1)
    summ["full+tiles_vs_full"] = dict(gained=len(gained), lost=len(lost))
    rng = np.random.default_rng(0)
    pick = lambda xs, k: sorted(rng.choice(xs, min(k, len(xs)), replace=False).tolist()) if len(xs) else []
    sheet(ra["source"], res["full+tiles"], pick(gained, 12), od / "sheet_gained.jpg", "gained")
    sheet(ra["source"], res["full+tiles"], pick(sorted(d1), 12), od / "sheet_random_detected.jpg", "random")
    (od / "summary.json").write_text(json.dumps(summ, indent=1))
    print(name, json.dumps(summ), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="whole-clip runs")
    ap.add_argument("--interval", action="append", default=[], help="name=run_dir:start:end")
    ap.add_argument("--provider", default="dml")
    ap.add_argument("--out", default="outputs/experiments/ball_tiles")
    a = ap.parse_args()
    det = WASBBallDetector(ROOT / "models/onnx/wasb_basketball_288x512.onnx", provider=a.provider)
    out_root = ROOT / a.out
    allsum = {}
    for r in a.runs:
        run = Path(r)
        n = len(pd.read_parquet(run / "tables" / "frames.parquet"))
        allsum[run.name] = run_case(run.name, run, 0, n - 1, det, out_root)
    for spec in a.interval:
        name, rest = spec.split("=", 1)
        rd, s, e = rest.rsplit(":", 2)
        allsum[name] = run_case(name, Path(rd), int(s), int(e), det, out_root)
    (out_root / "summary.json").write_text(json.dumps(allsum, indent=1))


if __name__ == "__main__":
    main()
