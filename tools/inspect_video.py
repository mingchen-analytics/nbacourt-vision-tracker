"""Inspect a broadcast video before processing: shot boundaries, court visibility, thumbnails.

Writes (to --out):
  frame_stats.csv   per frame: hist_dist (HSV histogram distance to the previous frame),
                    gray_diff, floor_frac (fraction of lower-2/3 pixels with a wood/paint
                    floor tone -- a rough court-visibility proxy, not a classifier)
  cuts.csv          candidate hard cuts (hist_dist peak above threshold) and gradual
                    transitions (sustained high hist_dist over several frames)
  sheet_XX.jpg      thumbnails every --every frames, labelled with frame index and time
  cut_XXXX.jpg      the frames just before / after each candidate cut

The view type of each shot (main broadcast camera / close-up / replay / other) is decided by
reviewing the sheets, not by this tool.

Usage: python tools/inspect_video.py data/raw/clip6_long_115s.mp4 --out outputs/real/clip6_long_115s/inspect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.io.video import iter_frames, probe  # noqa: E402


def hsv_hist(small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1)


def floor_fraction(small: np.ndarray) -> float:
    hsv = cv2.cvtColor(small[small.shape[0] // 3:], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = (h >= 5) & (h <= 30) & (s >= 30) & (s <= 200) & (v >= 120)  # hardwood + yellow/orange paint
    return float(wood.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--cut-thr", type=float, default=0.35)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(args.video)
    rows, thumbs = [], {}
    prev_h, prev_g, prev2_h = None, None, None
    keep_small = {}
    for fr in iter_frames(args.video):
        small = cv2.resize(fr.image, (320, 180), interpolation=cv2.INTER_AREA)
        h = hsv_hist(small)
        g = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        hd = 0.5 * float(np.abs(h - prev_h).sum()) if prev_h is not None else 0.0  # total variation
        gd = float(np.abs(g - prev_g).mean()) if prev_g is not None else 0.0
        hd2 = 0.5 * float(np.abs(h - prev2_h).sum()) if prev2_h is not None else 0.0  # vs frame f-2
        rows.append(dict(frame_index=fr.frame_index, timestamp_sec=fr.timestamp_sec, hist_dist=hd,
                         hist_dist2=hd2, gray_diff=gd, mean_gray=float(g.mean()), floor_frac=floor_fraction(small)))
        if fr.frame_index % args.every == 0:
            thumbs[fr.frame_index] = (fr.timestamp_sec, cv2.resize(fr.image, (256, 144), interpolation=cv2.INTER_AREA))
        keep_small[fr.frame_index] = small  # 170 KB per frame at 320x180; pruned below
        if len(keep_small) > 3:
            keep_small.pop(min(keep_small))
        prev2_h, prev_h, prev_g = prev_h, h, g
        if hd > args.cut_thr:  # save the boundary pair now (frames are not kept)
            pair = np.hstack([keep_small.get(fr.frame_index - 1, small), small])
            cv2.putText(pair, f"f{fr.frame_index - 1} | f{fr.frame_index}  d={hd:.2f}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(out / f"cut_{fr.frame_index:05d}.jpg"), pair)
    st = pd.DataFrame(rows)
    st.to_csv(out / "frame_stats.csv", index=False)
    # flash frame f: f differs from both neighbours but f-1 and f+1 match (single over-exposed
    # frame, same shot). hard cut: a peak that is not part of a flash. gradual: >= 4
    # consecutive frames above half threshold.
    d = st.hist_dist.to_numpy()
    d2 = st.hist_dist2.to_numpy()
    flash = set()
    for i in range(1, len(d) - 1):
        if d[i] > args.cut_thr and d[i + 1] > args.cut_thr and d2[i + 1] < args.cut_thr / 2:
            flash.add(i)
    cuts = []
    for i in np.where(d > args.cut_thr)[0]:
        if i in flash or (i - 1) in flash:
            continue
        cuts.append(dict(frame_index=int(st.frame_index[i]), timestamp_sec=float(st.timestamp_sec[i]),
                         kind="hard", hist_dist=float(d[i]), gray_diff=float(st.gray_diff[i])))
    for i in sorted(flash):
        cuts.append(dict(frame_index=int(st.frame_index[i]), timestamp_sec=float(st.timestamp_sec[i]),
                         kind="flash", hist_dist=float(d[i]), gray_diff=float(st.gray_diff[i])))
    run, start = 0, None
    for i, v in enumerate(d > args.cut_thr / 2):
        if v:
            run, start = run + 1, (i if run == 0 else start)
        else:
            if run >= 4:
                cuts.append(dict(frame_index=int(st.frame_index[start]), timestamp_sec=float(st.timestamp_sec[start]),
                                 kind=f"gradual({run}f)", hist_dist=float(d[start:i].max()),
                                 gray_diff=float(st.gray_diff[start:i].max())))
            run = 0
    pd.DataFrame(cuts).sort_values("frame_index").to_csv(out / "cuts.csv", index=False) if cuts else \
        pd.DataFrame(columns=["frame_index", "timestamp_sec", "kind", "hist_dist", "gray_diff"]).to_csv(out / "cuts.csv", index=False)
    keys = sorted(thumbs)
    per = 30
    for s in range(0, len(keys), per):
        tiles = []
        for k in keys[s:s + per]:
            ts, im = thumbs[k]
            im = im.copy()
            ff = st.floor_frac[st.frame_index == k].iloc[0]
            cv2.putText(im, f"f{k} {ts:.1f}s fl={ff:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(im, f"f{k} {ts:.1f}s fl={ff:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{info.width}x{info.height} fps={info.fps_nominal:.3f} frames={len(st)} duration={st.timestamp_sec.iloc[-1]:.2f}s")
    print(pd.DataFrame(cuts).sort_values("frame_index").to_string(index=False) if cuts else "no cuts")


if __name__ == "__main__":
    main()
