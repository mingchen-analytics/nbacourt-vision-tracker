"""Manual team / referee initialization helper (no GUI needed).

Step 1 - descriptors + numbered pictures:
  python tools/team_init_helper.py render --run outputs/real/clip2_ptz --video data/raw/X.mp4 --frames 60 300
  -> <run>/cache/appearance.npy and <run>/team_init_frames/frame_0060.jpg (boxes numbered)

Step 2 - look at the pictures and write the assignments:
  python tools/team_init_helper.py write --run outputs/real/clip2_ptz \
      --assign "60:0=A,1=A,2=B,6=REF" "300:3=REF" --team-a "Portland (black)" --team-b "GSW (white)" \
      --out data/annotations/clip2/teams.json
  Classes: A, B, REF. Pick clearly visible, unoccluded people (yellow boxes), ideally >= 2 per class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nbatrack.io.video import iter_frames
from nbatrack.track.appearance import extract_descriptors
from nbatrack.track.tracker import iou_matrix


def _frame_dets(det: pd.DataFrame, f: int) -> pd.DataFrame:
    return det[(det.frame_index == f) & (det.cls == 1) & (det.score >= 0.5)].sort_values("x1").reset_index()


def render(run: Path, video: str, frames: list[int]) -> None:
    det = pd.read_parquet(run / "cache" / "detections.parquet")
    app = run / "cache" / "appearance.npy"
    if not app.exists():
        extract_descriptors(video, det, out_path=app)
        print("descriptors saved", app)
    out = run / "team_init_frames"
    out.mkdir(exist_ok=True)
    for fr in iter_frames(video):
        if fr.frame_index not in frames:
            continue
        d = _frame_dets(det, fr.frame_index)
        img = fr.image.copy()
        b = d[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ov = iou_matrix(b, b)
        np.fill_diagonal(ov, 0)
        for i, r in d.iterrows():
            col = (0, 255, 255) if ov[i].max() < 0.1 else (0, 0, 255)
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
            cv2.putText(img, str(i), (int(r.x1), int(r.y1) - 4), 0, 0.8, (0, 0, 0), 4)
            cv2.putText(img, str(i), (int(r.x1), int(r.y1) - 4), 0, 0.8, col, 2)
        cv2.putText(img, f"frame {fr.frame_index} (yellow = unoccluded, red = overlapping)", (10, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"frame_{fr.frame_index:04d}.jpg"), img)
    print("pictures in", out)


def write(run: Path, assigns: list[str], team_a: str, team_b: str, out: Path, annotator: str) -> None:
    det = pd.read_parquet(run / "cache" / "detections.parquet")
    ex = []
    for spec in assigns:
        f, pairs = spec.split(":", 1)
        d = _frame_dets(det, int(f))
        for p in pairs.split(","):
            i, c = p.split("=")
            c = c.strip().upper()
            if c not in ("A", "B", "REF"):
                raise ValueError(f"class must be A, B or REF, got {c}")
            r = d.loc[int(i)]
            ex.append({"frame": int(f), "box": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)], "class": c})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"video_id": json.loads((run / "run_analyze.json").read_text())["video_id"],
                               "team_A": team_a, "team_B": team_b, "annotator": annotator,
                               "exemplars": ex}, indent=1), encoding="utf-8")
    counts = pd.Series([e["class"] for e in ex]).value_counts().to_dict()
    print("wrote", out, counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--run", required=True)
    r.add_argument("--video", required=True)
    r.add_argument("--frames", type=int, nargs="+", required=True)
    w = sub.add_parser("write")
    w.add_argument("--run", required=True)
    w.add_argument("--assign", nargs="+", required=True)
    w.add_argument("--team-a", required=True)
    w.add_argument("--team-b", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--annotator", default="user")
    a = ap.parse_args()
    if a.cmd == "render":
        render(Path(a.run), a.video, a.frames)
    else:
        write(Path(a.run), a.assign, a.team_a, a.team_b, Path(a.out), a.annotator)


if __name__ == "__main__":
    main()
