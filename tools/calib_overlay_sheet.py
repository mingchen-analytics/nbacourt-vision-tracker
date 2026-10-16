"""Contact sheet of the STORED calibration: court lines projected onto sampled frames, labelled
with frame, segment, quality, grade reason, inlier fraction and floor IoU. For visual review.

  python tools/calib_overlay_sheet.py outputs/real/clip6_long_115s --n 24 [--frames 100 2000 ...]
  -> <run>/calibration_overlays/sheet_XX.jpg
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

from nbatrack.court.render import draw_projected_court  # noqa: E402
from nbatrack.io.video import iter_frames  # noqa: E402
from nbatrack.pipeline.video_pipeline import camera_from_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = Path(a.run)
    ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
    cal = pd.read_parquet(run / "tables" / "calibration.parquet").set_index("frame_index")
    diag = pd.read_csv(run / "calibration_diagnostics.csv").set_index("frame")
    size = (ra["width"], ra["height"])
    if a.frames:
        pick = sorted(a.frames)
    else:
        ok = cal[cal.model != "none"].index.to_numpy()
        pick = sorted(set(ok[np.linspace(0, len(ok) - 1, a.n).round().astype(int)].tolist()))
    want = set(pick)
    tiles = []
    for fr in iter_frames(ra["source"]):
        if fr.frame_index > max(want):
            break
        if fr.frame_index not in want:
            continue
        c = cal.loc[fr.frame_index]
        img = fr.image.copy()
        if c.model != "none":
            cam = camera_from_row(c, size)
            draw_projected_court(img, lambda p: cam.project(np.hstack([p, np.zeros((len(p), 1))]))[0],
                                 color=(0, 255, 255), thickness=2)
        d = diag.loc[fr.frame_index] if fr.frame_index in diag.index else None
        txt = f"f{fr.frame_index} s{int(c.camera_segment_id)} {c.quality}"
        if d is not None:
            txt += f" {d.get('grade_reason', '')} inl={d.get('inlier', float('nan')):.2f} iou={d.get('floor_iou', float('nan')):.2f}"
        cv2.putText(img, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(img, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(cv2.resize(img, (640, 360)))
    od = run / "calibration_overlays"
    od.mkdir(exist_ok=True)
    per = 12
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        while len(t) % 3:
            t.append(np.zeros_like(t[0]))
        sheet = np.vstack([np.hstack(t[i:i + 3]) for i in range(0, len(t), 3)])
        p = od / f"sheet{a.tag}_{s // per:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(p)


if __name__ == "__main__":
    main()
