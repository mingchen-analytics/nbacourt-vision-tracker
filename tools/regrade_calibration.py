"""Re-apply the current calibration grading (grade_calibration) to an existing run without
re-running stage A, using the per-frame inlier / floor_iou / pan / tilt / focal values logged
in calibration_diagnostics.csv. Rewrites tables/calibration.parquet (quality column only) and
adds grade_reason to the diagnostics CSV; the previous quality is kept as quality_v1.

Needs floor_iou in the diagnostics (runs made after 2026-10-15 23:00). Afterwards re-run stage
B (track_and_export) so positions follow the new grades.

Usage: python tools/regrade_calibration.py outputs/real/clip6_long_115s_diag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.pipeline.video_pipeline import AnalyzeConfig, grade_calibration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = AnalyzeConfig()
    d = pd.read_csv(run / "calibration_diagnostics.csv")
    if "floor_iou" not in d:
        raise SystemExit("no floor_iou in diagnostics: re-run stage A")
    fps = float(pd.read_parquet(run / "tables" / "videos.parquet").fps_nominal.iloc[0])
    d = d.sort_values("frame").reset_index(drop=True)
    if "quality_v1" not in d:
        d["quality_v1"] = d["quality"]
    q_new, reason = [], []
    prev = {}  # segment -> (t, pan, tilt, f) of the last normal frame
    for r in d.itertuples():
        if r.method in ("unsupported_view",):
            q_new.append(r.quality_v1)
            reason.append("unsupported_view")
            continue
        if r.method == "held_flash":
            q_new.append("poor")
            reason.append("flash_frame_held")
            continue
        imp = False
        p = prev.get(r.segment)
        if p is not None and r.method != "keypoint":
            dt = abs(r.t - p[0]) or 1.0 / fps
            imp = (abs(r.pan - p[1]) / dt > cfg.max_pan_deg_s or abs(r.tilt - p[2]) / dt > cfg.max_tilt_deg_s
                   or abs(np.log(r.f / p[3])) / dt > cfg.max_zoom_frac_s)
        q, why = grade_calibration(r.inlier, int(r.n), r.floor_iou, cfg, imp)
        q_new.append(q)
        reason.append(why)
        prev[r.segment] = (r.t, r.pan, r.tilt, r.f)
    d["quality"] = q_new
    d["grade_reason"] = reason
    d.to_csv(run / "calibration_diagnostics.csv", index=False)
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = run / "tables" / "calibration.parquet"
    t = pq.read_table(p)
    qmap = dict(zip(d.frame, d.quality))
    old = t.column("quality").to_pylist()
    new = [qmap.get(f, q) for f, q in zip(t.column("frame_index").to_pylist(), old)]
    t = t.set_column(t.schema.get_field_index("quality"), t.schema.field("quality"), pa.array(new, pa.string()))
    pq.write_table(t, p, compression="zstd")
    csv = run / "tables" / "calibration.csv"
    if csv.exists():
        c = pd.read_csv(csv)
        c["quality"] = c.frame_index.map(qmap).fillna(c.quality)
        c.to_csv(csv, index=False)
    print(pd.crosstab(d.quality_v1, d.quality))
    print(d.grade_reason.value_counts().to_string())


if __name__ == "__main__":
    main()
