"""Synthetic tracking benchmark across tracker configurations.

Usage: python tools/tracking_study.py --out outputs/experiments/tracking_study
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from nbatrack.eval.tracking_study import CONFIGS, run_tracker, simulate
from nbatrack.track.tracker import TrackerConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/experiments/tracking_study")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rows = []
    for seed in range(a.seeds):
        t0 = time.time()
        frames = simulate(a.frames, seed=seed)
        t_sim = time.time() - t0
        variants = [(n, TrackerConfig(**kw), None) for n, kw in CONFIGS.items()]
        variants.append(("+cmc+court+offcourt_filter", TrackerConfig(use_cmc=True, use_court=True), 0.5))
        for name, cfg, filt in variants:
            t1 = time.time()
            r = run_tracker(frames, cfg, filt)
            rows.append(dict(seed=seed, config=name, **r.as_dict(),
                             track_ms_per_frame=1000 * (time.time() - t1) / a.frames))
        print(f"seed {seed} done (sim {t_sim:.0f}s)")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_seed.csv", index=False)
    cols = ["idf1", "mota", "id_switches", "frag", "precision", "recall", "fp", "fn", "track_ms_per_frame"]
    agg = df.groupby("config", sort=False)[cols].mean()
    agg.to_csv(out / "summary.csv")
    (out / "summary.txt").write_text(agg.round(3).to_string(), encoding="utf-8")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
