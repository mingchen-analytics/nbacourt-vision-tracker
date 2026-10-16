"""Synthetic study of quality flags and predicted sigma vs true error.

Usage: python tools/quality_flag_study.py --out outputs/experiments/quality_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nbatrack.eval.quality_study import run, summarize
from nbatrack.synthetic.scene import BroadcastScenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/experiments/quality_study")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--foot-noise-px", type=float, default=2.0)
    a = ap.parse_args()
    dfs = []
    for seed in range(a.seeds):
        sc = BroadcastScenario(n_frames=a.frames, seed=100 + seed, f_range=(1800, 4500))
        dfs.append(run(sc, noise_px=1.5, dropout=0.5, outlier_frac=0.08,
                       foot_noise_px=a.foot_noise_px, seed=100 + seed))
    df = pd.concat(dfs, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "records.parquet")
    s = summarize(df)
    s.to_csv(out / "summary.csv")
    (out / "summary.txt").write_text(s.round(3).to_string(), encoding="utf-8")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
