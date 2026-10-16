"""Run the synthetic calibration-model comparison and write a report.

Usage:
    python tools/compare_calibration_models.py --out outputs/experiments/calib_compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nbatrack.eval.calib_compare import run_scenario, stats_to_rows
from nbatrack.synthetic.scene import BroadcastScenario

SCENARIOS = {
    # name: (scenario kwargs, observation kwargs)
    "wide_fixed_noise1px": (dict(f_range=(1800, 2600)), dict(noise_px=1.0)),
    "wide_fixed_noise3px": (dict(f_range=(1800, 2600)), dict(noise_px=3.0)),
    "tight_zoom_noise1px": (dict(f_range=(3500, 4800)), dict(noise_px=1.0)),
    "outliers10pct": (dict(f_range=(1800, 3200)), dict(noise_px=1.0, outlier_frac=0.10)),
    "heavy_dropout": (dict(f_range=(2400, 3600)), dict(noise_px=1.0, dropout=0.6)),
    "center_drift_0.5mps": (dict(f_range=(1800, 3200), center_velocity=np.array([0.5, 0, 0])),
                            dict(noise_px=1.0)),
    "center_drift_2mps": (dict(f_range=(1800, 3200), center_velocity=np.array([2.0, 0, 0])),
                          dict(noise_px=1.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/experiments/calib_compare")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for name, (sk, ok) in SCENARIOS.items():
        for seed in range(args.seeds):
            sc = BroadcastScenario(n_frames=args.frames, seed=seed, **sk)
            for r in stats_to_rows(name, run_scenario(sc, seed=seed, **ok)):
                r["seed"] = seed
                rows.append(r)
        print(f"{name}: done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    agg = df.drop(columns=["seed", "frames"]).groupby(["scenario", "model"], sort=False).mean()
    agg.to_csv(out / "summary.csv")
    (out / "summary.txt").write_text(agg.round(3).to_string(), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"frames": args.frames, "seeds": args.seeds,
                                              "elapsed_sec": time.time() - t0}, indent=2))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
