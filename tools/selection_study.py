"""How sensitive is the PTZ-vs-free model selection to camera-center drift? (synthetic)

Usage: python tools/selection_study.py --out outputs/experiments/selection_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nbatrack.calib.select import select_model
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, observe_landmarks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/experiments/selection_study")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    rows = []
    for v in (0.0, 0.1, 0.25, 0.5, 1.0):
        for noise in (1.0, 3.0):
            for seed in range(a.seeds):
                sc = BroadcastScenario(n_frames=60, seed=seed, center_velocity=np.array([v, 0, 0]),
                                       f_range=(1800, 3200))
                rng = np.random.default_rng(seed)
                fr = [observe_landmarks(f.camera, rng, noise, 0.3, 0.05)[0] for f in generate_sequence(sc)]
                rep = select_model(fr, sc.image_size, seed=seed)
                rows.append(dict(drift_mps=v, noise_px=noise, seed=seed, choice=rep.choice,
                                 ptz_med=rep.heldout_median_px["ptz"],
                                 free_med=rep.heldout_median_px["camera_free"],
                                 homography_med=rep.heldout_median_px["homography"]))
        print(f"drift {v} done")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_seed.csv", index=False)
    agg = df.groupby(["drift_mps", "noise_px"]).agg(
        ptz_chosen=("choice", lambda c: float(np.mean(c == "ptz"))),
        ptz_med_px=("ptz_med", "mean"), free_med_px=("free_med", "mean"),
        homography_med_px=("homography_med", "mean"))
    (out / "summary.txt").write_text(agg.round(3).to_string(), encoding="utf-8")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
