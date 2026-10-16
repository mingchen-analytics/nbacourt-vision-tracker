"""Quality flags must be monotonic with TRUE error on synthetic data."""

import numpy as np

from nbatrack.eval.quality_study import run, summarize
from nbatrack.synthetic.scene import BroadcastScenario


def test_flags_monotonic_with_true_error():
    df = run(BroadcastScenario(n_frames=40, seed=21, f_range=(1800, 4200)), dropout=0.5,
             outlier_frac=0.05, seed=21)
    s = summarize(df)
    for model in ("homography", "camera_free"):
        q = s.loc[model]["median"]
        if "good" in q and "poor" in q:
            assert q["good"] < q["poor"], s
    emitted = df[df.quality.isin(["good", "fair"])]
    assert len(emitted) > 0
    assert np.mean(emitted.err_m > 0.5) < 0.05, s
