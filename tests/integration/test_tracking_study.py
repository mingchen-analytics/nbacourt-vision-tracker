"""Synthetic tracking regression: court-space association and the off-court filter must
not be worse than the IoU-only baseline (fixed seed, deterministic)."""

from nbatrack.eval.tracking_study import run_tracker, simulate
from nbatrack.track.tracker import TrackerConfig


def test_court_association_beats_iou_baseline():
    frames = simulate(150, seed=7)
    base = run_tracker(frames, TrackerConfig(use_cmc=False, use_court=False))
    full = run_tracker(frames, TrackerConfig(use_cmc=True, use_court=True), court_filter_m=0.5)
    assert full.idf1 > base.idf1
    assert full.id_switches <= base.id_switches
    assert full.precision > 0.99  # bench / sideline people removed by the court filter
    assert full.recall >= base.recall - 0.01
