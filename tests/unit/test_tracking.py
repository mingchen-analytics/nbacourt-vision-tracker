import numpy as np
import pytest

from nbatrack.eval.mot_metrics import evaluate
from nbatrack.track.kalman import ConstantVelocityKF
from nbatrack.track.tracker import Detection, Tracker, TrackerConfig, iou_matrix


def _box(cx, cy, w=40, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0


def test_kf_tracks_constant_velocity():
    kf = ConstantVelocityKF(np.zeros(2), q=1.0, r=0.1, v0_std=5.0)
    for t in range(1, 30):
        kf.predict(0.1)
        kf.update(np.array([t * 0.3, -t * 0.2]))
    assert kf.x[2:] == pytest.approx([3.0, -2.0], abs=0.05)


def test_metrics_perfect_and_switch():
    gt = {f: [(1, _box(100 + 5 * f, 300)), (2, _box(600, 300))] for f in range(10)}
    perfect = evaluate(gt, gt)
    assert perfect.idf1 == 1.0 and perfect.mota == 1.0 and perfect.id_switches == 0
    # predicted id for GT 1 changes at frame 5 -> one switch, IDF1 = 2*15/(20+20)
    pred = {f: [(1 if f < 5 else 3, _box(100 + 5 * f, 300)), (2, _box(600, 300))] for f in range(10)}
    r = evaluate(gt, pred)
    assert r.id_switches == 1 and r.idtp == 15 and r.idf1 == pytest.approx(0.75)
    # a false positive and a miss
    pred2 = {f: [(1, _box(100 + 5 * f, 300)), (9, _box(1500, 300))] for f in range(10)}
    r2 = evaluate(gt, pred2)
    assert r2.fp == 10 and r2.fn == 10 and r2.mota == pytest.approx(0.0)


def test_tracker_keeps_ids_through_short_miss():
    tr = Tracker(TrackerConfig(use_court=False, use_cmc=False))
    ids = {"A": set(), "B": set()}
    for f in range(40):
        players = [("A", _box(100 + 4 * f, 300)), ("B", _box(800 - 4 * f, 320))]
        if 15 <= f < 20:  # player A missed for 5 frames
            players = players[1:]
        dets = [Detection(b, 0.9) for _, b in players]
        for a in tr.update(dets):
            ids[players[a.det_index][0]].add(a.track_id)
    assert len(ids["A"]) == 1 and len(ids["B"]) == 1
    assert ids["A"] != ids["B"]


def test_cmc_handles_jerky_pan():
    """Static players; the camera pans back and forth 60 px/frame (boxes are 30 px wide),
    which a constant-velocity model cannot predict. With the true image motion M
    supplied, IDs survive; without it tracks never get confirmed (or fragment)."""
    def run(use_cmc):
        tr = Tracker(TrackerConfig(use_court=False, use_cmc=use_cmc))
        seen, n_assigned = set(), 0
        x = 0.0
        for f in range(30):
            dx = 0.0 if f == 0 else (60.0 if f % 2 else -60.0)
            x += dx
            M = np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]])
            dets = [Detection(_box(1000 + x, 400, 30, 60), 0.9),
                    Detection(_box(1100 + x, 420, 30, 60), 0.9)]
            for a in tr.update(dets, M if f > 0 else None):
                seen.add(a.track_id)
                n_assigned += 1
        return len(seen), n_assigned
    ids, n = run(True)
    assert ids == 2 and n >= 2 * 28
    ids, n = run(False)
    assert ids > 2 or n < 10
