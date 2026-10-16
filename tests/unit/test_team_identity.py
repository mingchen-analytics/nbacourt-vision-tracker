"""Manual team initialization, per-segment labels and tracker extensions."""

import numpy as np
import pandas as pd

from nbatrack.track.appearance import DIM, descriptor
from nbatrack.track.identity import label_change_rows, segment_ids
from nbatrack.track.team_init import TeamInit, classify_detections, prototypes, track_labels
from nbatrack.track.tracker import Detection, Tracker, TrackerConfig


def _person(top, shorts, legs, h=120, w=50):
    img = np.zeros((h, w, 3), np.uint8)
    img[: int(0.45 * h)] = top
    img[int(0.45 * h): int(0.65 * h)] = shorts
    img[int(0.65 * h):] = legs
    return img


def test_descriptor_separates_gray_ref_from_black_player():
    ref = descriptor(_person((128, 128, 128), (20, 20, 20), (20, 20, 20)), [0, 0, 50, 120])
    blk = descriptor(_person((20, 20, 20), (20, 20, 20), (150, 170, 200)), [0, 0, 50, 120])
    wht = descriptor(_person((235, 235, 235), (235, 235, 235), (150, 170, 200)), [0, 0, 50, 120])
    assert ref.shape == (DIM,)
    assert np.linalg.norm(ref - blk) > 0.3 and np.linalg.norm(blk - wht) > 0.3
    assert descriptor(np.zeros((10, 10, 3), np.uint8), [0, 0, 5, 10]) is None  # too small


def test_exemplar_classification_and_unknown():
    rng = np.random.default_rng(0)
    base = {"A": rng.random(DIM), "B": rng.random(DIM), "REF": rng.random(DIM)}
    det = pd.DataFrame([dict(frame_index=0, cls=1, x1=10 * i, y1=0, x2=10 * i + 8, y2=30) for i in range(3)])
    desc = np.stack([base["A"], base["B"], base["REF"]]).astype(np.float32)
    init = TeamInit([{"frame": 0, "box": [10 * i, 0, 10 * i + 8, 30], "class": c} for i, c in enumerate("A B REF".split())])
    protos = prototypes(init, det, desc)
    assert set(protos) == {"A", "B", "REF"}
    q = np.stack([base["A"] + 0.01, base["B"] + 0.01, (base["A"] + base["B"]) / 2]).astype(np.float32)
    lab, _ = classify_detections(q, protos, np.array([0.0, 0.5, 0.0]), max_dist=10, min_margin=0.05)
    assert lab[0] == "A"
    assert lab[1] == ""  # occluded -> abstain
    assert lab[2] in ("", "A", "B")  # ambiguous midpoint may abstain


def test_track_labels_keep_unknown_explicit():
    df = pd.DataFrame(dict(track_id=["t1"] * 10 + ["t2"] * 10 + ["t3"] * 3,
                           team_vote=["A"] * 9 + [""] + ["A", "B"] * 5 + ["B"] * 3))
    lbl = track_labels(df)
    assert lbl["t1"][0] == "A"
    assert lbl["t2"][0] == "UNK"  # 50/50 votes
    assert lbl["t3"][0] == "UNK"  # too few votes


def test_segments_split_only_at_persistent_change():
    votes = ["A"] * 40 + ["B"] * 3 + ["A"] * 20 + ["B"] * 40
    df = pd.DataFrame(dict(track_id="t", frame_index=range(len(votes)), team_vote=votes))
    seg = segment_ids(df, "team_vote", min_run=30, window=31)
    assert seg.iloc[:60].eq(0).all() and seg.iloc[-20:].eq(1).all()
    sus = label_change_rows(df, "team_vote", min_run=30, window=31)
    assert not sus.iloc[:55].any() and sus.iloc[-20:].all()


def _box(cx, cy=300, w=40, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def test_occlusion_aware_gate_uses_two_floors():
    t = Tracker(TrackerConfig(court_min_sigma_m=0.3, court_min_sigma_occluded_m=1.0))
    clean = Detection(_box(100), 0.9, np.zeros(2), 0.1, occlusion=0.0)
    occ = Detection(_box(100), 0.9, np.zeros(2), 0.1, occlusion=0.4)
    assert t._sig(clean) == 0.3 and t._sig(occ) == 1.0


def test_team_gate_blocks_cross_color_association():
    cfg = TrackerConfig(use_court=False, use_cmc=False, team_gate=30.0, team_min_samples=3)
    tr = Tracker(cfg)
    dark, light = np.array([20.0, 5, 0]), np.array([80.0, 5, 5])
    ids = set()
    for f in range(10):
        for a in tr.update([Detection(_box(200 + f), 0.9, color=dark)]):
            ids.add(a.track_id)
    # same place, very different jersey color: must not continue the dark track
    out = tr.update([Detection(_box(210), 0.9, color=light)])
    assert all(a.track_id not in ids for a in out)
