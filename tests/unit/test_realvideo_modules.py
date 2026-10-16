"""Unit tests for modules added for real-video processing."""

import cv2
import numpy as np
import pytest

from nbatrack.calib.camera import ptz_rotation
from nbatrack.calib.lines import build_evidence, floor_valid_mask, refine_camera, template_points
from nbatrack.filter.temporal import raw_speed_violations, rts_smooth
from nbatrack.segments.overlay import estimate_overlay_mask
from nbatrack.synthetic.render import SyntheticArena
from nbatrack.synthetic.scene import BroadcastScenario, generate_sequence, sample_players, visible
from nbatrack.track.teams import cluster_tracks, torso_histogram


def test_rts_smoother_reduces_noise_and_keeps_track():
    rng = np.random.default_rng(0)
    t = np.arange(90) / 30.0
    truth = np.stack([2.0 * t, 1.0 * np.sin(t)], 1)
    obs = truth + rng.normal(0, 0.15, truth.shape)
    sm, speed = rts_smooth(t, obs, np.full(len(t), 0.15))
    assert np.sqrt(np.mean((sm - truth) ** 2)) < 0.6 * np.sqrt(np.mean((obs - truth) ** 2))
    assert np.median(speed[10:-10]) == pytest.approx(2.1, abs=0.4)


def test_speed_violation_flags_teleport_only():
    t = np.arange(10) / 30.0
    xy = np.stack([np.linspace(0, 1, 10), np.zeros(10)], 1)
    xy[6:] += [3.0, 0.0]  # 3 m jump in one frame = 90 m/s
    f = raw_speed_violations(t, xy, np.full(10, 0.1))
    assert f.tolist() == [False] * 6 + [True] + [False] * 3


def test_overlay_mask_finds_static_corner_box():
    rng = np.random.default_rng(1)
    frames = []
    for k in range(20):
        img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        img[300:350, 480:630] = (20, 40, 200)  # static scoreboard
        frames.append(img)
    m = estimate_overlay_mask(frames)
    assert m[325, 550] > 0 and m[150, 300] == 0


def test_team_clustering_three_groups():
    rng = np.random.default_rng(2)
    hists = {}
    colors = {"black": (20, 20, 20), "white": (235, 235, 235), "gray": (128, 128, 128)}
    sizes = {"black": 5, "white": 5, "gray": 3}
    tid = 0
    truth = {}
    for name, n in sizes.items():
        for _ in range(n):
            img = np.full((120, 60, 3), colors[name], np.uint8)
            img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
            hists[tid] = torso_histogram(img, np.array([0, 0, 60, 120.0]))
            truth[tid] = name
            tid += 1
    lab = cluster_tracks(hists)
    refs = {t for t, (l, _) in lab.items() if l == "REF"}
    assert refs == {t for t, n in truth.items() if n == "gray"}
    a = {truth[t] for t, (l, _) in lab.items() if l == "A"}
    b = {truth[t] for t, (l, _) in lab.items() if l == "B"}
    assert len(a) == 1 and len(b) == 1 and a != b


def test_lab_clustering_gray_is_referee():
    from nbatrack.track.teams import cluster_tracks_lab
    rng = np.random.default_rng(3)
    feats, truth = {}, {}
    for name, center, n in [("black", (25, 2, 0), 5), ("white", (80, 3, 8), 5), ("gray", (52, 1, -2), 3)]:
        for _ in range(n):
            tid = f"t{len(feats)}"
            feats[tid] = np.array(center, float) + rng.normal(0, 2, 3)
            truth[tid] = name
    lab = cluster_tracks_lab(feats)
    assert {t for t, (l, _) in lab.items() if l == "REF"} == {t for t, n in truth.items() if n == "gray"}


def test_color_split_flags_cross_team_change():
    import pandas as pd
    from nbatrack.track.identity import split_on_color_change
    centers = np.array([[25, 0, 0], [80, 0, 0], [52, 0, 0]], float)
    rows = []
    for f in range(90):  # 45 frames dark jersey, then 45 frames light jersey: one switch
        L = 25 if f < 45 else 80
        rows.append(dict(track_id="t1", frame_index=f, torso_L=L, torso_a=0.0, torso_b=0.0))
    for f in range(90):  # referee track with noisy dark/light frames: must NOT split
        L = 52 if f % 7 else (25 if f % 2 else 80)
        rows.append(dict(track_id="t2", frame_index=f, torso_L=L, torso_a=0.0, torso_b=0.0))
    out, n = split_on_color_change(pd.DataFrame(rows), centers, min_run=30, window=31, ignore=(2,))
    assert n == 1
    assert set(out[out.track_id.str.startswith("t2")].track_id) == {"t2"}
    assert out[(out.track_id == "t1_2")].frame_index.min() >= 40


def test_line_refinement_recovers_perturbed_camera():
    fr = generate_sequence(BroadcastScenario(n_frames=1, seed=3, image_size=(1280, 720),
                                             f_range=(1300, 1300)))[0]
    gt = fr.camera
    img = SyntheticArena(2, floor_noise_amp=4, floor_seams=False).render(gt)
    p, t, r = gt.ptz
    init = gt.copy(f=gt.f * 1.05)
    init.R = ptz_rotation(p + 1.0, t - 0.7, r + 0.3)
    init.C = gt.C + np.array([0.5, 0.8, -0.4])
    ev = build_evidence(img, floor_valid_mask(init, img.shape[:2], margin_m=2.0))
    res = refine_camera(init, ev, template_points())
    probes = sample_players(np.random.default_rng(0), 200)
    uv, vis = visible(gt, probes)
    e0 = np.linalg.norm(init.backproject_to_plane(uv[vis])[0] - probes[vis], axis=1)
    e1 = np.linalg.norm(res.camera.backproject_to_plane(uv[vis])[0] - probes[vis], axis=1)
    assert np.median(e0) > 0.5 and np.median(e1) < 0.1
