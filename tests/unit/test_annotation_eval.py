"""Annotation exchange (CVAT) and evaluation tooling."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from nbatrack.annot.cvat import label_spec, parse_cvat_xml, predictions_to_cvat_xml, write_cvat_zip
from nbatrack.eval import trackeval_bridge
from nbatrack.eval.gt_eval import _best_team_mapping, detection_pr
from nbatrack.eval.mot_metrics import evaluate as mot_evaluate


def _toy_pl():
    rows = []
    for f in range(30):
        rows.append(dict(frame_index=f, track_id="s000_t001", role="player", team_id="A",
                         bbox_x1=100 + 3 * f, bbox_y1=200, bbox_x2=140 + 3 * f, bbox_y2=300))
        if not 10 <= f < 15:  # gap -> outside
            rows.append(dict(frame_index=f, track_id="s000_t002", role="referee", team_id=None,
                             bbox_x1=600, bbox_y1=220, bbox_x2=640, bbox_y2=320))
    return pd.DataFrame(rows)


def test_cvat_roundtrip_interpolation_gaps_and_review(tmp_path):
    xml = predictions_to_cvat_xml(_toy_pl(), 30, (1280, 720), "toy", keyframe_step=5)
    root = ET.fromstring(xml)
    assert root.find("version").text == "1.1"
    assert {t.attrib["label"] for t in root.iter("track")} == {"player", "referee"}
    z = write_cvat_zip(xml, tmp_path / "pre.zip")
    gt = parse_cvat_xml(z)
    assert gt.n_frames == 30 and gt.n_tracks_unreviewed == 2 and not gt.fully_reviewed
    p = gt.boxes[gt.boxes.label == "player"].sort_values("frame")
    assert len(p) == 30  # keyframes every 5 + linear interpolation restores every frame
    assert np.allclose(p.x1.to_numpy(), 100 + 3 * np.arange(30))
    assert (p.team == "A").all()
    r = gt.boxes[gt.boxes.label == "referee"]
    assert sorted(r.frame) == [f for f in range(30) if not 10 <= f < 15]  # gap respected
    # mark reviewed (what the annotator does) -> fully reviewed
    gt2 = parse_cvat_xml_bytes(tmp_path, xml.replace(b">unreviewed<", b">reviewed<"))
    assert gt2.fully_reviewed


def parse_cvat_xml_bytes(tmp_path, b):
    p = Path(tmp_path) / "x.xml"
    p.write_bytes(b)
    return parse_cvat_xml(p)


def test_points_parsed_as_feet_and_landmarks(tmp_path):
    xml = b"""<?xml version='1.0' encoding='utf-8'?>
<annotations><version>1.1</version>
 <track id="0" label="foot_contact" source="manual">
  <points frame="15" outside="0" occluded="0" keyframe="1" points="120.5,300.0"><attribute name="state">planted</attribute></points>
  <points frame="16" outside="1" occluded="0" keyframe="1" points="120.5,300.0"><attribute name="state">planted</attribute></points>
 </track>
 <track id="1" label="court_landmark" source="manual">
  <points frame="0" outside="0" occluded="0" keyframe="1" points="427,397"><attribute name="landmark">L_lane_ft_far</attribute></points>
 </track>
</annotations>"""
    gt = parse_cvat_xml_bytes(tmp_path, xml)
    assert gt.feet.to_dict("records") == [dict(frame=15, x=120.5, y=300.0, state="planted")]
    assert gt.landmarks.iloc[0]["name"] == "L_lane_ft_far"


def test_ball_track_roundtrip_with_gap_and_review(tmp_path):
    ball = pd.DataFrame(dict(frame_index=[f for f in range(30) if not 12 <= f < 20],
                             image_x=[500.0 + 2 * f for f in range(30) if not 12 <= f < 20], image_y=300.0))
    xml = predictions_to_cvat_xml(_toy_pl(), 30, (1280, 720), "toy", keyframe_step=5, ball=ball)
    gt = parse_cvat_xml_bytes(tmp_path, xml)
    assert gt.n_tracks_unreviewed == 3  # 2 person tracks + the ball track
    assert sorted(gt.ball.frame) == [f for f in range(30) if not 12 <= f < 20]  # ball not visible in the gap
    assert np.allclose(gt.ball.sort_values("frame").x, [500.0 + 2 * f for f in range(30) if not 12 <= f < 20])
    assert (gt.ball.review == "unreviewed").all()
    gt2 = parse_cvat_xml_bytes(tmp_path, xml.replace(b">unreviewed<", b">reviewed<"))
    assert gt2.fully_reviewed and (gt2.ball.review == "reviewed").all()


def test_label_spec_has_required_labels():
    names = {l["name"] for l in label_spec()}
    assert names == {"player", "referee", "other_person", "foot_contact", "court_landmark", "ball"}
    json.dumps(label_spec())


def test_trackeval_matches_own_metrics_on_toy_switch():
    box = lambda x: (x, 100.0, x + 40.0, 200.0)
    gt = {f: [(1, box(100 + 2 * f)), (2, box(600))] for f in range(20)}
    pred = {f: [(10 if f < 10 else 11, box(100 + 2 * f)), (12, box(600))] for f in range(20)}
    te = trackeval_bridge.evaluate(gt, pred, 20)
    own = mot_evaluate(gt, pred)
    assert te["IDSW"] == own.id_switches == 1
    assert te["IDF1"] == pytest.approx(own.idf1)
    assert te["MOTA"] == pytest.approx(own.mota)
    assert 0 < te["AssA"] < 1 and te["DetA"] == pytest.approx(1.0)


def test_team_mapping_is_permutation_invariant():
    acc, mapping, n = _best_team_mapping([("A", "B"), ("A", "B"), ("B", "A"), ("B", "B")])
    assert acc == pytest.approx(0.75) and mapping == {"A": "B", "B": "A"} and n == 4


def test_detection_pr():
    gt = pd.DataFrame(dict(frame=[0, 0], x1=[0, 100], y1=[0, 0], x2=[10, 110], y2=[10, 10]))
    det = pd.DataFrame(dict(frame=[0, 0, 0], x1=[0, 300, 101], y1=[0, 0, 0], x2=[10, 310, 111], y2=[10, 10, 10]))
    r = detection_pr(gt, det)
    assert r["tp"] == 2 and r["fp"] == 1 and r["recall"] == 1.0
