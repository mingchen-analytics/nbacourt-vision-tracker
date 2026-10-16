"""CVAT ("CVAT for video 1.1") annotation exchange.

Export: model predictions -> pre-annotation XML that a human CORRECTS (never ground truth
as-is). Every track carries `review = unreviewed`; the annotator switches it to
`reviewed` after checking the whole track.

Import: a CVAT export (zip or annotations.xml) -> ground-truth tables:
  boxes     frame, gt_id, label (player|referee|other_person), team (A|B|unknown),
            occluded, review
  feet      frame, x, y, state (planted|airborne|not_visible)
  landmarks frame, name, x, y

Frame numbers are CVAT task frames, 0-based, equal to our `frame_index` when the task is
created from the original video file with default settings (no frame step / range);
`check_frame_count` verifies this.

Points (foot contact, landmarks) are exported by CVAT as single-frame point tracks; both
that form and plain <points> shapes under <image> elements are accepted.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from nbatrack.court.geometry import court_landmarks

REVIEW_VALUES = ["unreviewed", "reviewed"]
TEAM_VALUES = ["unknown", "A", "B"]
FOOT_STATES = ["planted", "airborne", "not_visible"]


def label_spec() -> list[dict]:
    """CVAT label constructor ("Raw" tab) JSON."""
    review = {"name": "review", "input_type": "select", "mutable": False,
              "values": REVIEW_VALUES, "default_value": "unreviewed"}
    return [
        {"name": "player", "color": "#e53935", "type": "rectangle", "attributes": [
            {"name": "team", "input_type": "select", "mutable": False, "values": TEAM_VALUES,
             "default_value": "unknown"}, review]},
        {"name": "referee", "color": "#424242", "type": "rectangle", "attributes": [review]},
        {"name": "other_person", "color": "#9e9e9e", "type": "rectangle", "attributes": [review]},
        {"name": "foot_contact", "color": "#00e676", "type": "points", "attributes": [
            {"name": "state", "input_type": "select", "mutable": True, "values": FOOT_STATES,
             "default_value": "planted"}]},
        {"name": "court_landmark", "color": "#ffea00", "type": "points", "attributes": [
            {"name": "landmark", "input_type": "select", "mutable": True,
             "values": [lm.name for lm in court_landmarks() if lm.on_floor],
             "default_value": court_landmarks()[0].name}]},
    ]


# ------------------------------------------------------------------------------ export
def _box_el(parent, frame, b, outside, occluded, keyframe, attrs):
    e = ET.SubElement(parent, "box", frame=str(int(frame)), outside=str(int(outside)),
                      occluded=str(int(occluded)), keyframe=str(int(keyframe)),
                      xtl=f"{b[0]:.2f}", ytl=f"{b[1]:.2f}", xbr=f"{b[2]:.2f}", ybr=f"{b[3]:.2f}",
                      z_order="0")
    for k, v in attrs.items():
        a = ET.SubElement(e, "attribute", name=k)
        a.text = v
    return e


def _occluded_flags(df: pd.DataFrame, iou_thresh: float = 0.15) -> pd.Series:
    """Heuristic pre-annotation: a box overlapping another box (IoU > thresh) is marked
    occluded. The annotator must correct it."""
    from nbatrack.track.tracker import iou_matrix
    occ = pd.Series(False, index=df.index)
    for _, g in df.groupby("frame_index"):
        if len(g) < 2:
            continue
        b = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0)
        occ.loc[g.index] = m.max(1) > iou_thresh
    return occ


def predictions_to_cvat_xml(pl: pd.DataFrame, n_frames: int, image_size: tuple[int, int],
                            task_name: str, keyframe_step: int = 5) -> bytes:
    """Build pre-annotation XML from player_observations rows.

    Tracks get keyframes every `keyframe_step` frames, at their first/last frame and
    around gaps (with an `outside` box after each visible run), so CVAT interpolates the
    rest and the annotator edits only keyframes.
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(n_frames)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(n_frames - 1)
    orig = ET.SubElement(task, "original_size")
    ET.SubElement(orig, "width").text = str(image_size[0])
    ET.SubElement(orig, "height").text = str(image_size[1])
    occ = _occluded_flags(pl)
    tid_num = 0
    for tid, g in pl.sort_values("frame_index").groupby("track_id", sort=True):
        role = g.role.mode().iloc[0] if len(g.role.mode()) else "unknown"
        label = {"player": "player", "referee": "referee", "other": "other_person"}.get(role, "player")
        attrs = {"review": "unreviewed"}
        if label == "player":
            team = g.team_id.dropna().mode()
            attrs = {"team": team.iloc[0] if len(team) and team.iloc[0] in ("A", "B") else "unknown",
                     "review": "unreviewed"}
        tr = ET.SubElement(root, "track", id=str(tid_num), label=label, source="auto")
        tid_num += 1
        frames = g.frame_index.to_numpy()
        boxes = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
        occl = occ.loc[g.index].to_numpy()
        # split into visible runs (gaps -> outside)
        breaks = np.flatnonzero(np.diff(frames) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(frames) - 1]
        for s, e in zip(starts, ends):
            for k in range(s, e + 1):
                key = (k == s) or (k == e) or ((frames[k] - frames[s]) % keyframe_step == 0) \
                    or (occl[k] != occl[k - 1] if k > s else False)
                if key:
                    _box_el(tr, frames[k], boxes[k], 0, occl[k], 1, attrs)
            if frames[e] + 1 < n_frames:
                _box_el(tr, frames[e] + 1, boxes[e], 1, 0, 1, attrs)
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_cvat_zip(xml: bytes, out_zip: str | Path) -> Path:
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", xml)
    return out_zip


# ------------------------------------------------------------------------------ import
@dataclass
class GroundTruth:
    boxes: pd.DataFrame
    feet: pd.DataFrame
    landmarks: pd.DataFrame
    n_frames: int | None
    n_tracks_unreviewed: int
    source: str

    @property
    def fully_reviewed(self) -> bool:
        return self.n_tracks_unreviewed == 0 and len(self.boxes) > 0


def _read_xml_bytes(path: str | Path) -> bytes:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not names:
                raise ValueError("no XML file in zip")
            return z.read(names[0])
    return path.read_bytes()


def _attrs(el) -> dict[str, str]:
    return {a.attrib["name"]: (a.text or "") for a in el.findall("attribute")}


def _interp_track(shapes: list[dict], n_frames: int | None) -> list[dict]:
    """Expand keyframes to every frame (linear interpolation, like CVAT) when the export
    only contains keyframes; frames after an `outside` keyframe are invisible."""
    shapes = sorted(shapes, key=lambda s: s["frame"])
    have = {s["frame"] for s in shapes}
    out = []
    for a, b in zip(shapes, shapes[1:] + [None]):
        out.append(a)
        if b is None or a["outside"]:
            continue
        for f in range(a["frame"] + 1, b["frame"]):
            if f in have:
                continue
            t = (f - a["frame"]) / (b["frame"] - a["frame"])
            box = (1 - t) * np.asarray(a["box"]) + t * np.asarray(b["box"])
            out.append({**a, "frame": f, "box": box.tolist(), "keyframe": False})
    return out


def parse_cvat_xml(path: str | Path) -> GroundTruth:
    root = ET.fromstring(_read_xml_bytes(path))
    size = root.find("./meta/task/size")
    n_frames = int(size.text) if size is not None and size.text else None
    box_rows, feet_rows, lm_rows = [], [], []
    unreviewed = 0
    gid = 0
    for tr in root.iter("track"):
        label = tr.attrib["label"]
        boxes = [{"frame": int(b.attrib["frame"]), "outside": b.attrib.get("outside", "0") == "1",
                  "occluded": b.attrib.get("occluded", "0") == "1",
                  "keyframe": b.attrib.get("keyframe", "1") == "1",
                  "box": [float(b.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")],
                  "attrs": _attrs(b)} for b in tr.findall("box")]
        if boxes:
            if label in ("player", "referee", "other_person"):
                review = boxes[0]["attrs"].get("review", "unreviewed")
                if review != "reviewed":
                    unreviewed += 1
                for s in _interp_track(boxes, n_frames):
                    if s["outside"]:
                        continue
                    box_rows.append(dict(frame=s["frame"], gt_id=gid, label=label,
                                         team=s["attrs"].get("team", "unknown") if label == "player" else None,
                                         occluded=s["occluded"], keyframe=s["keyframe"],
                                         review=review, x1=s["box"][0], y1=s["box"][1],
                                         x2=s["box"][2], y2=s["box"][3]))
                gid += 1
            continue
        for p in tr.findall("points"):
            if p.attrib.get("outside", "0") == "1":
                continue
            _add_point(label, int(p.attrib["frame"]), p.attrib["points"], _attrs(p), feet_rows, lm_rows)
    for img in root.iter("image"):  # image-mode style shapes, if any
        f = int(img.attrib.get("id", img.attrib.get("frame", 0)))
        for p in img.findall("points"):
            _add_point(p.attrib["label"], f, p.attrib["points"], _attrs(p), feet_rows, lm_rows)
    boxes = pd.DataFrame(box_rows, columns=["frame", "gt_id", "label", "team", "occluded", "keyframe",
                                            "review", "x1", "y1", "x2", "y2"])
    feet = pd.DataFrame(feet_rows, columns=["frame", "x", "y", "state"])
    lms = pd.DataFrame(lm_rows, columns=["frame", "name", "x", "y"])
    return GroundTruth(boxes.drop_duplicates(["frame", "gt_id"]), feet, lms, n_frames, unreviewed, str(path))


def _add_point(label, frame, pts, attrs, feet_rows, lm_rows):
    for pair in pts.split(";"):
        x, y = (float(v) for v in pair.split(","))
        if label == "foot_contact":
            feet_rows.append(dict(frame=frame, x=x, y=y, state=attrs.get("state", "planted")))
        elif label == "court_landmark":
            lm_rows.append(dict(frame=frame, name=attrs.get("landmark", ""), x=x, y=y))


def check_frame_count(gt: GroundTruth, n_frames_video: int) -> list[str]:
    issues = []
    if gt.n_frames is not None and gt.n_frames != n_frames_video:
        issues.append(f"CVAT task has {gt.n_frames} frames but the video decodes to {n_frames_video}; "
                      "frame numbering would be misaligned (was a frame step or range used?)")
    if len(gt.boxes) and gt.boxes.frame.max() >= n_frames_video:
        issues.append("annotation frame index beyond the video length")
    return issues


def labels_json_text() -> str:
    return json.dumps(label_spec(), indent=2, ensure_ascii=False)
