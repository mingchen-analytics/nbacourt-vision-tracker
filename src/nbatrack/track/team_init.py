"""Per-video manually initialized team / referee classification.

A small JSON file names a few exemplar people per class on one or more frames:

  {"video_id": "...", "team_A": "Portland (black)", "team_B": "Golden State (white)",
   "annotator": "...",
   "exemplars": [{"frame": 60, "box": [x1, y1, x2, y2], "class": "A"}, ...]}

classes: "A", "B", "REF". Each exemplar is matched to the cached detection with the
highest IoU on that frame (>= 0.5); its appearance descriptor becomes a prototype.

Per detection: distance to the nearest prototype of each class. The detection votes for
the nearest class only if it is unoccluded, its nearest distance is below `max_dist`
and the margin to the second-best class is at least `min_margin`; otherwise it abstains.
Per track: the label is the class with >= `min_vote_frac` of the votes and at least
`min_votes` votes; otherwise "UNK" (kept explicit, never forced).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nbatrack.track.tracker import iou_matrix

CLASSES = ("A", "B", "REF")


@dataclass
class TeamInit:
    exemplars: list[dict]
    team_A: str = ""
    team_B: str = ""
    annotator: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "TeamInit":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["exemplars"], d.get("team_A", ""), d.get("team_B", ""), d.get("annotator", ""))


def prototypes(init: TeamInit, det: pd.DataFrame, desc: np.ndarray) -> dict[str, np.ndarray]:
    protos: dict[str, list] = {c: [] for c in CLASSES}
    for ex in init.exemplars:
        f = det[(det.frame_index == ex["frame"]) & (det.cls == 1)]
        if not len(f):
            continue
        m = iou_matrix(np.array([ex["box"]], float), f[["x1", "y1", "x2", "y2"]].to_numpy(float))[0]
        k = int(m.argmax())
        if m[k] < 0.5:
            continue
        v = desc[det.index.get_loc(f.index[k])]
        if np.all(np.isfinite(v)):
            protos[ex["class"]].append(v)
    return {c: np.stack(v) for c, v in protos.items() if v}


def classify_detections(desc: np.ndarray, protos: dict[str, np.ndarray], occlusion: np.ndarray,
                        max_dist: float = 0.9, min_margin: float = 0.08,
                        max_occ: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vote label or "" for abstain, nearest distance) per row."""
    names = list(protos)
    d = np.stack([np.min(np.linalg.norm(desc[:, None, :] - protos[c][None], axis=2), axis=1) for c in names], 1)
    order = np.argsort(d, axis=1)
    best = d[np.arange(len(d)), order[:, 0]]
    second = d[np.arange(len(d)), order[:, 1]] if len(names) > 1 else np.full(len(d), np.inf)
    ok = np.isfinite(best) & (best < max_dist) & (second - best >= min_margin) & (occlusion < max_occ)
    lab = np.where(ok, np.asarray(names, object)[order[:, 0]], "")
    return lab, best


def track_labels(df: pd.DataFrame, vote_col: str = "team_vote", min_votes: int = 5,
                 min_vote_frac: float = 0.7) -> dict:
    out = {}
    for tid, g in df.groupby("track_id"):
        v = g[vote_col][g[vote_col] != ""]
        if len(v) < min_votes:
            out[tid] = ("UNK", 0.0, len(v))
            continue
        cnt = v.value_counts()
        frac = cnt.iloc[0] / len(v)
        out[tid] = (cnt.index[0], float(frac), len(v)) if frac >= min_vote_frac else ("UNK", float(frac), len(v))
    return out
