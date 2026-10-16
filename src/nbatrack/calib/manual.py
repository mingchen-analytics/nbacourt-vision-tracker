"""Manual / semi-automatic calibration.

Core (testable, no GUI):  ManualCalibrationSession holds clicked (landmark_id, u, v)
pairs for a frame, fits homography and camera models, and renders a verification overlay.
Annotations are saved as JSON so they double as calibration ground truth for evaluation.

GUI (`run_gui`): OpenCV window. Keys:
  n / p      next / previous landmark in the list
  click      place the current landmark at the cursor
  x          delete current landmark's point
  f          fit and show projected court lines
  s          save JSON and quit        q  quit without saving
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from nbatrack.calib.fit import fit_camera
from nbatrack.calib.homography import fit_homography
from nbatrack.calib.models import CalibrationModel, CameraCalibration, Correspondences, HomographyCalibration
from nbatrack.calib.quality import QualityReport, assess
from nbatrack.court.geometry import court_landmarks
from nbatrack.court.render import draw_projected_court

LANDMARKS = court_landmarks()
LM_BY_ID = {lm.landmark_id: lm for lm in LANDMARKS}


@dataclass
class ManualCalibrationSession:
    video_id: str
    frame_index: int
    timestamp_sec: float
    image_size: tuple[int, int]
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    annotator: str = ""

    def set_point(self, landmark_id: int, u: float, v: float) -> None:
        if landmark_id not in LM_BY_ID:
            raise KeyError(landmark_id)
        self.points[landmark_id] = (float(u), float(v))

    def correspondences(self) -> Correspondences:
        ids = sorted(self.points)
        return Correspondences(np.array([LM_BY_ID[i].xyz for i in ids]).reshape(-1, 3),
                               np.array([self.points[i] for i in ids]).reshape(-1, 2),
                               np.array(ids))

    def fit(self, model: str = "camera") -> tuple[CalibrationModel | None, QualityReport]:
        """model: 'homography' (floor points, >= 4) or 'camera' (pinhole, >= 4 floor
        points for initialization; can use the rim)."""
        c = self.correspondences()
        if model == "homography":
            fm = c.floor_mask
            if fm.sum() < 4:
                return None, QualityReport("failed", 0.0, ["need >= 4 floor landmarks"])
            hf = fit_homography(c.world[fm, :2], c.image[fm])
            q = assess("homography", hf.n_inliers, hf.rmse_px, c.image[fm][hf.inliers], self.image_size)
            return HomographyCalibration(hf.H), q
        cf = fit_camera(c, self.image_size)
        if not cf.success:
            return None, QualityReport("failed", 0.0, [cf.message])
        q = assess("camera", int(cf.inliers.sum()), cf.rmse_px, c.image[cf.inliers],
                   self.image_size, cf.camera)
        return CameraCalibration(cf.camera, "camera_free"), q

    def to_json(self) -> dict:
        return {
            "video_id": self.video_id, "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec, "image_size": list(self.image_size),
            "annotator": self.annotator,
            "points": [{"landmark_id": i, "name": LM_BY_ID[i].name, "u": uv[0], "v": uv[1]}
                       for i, uv in sorted(self.points.items())],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ManualCalibrationSession":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls(d["video_id"], d["frame_index"], d["timestamp_sec"], tuple(d["image_size"]),
                annotator=d.get("annotator", ""))
        for p in d["points"]:
            s.set_point(p["landmark_id"], p["u"], p["v"])
        return s


def overlay(image: np.ndarray, session: ManualCalibrationSession,
            model: CalibrationModel | None, current_id: int | None = None) -> np.ndarray:
    out = image.copy()
    if model is not None:
        draw_projected_court(out, lambda p: model.court_to_image(p), (0, 255, 255), 1)
        if model.supports_3d:
            for lm in LANDMARKS:
                if not lm.on_floor:
                    uv = model.court_to_image(np.array([lm.xyz]))[0]
                    if np.all(np.isfinite(uv)):
                        cv2.drawMarker(out, tuple(int(v) for v in uv), (255, 0, 255),
                                       cv2.MARKER_TILTED_CROSS, 16, 2)
    for i, (u, v) in session.points.items():
        col = (0, 0, 255) if i == current_id else (0, 255, 0)
        cv2.circle(out, (int(round(u)), int(round(v))), 5, col, 2)
        cv2.putText(out, str(i), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return out


def run_gui(image: np.ndarray, session: ManualCalibrationSession, out_json: str | Path,
            max_display_w: int = 1600) -> bool:  # pragma: no cover - interactive
    scale = min(1.0, max_display_w / image.shape[1])
    cur = [0]
    model: list[CalibrationModel | None] = [None]
    status = [""]
    win = "nbatrack calibrate"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            session.set_point(LANDMARKS[cur[0]].landmark_id, x / scale, y / scale)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        lm = LANDMARKS[cur[0]]
        img = overlay(image, session, model[0], lm.landmark_id)
        disp = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
        txt = f"[{lm.landmark_id}] {lm.name}: {lm.description} | {status[0]}"
        cv2.putText(disp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(disp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("n"):
            cur[0] = (cur[0] + 1) % len(LANDMARKS)
        elif k == ord("p"):
            cur[0] = (cur[0] - 1) % len(LANDMARKS)
        elif k == ord("x"):
            session.points.pop(lm.landmark_id, None)
        elif k == ord("f"):
            model[0], q = session.fit("camera")
            if model[0] is None:
                model[0], q = session.fit("homography")
            status[0] = f"{q.quality} rmse*={q.rmse_corrected_px:.2f}px {';'.join(q.reasons)}"
        elif k == ord("s"):
            session.save(out_json)
            cv2.destroyWindow(win)
            return True
        elif k == ord("q"):
            cv2.destroyWindow(win)
            return False
