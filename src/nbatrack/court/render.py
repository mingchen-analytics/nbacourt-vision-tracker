"""Drawing utilities: top-down court diagram and projected court-line overlays."""

from __future__ import annotations

import cv2
import numpy as np

from nbatrack.court.geometry import NBA_COURT, CourtSpec, court_polylines, densify

FLOOR_BGR = (120, 170, 214)
LINE_BGR = (255, 255, 255)


class TopDownCanvas:
    """Maps court meters to canvas pixels. +Y (far sideline) is drawn at the top."""

    def __init__(self, px_per_m: float = 25.0, margin_m: float = 1.5,
                 spec: CourtSpec = NBA_COURT):
        self.s = px_per_m
        self.m = margin_m
        self.spec = spec
        self.w = int(round((2 * spec.half_length + 2 * margin_m) * px_per_m))
        self.h = int(round((2 * spec.half_width + 2 * margin_m) * px_per_m))

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        u = (xy[:, 0] + self.spec.half_length + self.m) * self.s
        v = (self.spec.half_width + self.m - xy[:, 1]) * self.s
        return np.stack([u, v], 1)

    def background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), FLOOR_BGR, np.uint8)
        for pl in court_polylines(self.spec).values():
            p = self.to_px(densify(pl, 0.1)).round().astype(np.int32)
            cv2.polylines(img, [p], False, LINE_BGR, 2, cv2.LINE_AA)
        return img


def draw_projected_court(img: np.ndarray, court_to_image, color=(0, 255, 255),
                         thickness: int = 2, step_m: float = 0.1) -> np.ndarray:
    """Overlay court lines projected by `court_to_image((N,2)) -> (N,2)` onto `img`."""
    h, w = img.shape[:2]
    for pl in court_polylines().values():
        uv = court_to_image(densify(pl, step_m))
        ok = np.all(np.isfinite(uv), 1) & (np.abs(uv).max(1) < 4 * max(w, h))
        # draw contiguous valid runs only
        start = None
        for i, v in enumerate(np.append(ok, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= 2:
                    cv2.polylines(img, [uv[start:i].round().astype(np.int32)], False, color,
                                  thickness, cv2.LINE_AA)
                start = None
    return img


def render_synthetic_view(camera, floor_bgr=FLOOR_BGR) -> np.ndarray:
    """Render the court floor + lines as seen by a PinholeCamera (debug / synthetic tests)."""
    w, h = camera.image_size
    img = np.full((h, w, 3), (40, 40, 40), np.uint8)
    hl, hw = NBA_COURT.half_length + 1.5, NBA_COURT.half_width + 1.5
    poly = densify(np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]]), 0.25)
    uv = camera.project(poly)[0]
    ok = np.all(np.isfinite(uv), 1)
    if ok.sum() >= 3:
        cv2.fillPoly(img, [uv[ok].round().astype(np.int32)], floor_bgr)
    return draw_projected_court(img, lambda p: camera.project(p)[0], LINE_BGR, 3)
