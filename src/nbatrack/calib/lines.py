"""Court-marking evidence from an image, and line-alignment calibration refinement.

Evidence map: thin painted lines (dark OR bright, via morphological black-hat / top-hat)
plus strong color edges (painted-area boundaries such as the lane), restricted to the
floor region and with players / broadcast overlays masked out. Its distance transform
gives, for any pixel, the distance to the nearest marking.

Refinement: court polylines (template) are densely sampled, projected with the current
camera, and the camera is optimized so projected template points land on markings
(robust Cauchy loss so logos, ads and unpainted template lines are tolerated). This is a
chamfer-matching variant of the point+line refinement used in TVCalib / PnLCalib.

The alignment residual is a SELF-CONSISTENCY measure (does the model explain the visible
markings?), not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from nbatrack.calib.camera import PinholeCamera, ptz_rotation
from nbatrack.court.geometry import NBA_COURT, court_polylines, densify

# Lines that are physically painted on (almost) every NBA court. The restricted-area arc
# and the dashed inner free-throw half circle are less reliable and are excluded.
TEMPLATE_KEYS_EXCLUDE = ("restricted",)


def template_points(step_m: float = 0.05) -> np.ndarray:
    pts = [densify(pl, step_m) for k, pl in court_polylines().items()
           if not any(x in k for x in TEMPLATE_KEYS_EXCLUDE)]
    return np.vstack(pts)


@dataclass
class Evidence:
    mask: np.ndarray  # uint8 marking pixels
    dist: np.ndarray  # float32 distance (px) to nearest marking
    valid: np.ndarray  # uint8 region where evidence can exist (floor, not occluded)


def marking_mask(bgr: np.ndarray, line_px: int = 9, thresh: float = 18.0,
                 edge_lo: int = 60, edge_hi: int = 140) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    thin = ((blackhat > thresh) | (tophat > thresh)).astype(np.uint8) * 255
    # Color edges (painted lane etc.): Canny on saturation and on a/b chroma.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    edges = cv2.Canny(sat, edge_lo, edge_hi)
    return cv2.bitwise_or(thin, edges)


def build_evidence(bgr: np.ndarray, valid: np.ndarray, **kw) -> Evidence:
    m = marking_mask(bgr, **kw)
    m = cv2.bitwise_and(m, valid)
    # remove tiny speckles
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 12
    small[0] = False
    m[small[lab]] = 0
    dist = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5)
    return Evidence(m, dist, valid)


FLOOR_IOU_SCALE = 0.25  # floor-IoU check runs at 1/4 resolution


def floor_tone_mask(bgr: np.ndarray, scale: float = FLOOR_IOU_SCALE) -> np.ndarray:
    """Pixels with a hardwood / light painted-floor tone (HSV hue 5-30, sat 30-200, value
    >= 120) at `scale` resolution. Arena-dependent heuristic: dark painted areas and
    logos are missed; it is used only as an INDEPENDENT sanity check of a calibration
    (line alignment alone can be fooled by dense crowd texture), not as evidence."""
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 5) & (h <= 30) & (s >= 30) & (s <= 200) & (v >= 120)).astype(np.uint8)


def floor_iou(cam: PinholeCamera, tone: np.ndarray, overlay: np.ndarray | None = None,
              scale: float = FLOOR_IOU_SCALE) -> float:
    """IoU between the projected court rectangle (inside the image) and the floor-tone mask.
    A correct calibration of a main broadcast view gives a high value (most of the
    wood-toned area lies inside the court and vice versa); a calibration that puts the court
    on the stands gives a low one."""
    hh, ww = tone.shape
    hl, hw = NBA_COURT.half_length, NBA_COURT.half_width
    border = densify(np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]]), 0.25)
    uv, z = cam.project(border)
    ok = (z > 0) & np.all(np.isfinite(uv), 1)
    poly = np.zeros_like(tone)
    if ok.sum() >= 3:
        cv2.fillPoly(poly, [np.clip(uv[ok] * scale, -10 * ww, 10 * ww).round().astype(np.int32)], 1)
    t = tone
    if overlay is not None:
        keep = cv2.resize(overlay, (ww, hh), interpolation=cv2.INTER_NEAREST) == 0
        poly, t = poly & keep, t & keep
    inter = int((poly & t).sum())
    union = int((poly | t).sum())
    return inter / union if union else 0.0


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(uv[:, 0], 0, w - 1.001)
    y = np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


@dataclass
class RefineResult:
    camera: PinholeCamera
    n_points: int  # template points evaluated
    inlier_frac: float  # fraction within `inlier_px` of a marking
    median_dist_px: float  # median distance of inlier template points
    success: bool


def alignment_stats(cam: PinholeCamera, ev: Evidence, tpl: np.ndarray,
                    inlier_px: float = 3.0) -> tuple[int, float, float]:
    uv, z = cam.project(tpl)
    h, w = ev.dist.shape
    ok = (z > 0) & np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if ok.any():
        vi = ev.valid[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] > 0
        idx = np.flatnonzero(ok)[vi]
    else:
        idx = np.zeros(0, int)
    if len(idx) == 0:
        return 0, 0.0, float("nan")
    d = _bilinear(ev.dist, uv[idx])
    inl = d < inlier_px
    return len(idx), float(inl.mean()), float(np.median(d[inl])) if inl.any() else float("nan")


def refine_camera(cam: PinholeCamera, ev: Evidence, tpl: np.ndarray | None = None,
                  fixed_C: np.ndarray | None = None, scales=(12.0, 5.0, 2.0),
                  min_points: int = 80, anchors=None, anchor_sigma_px: float = 6.0,
                  anchor_weight: float = 1.0) -> RefineResult:
    """Coarse-to-fine robust chamfer alignment of the court template to markings.

    Free parameters: pan, tilt, roll, f, and C (unless `fixed_C`, i.e. PTZ-constrained).
    `anchors` (Correspondences) adds point residuals (joint point + line optimization) so
    that the solution cannot slide to a wrong set of lines; their total weight is
    comparable to the whole line term.
    """
    tpl = template_points() if tpl is None else tpl
    w, h = cam.image_size
    pan0, tilt0, roll0 = cam.ptz
    x0 = [pan0, tilt0, roll0, cam.f / 1000.0]
    if fixed_C is None:
        x0 += list(cam.C)
    x = np.array(x0, float)

    def make(xv):
        C = fixed_C if fixed_C is not None else xv[4:7]
        return PinholeCamera(xv[3] * 1000.0, cam.cx, cam.cy, ptz_rotation(xv[0], xv[1], xv[2]),
                             np.asarray(C, float), cam.k1, cam.image_size)

    # The set of template points that participate is fixed per stage (chosen at the
    # stage's starting estimate) so the objective is smooth for the optimizer.
    for s in scales:
        c0 = make(x)
        uv, z = c0.project(tpl)
        ok = (z > 0) & np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if ok.sum() == 0:
            return RefineResult(c0, 0, 0.0, float("nan"), False)
        vi = ev.valid[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] > 0
        P = tpl[np.flatnonzero(ok)[vi]]
        if len(P) < min_points:
            return RefineResult(c0, len(P), 0.0, float("nan"), False)
        P3 = np.hstack([P, np.zeros((len(P), 1))])
        trunc = 4.0 * s

        def resid(xv):
            c = make(xv)
            Xc = (P3 - c.C) @ c.R.T
            zz = np.where(Xc[:, 2] > 1e-6, Xc[:, 2], 1e-6)
            u = c.f * Xc[:, 0] / zz + c.cx
            v = c.f * Xc[:, 1] / zz + c.cy
            d = _bilinear(ev.dist, np.stack([u, v], 1))
            out = (u < 0) | (u > w - 1) | (v < 0) | (v > h - 1)
            d = np.where(out, trunc, np.minimum(d, trunc))
            r_line = d / np.sqrt(len(P))
            if anchors is None or len(anchors) == 0:
                return r_line
            ua, _ = c.project(anchors.world)
            ea = (ua - anchors.image).ravel() / anchor_sigma_px * (s * anchor_weight)
            return np.concatenate([r_line, ea / np.sqrt(len(anchors))])

        r = least_squares(resid, x, loss="cauchy", f_scale=s / np.sqrt(len(P)),
                          diff_step=1e-4, method="trf", max_nfev=200)
        x = r.x
    c = make(x)
    n, frac, med = alignment_stats(c, ev, tpl)
    ok = n >= min_points and c.C[2] > 0 and c.f > 0
    return RefineResult(c, n, frac, med, bool(ok))


def initialize_from_landmarks(bgr: np.ndarray, corr, overlay: np.ndarray | None = None,
                              boxes: np.ndarray | None = None) -> tuple[RefineResult | None, list]:
    """Semi-automatic keyframe calibration: a few approximate landmarks + court markings.

    1. Grid of sideline-camera priors, each fit to the landmarks (pan, tilt, f; roll 0).
    2. Every candidate is refined against the image markings (center fixed, then free).
    3. The candidate explaining the most markings (inlier fraction) wins.
    Returns (best result, all candidate summaries for diagnostics).
    """
    from nbatrack.calib.fit import prior_candidates

    h, w = bgr.shape[:2]
    tpl = template_points()
    cands = prior_candidates(corr, (w, h))
    summaries = []
    best = None
    for fit in cands:
        c = fit.camera
        ev = build_evidence(bgr, floor_valid_mask(c, (h, w), 1.5, boxes, overlay))
        r1 = refine_camera(c, ev, tpl, fixed_C=c.C, scales=(20.0, 10.0, 5.0), anchors=corr)
        ev = build_evidence(bgr, floor_valid_mask(r1.camera, (h, w), 1.5, boxes, overlay))
        r2 = refine_camera(r1.camera, ev, tpl, scales=(5.0, 2.0), anchors=corr)
        lm_err = float(np.median(np.linalg.norm(r2.camera.project(corr.world)[0] - corr.image, axis=1)))
        summaries.append(dict(C=np.round(c.C, 1).tolist(), landmark_rmse=fit.rmse_px,
                              inlier_frac=r2.inlier_frac, n=r2.n_points, landmark_err_after=lm_err))
        # landmarks must still agree after refinement (guards against snapping to wrong lines)
        if r2.success and lm_err < 25.0 and (best is None or r2.inlier_frac > best.inlier_frac):
            best = r2
    return best, summaries


def floor_valid_mask(cam: PinholeCamera, shape: tuple[int, int], margin_m: float = 0.6,
                     boxes: np.ndarray | None = None, overlay: np.ndarray | None = None,
                     box_pad: int = 4) -> np.ndarray:
    """Pixels where court markings may be observed: projected floor (+margin), minus
    player boxes and broadcast overlays."""
    h, w = shape
    hl, hw = NBA_COURT.half_length + margin_m, NBA_COURT.half_width + margin_m
    border = densify(np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]]), 0.25)
    uv, z = cam.project(border)
    m = np.zeros((h, w), np.uint8)
    ok = (z > 0) & np.all(np.isfinite(uv), 1)
    if ok.sum() >= 3:
        cv2.fillPoly(m, [np.clip(uv[ok], -10 * w, 10 * w).round().astype(np.int32)], 255)
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes, float):
            m[max(0, int(y1) - box_pad):min(h, int(y2) + box_pad),
              max(0, int(x1) - box_pad):min(w, int(x2) + box_pad)] = 0
    if overlay is not None:
        m[overlay > 0] = 0
    return m
