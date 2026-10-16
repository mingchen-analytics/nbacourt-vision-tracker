"""Multi-object tracker: ByteTrack-style two-stage association, extended with

  * camera-motion compensation (CMC): predicted boxes are warped by the frame-to-frame
    image homography, as in BoT-SORT (re-implemented; boxmot is AGPL and not used);
  * court-space gating/cost: when both the track and the detection have calibrated court
    positions, a Mahalanobis distance in meters is added to the cost. Court space is
    unaffected by pan/zoom, and its motion model is physical (players <= ~10 m/s);
  * optional appearance features (cosine distance), unused until a re-ID model exists.

Identities are only maintained within one camera segment; call `reset()` at a cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from nbatrack.calib.homography import apply_h
from nbatrack.track.kalman import ConstantVelocityKF


@dataclass
class Detection:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 px
    score: float
    court_xy: np.ndarray | None = None  # meters, only if calibration emitted coordinates
    court_sigma_m: float | None = None
    feature: np.ndarray | None = None  # L2-normalized appearance embedding
    occlusion: float = 0.0  # max IoU with other person boxes in the frame
    color: np.ndarray | None = None  # jersey color (e.g. median torso Lab)


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8  # max cost accepted in stage 1
    low_match_iou: float = 0.5  # min IoU accepted in stage 2
    max_time_lost_s: float = 1.0
    min_hits: int = 2  # confirmations before a track is reported
    use_cmc: bool = True
    use_court: bool = True
    court_gate_chi2: float = 13.8  # 99.9% for 2 dof
    w_iou: float = 0.5
    w_court: float = 0.5
    w_app: float = 0.0
    court_accel_std: float = 8.0  # m/s^2, player maneuvering
    court_min_sigma_m: float = 0.05  # floor on per-detection court sigma used for gating
    # appearance: "cosine" for L2-normalized embeddings, "euclidean" for color features
    # (e.g. torso Lab); app_gate forbids associations whose distance exceeds it
    app_metric: str = "cosine"
    app_scale: float = 30.0
    app_gate: float | None = None
    img_accel_frac: float = 2.0  # image accel std as multiple of box height per s^2
    # occlusion-aware court gating: if set, `court_min_sigma_m` applies to unoccluded
    # detections and this floor to detections overlapping another box (IoU >= occ_thresh)
    court_min_sigma_occluded_m: float | None = None
    occ_thresh: float = 0.1
    # team-aware association: compare an UNOCCLUDED detection's jersey color with the
    # track's color prototype (median of its unoccluded detections)
    w_team: float = 0.0
    team_scale: float = 30.0
    team_gate: float | None = None
    team_min_samples: int = 8
    team_max_occ: float = 0.1


@dataclass
class Track:
    track_id: int
    kf_img: ConstantVelocityKF  # [cx, cy, w, h]
    kf_court: ConstantVelocityKF | None
    hits: int = 1
    time_since_update: float = 0.0
    confirmed: bool = False
    feature: np.ndarray | None = None
    last_score: float = 0.0
    colors: list = field(default_factory=list)

    def color_proto(self, min_samples: int) -> np.ndarray | None:
        if len(self.colors) < min_samples:
            return None
        return np.median(np.asarray(self.colors[-60:]), axis=0)

    @property
    def bbox(self) -> np.ndarray:
        cx, cy, w, h = self.kf_img.pos
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Assignment:
    track_id: int
    det_index: int
    cost: float  # 0 (perfect) .. match_thresh

    @property
    def confidence(self) -> float:
        return float(max(0.0, 1.0 - self.cost))


def _xywh(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class Tracker:
    def __init__(self, config: TrackerConfig | None = None, fps: float = 30.0):
        self.cfg = config or TrackerConfig()
        self.fps = fps
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Camera cut: drop all tracks (identities are not carried across segments)."""
        self.tracks = []

    # ---------------------------------------------------------------- prediction
    def _new_track(self, d: Detection) -> Track:
        z = _xywh(d.bbox)
        h = max(z[3], 1.0)
        kf_img = ConstantVelocityKF(z, q=self.cfg.img_accel_frac * h, r=np.array([0.05, 0.05, 0.1, 0.05]) * h,
                                    v0_std=2.0 * h)
        kf_court = None
        if d.court_xy is not None:
            kf_court = ConstantVelocityKF(d.court_xy, q=self.cfg.court_accel_std,
                                          r=self._sig(d), v0_std=4.0)
        t = Track(self._next_id, kf_img, kf_court, feature=d.feature, last_score=d.score)
        if d.color is not None and d.occlusion < self.cfg.team_max_occ and np.all(np.isfinite(d.color)):
            t.colors.append(np.asarray(d.color, float))
        self._next_id += 1
        return t

    def _predict(self, dt: float, M: np.ndarray | None) -> None:
        for t in self.tracks:
            t.kf_img.predict(dt)
            if self.cfg.use_cmc and M is not None:
                self._warp(t.kf_img, M)
            if t.kf_court is not None:
                t.kf_court.predict(dt)
            t.time_since_update += dt

    @staticmethod
    def _warp(kf: ConstantVelocityKF, M: np.ndarray) -> None:
        """Apply image motion M (prev -> cur) to a box state: center by M, size by the
        local scale of M, velocities by its linear part."""
        cx, cy, w, h = kf.pos
        c = apply_h(M, np.array([[cx, cy]]))[0]
        # local Jacobian of M at the center
        eps = 1.0
        jx = (apply_h(M, np.array([[cx + eps, cy]]))[0] - c) / eps
        jy = (apply_h(M, np.array([[cx, cy + eps]]))[0] - c) / eps
        J = np.column_stack([jx, jy])
        s = np.sqrt(abs(np.linalg.det(J)))
        kf.x[:4] = [c[0], c[1], w * s, h * s]
        kf.x[4:6] = J @ kf.x[4:6]
        kf.x[6:8] *= s
        A = np.eye(8)
        A[:2, :2] = J
        A[4:6, 4:6] = J
        A[2:4, 2:4] *= s
        A[6:8, 6:8] *= s
        kf.P = A @ kf.P @ A.T

    def _sig(self, d: Detection) -> float:
        c = self.cfg
        floor = c.court_min_sigma_m
        if c.court_min_sigma_occluded_m is not None and d.occlusion >= c.occ_thresh:
            floor = c.court_min_sigma_occluded_m
        return max(d.court_sigma_m or 0.3, floor)

    # ---------------------------------------------------------------- costs
    def _cost(self, tracks: list[Track], dets: list[Detection]) -> np.ndarray:
        c = self.cfg
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix(np.array([t.bbox for t in tracks]), np.array([d.bbox for d in dets]))
        cost = np.full(iou.shape, np.inf)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                use_court = (c.use_court and t.kf_court is not None and d.court_xy is not None)
                if use_court:
                    r = self._sig(d)
                    m2 = t.kf_court.mahalanobis2(d.court_xy, r)
                    if m2 > c.court_gate_chi2:
                        continue  # physically impossible association
                    cc = min(1.0, m2 / c.court_gate_chi2)
                    val = c.w_iou * (1 - iou[i, j]) + c.w_court * cc
                    val /= (c.w_iou + c.w_court)
                else:
                    if iou[i, j] <= 0:
                        continue
                    val = 1 - iou[i, j]
                if t.feature is not None and d.feature is not None:
                    if c.app_metric == "euclidean":
                        dist = float(np.linalg.norm(t.feature - d.feature))
                        if c.app_gate is not None and dist > c.app_gate:
                            continue  # clearly different jersey / shirt color
                        app = min(1.0, dist / c.app_scale)
                    else:
                        app = 1 - float(t.feature @ d.feature)
                    if c.w_app > 0:
                        val = (1 - c.w_app) * val + c.w_app * app
                if (c.w_team > 0 or c.team_gate is not None) and d.color is not None \
                        and d.occlusion < c.team_max_occ:
                    proto = t.color_proto(c.team_min_samples)
                    if proto is not None:
                        dist = float(np.linalg.norm(d.color - proto))
                        if c.team_gate is not None and dist > c.team_gate:
                            continue  # clearly a different jersey
                        val = (1 - c.w_team) * val + c.w_team * min(1.0, dist / c.team_scale)
                cost[i, j] = val
        return cost

    @staticmethod
    def _match(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        big = 1e6
        c = np.where(np.isfinite(cost), cost, big)
        r, k = linear_sum_assignment(c)
        matches = [(i, j, float(cost[i, j])) for i, j in zip(r, k) if cost[i, j] <= thresh]
        mi = {m[0] for m in matches}
        mj = {m[1] for m in matches}
        return (matches, [i for i in range(cost.shape[0]) if i not in mi],
                [j for j in range(cost.shape[1]) if j not in mj])

    def _apply(self, t: Track, d: Detection) -> None:
        t.kf_img.update(_xywh(d.bbox))
        if d.court_xy is not None:
            r = self._sig(d)
            if t.kf_court is None:
                t.kf_court = ConstantVelocityKF(d.court_xy, q=self.cfg.court_accel_std, r=r, v0_std=4.0)
            else:
                t.kf_court.update(d.court_xy, r)
        if d.feature is not None:
            if t.feature is None:
                t.feature = np.asarray(d.feature, float)
            elif self.cfg.app_metric == "euclidean":
                t.feature = 0.9 * t.feature + 0.1 * d.feature
            else:
                t.feature = _normalize(0.9 * t.feature + 0.1 * d.feature)
        if d.color is not None and d.occlusion < self.cfg.team_max_occ and np.all(np.isfinite(d.color)):
            t.colors.append(np.asarray(d.color, float))
        t.hits += 1
        t.time_since_update = 0.0
        t.last_score = d.score
        if t.hits >= self.cfg.min_hits:
            t.confirmed = True

    # ---------------------------------------------------------------- main step
    def update(self, dets: list[Detection], M_prev_to_cur: np.ndarray | None = None,
               dt: float | None = None) -> list[Assignment]:
        """Advance one frame. Returns assignments for detections that belong to
        confirmed tracks (tentative tracks are not reported)."""
        c = self.cfg
        dt = 1.0 / self.fps if dt is None else dt
        self._predict(dt, M_prev_to_cur)
        hi = [j for j, d in enumerate(dets) if d.score >= c.high_thresh]
        lo = [j for j, d in enumerate(dets) if c.low_thresh <= d.score < c.high_thresh]
        confirmed = [t for t in self.tracks if t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]
        out: list[tuple[Track, int, float]] = []

        # Stage 1: confirmed (incl. recently lost) tracks vs high-score detections
        m, ut, ud = self._match(self._cost(confirmed, [dets[j] for j in hi]), c.match_thresh)
        for i, j, cst in m:
            out.append((confirmed[i], hi[j], cst))
        rem_tracks = [confirmed[i] for i in ut if confirmed[i].time_since_update <= 1.5 / self.fps + dt]
        rem_hi = [hi[j] for j in ud]

        # Stage 2: still-active tracks vs low-score detections, IoU only
        if rem_tracks and lo:
            iou = iou_matrix(np.array([t.bbox for t in rem_tracks]), np.array([dets[j].bbox for j in lo]))
            m2, _, _ = self._match(np.where(iou > 0, 1 - iou, np.inf), 1 - c.low_match_iou)
            for i, j, cst in m2:
                out.append((rem_tracks[i], lo[j], cst))

        # Stage 3: tentative tracks vs remaining high detections
        m3, ut3, ud3 = self._match(self._cost(tentative, [dets[j] for j in rem_hi]), c.match_thresh)
        for i, j, cst in m3:
            out.append((tentative[i], rem_hi[j], cst))
        dead_tentative = {id(tentative[i]) for i in ut3}
        unmatched_hi = [rem_hi[j] for j in ud3]

        for t, j, _ in out:
            self._apply(t, dets[j])
        # Remove unmatched tentative tracks and long-lost tracks; start new tracks.
        self.tracks = [t for t in self.tracks if id(t) not in dead_tentative
                       and t.time_since_update <= c.max_time_lost_s]
        for j in unmatched_hi:
            if dets[j].score >= c.new_track_thresh:
                self.tracks.append(self._new_track(dets[j]))
        return [Assignment(t.track_id, j, cst) for t, j, cst in out if t.confirmed]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)
