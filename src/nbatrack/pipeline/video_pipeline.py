"""Real-video pipeline.

Stage A  analyze_video : decode -> RF-DETR detection -> camera calibration (keyframe
         landmarks + court-line alignment, propagated by KLT camera motion and re-aligned
         every frame) -> cached detections + calibration tables.      (slow: neural net)
Stage B  track_and_export : cached detections -> bbox-bottom ground points -> court
         coordinates with sigma -> tracking (CMC + court-space) -> off-court / team /
         referee labeling -> temporal smoothing -> validated tables.   (fast: re-runnable)
Stage C  viz/render.py renders videos from the tables.

Calibration quality here is a SELF-CONSISTENCY measure (fraction of the projected court
template that lands on detected markings); it is not ground-truth accuracy.
"""

from __future__ import annotations

import bisect
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from nbatrack import SCHEMA_VERSION, __version__
from nbatrack.calib.camera import PinholeCamera, ptz_rotation
from nbatrack.calib.lines import (
    alignment_stats,
    build_evidence,
    floor_iou,
    floor_tone_mask,
    floor_valid_mask,
    initialize_from_landmarks,
    refine_camera,
    template_points,
)
from nbatrack.calib.manual import ManualCalibrationSession
from nbatrack.calib.models import CameraCalibration
from nbatrack.calib.propagate import camera_from_motion, estimate_motion, mask_boxes
from nbatrack.court.geometry import court_landmarks, is_on_court
from nbatrack.detect.rfdetr_onnx import COCO_PERSON, COCO_SPORTS_BALL, RFDETRDetector, nms
from nbatrack.export.writers import write_rows
from nbatrack.filter.temporal import raw_speed_violations, rts_smooth
from nbatrack.io.video import iter_frames, probe
from nbatrack.project.projector import bbox_bottom_points, project_ground_points
from nbatrack.segments.overlay import estimate_overlay_mask
from nbatrack.track.identity import label_change_rows, segment_ids, split_on_color_change
from nbatrack.track.ball import track_ball
from nbatrack.track.offline import StitchConfig, interpolate_gaps, stitch_tracklets
from nbatrack.track.team_init import TeamInit, classify_detections, prototypes, track_labels
from nbatrack.track.teams import cluster_tracks_lab, torso_lab
from nbatrack.track.tracker import Detection, Tracker, TrackerConfig, iou_matrix


@dataclass
class AnalyzeConfig:
    detector_onnx: str = "models/onnx/rfdetr_small_512x896/rfdetr-small.onnx"
    provider: str = "dml"
    det_score_min: float = 0.1
    calib_mode: str = "ptz"  # "ptz": center fixed at the keyframe estimate; "free": per frame
    # Calibration grading (grade_calibration). Line-alignment inliers alone accepted 27.6% of
    # deliberately wrong poses on clips 1-5 (wrong tilt reaches 0.85: crowd texture "explains"
    # court lines); floor IoU rejects tilt errors but not pan/focal errors. Requiring both
    # (inlier >= 0.70 AND floor IoU >= 0.45) accepted 0/980 wrong poses and rejected 1/98
    # visually verified ones (tools/calib_gate_study.py, docs/OVERNIGHT_REPORT.md).
    q_good: float = 0.75  # thresholds on alignment inlier fraction (self-consistency)
    q_fair: float = 0.70  # was 0.55 before the gate study
    q_poor: float = 0.35
    floor_iou_ok: float = 0.45  # good/fair also require this floor IoU
    floor_iou_poor: float = 0.30  # below: failed whatever the inliers
    # temporal plausibility of the per-frame camera change (broadcast PTZ heads)
    max_pan_deg_s: float = 90.0
    max_tilt_deg_s: float = 45.0
    max_zoom_frac_s: float = 1.0
    min_template_points: int = 150
    max_frames: int | None = None


# Per-detection torso color is too noisy for association gating (clip-2 study: gating at
# 30 Lab units fragmented tracks 2x); color is used post hoc instead (track/identity.py).
# Birth thresholds lowered after the 2026-10-15 user review: players entering the view or near
# the cluttered far sideline score 0.5-0.6 and were never given a track (untracked on-court
# detections -60..-80%, IDs and interior births not increased; see REAL_VIDEO_REPORT).
DEFAULT_TRACKER = TrackerConfig(use_cmc=True, use_court=True, court_min_sigma_m=1.0,
                                new_track_thresh=0.4, high_thresh=0.4, min_hits=3, max_time_lost_s=1.5)


@dataclass
class TrackConfig:
    on_court_margin_m: float = 1.0  # detections farther off-court are not tracked
    offcourt_track_margin_m: float = 0.3  # tracks whose MEDIAN position is off-court -> "other"
    calib_min_sigma_px: float = 2.0
    min_track_frames_for_team: int = 5
    # Color-based identity check: by default only FLAG rows after a suspected cross-class
    # switch ("suspected_id_switch"); splitting is opt-in because on clip 2 it also split
    # referee tracks whose shirt numbers confuse the color class (unverifiable without GT).
    split_on_color: bool = False
    # Manual per-video team initialization (teams.json with exemplars); requires
    # cache/appearance.npy (tools/team_init_helper.py render). None -> automatic k-means.
    team_init: str | None = None
    # Offline reconstruction (docs/OFFLINE_RECONSTRUCTION.md). Stitching is OFF by default
    # until validated on reviewed ground truth; short-gap interpolation adds separate
    # rows with position_status="interpolated" and never alters observed rows.
    stitch: bool = False
    stitch_cfg: StitchConfig | None = None
    interpolate_max_gap_s: float | None = 1.0
    # Real-footage default (clip-2 study): bbox-bottom court positions jump during
    # occlusions, so gating needs a 1 m sigma floor (0.05 m fragmented tracks 2.5x).
    tracker: TrackerConfig | None = None
    # Ball candidate source for the offline ball tracker (docs/OPEN_SOURCE_TRACKING_REVIEW.md):
    # "wasb" = WASB basketball heatmap model (ONNX; candidates cached in
    # cache/wasb_ball_candidates.parquet), "rfdetr" = COCO `sports ball` from the cached
    # RF-DETR detections. A random-frame visual spot check (assistant-judged, not GT) found
    # 0.98 vs 0.59 precision; falls back to "rfdetr" if the WASB ONNX model is missing.
    ball_detector: str = "wasb"
    ball_provider: str = "dml"


def _is_ref_center(c, centers) -> bool:
    from nbatrack.track.teams import REF_CHROMA_TOL, REF_L, REF_L_TOL
    score = lambda x: abs(x[0] - REF_L) / REF_L_TOL + np.hypot(x[1], x[2]) / REF_CHROMA_TOL
    return bool(np.argmin([score(x) for x in centers]) == [tuple(x) for x in centers].index(tuple(c)))


def grade_calibration(inlier: float, n: int, floor_iou_v: float | None, cfg: AnalyzeConfig,
                      motion_implausible: bool = False) -> tuple[str, str]:
    """(quality, reason). Only good / fair frames get court coordinates downstream; poor =
    calibration_uncertain, failed = calibration_failed. Image-space tracking is unaffected.
    Each check is necessary, none is sufficient: they are self-consistency checks, not GT."""
    if n < cfg.min_template_points:
        return "failed", "too_few_template_points"
    iou_ok = floor_iou_v is None or not np.isfinite(floor_iou_v) or floor_iou_v >= cfg.floor_iou_ok
    if floor_iou_v is not None and np.isfinite(floor_iou_v) and floor_iou_v < cfg.floor_iou_poor:
        return "failed", "floor_iou_very_low"
    if inlier < cfg.q_poor:
        return "failed", "line_inliers_very_low"
    if inlier < cfg.q_fair:
        return "poor", "line_inliers_low"
    if not iou_ok:
        return "poor", "floor_iou_low"
    if motion_implausible:
        return "poor", "implausible_camera_motion"
    return ("good" if inlier >= cfg.q_good else "fair"), "ok"


def _cam_row(video_id, fi, ts, cam, method, model, quality, inlier, n_pts, seg=0) -> dict:
    r = dict(video_id=video_id, frame_index=fi, timestamp_sec=ts, camera_segment_id=seg,
             method=method, model=model if cam is not None else "none", quality=quality,
             quality_score=inlier, n_inliers=None, n_landmarks=None)
    if cam is not None:
        H = cam.ground_homography()
        Hi = np.linalg.inv(H)
        pan, tilt, roll = cam.ptz
        r.update(H_court_to_img=(H / H[2, 2]).ravel(), H_img_to_court=(Hi / Hi[2, 2]).ravel(),
                 focal_px=cam.f, cx=cam.cx, cy=cam.cy, k1=cam.k1, pan_deg=pan, tilt_deg=tilt,
                 roll_deg=roll, cam_x_m=cam.C[0], cam_y_m=cam.C[1], cam_z_m=cam.C[2])
    return r


def camera_from_row(r, image_size) -> PinholeCamera:
    return PinholeCamera(r.focal_px, r.cx, r.cy, ptz_rotation(r.pan_deg, r.tilt_deg, r.roll_deg),
                         np.array([r.cam_x_m, r.cam_y_m, r.cam_z_m]), r.k1, image_size)


class _Mem:
    def __init__(self):
        self.p = psutil.Process()
        self.peak = 0.0

    def __call__(self):
        self.peak = max(self.peak, self.p.memory_info().private / 2**20)


def load_segments(segments_json: str | Path | None, keyframe_json: str | Path | None, n_frames: int):
    """Camera segments + flash frames. Without a segments file: one supported main-broadcast
    segment calibrated from `keyframe_json` (the single-shot behaviour of clips 1-5).

    segments.json (see data/annotations/clip6/segments.json):
      {"segments": [{"camera_segment_id", "start_frame", "end_frame", "view_type",
                     "supported", "keyframe"}...], "flash_frames": [...]}
    Unsupported segments (close-ups, replays, ...) get no calibration and no court positions.
    """
    if segments_json is None:
        return [dict(camera_segment_id=0, start_frame=0, end_frame=10**12, view_type="main_broadcast",
                     supported=True, keyframe=str(keyframe_json), boundary_method="manual_review_single_shot")], set()
    d = json.loads(Path(segments_json).read_text(encoding="utf-8"))
    segs = sorted(d["segments"], key=lambda s: s["start_frame"])
    for a, b in zip(segs, segs[1:]):
        if b["start_frame"] != a["end_frame"] + 1:
            raise ValueError(f"segments must be contiguous: {a} / {b}")
    for s in segs:
        s.setdefault("boundary_method", "hist_cut_detection+visual_review")
        if s["supported"] and not s.get("keyframe"):
            raise ValueError(f"supported segment {s['camera_segment_id']} needs a keyframe")
    return segs, set(int(f) for f in d.get("flash_frames", []))


_DET_COLS = ["frame_index", "timestamp_sec", "cls", "score", "x1", "y1", "x2", "y2", "torso_L", "torso_a", "torso_b"]


# ======================================================================== stage A
def analyze_video(video_path: str | Path, out_dir: str | Path, keyframe_json: str | Path | None,
                  video_id: str, cfg: AnalyzeConfig | None = None,
                  segments_json: str | Path | None = None) -> dict:
    """Stage A. Camera segments are calibrated independently (own keyframe, forward and
    backward propagation inside the segment only); nothing is propagated across a cut.
    Flash frames (single over-exposed frames) keep the previous frame's camera
    (method "held_flash", quality "poor" so no court positions are emitted for them) and are
    skipped as motion references."""
    cfg = cfg or AnalyzeConfig()
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    mem = _Mem()
    t_all = time.time()
    info = probe(video_path, with_hash=True)
    W, H = info.width, info.height
    fps = info.fps_nominal

    n_rep = info.n_frames_reported or int(info.duration_sec * fps)
    sample = [f.image for f in iter_frames(video_path, stride=max(1, n_rep // 30))]
    overlay = estimate_overlay_mask(sample)
    del sample
    cv2.imwrite(str(out / "overlay_mask.png"), overlay)

    segs, flashes = load_segments(segments_json, keyframe_json, n_rep)
    seg_starts = [s["start_frame"] for s in segs]
    kfs = {s["camera_segment_id"]: ManualCalibrationSession.load(s["keyframe"]) for s in segs if s["supported"]}
    for s in segs:
        if s["supported"] and not s["start_frame"] <= kfs[s["camera_segment_id"]].frame_index <= s["end_frame"]:
            raise ValueError(f"keyframe of segment {s['camera_segment_id']} lies outside the segment")
    det = RFDETRDetector(cfg.detector_onnx, cfg.provider)
    tpl = template_points()
    t_stage = {"decode": [], "detect": [], "calib": []}
    rows_frames, rows_cal, rows_lm, calib_log = [], [], [], []
    det_chunks: list[np.ndarray] = []  # per-frame (k, 11) arrays; dicts would need ~1 KB per row
    model_name = "ptz" if cfg.calib_mode == "ptz" else "camera_free"
    state = {"cam": None, "fixed_C": None, "gray": None, "boxes": None, "prev": None}

    def calibrate(fi, ts, img, occl, direction, seg):
        """One calibration step inside camera segment `seg`. direction: "key" (keyframe),
        "fwd" or "bwd" (propagate from the neighbouring, already calibrated frame)."""
        t0 = time.time()
        sid = seg["camera_segment_id"]
        if direction != "key" and fi in flashes:
            # over-exposed frame: markings and KLT motion are unreliable -> hold the camera; the
            # next frame's motion is measured against the last normal frame (state unchanged)
            cam = state["cam"]
            rows_cal.append(_cam_row(video_id, fi, ts, cam, "held_flash", model_name, "poor", None, None, sid))
            calib_log.append(dict(frame=fi, t=ts, segment=sid, method="held_flash", inlier=np.nan, n=0, quality="poor",
                                  grade_reason="flash_frame_held", motion_ok=False, direction=direction, floor_iou=np.nan, pan=cam.ptz[0],
                                  tilt=cam.ptz[1], roll=cam.ptz[2], f=cam.f, Cx=cam.C[0], Cy=cam.C[1], Cz=cam.C[2]))
            t_stage["calib"].append(time.time() - t0)
            return
        prev = None if direction == "key" else state["prev"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        method, motion_ok = "propagated", True
        if direction == "key":
            kf = kfs[sid]
            best, _ = initialize_from_landmarks(img, kf.correspondences(), overlay, occl)
            if best is None:
                raise RuntimeError(f"keyframe calibration failed (segment {sid}, frame {fi})")
            state["cam"] = best.camera
            state["fixed_C"] = best.camera.C.copy() if cfg.calib_mode == "ptz" else None
            method = "keypoint"
            for lid, (u, v) in kf.points.items():
                rows_lm.append(dict(video_id=video_id, frame_index=fi, timestamp_sec=ts,
                                    landmark_id=int(lid), image_x=u, image_y=v, source="manual"))
        else:
            me = estimate_motion(state["gray"], gray, mask_boxes(255 - overlay, state["boxes"], pad=8))
            motion_ok = me.ok
            if me.ok:
                pred = camera_from_motion(state["cam"], me.M, state["fixed_C"])
                if pred is not None:
                    state["cam"] = pred
        cam = state["cam"]
        ev = build_evidence(img, floor_valid_mask(cam, (H, W), 1.0, occl, overlay))
        if method != "keypoint":
            r = refine_camera(cam, ev, tpl, fixed_C=state["fixed_C"], scales=(6.0, 3.0, 1.5))
            _, f0, _ = alignment_stats(cam, ev, tpl)
            if r.success and r.inlier_frac >= f0:
                cam = state["cam"] = r.camera
        n_pts, inl, _ = alignment_stats(cam, ev, tpl)
        # floor IoU: check independent of the line alignment (which crowd texture can fool)
        fiou = floor_iou(cam, floor_tone_mask(img), overlay)
        implausible = False
        if prev is not None:  # camera change vs. the neighbouring frame (fwd or bwd)
            dt = abs(ts - prev[0]) or 1.0 / fps
            implausible = (abs(cam.ptz[0] - prev[1][0]) / dt > cfg.max_pan_deg_s
                           or abs(cam.ptz[1] - prev[1][1]) / dt > cfg.max_tilt_deg_s
                           or abs(np.log(cam.f / prev[2])) / dt > cfg.max_zoom_frac_s)
        q, reason = grade_calibration(inl, n_pts, fiou, cfg, implausible)
        rows_cal.append(_cam_row(video_id, fi, ts, cam, method, model_name, q, inl, n_pts, sid))
        state["prev"] = (ts, cam.ptz, cam.f)
        calib_log.append(dict(frame=fi, t=ts, segment=sid, method=method, inlier=inl, n=n_pts, quality=q,
                              grade_reason=reason, motion_ok=motion_ok, direction=direction,
                              floor_iou=fiou,
                              pan=cam.ptz[0], tilt=cam.ptz[1], roll=cam.ptz[2],
                              f=cam.f, Cx=cam.C[0], Cy=cam.C[1], Cz=cam.C[2]))
        state["gray"], state["boxes"] = gray, occl
        t_stage["calib"].append(time.time() - t0)

    # Single streaming pass: decode + detect every frame; per camera segment, calibrate from
    # its keyframe forward; frames of the segment before its keyframe are kept JPEG-compressed
    # and calibrated backward right after the keyframe.
    before: list = []
    cur = None
    t_dec = time.time()
    for fr in iter_frames(video_path):
        if cfg.max_frames is not None and fr.frame_index >= cfg.max_frames:
            break
        t_stage["decode"].append(time.time() - t_dec)
        fi, ts, img = fr.frame_index, fr.timestamp_sec, fr.image
        seg = segs[max(0, bisect.bisect_right(seg_starts, fi) - 1)]
        sid = seg["camera_segment_id"]
        if seg is not cur:  # cut: nothing carries over
            if before:
                raise ValueError(f"keyframe of segment {cur['camera_segment_id']} was not reached")
            cur = seg
            state.update(cam=None, fixed_C=None, gray=None, boxes=None, prev=None)
        rows_frames.append(dict(video_id=video_id, frame_index=fi, timestamp_sec=ts, pts=fr.pts,
                                pts_sec=fr.pts_sec, camera_segment_id=sid))
        t0 = time.time()
        d = det(img, score_thresh=cfg.det_score_min)
        keep = nms(d.boxes, d.scores, 0.7)
        boxes, scores, classes = d.boxes[keep], d.scores[keep], d.classes[keep]
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2).clip(0, W - 1).astype(int)
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2).clip(0, H - 1).astype(int)
        in_overlay = overlay[cy, cx] > 0 if len(boxes) else np.zeros(0, bool)
        rows = []
        for k in range(len(boxes)):
            if in_overlay[k] or classes[k] not in (COCO_PERSON, COCO_SPORTS_BALL):
                continue
            L, a, b = torso_lab(img, boxes[k]) if classes[k] == COCO_PERSON else (np.nan,) * 3
            rows.append((fi, ts, classes[k], scores[k], *boxes[k, :4], L, a, b))
        if rows:
            det_chunks.append(np.array(rows, np.float64))
        pm = (classes == COCO_PERSON) & ~in_overlay
        occl = boxes[pm][scores[pm] >= 0.3]
        t_stage["detect"].append(time.time() - t0)
        if not seg["supported"]:
            rows_cal.append(_cam_row(video_id, fi, ts, None, "none", model_name, "failed", None, None, sid))
            calib_log.append(dict(frame=fi, t=ts, segment=sid, method="unsupported_view", quality="failed",
                                  direction="none"))
        elif fi < kfs[sid].frame_index:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            before.append((fi, ts, buf, occl))
        else:
            is_key = fi == kfs[sid].frame_index
            calibrate(fi, ts, img, occl, "key" if is_key else "fwd", seg)
            if is_key and before:  # backward to the segment start, then resume forward
                saved = dict(state)
                for bfi, bts, buf, bocc in reversed(before):
                    calibrate(bfi, bts, cv2.imdecode(buf, cv2.IMREAD_COLOR), bocc, "bwd", seg)
                    mem()
                before.clear()
                state.update(saved)
        mem()
        if (fi + 1) % 300 == 0:  # progress (a stalled or suspended run is visible)
            el = time.time() - t_all
            tot = cfg.max_frames or n_rep
            prog = dict(frame=fi, of=tot, elapsed_s=round(el), eta_s=round(el / (fi + 1) * (tot - fi - 1)),
                        peak_private_mb=round(mem.peak), at=datetime.now().isoformat(timespec="seconds"))
            (out / "progress.json").write_text(json.dumps(prog), encoding="utf-8")
            print(f"[stage A] {prog}", flush=True)
        t_dec = time.time()
    if before:
        raise ValueError(f"keyframe of segment {cur['camera_segment_id']} was not reached")
    rows_cal.sort(key=lambda r: r["frame_index"])
    calib_log.sort(key=lambda r: r["frame"])
    del det
    n_frames = len(rows_frames)
    last = rows_frames[-1]["frame_index"]
    ts_of = {r["frame_index"]: r["timestamp_sec"] for r in rows_frames}
    tables = out / "tables"
    write_rows("videos", [dict(video_id=video_id, source_path=str(video_path), sha256=info.sha256,
                               width=W, height=H, fps_nominal=fps, time_base=info.time_base,
                               start_sec=info.start_sec, duration_sec=info.duration_sec,
                               n_frames=n_frames, codec=info.codec, pipeline_version=__version__,
                               created_at=datetime.now(timezone.utc).isoformat())], tables)
    write_rows("frames", rows_frames, tables)
    write_rows("camera_segments", [dict(video_id=video_id, camera_segment_id=s["camera_segment_id"],
                                        start_frame=s["start_frame"], end_frame=min(s["end_frame"], last),
                                        start_sec=ts_of[s["start_frame"]], end_sec=ts_of[min(s["end_frame"], last)],
                                        view_type=s["view_type"] if s["view_type"] in ("main_broadcast", "unknown") else "other",
                                        supported=bool(s["supported"]), boundary_method=s["boundary_method"])
                                   for s in segs if s["start_frame"] <= last], tables)
    write_rows("calibration", rows_cal, tables)
    write_rows("court_landmarks", [dict(landmark_id=lm.landmark_id, name=lm.name, court_x_m=lm.xyz[0],
                                        court_y_m=lm.xyz[1], court_z_m=lm.xyz[2],
                                        description=lm.description) for lm in court_landmarks()], tables)
    write_rows("landmark_observations", rows_lm, tables)
    dets = pd.DataFrame(np.vstack(det_chunks) if det_chunks else np.zeros((0, len(_DET_COLS))), columns=_DET_COLS)
    dets["frame_index"] = dets.frame_index.astype(np.int64)
    dets["cls"] = dets.cls.astype(np.int64)
    dets.to_parquet(out / "cache" / "detections.parquet")
    pd.DataFrame(calib_log).to_csv(out / "calibration_diagnostics.csv", index=False)
    elapsed = time.time() - t_all
    run = dict(video_id=video_id, source=str(video_path), sha256=info.sha256, n_frames=n_frames,
               fps=fps, width=W, height=H, config=asdict(cfg),
               ms_per_frame={k: 1000 * float(np.mean(v)) for k, v in t_stage.items() if v},
               ms_per_frame_p95={k: 1000 * float(np.percentile(v, 95)) for k, v in t_stage.items() if v},
               total_seconds=elapsed, processing_fps=n_frames / elapsed,
               peak_private_memory_mb=mem.peak,
               keyframe_annotator="; ".join(sorted({k.annotator for k in kfs.values()})),
               keyframe=str(keyframe_json) if segments_json is None else None,
               segments=str(segments_json) if segments_json is not None else None,
               camera_segments=[{k: s[k] for k in ("camera_segment_id", "start_frame", "end_frame", "view_type", "supported")}
                                for s in segs], flash_frames=sorted(flashes))
    (out / "run_analyze.json").write_text(json.dumps(run, indent=2, default=float), encoding="utf-8")
    return run


# ======================================================================== stage B
def _ball_candidates(cfg: TrackConfig, out: Path, run_a: dict, dets: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Ball candidate rows (detections-cache layout) for track_ball, and their source."""
    if cfg.ball_detector == "rfdetr":
        return dets, "rfdetr"
    if cfg.ball_detector != "wasb":
        raise ValueError(f"unknown ball_detector {cfg.ball_detector!r}")
    from nbatrack.detect.wasb_ball import DEFAULT_ONNX, WASBBallDetector, detect_video, to_detection_rows
    cache = out / "cache" / "wasb_ball_candidates.parquet"
    if cache.exists():
        return to_detection_rows(pd.read_parquet(cache)), "wasb"
    onnx = Path(__file__).resolve().parents[3] / DEFAULT_ONNX
    if not onnx.exists():
        print(f"[ball] WASB model {onnx} missing -> falling back to RF-DETR sports-ball candidates")
        return dets, "rfdetr (fallback)"
    try:
        det = WASBBallDetector(onnx, provider=cfg.ball_provider)
    except Exception as e:  # DirectML unavailable
        print(f"[ball] {cfg.ball_provider} provider failed ({e}); using CPU")
        det = WASBBallDetector(onnx, provider="cpu")
    c, ms = detect_video(run_a["source"], det, step=1, max_frames=run_a["n_frames"])
    c.to_parquet(cache)
    print(f"[ball] WASB: {len(c)} candidates, {ms:.1f} ms/frame")
    return to_detection_rows(c), "wasb"


def track_and_export(out_dir: str | Path, cfg: TrackConfig | None = None) -> dict:
    cfg = cfg or TrackConfig()
    out = Path(out_dir)
    tables = out / "tables"
    t0 = time.time()
    run_a = json.loads((out / "run_analyze.json").read_text(encoding="utf-8"))
    video_id, fps = run_a["video_id"], run_a["fps"]
    size = (run_a["width"], run_a["height"])
    cal = pd.read_parquet(tables / "calibration.parquet").set_index("frame_index")
    dets = pd.read_parquet(out / "cache" / "detections.parquet")
    by_frame = dict(tuple(dets.groupby("frame_index")))
    frames = pd.read_parquet(tables / "frames.parquet")
    if "camera_segment_id" not in frames.columns:
        frames["camera_segment_id"] = 0
    rows_pl, rows_ball, lab_feats = [], [], {}
    H_prev, seg_cur, tracker = None, None, None
    for fi, ts, sid in zip(frames.frame_index, frames.timestamp_sec, frames.camera_segment_id):
        if sid != seg_cur:  # camera cut: new tracker, IDs never continue across segments
            seg_cur, H_prev = sid, None
            tracker = Tracker(cfg.tracker or DEFAULT_TRACKER, fps)
        c = cal.loc[fi]
        q = c.quality
        cam = camera_from_row(c, size) if c.model != "none" else None
        Hc = np.asarray(c.H_court_to_img, float).reshape(3, 3) if cam is not None else None
        M = None if (H_prev is None or Hc is None) else Hc @ np.linalg.inv(H_prev)
        H_prev = Hc
        emit = q in ("good", "fair")
        sigma_px = cfg.calib_min_sigma_px
        g = by_frame.get(fi)
        persons = g[g.cls == COCO_PERSON] if g is not None else None
        tdets, meta = [], []
        if persons is not None and len(persons) and cam is not None:
            boxes = persons[["x1", "y1", "x2", "y2"]].to_numpy(float)
            gp = bbox_bottom_points(boxes)
            strong = persons.score.to_numpy() >= 0.3
            ovl = iou_matrix(boxes, boxes[strong]) if strong.any() else np.zeros((len(boxes), 0))
            if ovl.size:
                ovl[np.arange(len(boxes))[strong], np.arange(strong.sum())] = 0.0  # ignore self
            occlusion = ovl.max(1) if ovl.size else np.zeros(len(boxes))
            cp = project_ground_points(CameraCalibration(cam), gp, calib_rmse_px=sigma_px,
                                       on_court_margin_m=cfg.on_court_margin_m)
            for k, row in enumerate(persons.itertuples()):
                xy = cp.xy[k]
                finite = bool(np.all(np.isfinite(xy)))
                if q != "failed" and finite and not cp.on_court[k]:
                    continue  # bench / spectators / staff
                lab = np.array([row.torso_L, row.torso_a, row.torso_b], float)
                use_app = (cfg.tracker or DEFAULT_TRACKER).w_app > 0 and np.all(np.isfinite(lab))
                tdets.append(Detection(boxes[k], float(row.score), xy if (emit and finite) else None,
                                       float(cp.sigma_m[k]) if (emit and finite) else None,
                                       lab if use_app else None, occlusion=float(occlusion[k]),
                                       color=lab if np.all(np.isfinite(lab)) else None))
                meta.append((gp.uv[k], xy if finite else None, float(cp.sigma_m[k]) if finite else None,
                             (row.torso_L, row.torso_a, row.torso_b), row.Index, float(occlusion[k])))
        for a in tracker.update(tdets, M):
            d = tdets[a.det_index]
            uv, xy, sig, lab, det_idx, occ = meta[a.det_index]
            rows_pl.append(dict(
                _det_idx=det_idx, _occ=occ, torso_L=lab[0], torso_a=lab[1], torso_b=lab[2], video_id=video_id, frame_index=fi, timestamp_sec=ts, camera_segment_id=int(sid),
                track_id=f"s{int(sid):03d}_t{a.track_id:03d}", role="unknown",
                bbox_x1=d.bbox[0], bbox_y1=d.bbox[1], bbox_x2=d.bbox[2], bbox_y2=d.bbox[3],
                det_confidence=d.score, assoc_confidence=a.confidence, ground_method="bbox_bottom",
                image_x=uv[0], image_y=uv[1],
                court_x_m_raw=xy[0] if (xy is not None and q != "failed") else None,
                court_y_m_raw=xy[1] if (xy is not None and q != "failed") else None,
                court_sigma_m=sig if (emit and xy is not None) else None,
                position_status="observed_projected" if (emit and xy is not None) else "unknown",
                calibration_quality=q, is_visible=True,
                is_on_court=bool(is_on_court(xy[None], cfg.on_court_margin_m)[0]) if xy is not None else None))
        if g is not None:
            for b in g[g.cls == COCO_SPORTS_BALL].itertuples():
                rows_ball.append(dict(video_id=video_id, frame_index=fi, timestamp_sec=ts,
                                      camera_segment_id=int(sid), bbox_x1=b.x1, bbox_y1=b.y1, bbox_x2=b.x2,
                                      bbox_y2=b.y2, image_x=(b.x1 + b.x2) / 2, image_y=(b.y1 + b.y2) / 2,
                                      det_confidence=b.score, is_visible=True,
                                      position_3d_status="unknown"))
    pl = pd.DataFrame(rows_pl)
    # ---- off-court tracks
    off = set()
    for tid, g in pl.groupby("track_id"):
        xy = g[["court_x_m_raw", "court_y_m_raw"]].dropna().to_numpy(float)
        if len(xy) and not is_on_court(np.median(xy, axis=0)[None], cfg.offcourt_track_margin_m)[0]:
            off.add(tid)

    def track_feats(df):
        out = {}
        for tid, g in df.groupby("track_id"):
            X = g[["torso_L", "torso_a", "torso_b"]].dropna().to_numpy(float)
            if len(X) >= cfg.min_track_frames_for_team and tid not in off:
                out[tid] = np.median(X, axis=0)
        return out

    # ---- team / referee labels; class changes inside a track = suspected ID switch
    n_splits = 0
    suspect = pd.Series(False, index=pl.index)
    team_info: dict = {}
    pl["source_track_id"] = pl["track_id"]
    app_path = out / "cache" / "appearance.npy"
    desc_all = np.load(app_path, mmap_mode="r") if app_path.exists() else None  # disk-backed
    protos = {}
    if cfg.team_init and desc_all is not None:
        protos = prototypes(TeamInit.load(cfg.team_init), dets, desc_all)
        D = desc_all[dets.index.get_indexer(pl["_det_idx"])]
        votes, _ = classify_detections(D, protos, pl["_occ"].to_numpy(float))
        pl["team_vote"] = votes
    # ---- offline tracklet stitching (bidirectional: uses fragment ends AND starts)
    stitch_links: list = []
    if cfg.stitch:
        posmap = dict(zip(pl.index, dets.index.get_indexer(pl["_det_idx"])))

        def desc_of_row(i):
            if desc_all is None:
                return None
            v = desc_all[posmap[i]]
            return v if np.all(np.isfinite(v)) else None

        pl, stitch_links = stitch_tracklets(pl, fps, desc_of_row,
                                            "team_vote" if "team_vote" in pl.columns else None,
                                            cfg.stitch_cfg, exclude=off)
    if cfg.team_init and desc_all is not None:
        on = ~pl.track_id.isin(off)
        # labels per (track, segment): a segment ends at a persistent class change
        pl["_seg"] = segment_ids(pl[on], "team_vote", min_run=int(round(fps)),
                                 window=int(round(fps)) | 1).reindex(pl.index, fill_value=0)
        pl["_tseg"] = pl.track_id + "#" + pl["_seg"].astype(str)
        lbl = track_labels(pl[on].assign(track_id=pl.loc[on, "_tseg"]))
        teams = {tid: (v[0], v[1]) for tid, v in lbl.items()}
        suspect = label_change_rows(pl[on], "team_vote", min_run=int(round(fps)), window=int(round(fps)) | 1)
        suspect = suspect.reindex(pl.index, fill_value=False)
        n_splits = int(sum(g.astype(int).diff().fillna(0).gt(0).sum() + int(g.iloc[0])
                           for _, g in suspect.groupby(pl.track_id)))
        team_info = {"team_method": "manual_exemplars", "team_init": str(cfg.team_init),
                     "prototypes": {k: int(len(v)) for k, v in protos.items()},
                     "votes_per_track": {k: {"label": v[0], "vote_frac": round(v[1], 3), "n_votes": v[2]}
                                         for k, v in lbl.items()}}
    else:
        teams, centers = cluster_tracks_lab(track_feats(pl), return_centers=True)
        team_info = {"team_method": "auto_kmeans"}
        if centers is not None:
            on = ~pl.track_id.isin(off)
            ref_idx = [i for i, c in enumerate(centers) if _is_ref_center(c, centers)]
            part, n_splits = split_on_color_change(pl[on], centers, min_run=int(round(fps)),
                                                   window=int(round(fps)) | 1, ignore=tuple(ref_idx))
            changed = part.track_id != pl.loc[part.index, "track_id"]
            suspect.loc[changed[changed].index] = True
            if cfg.split_on_color:
                pl = pd.concat([part, pl[~on]]).sort_values(["frame_index", "track_id"])
                suspect = pd.Series(False, index=pl.index)
                teams = cluster_tracks_lab(track_feats(pl))
    for k in off:
        teams[k] = ("OFF", 0.0)
    pl["team_id"] = None
    pl["team_confidence"] = None
    pl["flags"] = None
    pl["court_x_m"] = np.nan
    pl["court_y_m"] = np.nan
    role_map = {"REF": "referee", "A": "player", "B": "player", "OFF": "other"}
    if "_tseg" in pl.columns:  # manual exemplars: per-segment labels
        for ts_id, gs in pl.groupby("_tseg"):
            tid = ts_id.split("#")[0]
            lab, conf = teams.get(tid, None) or teams.get(ts_id, ("UNK", 0.0))
            pl.loc[gs.index, "role"] = role_map.get(lab, "unknown")
            if lab in ("A", "B"):
                pl.loc[gs.index, "team_id"] = lab
                pl.loc[gs.index, "team_confidence"] = conf
    for tid, g in pl.groupby("track_id"):
        lab, conf = teams.get(tid, ("UNK", 0.0))
        if "_tseg" not in pl.columns or lab == "OFF":
            pl.loc[g.index, "role"] = role_map.get(lab, "unknown")
            if lab in ("A", "B"):
                pl.loc[g.index, "team_id"] = lab
                pl.loc[g.index, "team_confidence"] = conf
        flags = pd.Series([None] * len(g), index=g.index, dtype=object)
        if lab == "OFF":
            flags[:] = "off_court_track"
        for i in g.index[suspect.reindex(g.index, fill_value=False).to_numpy()]:
            flags[i] = "suspected_id_switch" if flags[i] is None else flags[i] + ";suspected_id_switch"
        e = g[g.position_status == "observed_projected"].sort_values("timestamp_sec")
        if len(e) >= 2:
            t = e.timestamp_sec.to_numpy()
            xy = e[["court_x_m_raw", "court_y_m_raw"]].to_numpy(float)
            sg = e.court_sigma_m.to_numpy(float)
            sm, _ = rts_smooth(t, xy, sg)
            pl.loc[e.index, "court_x_m"] = sm[:, 0]
            pl.loc[e.index, "court_y_m"] = sm[:, 1]
            viol = raw_speed_violations(t, xy, sg)
            for i in e.index[viol]:
                flags[i] = "speed_violation" if flags[i] is None else flags[i] + ";speed_violation"
        elif len(e) == 1:
            pl.loc[e.index, "court_x_m"] = e.court_x_m_raw.to_numpy()
            pl.loc[e.index, "court_y_m"] = e.court_y_m_raw.to_numpy()
        pl.loc[g.index, "flags"] = flags
    # ---- short-gap interpolation (new rows only; observed rows untouched)
    n_interp = 0
    if cfg.interpolate_max_gap_s:
        add = interpolate_gaps(pl[pl.role != "other"], frames, cfg.interpolate_max_gap_s)
        if len(add):
            pl = pd.concat([pl, add], ignore_index=True).sort_values(["frame_index", "track_id"])
            n_interp = len(add)
    pl = pl.drop(columns=[c for c in ("torso_L", "torso_a", "torso_b", "_det_idx", "_occ", "team_vote", "_seg", "_tseg")
                          if c in pl.columns])
    write_rows("player_observations", pl.replace({np.nan: None}).to_dict("records"), tables)
    ball_cands, ball_source = _ball_candidates(cfg, out, run_a, dets)
    # one ball track per supported camera segment: the Viterbi never links across a cut, and
    # unsupported views (close-ups) get no ball rows (no calibration, no tracked players)
    segp = tables / "camera_segments.parquet"
    supported = dict(zip(*pd.read_parquet(segp)[["camera_segment_id", "supported"]].T.values)) if segp.exists() else {}
    cal_r = cal.reset_index()
    parts = []
    for sid, fg in frames.groupby("camera_segment_id"):
        if not supported.get(sid, True):
            continue
        fs = set(fg.frame_index)
        b = track_ball(ball_cands[ball_cands.frame_index.isin(fs)], fg, cal_r[cal_r.frame_index.isin(fs)],
                       pl[pl.frame_index.isin(fs)], video_id)
        if len(b):
            b["camera_segment_id"] = int(sid)
            b["ball_track_id"] = f"s{int(sid):03d}_ball"
            parts.append(b)
    ball_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_rows("ball_observations", ball_df.replace({np.nan: None}).to_dict("records") if len(ball_df) else [], tables)
    res = dict(n_tracks=int(pl.track_id.nunique()),
               suspected_cross_class_switches=int(n_splits),
               rows_flagged_suspected_id_switch=int(suspect.sum()),
               stitch_enabled=bool(cfg.stitch), stitch_links=stitch_links,
               interpolated_rows=int(n_interp),
               ball_detector=ball_source,
               ball_frames=int(len(ball_df)),ball_detected_frames=int((ball_df.image_source == "detected").sum()) if len(ball_df) else 0,
               ball_possession_proxy_frames=int((ball_df.position_3d_status == "possession_proxy").sum()) if len(ball_df) else 0,
               **team_info,
               roles=pl.groupby("role").track_id.nunique().to_dict(),
               rows=len(pl), team_labels={str(k): v for k, v in teams.items()},
               tracker_config=asdict(cfg.tracker or DEFAULT_TRACKER),
               seconds=time.time() - t0)
    (out / "run_track.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def process_video(video_path, out_dir, keyframe_json, video_id,
                  acfg: AnalyzeConfig | None = None, tcfg: TrackConfig | None = None,
                  segments_json=None) -> dict:
    a = analyze_video(video_path, out_dir, keyframe_json, video_id, acfg, segments_json=segments_json)
    b = track_and_export(out_dir, tcfg)
    return {**a, "tracking": b}
