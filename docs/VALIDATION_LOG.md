# Validation log

Every number below was produced by code in this repo on CPU (Ryzen 7 7840HS). The commands to reproduce each one are given. Last full refresh: 2026-10-15, after the PTZ outlier-rejection fix (bug 5).

> **Scope warning.** All results so far are **synthetic**. The simulator provides exact, independent ground truth (true camera, true world positions, true identities), so errors are meaningful *for the simulated conditions only*. The simulator does **not** model:
> - real landmark-detector error, which is correlated and non-Gaussian;
> - lens distortion;
> - players' real foot-ground contact (bbox bottom = true foot point here);
> - jumping;
> - occlusion of court lines;
> - real floor texture;
> - appearance;
> - real detector failure modes.
>
> **No real-footage accuracy has been measured yet.**

> **Real footage (2026-10-15):** see [REAL_VIDEO_REPORT.md](REAL_VIDEO_REPORT.md) for the first real-video results.
> - Calibration self-consistency, plus independent landmark and rim checks.
> - Tracking proxies, coordinate stability, speed and memory.

## Unit + integration tests
`D:\conda\envs\nbatrack\python.exe -m pytest` → **89 passed, 0 skipped** (2026-10-15, milestone 2). Since the 72-test count, these were added:
- CVAT import/export;
- the TrackEval cross-check;
- the review guard;
- team exemplars and segments;
- offline stitching and interpolation;
- mid-clip keyframe calibration (forward and backward).

The earlier 72-test count (2026-10-15) covered the following. Since the previous count, these were added:
- line-alignment refinement;
- RTS smoother;
- overlay mask;
- Lab team clustering;
- the color identity check;
- a stage-A calibration-tracking test on a rendered synthetic video.

The earlier count of **64 passed** covered the following. Coverage:
- geometry, camera, homography, estimators;
- model selection;
- schema/export;
- projection, quality flags;
- video timestamps;
- propagation;
- manual calibration;
- tracker and MOT metrics;
- end-to-end synthetic pipeline.

Key exactness checks (noise-free synthetic):
- Homography decomposition recovers R and C exactly: C to 1e-6 m, f to 1e-8 relative.
- Unconstrained camera fit recovers C to < 1e-5 m.
- A PTZ frame fit with the center known recovers pan, tilt, and f from **2** landmarks.
- Projection → back-projection round trip on the floor and at rim height: < 1e-6 m, with and without radial distortion.
- Decoded timestamps of CFR and VFR (non-zero start) H.264 files match the written PTS and **ffprobe** frame by frame.
- MOT metrics reproduce hand-computed IDF1, ID switch, FP, and FN values on toy sequences.

Bugs found and fixed by tests or experiments:
1. The focal-from-homography normalization amplified a degenerate equation (pure-tilt camera).
2. In OpenCV 5.0.0, `cv2.USAC_MAGSAC` / `USAC_DEFAULT` return `None` on valid data, so `cv2.RANSAC` is used.
3. Optimizer early termination: the focal length dominated the parameter norm. It is now optimized in kilo-pixels.
4. scipy TRF + sparse Jacobian + robust loss never met its tolerance. The solver now runs a capped robust stage, then linear least squares. This took segment bundle adjustment from 44 s to < 1 s.
5. *(Found by the model-selection study.)* PTZ segment outlier rejection used a fixed 8 px threshold on residuals from the capped, not-yet-converged robust stage. With fewer landmarks it discarded most valid points: median inlier fraction 0–33%, and a fixed-center camera was sometimes "rejected".
   - Fix: iterative rejection whose threshold starts at 3× the median residual and tightens to 8 px.
   - This removed most of the heavy PTZ error tail reported on 2026-10-10. The tables below are post-fix.

## Calibration model comparison (synthetic)
`python tools/compare_calibration_models.py --frames 90 --seeds 3` → `outputs/experiments/calib_compare/`

Setup:
- 1080p, simulated landmark detector with Gaussian noise, 20% dropout (unless stated), and 25% of landmarks held out from fitting.
- court_err = true floor point at its true pixel, back-projected by the estimated model, compared with the true world position.

| Scenario | Model | fail | held-out reproj px (med / p95) | court err m (med / p95) | rim px (med) | static jitter m |
|---|---|---|---|---|---|---|
| wide, 1 px noise | homography | 0 | 0.84 / 3.42 | 0.015 / 0.049 | n/a | 0.022 |
| | camera_free | 0 | 0.73 / 2.19 | 0.014 / 0.043 | 0.59 | 0.018 |
| | ptz_segment | 0 | **0.44 / 1.07** | **0.009 / 0.023** | 0.46 | **0.011** |
| wide, 3 px noise | homography | 0 | 3.76 / 20.1 | 0.067 / 0.284 | n/a | 0.118 |
| | camera_free | 0 | 2.25 / 7.14 | 0.045 / 0.140 | 2.01 | 0.060 |
| | ptz_segment | 0 | **1.33 / 3.16** | **0.026 / 0.071** | 1.39 | **0.034** |
| tight zoom | homography | 0.7% | 1.41 / 30.2 | 0.018 / 0.452 | n/a | 0.417 |
| | camera_free | 6.3% | 1.13 / 4.61 | 0.013 / 0.056 | 0.86 | 0.024 |
| | ptz_segment | 0 | **0.63 / 1.49** | **0.006 / 0.021** | 0.58 | **0.015** |
| 10% outliers | homography | 0 | 0.95 / 8.70 | 0.018 / 0.291 | n/a | 0.350 |
| | camera_free | 1.1% | 0.83 / 3.57 | 0.016 / 0.079 | 0.63 | 0.106 |
| | ptz_segment | 0 | **0.50 / 1.13** | **0.009 / 0.023** | 0.51 | **0.011** |
| 60% dropout | homography | 31.9% | 2.93 / 404 | 0.044 / 3.79 | n/a | 2.37 |
| | camera_free | 40.0% | 1.40 / 6.21 | 0.025 / 0.129 | 1.20 | 0.185 |
| | ptz_segment | 1.1% | 0.72 / 2.19 | 0.013 / 1.49 | 0.80 | 1.34 |
| center drift 0.5 m/s | homography | 0 | 0.88 / 3.94 | 0.015 / 0.052 | n/a | 0.027 |
| | camera_free | 0 | **0.73 / 2.43** | **0.013 / 0.043** | 0.66 | **0.018** |
| | ptz_segment | 0 | 2.99 / 14.0 | 0.045 / 0.216 | 1.49 | 0.074 |
| center drift 2 m/s | homography | 0 | 1.01 / 4.92 | 0.018 / 0.071 | n/a | 0.036 |
| | camera_free | 0 | **0.81 / 3.08** | **0.016 / 0.051** | 0.61 | **0.023** |
| | ptz_segment | 0 | 11.6 / 65.0 | 0.182 / 1.03 | 5.88 | 0.363 |

Conclusions (synthetic):
- **When the center is fixed**, PTZ roughly halves the error and jitter. It is robust to outliers and tight zoom, and keeps calibrating frames with too few landmarks for a homography (only 1.1% failures at 60% dropout, vs 32–40%).
- **Remaining PTZ weakness**: at 60% dropout the p95 court error is 1.49 m, from frames with only 2–3 landmarks. Quality gating is required.
- **When the center moves**, PTZ is clearly worse, so it must not be assumed (see model selection below).

## Model selection: is the fixed-center assumption valid? (synthetic)
`python tools/selection_study.py` → `outputs/experiments/selection_study/`. Method: `calib/select.py`.

How it works:
- Per segment, landmarks are held out (30%, 3 repeats).
- PTZ, free camera, and homography are each fit on the rest, and the held-out reprojection error is compared on common frames.
- PTZ is chosen only if its held-out median is ≤ 1.15 × the free camera's and its p90 ≤ 1.73 × the free camera's.

Setup: 60 frames, 30% dropout, 5% outliers, 6 seeds per cell.

| center drift | noise | PTZ chosen | held-out median px: PTZ / free / homography |
|---|---|---|---|
| 0 | 1 px | **100%** | 1.40 / 1.78 / 2.03 |
| 0 | 3 px | **100%** | 4.20 / 5.49 / 8.08 |
| 0.1 m/s | 1 px | 100% | 1.57 / 1.78 / 2.04 |
| 0.25 m/s | 1 px | 50% | 2.17 / 1.80 / 1.98 |
| 0.5 m/s | 1 px | **0%** | 3.16 / 1.78 / 2.16 |
| 0.5 m/s | 3 px | 100% (not detected) | 5.62 / 5.53 / 8.34 |
| 1.0 m/s | 1 px | **0%** | 5.59 / 1.79 / 2.10 |
| 1.0 m/s | 3 px | 33% | 7.28 / 5.42 / 8.26 |

Interpretation:
- A real fixed camera is accepted.
- With accurate landmarks, drift of 0.5 m/s or more is rejected.
- With noisy (3 px) landmarks, moderate drift goes undetected. In that regime, though, PTZ's held-out error is within 2% of the free camera's, so the cost of the wrong choice is small.
- Real broadcast rigs and real detector noise must still be tested on footage.

## Quality flags vs true error (synthetic)
`python tools/quality_flag_study.py` → `outputs/experiments/quality_study/`

Setup:
- Focal length 1800–4500 px, landmark noise 1.5 px, 50% dropout, 8% outliers.
- Ground-point noise 2 px.
- Thresholds use the DOF-corrected residual.

| Model | Quality | n | med err m | p95 err m | frac > 0.5 m | P(err ≤ 2σ) |
|---|---|---|---|---|---|---|
| homography | good | 1314 | 0.052 | 0.159 | 0.2% | 0.951 |
| | fair | 1028 | 0.069 | 0.300 | 1.8% | 0.822 |
| | failed | 220 | 0.113 | 6.34 | 17.3% | n/a (not emitted) |
| camera_free | good | 1378 | 0.050 | 0.152 | 0.0% | 0.964 |
| | fair | 1087 | 0.056 | 0.202 | 1.5% | 0.890 |
| | poor | 50 | 0.324 | 2.18 | 30% | n/a (not emitted) |
| ptz_segment | good | 1596 | 0.044 | 0.135 | 0.0% | 0.989 |
| | fair | 1078 | 0.039 | 0.113 | 0.0% | 0.989 |
| | poor | 61 | 0.037 | 0.112 | 0.0% | n/a (not emitted) |
| | failed | 54 | 1.18 | 51.8 | 66.7% | n/a (not emitted) |

How to read this:
- For a 2D Gaussian, P(r ≤ 2σ_major) ≥ 0.865.
- Flags are conservative. After the bug-5 fix, PTZ frames rated `poor` are in fact accurate, so the PTZ thresholds discard some good data.
- Thresholds remain engineering defaults and **must be re-tuned on annotated real footage**.

## Camera-motion propagation (synthetic rendered frames)
Setup: frame 0 has the true calibration, and only frame-to-frame KLT plus RANSAC chains it forward (no re-anchoring), at 30 fps with a panning and zooming camera.

| Features used | median / p95 floor error after 30 frames | after 89 frames |
|---|---|---|
| floor only (valid for any camera motion) | 0.015 / 0.038 m | 0.059 / 0.187 m |
| all static scene (valid only for a fixed-center camera) | 0.0015 / 0.0026 m | 0.0036 / 0.0091 m |

Cost: about 0.05 s/frame at 1080p on CPU. The synthetic floor texture is unlike real hardwood, so these numbers are not transferable. They show that drift accumulates and that re-anchoring is required.

## Tracking (synthetic)
`python tools/tracking_study.py --seeds 5 --frames 300` → `outputs/experiments/tracking_study/`

Simulated detector, derived from ground truth:
- Boxes more than 70% covered by a nearer player are missed. Partially occluded boxes get scores of 0.2–0.5 and 3× box noise.
- 3% random misses; 2 px box noise.
- 4 bench people just off the court are detected but are not ground truth.

Scene:
- 10 players and 3 referees crowded into one half court.
- Panning and zooming camera, 720p, 10 s at 30 fps.
- Calibration from simulated landmarks (PTZ segment); ground point = bbox bottom.

| Tracker config | IDF1 | MOTA | ID switches | Precision | Recall | FP | ms/frame |
|---|---|---|---|---|---|---|---|
| ByteTrack-style, IoU only | 0.593 | 0.773 | 39.4 | 0.918 | 0.863 | 275 | 1.8 |
| + camera-motion compensation | 0.589 | 0.780 | 34.0 | 0.919 | 0.868 | 275 | 2.7 |
| + court-space association | 0.677 | 0.790 | 21.2 | 0.919 | 0.874 | 275 | 5.9 |
| + off-court detection filter | **0.706** | **0.868** | **21.2** | **0.999** | 0.874 | 2 | 5.1 |

(Means over 5 seeds.)

Interpretation:
- Court-space association is the largest gain: about −46% ID switches.
- The off-court filter removes bench false positives.
- CMC alone does not change IDF1 here, because the synthetic pan is smooth enough for the box Kalman velocity. A unit test confirms that CMC is essential for jerky pans.
- Recall is capped by occlusion misses, which association cannot recover.
- There is **no appearance model**, so identity recovery after long occlusions is not tested.
- **None of this is evidence about real tracking accuracy.** Real IDF1/HOTA require annotated footage.

## Synthetic end-to-end demo (150 frames, 720p)
`nbatrack synth-demo --calib {ptz,camera,homography}` → `outputs/samples/synthetic_demo_*`

Setup:
- Detections are noisy ground-truth boxes (2 px noise, 5% misses).
- Track IDs are the true identities.
- Ground point = bbox bottom, which equals the exact foot point in this simulator.
- Landmark noise 1 px, 40% dropout, 5% outliers.

| Calibration | frames good / fair / poor / failed | player err m (med / p95 / max) | frac > 0.5 m | P(err ≤ 2σ) | visible players with coords |
|---|---|---|---|---|---|
| ptz | 17 / 131 / 2 / 0 | 0.059 / 0.184 / 0.399 | 0.0% | 0.997 | 98.5% |
| camera_free | 21 / 124 / 4 / 1 | 0.069 / 0.201 / 0.453 | 0.0% | 0.992 | 96.7% |
| homography | 28 / 113 / 2 / 7 | 0.081 / 0.249 / 1.203 | 0.9% | 0.957 | 94.9% |

## Not yet validated (requires real footage and/or annotation)
- Player detection precision and recall (no detector integrated yet).
- Real tracking accuracy (IDF1, HOTA, ID switches) and team classification.
- Real landmark detection, and real-footage calibration reprojection error on held-out manual landmarks.
- Bbox-bottom vs pose-based ground-contact error. Requires annotated foot points; whether RTMPose Halpe26 exposes heel and toe keypoints is not yet verified on the checkpoint.
- Whether real NBA main cameras satisfy the fixed-center PTZ assumption. The selector exists; it needs footage.
- Real court-coordinate error in meters. This needs independent geometric ground truth (see the annotation plan in `docs/PHASE1_ARCHITECTURE.md` §9.2); projecting points through the same calibration model would **not** count.
- Processing FPS of the full pipeline with neural models.
