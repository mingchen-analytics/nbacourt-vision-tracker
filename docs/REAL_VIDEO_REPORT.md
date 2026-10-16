# First real-video experiment: technical evaluation (2026-10-15)

**Scope.** Results of the first end-to-end runs on real NBA broadcast footage:

- RF-DETR detection;
- semi-automatic camera calibration;
- tracking;
- bbox-bottom ground contact;
- court coordinates in meters;
- export;
- visualization.

**How to read the numbers.** Every number below was measured by running the code in this repo. The verification method for each claim is stated next to it.

- No ground-truth annotation exists yet. Detection precision and recall, IDF1 / HOTA, and absolute court-position error in meters are therefore **not measured**.
- Where only a proxy or a self-consistency measure is available, it is labeled as such.

## 1. Input videos

| # | File | Res | FPS | Duration | Bitrate | Venue / view | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `Kuminga 切入 Closeout 硬上中鋒Plumlee.mp4` | 1280×720 | 60 | 11.2 s (673 fr) | 6.0 Mb/s | PHX (Footprint Center), main sideline camera, left half court | Purple lane, GSW white vs PHX purple; "NBA Advanced Stats" overlay |
| 2 | `拓荒者 進攻 戰術 AI Ghost Flare Sharpe 擋拆助攻外線Camara.mp4` | 1280×720 | 30 | 15.0 s (451 fr) | 6.0 Mb/s | GSW (Chase Center), main camera, pans from left half across midcourt to right half (pan −9.8° → +18°) | **Chosen for the first test** |
| 3 | `勇士 防守 擋拆 Ice+High Tag.mp4` | 1280×720 | 60 | 15.9 s (951 fr) | **1.2 Mb/s** | GSW, main camera, midcourt then left half | One-frame exposure flash at 12.0 s (not a cut) |
| 4 | `勇士 進攻 戰術 35Empty Kuminga 低位 傳導很順 Reath 火鍋.mp4` | 1280×720 | 30 | 15.0 s (450 fr) | 6.0 Mb/s | GSW, main camera, pans from midcourt into a half court (pan −19.5° → +9.8°) | |
| 5 | `勇士 進攻 戰術 邊線 Double Gap Butler切入製造犯規.mp4` | 1280×720 | 30 | 11.1 s (332 fr) | 6.0 Mb/s | GSW, main camera, left half court, small pan | |

Common characteristics of all five clips:
- 720p.
- Continuous main-camera shots with no cuts (ffmpeg scene score > 0.3 fired only on clip 3's flash).
- The game clock runs continuously.
- They are 11–16 s long, shorter than the 30–60 s originally planned.
- Clips 2–5 come from the same arena and opponent; only clip 1 is a different venue.

**Why clip 2 first:**
- standard main camera with no cuts;
- normal bitrate;
- the largest camera motion, a 28° pan across midcourt, which stresses calibration propagation;
- the widest variety of court markings.

**Clip 3 was not processed.** Its first frame shows mostly midcourt, where the center circle is covered by the logo, plus only a sliver of the lane. That gives too few unambiguous landmarks for keyframe initialization, and the pipeline currently needs the keyframe to be the first frame. It also has the lowest bitrate.

## 2. What was built and run

| Stage | Implementation | Status |
|---|---|---|
| Detection | RF-DETR Small (Apache-2.0, COCO), exported to ONNX at 512×896 (16:9, no squashing); ONNX Runtime with DirectML (AMD 780M) or CPU; classes `person` and `sports ball` | Run on clips 1, 2, 4, 5. ONNX vs official torch output checked on a real frame: same people, boxes within a few px |
| Overlay masking | Scoreboard and network logo found from temporal pixel variance | Run |
| Keyframe calibration | 4–6 approximate landmark clicks on frame 0. A grid of sideline-camera priors (zero roll, fixed center) is fit to them, then refined by **joint point + court-line alignment** (chamfer distance to detected painted lines and paint edges, Cauchy loss). The candidate explaining the most markings wins. | Run |
| Calibration tracking | Per frame: KLT camera motion on static pixels (players and overlays masked) predicts the camera, then line-alignment refinement. Default **PTZ**: center fixed at the keyframe estimate. Alternative `--calib-mode free`. | Run |
| Ground contact | Bounding-box bottom-center (baseline) | Run. Pose-based estimation not yet run. |
| Court coordinates | Back-projection to the floor plane, with first-order sigma | Run |
| Tracking | ByteTrack-style + camera-motion compensation + court-space gating | Run |
| Off-court filter | A track whose median position is > 0.3 m outside the court gets role `other` (spectators/staff behind boards) | Run |
| Team / referee | k-means (k=3) on per-track median torso Lab color; the referee cluster is chosen by an "NBA referee gray" prior | Run (unreliable, see §5) |
| Identity check | Per-frame jersey-color class; a cross-team change lasting ≥ 1 s flags `suspected_id_switch` | Run (flags only) |
| Smoothing | Per-track constant-velocity Kalman + RTS; raw coordinates kept; speed violations flagged | Run |
| Export | Schema-validated Parquet + CSV: calibration, frames, players, ball, landmarks, detections cache | Run |
| Visualization | `annotated.mp4` (boxes, IDs, teams, projected court lines) and `side_by_side.mp4` (video \| top-down court with 2σ rings and 1 s trails), both on source PTS | Run; all 8 files decode with 0 errors |

Keyframe landmarks were **read by Claude from gridded frames** (`data/annotations/clip*/kf_0000.json`, annotator field says so). They are initialization only, accurate to roughly ±3–5 px, and are not ground truth.

## 3. Calibration

| Clip | Frames | Quality flag (self-consistency) | Line-alignment inlier fraction (median / p05 / min) | KLT motion success | Pan range | Focal range (px) |
|---|---|---|---|---|---|---|
| 2 | 451 | 100% good | 0.970 / 0.943 / 0.908 | 100% | −9.8° … 18.0° | 3499–3991 |
| 5 | 332 | 100% good | 0.962 / 0.940 / 0.863 | 100% | −18.0° … −12.7° | 3041–4181 |
| 4 | 450 | 100% good | 0.937 / 0.881 / 0.808 | 100% | −19.5° … 9.8° | 3198–4671 |
| 1 | 673 | 100% good | 0.915 / 0.896 / 0.884 | 100% | −15.7° … −11.1° | 3164–4215 |

**Self-consistency is not accuracy.** The inlier fraction says the projected court template lies on painted markings. It cannot detect a solution that snaps consistently to the wrong lines. Independent checks made so far:

1. **Independent landmarks, clip 2, frame 300.** This frame is 10 s and an 18° pan away from the keyframe, and its landmarks were annotated separately; they did not enter calibration.
   - Lane corners: **3.3, 4.5, 5.9 px** reprojection error.
   - Arc landmarks, measured as point-to-curve distance: **1.1 px** (free-throw circle) and **3.9 px** (3-pt arc).
   - These errors are at the level of the annotation uncertainty (±3–5 px).
   - Lesson: the first point-to-point check reported 49.7 px on the 3-pt "apex". That was an annotation-definition error: the leftmost image point of an arc is not the court apex. Arc landmarks must be checked point-to-curve.
2. **3D rim check, clip 2, frame 300.** The rim center at 3.05 m projects onto the visible rim (visual check, within a few px). A floor homography cannot make this prediction, so it supports the 3D camera model.
3. **Visual overlay inspection.** On sampled frames of all four clips, the projected lines follow the lane, free-throw circle, 3-pt arc, sidelines and baseline.

**PTZ vs free camera (clip 2, adjustment A):**

| Metric | PTZ (fixed center) | Free camera |
|---|---|---|
| Alignment inlier fraction, median | 0.970 | 0.965 |
| Camera center over 15 s | fixed at (0.0, −34.5, 10.6) m | **drifts 9 m in y and 2.5 m in z** (physically implausible for a broadcast camera; zoom/dolly ambiguity) |
| Independent corner errors at frame 300 (px) | 3.3 / 5.9 / 4.5 | 10.7 / 3.9 / 5.6 |
| Tilt jitter, frame-to-frame p95 | 0.021° | 0.040° |
| Floor-position difference between the two models | median 0.074 m, max 0.126 m | |

Conclusion for clip 2: the data does not reject the fixed-center assumption. The free model fits the markings no better while its center wanders. This is **one 15 s clip**; real camera rigs vary, and the cross-validated selector (`calib/select.py`) needs per-frame landmarks that we do not yet have on real footage.

## 4. Detection (counts only; precision/recall not measurable without annotation)

| Clip | Persons with score ≥ 0.5, per frame (median) | On-court persons tracked per frame (median) | Frames with a ball detection (score ≥ 0.3) |
|---|---|---|---|
| 2 | 9 | 9 | 47% |
| 5 | 8 | 9 | 21% |
| 4 | 8 | 7 | 44% |
| 1 | 9 | 10 | 70% |

Ten players plus three referees are on court, but not all are in view. These counts indicate that most visible people are detected, but **the per-frame miss rate is unknown**.

## 5. Tracking (proxies only)

| Clip | On-court track IDs | Roles (tracks) | Interior births* | Median track length | Suspected cross-team switches† |
|---|---|---|---|---|---|
| 2 | 13 | 10 player / 3 referee (+3 `other`) | 5 | 344 fr | 16 events (1661 rows flagged) |
| 5 | 16 | 12 / 4 | 9 | 186 fr | 9 |
| 4 | 15 | 7 / 8 (+2 `other`) | 6 | 180 fr | 12 |
| 1 | 16 | 13 / 3 | 5 | 428 fr | 5 |

\* A track starting after frame 5 away from the left and right image edges. A person cannot appear from nowhere, so this proxies fragmentation.
† The per-frame jersey color changes class for ≥ 1 s. Some flags are false positives; for example, referee shirt numbers can read as a dark jersey.

Tracker tuning on clip 2 (cached detections, `nbatrack retrack`):

| Court-gating sigma floor | On-court IDs | Interior births |
|---|---|---|
| 0.05 m (the best setting on synthetic data) | 32 | 23 |
| 0.3 m | 20 | 11 |
| 0.6 m | 15 | 6 |
| **1.0 m (chosen)** | **13** | **5** |
| 1.0 m + 2 s lost buffer | 11 (fewer than the 13 people: merges) | 2 |
| IoU only (no court cost) | 17 | 9 |

**Synthetic-optimal settings did not transfer.** Real bbox-bottom positions jump during occlusions.

**Observed ID switches.** Visual inspection of clip 2 at frame 440 found two cross-identity switches at crossings: t018 went from referee to player, and t006 from a POR player to a GSW player.

**Appearance cue tried and rejected.** Using per-detection torso color directly in association made things worse: a 30-unit Lab gate fragmented tracks to 24 IDs. Color is therefore used post hoc for flagging only.

**Team / referee labels are unreliable.**
- Clip 2: mostly right after switching to Lab features.
- Clip 1: mostly right.
- Clips 4 and 5: visibly wrong for several players (for example, white GSW players labeled team A or referee).
- **Do not use `team_id` / `role` for analysis yet.**

## 6. Court coordinates (stability; not absolute accuracy)

| Clip | On-court rows with court XY | Median σ (m) | Raw − smoothed (median / p95 m) | Smoothed speed (median / p95 / max m/s) | Raw speed-violation rows | Outside court > 0.5 m |
|---|---|---|---|---|---|---|
| 2 | 100% | 0.144 | 0.10 / 0.52 | 1.7 / 5.2 / 9.2 | 84 | 2.3% |
| 5 | 100% | 0.149 | 0.09 / 0.61 | 1.3 / 5.6 / **12.0** | 75 | 0.5% |
| 4 | 100% | 0.148 | 0.12 / 0.66 | 1.9 / 5.5 / 8.6 | 103 | 1.0% |
| 1 | 100% | 0.122 | 0.07 / 0.44 | 1.3 / 4.0 / 8.7 | 68 | 0.0% |

How to read this:
- Median speeds of 1.3–1.9 m/s and p95 speeds of 4–5.6 m/s are physically plausible for basketball.
- The 12 m/s maximum on clip 5 is not plausible and points to an identity or ground-point error.
- The p95 raw-vs-smoothed residual of 0.4–0.7 m shows bbox-bottom noise during occlusions.
- **`court_sigma_m` (median 0.12–0.15 m) is underestimated:** it uses the uncalibrated bbox prior. The tracker study showed that real bbox-bottom positions need a ≈ 1 m gate.
- **No absolute position error in meters can be stated.** No independent player-position ground truth exists.

## 7. Speed and memory (measured on this laptop)

| | Clip 2 | Clip 5 | Clip 4 | Clip 1 |
|---|---|---|---|---|
| Stage A (detect + calibrate), frames/s | 2.25 | 0.69 | 1.77 | 2.39 |
| Detection ms/frame (DirectML) | 318 | **1239** | 405 | 285 |
| Calibration ms/frame | 117 | 197 | 149 | 123 |
| Stage B (track + export), whole clip | 4.5 s | 3.0 s | 3.7 s | 7.4 s |
| Peak private memory | 2.05 GB | 2.04 GB | 2.06 GB | 2.05 GB |

Controlled detector benchmark (`outputs/real/detector_benchmark.json`): 60 frames each, on AC power, balanced power plan.

| Backend | Median ms/frame | Notes |
|---|---|---|
| CPU | 413 | p10–p90 405–426: stable |
| DirectML, first session | 348 | |
| DirectML, second session | 186 | |

DirectML is faster on average, but its latency is **highly variable**: ~1.2 s/frame was seen in two pipeline runs. CPU is slower but predictable.

At about 2 fps end to end, a 2.5 h broadcast (~270k frames at 30 fps) would take roughly 37 h on this laptop. That is not practical without frame sub-sampling or a discrete GPU.

## 8. Failure cases observed
1. Cross-identity switches at player crossings. The court gate had to be loosened to 1 m to stop fragmentation, which permits swaps.
2. Team / referee misclassification (clips 4 and 5), and referee tracks falsely flagged as switches.
3. Spectators and staff behind advertising boards look like people on the sideline. The track-level off-court rule removes the ones seen so far, but a staff member standing on the floor apron could pass.
4. bbox-bottom ground points jump when feet are occluded or a box covers two people. Jumping is not handled.
5. A physically implausible 12 m/s maximum smoothed speed (clip 5).
6. DirectML latency spikes.
7. Keyframe initialization needs a frame with enough well-defined landmarks (clip 3).
8. Ball: detected in 21–70% of frames in image space only. 3D position is `unknown` by design.

## 9. Not yet validated
- Detection precision and recall; tracking IDF1 / HOTA / ID switches. These need CVAT annotation (see `docs/PHASE1_ARCHITECTURE.md` §9.2).
- Absolute court-coordinate error in meters. This needs annotated foot points plus independent calibration.
- Pose-based ground contact. RTMPose heel/toe availability is still to be verified.
- Generalization beyond two arenas, and camera cuts within a clip.

## Output locations
- `outputs/real/clip{1,2,4,5}_ptz/`:
  - `annotated.mp4`, `side_by_side.mp4`;
  - `tables/*.parquet|csv`;
  - `calibration_diagnostics.csv`, `diagnostics.png`;
  - `report_metrics.json`, `overlay_mask.png`;
  - `cache/detections.parquet`.
- `outputs/real/clip2_free/`: free-camera comparison run (tables only).


---

# Milestone 2: validation tooling and identity work (2026-10-15)

**Ground-truth status: no reviewed annotations exist yet.** All tracking, detection, classification and ground-contact accuracy metrics on real footage are **NOT MEASURED**. The tooling that will measure them is built and tested; see EVALUATION.md.

## Deliverables
| Deliverable | Location | Status |
|---|---|---|
| CVAT annotation packages: clip 2 (primary) and clip 5 (hard case) | `outputs/annotation/clip{2,5}/` | ready. Pre-annotations: 27 and 23 tracks, split at suspected switches; all `review=unreviewed`. Foot points and landmarks are deliberately not pre-filled. |
| Annotation guide (Chinese, step by step) | `docs/ANNOTATION_GUIDE.md`, plus `docs/court_landmarks.png` | ready |
| Evaluation script (TrackEval HOTA/CLEAR/Identity, detection P/R, failure decomposition, role/team accuracy, coverage, ground contact, landmarks) | `tools/evaluate_against_gt.py`, `docs/EVALUATION.md` | tested on toy data, synthetic data and a self-consistency round trip; awaiting GT |
| Tracker ablation runner (baseline kept) | `tools/tracker_ablation.py` | proxies now; GT metrics automatic once exports exist |
| Root-cause diagnostics and crop strips | `tools/identity_diagnostics.py` → `outputs/real/*/diagnostics_identity/` | done for all 5 clips |
| Manual team/referee initialization | `tools/team_init_helper.py`, `data/annotations/*/teams.json` | exemplars picked by Claude from numbered frames; **to be verified by the user** |
| Offline reconstruction (stitching, interpolation; off-screen inference as design) | `docs/OFFLINE_RECONSTRUCTION.md` | interpolation on; stitching off (see below) |
| Prioritized improvement plan | `docs/IDENTITY_IMPROVEMENT_PLAN.md` | |

## Tooling validation (measured)
- **TrackEval vs own metrics** on a synthetic sequence: identical IDF1 (0.5577 / 0.7649); MOTA within 0.001; IDSW 46 vs 48.
- **CVAT round trip** (export → parse): keyframes, linear interpolation and gaps are reproduced exactly; the review guard refuses unreviewed data.
- **Keyframe density of the pre-annotations.**
  - At 5 frames, the median interpolated-vs-original box IoU is 0.94–0.95, with IoU < 0.7 for 2–4% of boxes.
  - At 10 frames, the median is 0.91, with IoU < 0.7 for 7–8%.
  - 5 frames is the default.
- **Self-evaluation** of clip 2's pipeline output against its own pre-annotation (labeled NOT GROUND TRUTH): IDF1 0.99, MOTA 0.98. The gap is pure interpolation error.

## Clip 3 recovered
- **Method:** the keyframe is now allowed at any frame. Calibration propagates forward and backward from it.
- **Keyframe:** frame 160, with 6 approximate landmarks.
- **Coverage:** all 951 frames calibrated; 845 good, 106 fair, 0 failed.
- **Fair frames:**
  - Frames 0–~50 (midcourt view, where the logo covers most lines) are rated fair.
  - The one-frame exposure flash at frame 721 drops the alignment inlier fraction from 0.86 to 0.62 and is rated fair.
- **Regression check:** a synthetic integration test covers keyframe 0 and keyframe 12 (forward + backward); the error is < 0.15 m in every frame.

## Calibration: independent landmark checks across clips (annotations by Claude, approximate)
Only well-defined intersections are trustworthy as checks.

| Clip, frame (s from keyframe) | Free-throw-line corners: far / near (px) | Baseline lane corners (px) | Notes |
|---|---|---|---|
| clip 2, f300 (10.0 s) | 4.5 / 5.9 | 3.3 | arc landmarks point-to-curve: 1.1 and 3.9 |
| clip 1, f500 (8.3 s) | 8.1 / 3.9 | 22.2 / 13.5 | |
| clip 3, f600 (7.3 s) | 3.2 / 2.3 | 4.0 / 14.6 | |
| clip 4, f300 (10.0 s) | 5.7 / 10.6 | 52.6 / 18.7 | overlay shows lines aligned; GSW's yellow apron hides the lane–baseline corner, so these readings are annotation errors |
| clip 5, f250 (8.3 s) | 3.6 / 32.4 | 35.9 / 1.3 | near corner occluded by a falling player; same apron problem |

- Overlay evidence: `outputs/eval/calibration_checks/`.
- Clear free-throw-line corners: 2.3–10.6 px. The large values trace to ambiguous or occluded annotation points.
- This is exactly why reviewed landmark annotation (ANNOTATION_GUIDE §6) is needed before stating calibration accuracy.
- The PTZ-vs-free comparison from milestone 1 is preserved (`outputs/real/clip2_free/`).

## Team / referee classification
- **Method:** manual exemplars (2 frames per clip), kNN on 3-region HSV descriptors, abstention when occluded or ambiguous, track-segment majority votes. `unknown` is explicit.
- **Development spot checks on held-out frames** (by eye, not GT):
  - clip 2, frame 420: 9/9 plausible;
  - clip 5, frame 300: 5 correct, 2 unknown, 2 wrong, both explained by identity switches.
- **Clip 3** still uses automatic k-means: no exemplars yet.

## Speed-spike root causes (heuristic classification of every speed-violation row)
| Clip | Rows | Crossing overlap | Ground-point (box height) jump | Re-acquired after gap | Calibration jump | Other |
|---|---|---|---|---|---|---|
| 1 | 68 | 31 | 12 | 0 | 0 | 25 |
| 2 | 84 | 36 | 8 | 3 | 0 | 37 |
| 3 | 105 | 15 | 26 | 0 | 2 | 62 |
| 4 | 103 | 32 | 22 | 0 | 1 | 48 |
| 5 | 75 | 27 | 9 | 6 | 0 | 33 |

- Calibration contributes ≤ 2 rows per clip. Spikes come mostly from ground-point localization under occlusion and from crossings (possible switches).
- The "other" rows have a median raw step of about 1.5 m per frame with no single trigger, consistent with bbox-bottom jitter at distance.
- The V1 occlusion-aware gate cuts speed-violation rows by 14–44% (proxy).

## Uncertainty
`court_sigma_m` is now marked **UNCALIBRATED** in the Parquet metadata, the schema documentation and the visualization legend. It will be calibrated against annotated foot points (image-space error) once they exist.
