# Evaluation protocol (ground-truth based)

Implementation:
- `src/nbatrack/eval/gt_eval.py`
- `src/nbatrack/eval/trackeval_bridge.py`
- `tools/evaluate_against_gt.py`
- `tools/tracker_ablation.py`

Annotation procedure: [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

## Ground-truth status rule
Every annotated track carries `review = unreviewed | reviewed`.
- If any track is unreviewed, `evaluate_run` raises an error.
- The `--allow-unreviewed` flag exists only for tooling checks; every output it produces is stamped **NOT GROUND TRUTH**.
- Pre-annotations are model output and are never ground truth.

## Conventions
| Item | Rule |
|---|---|
| Frame numbering | CVAT task frame = pipeline `frame_index`: 0-based, presentation order, whole file. The task must be created from the original file with no frame step or range. `check_frame_count` compares the CVAT task size with the decoded frame count and reports any mismatch. |
| Timestamps | Taken from the pipeline `frames` table (PTS-based). CVAT does not supply timestamps. |
| Evaluated objects | GT `player` and `referee` boxes in frames where the track is not `outside`. Occluded boxes are included and also reported separately. |
| Ignore regions | Predictions overlapping GT `other_person` boxes (IoU ≥ 0.5) are removed before matching. Bench and staff therefore count as neither TP nor FP. |
| Predictions | Pipeline rows with `role != other`. Off-court tracks are excluded, as in the pipeline output. |
| Matching | IoU ≥ 0.5 for CLEAR (MOTA, IDSW, Frag), Identity (IDF1) and detection P/R. HOTA averages over IoU thresholds 0.05–0.95, as in TrackEval. |
| Interpolation | CVAT keyframes are linearly interpolated, as CVAT does, when an export contains keyframes only. |

## Metrics
| Group | Metric | Source |
|---|---|---|
| Tracking | HOTA, DetA, AssA, LocA; MOTA, MOTP, IDSW, Frag, MT, ML; IDF1, IDP, IDR | **TrackEval** (reference implementation, MIT). Cross-checked against our own `mot_metrics.py`: identical IDF1, MOTA within 0.001, IDSW within 2 on a 150-frame synthetic sequence (§Validation). |
| Detection | Precision and recall of raw RF-DETR person boxes with score ≥ 0.5; recall at score ≥ 0.1 | `detection_pr` (Hungarian, IoU 0.5). Raw-detector precision is pessimistic, because spectators in the stands are not annotated. |
| Failure decomposition | For each GT box: covered by output / never detected / detected but dropped by tracking or filters | `miss_decomposition`. Together with DetA vs AssA, this separates detection failures from association failures. |
| Classification | Player-vs-referee accuracy on matched boxes; team accuracy under the best A/B permutation; unknown rate | `classification` |
| Coverage | Fraction of GT boxes with a matched output that has court coordinates | `coverage` |
| Ground contact | Image-space distance (px, and relative to box height) between the matched output's bbox-bottom and the annotated foot point; vertical bias; occluded vs not | `ground_contact`. This is independent of calibration. |
| Ground contact in meters | The same difference projected through the estimated camera | Labeled *"ground-contact component through the estimated camera; NOT absolute accuracy"* |
| Calibration | Reprojection error of annotated landmarks: point-to-point for intersections, point-to-curve for arc landmarks | `calibration_landmarks` |

## What cannot be measured even with these annotations
- **Absolute court-position error in meters.** Annotated foot points are mapped to the court through the same estimated camera, so only the ground-contact component can be isolated.
- **True 3D ball position.**

## Validation of the tooling (no human GT needed)
- CVAT XML round trip: export → parse → identical boxes on keyframes, correct linear interpolation, gaps respected, review status parsed (unit tests).
- Evaluating the pipeline against its own pre-annotation (`--allow-unreviewed`) gives IDF1 0.99, MOTA 0.98 rather than 1.0. The gap is the interpolation error of 5-frame keyframes (median IoU 0.94; IoU < 0.7 for 2–4% of boxes), which the annotator corrects.
- TrackEval vs own metrics on synthetic tracking: IDF1 0.5577 vs 0.5577 and 0.7649 vs 0.7649; MOTA 0.7266 vs 0.7277.

## Current status (2026-10-15)
**No reviewed ground truth exists yet.** Detection P/R, HOTA, IDF1, IDSW, Frag, classification accuracy, coverage against GT, and ground-contact error are therefore all **not measured** on real footage. The tools are ready and tested.
