# Output data schema (v1.2.0)

Changes in 1.2.0 (ball):
- `ball_observations.image_source`: `detected` (a candidate chosen by the offline Viterbi tracker) or `interpolated` (linear image-space fill of a gap of at most 8 frames between two detections). A frame with **no row** means the ball is **missing** (not found, not invented).
- `position_3d_status = possession_proxy`: the ball point lies inside a tracked player's box, and that player's court XY is reported as a proxy with `ball_z_m` NULL. Otherwise the status is `unknown` and the court XY stays NULL. No 3D ball position is ever invented.
- `holder_track_id`: the player used for the proxy.
- `det_confidence` is the detector score (RF-DETR score or WASB heatmap peak, depending on `run_track.json: ball_detector`).

Changes in 1.1.0:
- `position_status` values are `observed_projected`, `interpolated`, `extrapolated` (reserved) and `unknown`.
- New column `source_track_id` holds the pre-stitching fragment ID.
- Parquet metadata now carries `nbatrack.court_sigma_m.status`, which states that the sigma is **uncalibrated**.

Source of truth: `src/nbatrack/export/schema.py`. Every written table is validated against it: unknown fields, NULLs in required fields, and invalid enum values all raise `SchemaError`.

## Conventions
- **Units**: meters, seconds, pixels (source resolution, origin at the top-left).
- **Missing values are NULL**: never 0, −1, or NaN. NaN floats are converted to NULL on write.
- **Formats**:
  - Parquet (zstd) is primary. Its schema metadata carries the schema version and the coordinate system.
  - CSV has the same columns; matrix columns are JSON strings.
  - JSON is optional.
- **Time**: `timestamp_sec` = PTS × time_base − stream start, so the first frame ≈ 0. Raw `pts` and `pts_sec` are kept in `frames`. `frame_index` is the 0-based presentation order over the *whole file*; it is stable under striding or windowing.
- **Identity**: `track_id` is unique only within a `camera_segment_id` (format `sNNN_tNNN`). **No identity consistency across camera cuts is claimed.**
  - `source_track_id` is the online-tracker fragment ID. It differs from `track_id` only where offline stitching linked fragments; stitching is off by default.
- **Role and team** are assigned per track segment. A segment ends at a persistent change of the per-frame jersey class, so the label can change at a suspected identity switch; such rows are flagged `suspected_id_switch`.

## Tables
| Table | Grain | Key fields |
|---|---|---|
| `videos` | one per source | video_id, sha256, width, height, fps_nominal, time_base, pipeline_version, config_hash |
| `frames` | one per decoded frame | video_id, frame_index, timestamp_sec, pts, pts_sec, camera_segment_id |
| `camera_segments` | continuous shot | start/end frame and time, view_type (`main_broadcast`/`other`/`unknown`), supported |
| `calibration` | one per frame | method (`keypoint`/`propagated`/`manual`/`smoothed`/`none`), model (`homography`/`camera_free`/`ptz`/`none`), H_img_to_court, H_court_to_img (9 floats, row-major), camera intrinsics and extrinsics (NULL for homography), n_landmarks, n_inliers, reproj_rmse_px, quality (`good`/`fair`/`poor`/`failed`), quality_score |
| `court_landmarks` | static reference | landmark_id, name, court_x/y/z_m |
| `landmark_observations` | per frame per landmark | image_x/y, confidence, is_inlier, reproj_error_px, source (`model`/`manual`/`synthetic`) |
| `player_observations` | per frame per detected person | see below |
| `ball_observations` | per frame with a detected or interpolated ball (no row = missing) | image-space bbox/point; `image_source` (`detected`/`interpolated`); `position_3d_status` (`unknown`/`possession_proxy`; `trajectory_fit`/`floor_contact` reserved); ball_x/y_m only for `possession_proxy`; ball_z_m always NULL; `holder_track_id` |
| `hoop_observations` | per frame per visible rim | detection plus `predicted_image_x/y` from the 3D camera model, and the residual (an independent calibration check) |

### `player_observations` — the four kinds of data are kept apart
| Kind | Columns |
|---|---|
| Direct image observation | bbox_x1..y2, det_confidence, image_x/image_y (ground-contact point), ground_method (`bbox_bottom`/`feet_keypoints`) |
| Court coordinates through calibration | court_x_m_raw, court_y_m_raw (unsmoothed projection), court_sigma_m |
| Temporal estimate | court_x_m, court_y_m after RTS smoothing of observed rows. Gaps of 1 s or less are filled with **new rows**: position_status = `interpolated`, is_visible = false, raw, bbox and image columns NULL, flag `interpolated`. |
| Unknown | position_status = `unknown`, with court_x_m / court_y_m / court_sigma_m all NULL |

Emission rules (implemented in `calib/quality.py` and the pipeline):
- Court coordinates are emitted only when calibration_quality ∈ {good, fair}.
- For `poor` frames, `court_*_raw` is kept for diagnostics, but position_status = `unknown`.
- For `failed` frames, everything court-related is NULL.

`court_sigma_m` is **UNCALIBRATED** (Parquet metadata `nbatrack.court_sigma_m.status`). Do not use it as a confidence interval. Real bbox-bottom errors during occlusion are several times larger. It is the 1σ length of the major axis of the first-order propagated covariance, and it combines:
- ground-point pixel uncertainty;
- the DOF-corrected calibration residual.

See `project/projector.py`. Its synthetic coverage is reported in `docs/VALIDATION_LOG.md`.

`flags` is a `;`-separated string. Values in use: `speed_violation`, `suspected_id_switch`, `off_court_track`, `interpolated`, `synthetic_gt_identity`. Reserved: `possibly_airborne`, `feet_occluded`.

## Example (Python / SQL)
```python
import pandas as pd
p = pd.read_parquet("outputs/samples/synthetic_demo_ptz/tables/player_observations.parquet")
p[p.position_status == "observed_projected"][["timestamp_sec", "track_id", "court_x_m", "court_y_m", "court_sigma_m"]]
```
```sql
-- DuckDB
SELECT track_id, avg(court_sigma_m) FROM 'outputs/samples/synthetic_demo_ptz/tables/player_observations.parquet'
WHERE position_status = 'observed_projected' GROUP BY 1;
```
