# nbatrack — NBA broadcast video → court-coordinate tracking data

Goal: produce time-synchronized, physically meaningful player (and ball) coordinates on a canonical NBA court, in meters, from ordinary broadcast video. The output includes explicit quality and uncertainty, and never invents positions.

**Status (2026-10-15):**
- Implemented and tested on synthetic data:
  - geometry and calibration core;
  - data-driven PTZ-vs-free-camera model selection;
  - video timestamps;
  - data schema;
  - quality flags;
  - camera-motion propagation;
  - manual calibration tool;
  - multi-object tracker (ByteTrack-style + CMC + court-space association) with MOT metrics;
  - diagnostic visualization.
- Real footage, since 2026-10-15: RF-DETR detection, semi-automatic calibration, tracking, court coordinates, export, and side-by-side videos run end-to-end on 4 of the 5 supplied clips.
- Unreliable: team/referee labels and identities at player crossings.
- Not done: ball 3D, pose-based ground contact.
- **No real-footage accuracy has been measured.** See [docs/VALIDATION_LOG.md](docs/VALIDATION_LOG.md).

## Install (Windows; everything on D:)
```powershell
$env:CONDA_PKGS_DIRS = "D:\conda\pkgs"
conda create -y -p D:\conda\envs\nbatrack python=3.11
$env:PIP_CACHE_DIR = "D:\pip_cache"
D:\conda\envs\nbatrack\python.exe -m pip install -e ".[video,dev]"
```
FFmpeg/ffprobe on PATH is optional; the integration test uses it to cross-check timestamps.

## Use
```powershell
$py = "D:\conda\envs\nbatrack\python.exe"
& $py -m pytest                                   # all tests
& $py -m nbatrack.cli probe data\raw\clip.mp4 --hash
& $py -m nbatrack.cli calibrate data\raw\clip.mp4 --frame 0 --out data\annotations\calib_f0.json
& $py -m nbatrack.cli synth-demo --calib ptz --out outputs\samples\synthetic_demo_ptz
& $py tools\compare_calibration_models.py           # PTZ vs homography vs free camera
& $py tools\quality_flag_study.py                   # quality flags vs true error
& $py tools\selection_study.py                      # does model selection detect a moving camera?
& $py tools\tracking_study.py                       # tracker variants, IDF1 / MOTA / ID switches
```
Manual calibration keys: `n`/`p` next/previous landmark, click to place, `x` delete, `f` fit and overlay, `s` save, `q` quit.

## Processing a new video (works end-to-end since 2026-10-15)
1. Put the MP4 in `data/raw/` (git-ignored; you are responsible for the footage's legality).
2. One-time setup: export the detector with `python tools/export_rfdetr.py --size small --shape 512 896`. This downloads Apache-2.0 weights to `models/`.
3. `nbatrack probe data\raw\clip.mp4`
4. Annotate frame 0 with 4–6 court landmarks: `nbatrack calibrate VIDEO --frame 0 --out data\annotations\<clip>\kf_0000.json`.
   - Lane corners are best, because they are unambiguous intersections.
   - Arc "tops" are ambiguous along the arc.
5. Run the full pipeline:
   ```
   nbatrack process VIDEO --keyframe data\annotations\<clip>\kf_0000.json --video-id <id> --out outputs\real\<clip>
   ```
   - Options: `--provider cpu|dml` (default dml) and `--calib-mode ptz|free` (default ptz).
6. `nbatrack retrack outputs\real\<clip> [--court-min-sigma 1.0] [--render]` re-runs tracking and export from cached detections in seconds.
7. `python tools/real_video_report.py outputs\real\<clip> [--check data\annotations\<clip>\check_XXXX.json]` writes `report_metrics.json` and `diagnostics.png`.

Results of the first real runs, including what is and is not validated: [docs/REAL_VIDEO_REPORT.md](docs/REAL_VIDEO_REPORT.md).

### Validation and identity tools (milestone 2)
- **Ground-truth annotation.** Build a package with `python tools/make_cvat_package.py ...`, then follow [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md).
- **Evaluation.** Run `python tools/evaluate_against_gt.py --gt <cvat export zip> --runs <run dirs> --out outputs/eval/<clip>`. It uses TrackEval and refuses unreviewed annotations. Protocol: [docs/EVALUATION.md](docs/EVALUATION.md).
- **Tracker variants.** `python tools/tracker_ablation.py --runs ... [--gt clip2=...]`
- **Manual team/referee initialization.** `python tools/team_init_helper.py render|write ...`, then `nbatrack retrack RUN --team-init data/annotations/<clip>/teams.json`.
- **Root-cause strips** for switches and speed spikes: `python tools/identity_diagnostics.py RUN --video VIDEO`
- **Keyframe on any frame.** Calibration propagates forward and backward from it (used to recover clip 3).
- **Offline reconstruction.** Short-gap interpolation is on; stitching is experimental (`--stitch`). See [docs/OFFLINE_RECONSTRUCTION.md](docs/OFFLINE_RECONSTRUCTION.md) and [docs/IDENTITY_IMPROVEMENT_PLAN.md](docs/IDENTITY_IMPROVEMENT_PLAN.md).

## Layout
```
src/nbatrack/
  court/      geometry.py (NBA court in meters, 32 landmarks), render.py
  calib/      camera.py (pinhole + PTZ), homography.py, fit.py (free / PTZ frame / PTZ segment BA),
              models.py (common CalibrationModel interface), quality.py, propagate.py, manual.py,
              select.py (cross-validated PTZ vs free-camera choice)
  track/      kalman.py, tracker.py (ByteTrack-style + CMC + court-space gating)
  io/         video.py (PyAV, exact PTS)
  project/    projector.py (ground point -> court + sigma)
  export/     schema.py, writers.py (Parquet/CSV/JSON, validated)
  viz/        side_by_side.py
  synthetic/  scene.py, render.py, players.py, video.py (ground-truth simulator)
  eval/       calib_compare.py, quality_study.py, mot_metrics.py, tracking_study.py
  pipeline/   synthetic_demo.py
  cli.py
tools/  tests/{unit,integration}/  docs/
```

## Docs
- [docs/PHASE1_ARCHITECTURE.md](docs/PHASE1_ARCHITECTURE.md): architecture and plan
- [docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)
- [docs/LICENSES.md](docs/LICENSES.md): third-party register
- [docs/VALIDATION_LOG.md](docs/VALIDATION_LOG.md): every measured number and how to reproduce it

## Known environment constraints
- No NVIDIA GPU (AMD 780M iGPU). Inference will be CPU / ONNX Runtime; DirectML is to be benchmarked, not assumed.
- **Low system commit memory:** about 1.5 GB was free during development, even though about 8 GB of RAM was physically free. The C: drive's free space limits the page file. Two simultaneous 1080p x264 encoders failed with out-of-memory, so the synthetic demo renders at 720p. Real processing with neural models will need more headroom. Options: close browsers, or move or enlarge the page file on D:.
