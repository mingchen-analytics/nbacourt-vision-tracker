# Development notes

Working notes for the author's development machine and the full tool set. The public overview is in the [README](../README.md).

## Development environment (Windows; everything on D:)
```powershell
$env:CONDA_PKGS_DIRS = "D:\conda\pkgs"
conda create -y -p D:\conda\envs\nbatrack python=3.11
$env:PIP_CACHE_DIR = "D:\pip_cache"
D:\conda\envs\nbatrack\python.exe -m pip install -e ".[video,dev]"
```
FFmpeg/ffprobe on PATH is optional; the integration test uses it to cross-check timestamps.

### Known environment constraints
- No NVIDIA GPU (AMD 780M iGPU). Inference is CPU / ONNX Runtime; DirectML is benchmarked per session, not assumed. After a sleep/resume it has been 3× slower than CPU.
- **Low system commit memory:** about 1.5 GB was free during development, even though about 8 GB of RAM was physically free. The C: drive's free space limits the page file. Two simultaneous 1080p x264 encoders failed with out-of-memory, so the synthetic demo renders at 720p. Do not run two pipelines at once.
- OpenCV 5.0.0: `cv2.USAC_MAGSAC` / `USAC_DEFAULT` return `None` on valid data, so the code uses `cv2.RANSAC`.
- Long runs: use `tools/run_long_video.py` (keep-awake request, EcoQoS opt-out, stage-A checkpoints every 1000 frames).

## Commands
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

## Processing a new video
1. Put the MP4 in `data/raw/` (git-ignored; you are responsible for the footage's legality).
2. One-time setup: export the detector with `python tools/export_rfdetr.py --size small --shape 512 896`. This downloads Apache-2.0 weights to `models/`.
3. `nbatrack probe data\raw\clip.mp4`
4. Annotate a keyframe with 4–6 court landmarks: `nbatrack calibrate VIDEO --frame 0 --out data\annotations\<clip>\kf_0000.json`.
   - Lane corners are best, because they are unambiguous intersections.
   - Arc "tops" are ambiguous along the arc.
   - Any frame can be the keyframe; calibration propagates forward and backward from it.
   - Multi-shot videos: pass `--segments segments.json` (one keyframe per supported camera segment) instead of `--keyframe`.
5. Run the full pipeline:
   ```
   nbatrack process VIDEO --keyframe data\annotations\<clip>\kf_0000.json --video-id <id> --out outputs\real\<clip>
   ```
   - Options: `--provider cpu|dml` (default dml) and `--calib-mode ptz|free` (default ptz).
6. `nbatrack retrack outputs\real\<clip> [--court-min-sigma 1.0] [--render]` re-runs tracking and export from cached detections in seconds.
7. `python tools/real_video_report.py outputs\real\<clip> [--check data\annotations\<clip>\check_XXXX.json]` writes `report_metrics.json` and `diagnostics.png`.

## Validation and identity tools
- **Ground-truth annotation.** Build a package with `python tools/make_cvat_package.py ...`, then follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) and [CVAT_USER_GUIDE.md](CVAT_USER_GUIDE.md).
- **Evaluation.** `python tools/evaluate_against_gt.py --gt <cvat export zip> --runs <run dirs> --out outputs/eval/<clip> [--interval]`. It uses TrackEval and refuses unreviewed annotations. Protocol: [EVALUATION.md](EVALUATION.md).
- **Tracker variants.** `python tools/tracker_ablation.py --runs ... [--gt clip2=...]`
- **Manual team/referee initialization.** `python tools/team_init_helper.py render|write ...`, then `nbatrack retrack RUN --team-init data/annotations/<clip>/teams.json`.
- **Root-cause strips** for switches and speed spikes: `python tools/identity_diagnostics.py RUN --video VIDEO`
- **Offline reconstruction.** Short-gap interpolation is on; stitching is experimental (`--stitch`). See [OFFLINE_RECONSTRUCTION.md](OFFLINE_RECONSTRUCTION.md) and [IDENTITY_IMPROVEMENT_PLAN.md](IDENTITY_IMPROVEMENT_PLAN.md).

## Video indexing
Design: [VIDEO_INDEXING.md](VIDEO_INDEXING.md). Review UI guide (Traditional Chinese): [VIDEO_INDEX_GUIDE.md](VIDEO_INDEX_GUIDE.md).
```
python tools/index_video.py extract <video> --scoreboard <scoreboard.json> --out outputs/index/<name>
python tools/index_video.py glyphs  --out outputs/index/<name> --labels <glyph_labels.json>
python tools/index_video.py index   --out outputs/index/<name> --scoreboard <scoreboard.json> --labels <glyph_labels.json>
python tools/index_server.py --out outputs/index/<name>     # http://127.0.0.1:8765
python tools/index_eval.py   --out outputs/index/<name>
```

## Session reports
- [REAL_VIDEO_REPORT.md](REAL_VIDEO_REPORT.md): first real-video runs (clips 1–5)
- [OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md), [OVERNIGHT_PROGRESS.md](OVERNIGHT_PROGRESS.md): 113 s multi-camera video, calibration gating, ball fixes
- [VALIDATION_LOG.md](VALIDATION_LOG.md): every measured number and how to reproduce it
