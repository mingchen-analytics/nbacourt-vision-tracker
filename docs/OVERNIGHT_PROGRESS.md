# Overnight progress log (2026-10-15/16)

Resumable checkpoint log. Newest entries at the bottom. Git: local repo, branch `dev`
(baseline commit `a4addca` on `main`). Nothing is pushed anywhere.

## How to resume
1. `git log --oneline` on branch `dev` shows the completed milestones.
2. Check the "Open / in progress" section below for running jobs and their output folders.
3. Long runs are resumable: `tools/run_long_video.py` skips every stage whose output already
   exists (stage A writes `run_analyze.json` last; WASB writes `cache/wasb_ball_candidates.parquet`).

## Inputs (not in git: data/raw is ignored; annotations are tracked)
- Video: `data/raw/clip6_long_115s.mp4`. 1280x720, 59.94 fps, 6756 frames, 112.7 s, H.264.
  SHA-256 before processing is in `outputs/clip6_sha256_before.txt`.
- `data/annotations/clip6/segments.json`: 3 camera segments + 11 flash frames
  (from tools/inspect_video.py + visual review).
- `data/annotations/clip6/kf_0000.json`, `kf_6100.json`: manual-estimate keyframes
  (assistant visual estimates, NOT ground truth).

## Milestones
- [x] M0: video inspection. Segments: 0 = main view f0-5817; 1 = sideline close-up f5818-6090
  (unsupported); 2 = main view f6091-6755. The 11 "cuts" at f927, 1880, 2375, 3346, 3739, 3914,
  4238, 4625, 4800, 5648, 6662 are single over-exposed flash frames, not cuts.
- [x] M1: multi-segment pipeline (per-segment calibration, flash-frame hold, tracker reset at
  cuts, per-segment ball Viterbi), floor-IoU diagnostic, memory-safe detection storage, issue
  detector, metrics tool, interval CVAT packages + interval GT evaluation. 99 tests pass.
- [ ] M2: 17 s diagnostic run (frames 0-1019) -> `outputs/real/clip6_long_115s_diag`
- [ ] M3: calibration safety gates + regression tests
- [ ] M4: full 112 s run -> `outputs/real/clip6_long_115s`
- [ ] M5: issue list, metrics, LONG_VIDEO_EVALUATION.md, CVAT intervals
- [ ] M6: focused tracking / ball experiments on short intervals
- [ ] M7: OVERNIGHT_REPORT.md

## Findings so far
- Automatic calibration re-initialization after a cut (same camera centre, pan/tilt/focal
  grid) FAILED on 3/3 test frames (f2400, f4000, f6091):
  - Line-alignment scoring picked a pose that draws the court onto the stands, with a
    line-alignment inlier fraction of 0.81.
  - Floor-IoU scoring hit the edges of the focal grid.
  - The code was removed; segment 2 uses a manual-estimate keyframe at f6100 instead.
- The Denver centre-court logo ring (~6 m wide) is NOT the 3.66 m centre circle; using it as a
  landmark gave 94-100 px residuals.

## Open / in progress
- 23:03: diagnostic run started (`tools/run_long_video.py ... --max-frames 1020`).
