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

- [x] M3a (commit 709ed03): calibration grading = line inliers AND floor IoU AND motion
  plausibility (`grade_calibration`). Gate study on clips 1-5 (DEVELOPMENT data: thresholds
  were chosen on it):
  - old rule accepted 27.6% of 980 synthetic wrong poses;
  - new rule accepted 0/980 and rejected 1/98 visually verified poses.
  - Regression tests use the real clip-6 failures f2400 / f6091 (DEVELOPMENT examples, not
    held-out).
  - Still to do: validation on unseen clip-6 frames with naturally occurring wrong solutions.

## Validation data split (user request 00:24)
- Development / regression: clips 1-5 gate study; clip-6 frames 0, 2400, 4000, 6091/6100
  (used to design the grading and the tests).
- Unseen validation: other clip-6 frames, chosen AFTER the thresholds were frozen
  (commit 709ed03). Naturally occurring wrong solutions are generated there by the
  automatic search, as an experiment only.

## Incidents
- ~23:20-00:21: the laptop apparently slept (system log), which suspended the diagnostic run.
  `run_long_video.py` now requests "system required" while it runs (per-process, no settings
  changed). Stage A writes `progress.json` every 300 frames.

- [x] M2 (02:11): diagnostic done (`outputs/real/clip6_long_115s_diag`).
  - Calibration: 1019/1020 good, 1 flash frame held; overlays visually OK on 12 frames.
  - Videos decode fully (1020/1020 packets each); peak memory 2.0 GB.
  - Slow run: RF-DETR on DirectML degraded to 1053 ms/frame (CPU now 373 ms; 318 ms on
    DirectML earlier). CPU stages ran 3-4x slower than normal after the sleep/resume.
    Full run therefore uses RF-DETR on CPU; the run script now also opts out of EcoQoS throttling.
- [x] Stage-B variants on the diagnostic interval (tools/variant_runner.py,
  outputs/experiments/clip6_diag_variants/variant_summary.json). All proxies, no GT:
  - manual team exemplars vs k-means: referees/frame p50 4 -> 2; players p50 8 -> 10;
    tracks with colour switch 9 -> 5. Adopted (as on clips 1-5).
  - stitching: 0 links (no effect).
  - lost-track 3 s: fewer tracks (23 -> 21), but merges are not yet verified -> not adopted.
  - New box-height ratio = box height / expected pixel height of a 1.95 m person at the
    ground point. Players: median 0.89-1.05; front-row spectator 0.45.
    - Per detection, output only (< 0.6): coordinate jumps 87 -> 44, speed violations
      20 -> 12 per 1000 rows, tracking unchanged. 12/12 sampled suppressed rows had hidden feet
      (AI visual check). Adopted.
    - Per track (median < 0.6 -> role other): frames with > 10 "players" 22% -> 3%. Both
      relabelled tracks are front-row spectators. Adopted.
    - Thresholds were chosen on this interval: to be checked on unseen intervals of the full run.
  - Hiding the point from the TRACKER as well fragmented tracks (23 -> 28): rejected.
  - Bug fixed: interpolation created duplicate (track, frame) rows where an observed row had an
    unknown position (11 duplicates at the flash frame). Now merged into the existing row
    (`merge_interpolated`, test_stage_b_rows.py).
- Found by visual check: track t004 switches from an on-court player to a front-row spectator;
  referee #45 at the near sideline is labelled player B for part of track t002.
- Floor-IoU limitation found by the synthetic integration test: a correct pose of a wide view
  scores 0.34-0.45 when the floor outside the lines is court-coloured (wood apron).
  `court_coverage` (share of the projected court with floor tone) is now logged as well; the
  final rule is decided after stage A from the logged values. 2 integration tests fail
  until then (not committed).

- [x] Ball (commit 70e4455): in the clip-6 diagnostic the last ball gap (f937-1019) had WASB
  candidates with peaks up to 0.88; they were deleted by the static-clutter rule
  "|floor y| > 12 m". A ball high in the air (a shot) projects far beyond the court.
  - Rule off -> ball frames gained: clip 2 +23, clip 4 +23, clip 5 +14, clip 6 diag +65,
    clip 3 +3, clip 1 0; frames lost: 0 everywhere.
  - 48/48 sampled gained frames were the real ball in flight (AI visual; sheets in
    outputs/experiments/ball_far_rule/). Rule off by default.
- [x] Lost-track 3 s variant: the track mapping shows merges across teams (e.g. a Lakers track
  continuing as a Nuggets player) -> rejected.
- [x] Stage-A checkpoints every 1000 frames (`cache/stageA_ckpt.pkl`, resume test written,
  run after the full run).

- [x] M4 (04:05): full run finished in 94 min wall time.
  - Stage times: stage A 73 min (1.53 frames/s; RF-DETR on CPU 501 ms/frame, calibration
    148 ms/frame); WASB 9.6 min; appearance 1.5 min; tracking 1.5 min; rendering 8.2 min.
  - Peak memory: 3.39 GB (tracking stage); stage A 2.07 GB.
  - Both videos decode completely (6756/6756 packets, 0 errors).
  - Original video SHA-256 unchanged.
  - Calibration: segment 0 99.21% accepted / 0.79% uncertain; segment 2 99.85% / 0.15%;
    segment 1 unsupported.
  - Overlays visually OK on 36 frames. All 11 flash frames recovered on the next frame.
- Incident: after 04:05 the session stopped without continuing (no process running) until the
  user asked at 12:45. Resumed at 12:45.
- 12:46: tracking + rendering re-run with the latest code (airborne-ball fix was not in the
  02:31 process). The first output is kept in outputs/experiments/clip6_full_stageB_v1.

## Open / in progress
- 02:31: FULL run started:
  `tools/run_long_video.py data/raw/clip6_long_115s.mp4 --video-id clip6_LAL_at_DEN_long
  --segments data/annotations/clip6/segments.json --out outputs/real/clip6_long_115s
  --det-provider cpu --provider dml --team-init data/annotations/clip6/teams.json
  --team-frames 60 400 860`
  - Log: outputs/real/clip6_long_115s/run_long.log; stage-A progress in progress.json.
  - If interrupted, rerun the same command: finished stages are skipped.
