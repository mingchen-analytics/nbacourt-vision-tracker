# Overnight report: 2026-10-15/16

Scope: clip 6 (`data/raw/clip6_long_115s.mp4`, 112.7 s, LAL @ DEN) plus re-checks on clips 1–5. Working log: `docs/OVERNIGHT_PROGRESS.md`. Code is on local git branch `dev` (latest commit at the end of this file; nothing pushed). Detailed clip-6 results: `outputs/real/clip6_long_115s/LONG_VIDEO_EVALUATION.md`.

**Evidence labels used throughout**
- **measured**: our tools computed it.
- **proxy**: a self-consistency or heuristic statistic, not accuracy.
- **AI visual**: the assistant judged it from image crops; this is not ground truth.
- **not measured**: no evidence exists.

No human-reviewed annotations exist yet. Nothing here is a measured HOTA, IDF1, precision/recall or metre error.

## 1. Executive summary
- **The full 112.7 s video was processed end to end.**
  - It has 3 camera segments (main view → sideline close-up → main view) and 11 single-frame flashes. Both are handled: calibration restarts at the cut, flash frames hold the previous camera, and nothing is carried across the cut.
  - All output videos decode completely. The original video is unchanged (SHA-256 identical).
- **Calibration safety is the main technical result.** The old quality rule (line inliers ≥ 0.55) accepted:
  - 27.6% of 980 deliberately wrong poses on clips 1–5;
  - **85% (232/272) of naturally occurring wrong solutions** on unseen clip-6 frames;
  - and a real pose that drew the court onto the stands (inliers 0.81).

  The new grading needs line inliers ≥ 0.70, floor agreement (IoU ≥ 0.45, or court coverage ≥ 0.80 when IoU ≥ 0.30) and plausible camera motion. It accepts:
  - **2/1060** synthetic wrong poses;
  - **9/272 (3.3%)** natural wrong solutions;
  - **17/17** verified references.

  It rejects 1/98 correct real poses. Clip 6 under the new grading: **99.2% and 99.85% of frames accepted** in the two main-camera segments, with no visible drift over 97 s (AI visual check on 36 overlay frames).
- **Ball: a bug deleted balls in flight during shots.** An old clutter rule dropped every candidate whose floor projection is more than 12 m from the court's long axis; a ball high in the air always projects that far.
  - After the fix, all 48 sampled restored frames on clips 2/4/5/6 were the real ball (AI visual), and 0 frames were lost on any clip.
  - Full video, segment 0: missing ball 35.7% → **29.2%**; longest gap 2.39 s → **1.30 s**.
- **New ground-contact check: box height vs the expected height of a standing person at that point.**
  - It stops front-row spectators and referees, whose feet are hidden by the crowd, from being placed on the court as players.
  - Diagnostic interval: frames with more than 10 "players" 22% → 3%; coordinate jumps 87 → 44 (proxies). Every sampled affected case was verified (AI visual).
- **Not improved: long-term identity.**
  - Segment 0 has 48 player tracks in 97 s, where about 10 is ideal.
  - Swaps still happen at close crossings, and tracks still switch between a player and a front-row spectator.
  - Offline stitching finds no safe links, and the one association change tested (a longer lost-track window) merged players of different teams, so it was rejected.
  - Progress here needs a re-ID model (which needs your approval) and reviewed annotations.

## 2. Experiments run

| # | Experiment | Result | Kept? |
|---|---|---|---|
| 1 | Video inspection + cut/flash detection (`tools/inspect_video.py`) | 2 true cuts, 11 flash frames (the first pass misread the flashes as cuts) | yes |
| 2 | Automatic calibration re-initialization after a cut (pan/tilt/focal grid, same camera centre) | **Failed** on 3/3 frames: line scoring chose a pose with the court on the stands; floor scoring hit the grid limits | **no**; manual keyframe instead |
| 3 | Calibration gate study (clips 1–5, 980 perturbed poses) | Old rule accepts 27.6% of wrong poses; inliers + floor IoU accept 0/980 | yes |
| 4 | Gate study v2 (+ coverage, + synthetic wood-apron arena) | IoU alone rejects 3/8 correct synthetic poses; adding the coverage alternative rejects 0/8, real clips unchanged | yes |
| 5 | Natural wrong solutions on 17 unseen clip-6 frames | The first coverage rule accepted 14/272; a loophole (IoU ≈ 0, coverage ≈ 1) was fixed **after** seeing this, giving 9/272 | yes (the fix is post-hoc) |
| 6 | Stage-B variants on the diagnostic interval (f0–1019) | See §5: exemplars yes; stitching no effect; lost 3 s rejected; occlusion sigma and birth 0.5 not better; height rules yes | mixed |
| 7 | Ball far-projection rule on vs off (clips 1–5 + clip-6 diagnostic) | +23 / +23 / +14 / +65 ball frames on clips 2 / 4 / 5 / 6, 0 lost, 48/48 gained frames real | yes (rule off) |
| 8 | RF-DETR DirectML vs CPU | DirectML 1053 ms vs CPU 373 ms (after a sleep/resume; DirectML had been 318 ms) | CPU for this run |
| 9 | Stage-A checkpoint / resume | The resumed run equals the uninterrupted run (test) | yes |
| 10 | AI visual audit of 10 high-severity swap flags | 2 probable swaps, 3 unresolved overlaps, 3 no swap, 2 unclear | review aid only |

**Failed or rejected:** experiment 2; the lost-track 3 s variant; hiding the hidden-feet detections from the tracker as well (tracks 23 → 28: more fragmentation); occlusion sigma 2 m and birth threshold 0.5 (no proxy improvement).

## 3. Before/after comparisons (all proxies or AI visual)

| Area | Before | After | Evidence |
|---|---|---|---|
| Wrong poses accepted (synthetic, clips 1–5) | 27.6% | 0.2% (2/1060, synthetic arena only) | measured on perturbed poses |
| Natural wrong solutions accepted (unseen clip-6 frames) | 85% | 3.3% | measured, labelled against verified references |
| Correct poses rejected (clips 1–5 / synthetic) | 0/98 / 0/8 | 1/98 / 0/8 | measured |
| Ball missing, clip-6 segment 0 | 35.7% | 29.2% | measured coverage; restored frames AI-visual-verified |
| Ball longest gap, clip-6 segment 0 | 2.39 s | 1.30 s | measured |
| "Referees" per frame, diagnostic (median) | 4 | 2 | proxy (3 referees exist) |
| Frames with > 10 players, diagnostic | 22% | 3% | proxy; relabelled tracks AI-visual-verified |
| Coordinate jumps, diagnostic | 87 | 44 | proxy; suppressed rows AI-visual-verified (12/12) |
| Duplicate (track, frame) rows | 11 | 0 | measured (bug fix) |

Thresholds for the height rules were chosen on the diagnostic interval, and the calibration thresholds on clips 1–5. The clip-6 calibration validation is partly post-hoc (see experiment 5).

## 4. Camera calibration

| Clip 6 segment | Frames | Accepted | Uncertain | Rejected / unsupported | Initialization |
|---|---|---|---|---|---|
| 0 main | 5818 | 99.21% | 0.79% (10 flash + 36 motion-blurred frames) | 0% | manual-estimate keyframe f0 |
| 1 close-up | 273 | — | — | 100% (unsupported view) | none |
| 2 main | 665 | 99.85% | 0.15% (1 flash) | 0% | manual-estimate keyframe f6100 (automatic re-init rejected, exp. 2) |

- **The 36 uncertain frames** (f3978–4016) are a fast blurred pan. Their overlays look correct, so the gate is cautious here (≈ 0.6% false rejections).
- **Flash frames.** All 11 recovered "good" on the very next frame.
- **PTZ vs homography.** PTZ (fixed centre) was kept. A held-out comparison on this video was **not measured**.
- **Uncertain or failed frames keep image-space tracking**, but get no court coordinates (positions `unknown`, or interpolated when the gap is < 1 s).
- **Regression tests from the real failures:** `tests/unit/test_calibration_gates.py`, with fixtures in `tests/fixtures/calib_clip6/`.

## 5. Player tracking

| Problem | What changed | Status |
|---|---|---|
| ID swaps at crossings | Nothing that helps safely. Exemplar team labels reduced colour-switch tracks on the diagnostic interval (9 → 5, proxy) | **open**: teammates are not separable by colour; needs re-ID |
| Re-entry / players entering later | No change. Lost tracks survive 1.5 s; a returning player gets a new ID. Longer windows merged different players | open, by design (no invented identity) |
| Bench / sideline / spectators as players | Box-height rules + manual team exemplars (§3) | improved (proxy); remaining: tracks switching between a player and a spectator at the image bottom |
| Fragmentation / wrong merges | Stitching still 0 links (strict); no aggressive merging was introduced | open: 48 player tracks / 97 s |

## 6. Ball tracking
- **WASB remains the baseline**, now with the airborne-ball fix (§1, §3).
- **Clip 6:** 64.7% / 72.2% of frames detected (segments 0 / 2); 6.1% / 2.6% interpolated; 29.2% / 25.3% missing.
- **Temporal stability.** 2 `ball_jump` flags (possible wrong links) and 0 static false-candidate flags.
- **No 3D ball positions are produced.** Only a possession proxy is reported, while the ball is inside a player's box.
- **Still missing:** the ball during dribbles inside crowds and in blurred pans.
  - Distant views (clip 3) are still weak: 13% detected.
  - Local high-resolution crops were **not tried** tonight.

## 7. Performance (Ryzen 7 7840HS, Radeon 780M, 31.2 GB RAM)
- **Full video:** 94 min wall time ≈ 0.84 s/frame ≈ 1.2 fps (50× slower than real time).
  - Stage A: 73 min. RF-DETR on CPU takes 501 ms/frame and calibration 148 ms/frame.
  - WASB: 9.6 min. Tracking: 1.5 min. Rendering: 8.2 min.
- **Peak memory:** 3.39 GB.
- **Resource limits seen:**
  - DirectML slowed to 3.3× slower than CPU after the laptop slept.
  - CPU stages ran 3–4× slower right after the resume.
  - The session stopped advancing between 04:05 and 12:45 without any process running (my orchestration stopped; the pipeline itself did not fail).
- **Mitigations added:**
  - keep-awake request and EcoQoS opt-out (per process; no system settings changed);
  - stage-A progress file;
  - resumable checkpoints every 1000 frames.

## 8. Remaining problems (ranked)
1. **Identity continuity.** About 5 tracks per player in 97 s, and swaps at close crossings. The biggest gap for "consistent player tracking"; it needs an appearance identity cue (re-ID) and reviewed GT to tune safely.
2. **Player ↔ front-row spectator switches** at the near sideline / image bottom. Positions are now suppressed, but the track identity is still polluted.
3. **Ball during crowded dribbles and distant views:** 25–30% missing on clip 6, 87% on clip 3.
4. **Implausible speeds / coordinate jumps** (272 / 256 flags, mostly at crossings and occlusions, not calibration). Ground contact is still the bounding-box bottom.
5. **Calibration re-initialization after a cut still needs a manual keyframe.** Natural wrong solutions are now mostly rejected (3.3% still accepted), but there is no reliable automatic initializer.
6. **The gate is over-cautious on blurred pans** (0.6% of frames).

## 9. Files to review

| What | Where |
|---|---|
| Clip-6 videos | `outputs/real/clip6_long_115s/annotated.mp4`, `side_by_side.mp4` |
| Suspected failures (start with these) | `outputs/real/clip6_long_115s/tracking_issues.csv`; clips in `outputs/real/clip6_long_115s/issues/` (60 short clips, named by issue id) |
| Tracking data | `tracking_data.parquet`, `ball_tracking.parquet`, `tables/` (schema 1.3.0) |
| Calibration | `calibration_diagnostics.csv`, `calibration_overlays/` |
| Clip-6 evaluation | `outputs/real/clip6_long_115s/LONG_VIDEO_EVALUATION.md` |
| Experiments | `outputs/experiments/` (`calib_gates*`, `calib_natural`, `clip6_diag_variants`, `ball_far_rule`, `swap_audit_*`) |
| Annotation packages | `outputs/annotation/clip6_A_identity`, `clip6_B_sideline`, `clip6_C_ball_calib` (10 s each, outside the tuning interval) |

## Addendum: 2026-10-16 afternoon session (user away)

| Item | Result | Evidence |
|---|---|---|
| Local CVAT 2.79.0 | Running on Docker/WSL2, data on D:. Tasks 1–3 = clip-6 intervals A/B/C, pre-annotations verified | measured |
| Interval GT evaluation | Works end to end on the user's partial clip6_A export (refuses to report as GT until all tracks are reviewed) | tooling check |
| Clips 1–5 rerun with the current pipeline (`outputs/real/clip{1..5}_v2`) | All videos decode fully. Speed-violation rows fell on clips 1–3; clip-2 ball 80% → 85%. 5 calibration rejections on flash frames (harmless); clip 3: 44 cautious false rejections (wide blurry view) | measured / AI visual |
| Tiled WASB (opt-in `TrackConfig.ball_tiles`) | Clip-3 ball 13.5% → 20.1% (11/12 gained real); clip-6 C 65.3% → 68.5% (9/12 real, 3 doubtful); +250–280 ms/frame | proxy + AI visual |
| Tracker height-consistency gate | Helped on the tuning interval, did **not** generalize to unseen clip-6 frames (fragmentation up) → kept off | proxy |
| Incidents | Unexpected reboot corrupted CVAT's Redis file. It was repaired after backups, and the user's annotations were verified intact. Memory pressure (CVAT VM ~8 GB) made pipeline runs fail, so CVAT was paused during runs and restarted after | measured |

## 10. Next actions that need you
1. **Annotate the three 10 s clip-6 intervals in CVAT.** Correct the pre-annotations and add ball points. This turns every proxy above into a measurement. The evaluation tool now supports interval packages: `tools/evaluate_against_gt.py --interval`.
2. **Approve (or not) downloading a re-ID model** (Deep-EIoU's OSNet, ~15–30 MB, weights licence not stated, research use only). This is the main path for problem 1.
3. **Verify the assistant-made inputs:** `data/annotations/clip6/teams.json` (team exemplars), `kf_0000.json` / `kf_6100.json` (keyframes), `segments.json`.
4. **Decide whether a more cautious gate is acceptable.** Raising the inlier threshold to about 0.76 would reject most of the remaining 9 natural wrong solutions, but also about 5% of clip 3's correct frames.
