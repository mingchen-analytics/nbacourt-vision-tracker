# Overnight report: 2026-10-15/16

Scope: clip 6 (`data/raw/clip6_long_115s.mp4`, 112.7 s, LAL @ DEN) plus re-checks on clips 1–5. Working log: `docs/OVERNIGHT_PROGRESS.md`. Code is on local git branch `dev` (nothing pushed).

**Evidence labels used throughout**
- **measured**: our tools computed it.
- **proxy**: a self-consistency or heuristic statistic, not accuracy.
- **AI visual**: the assistant judged it from image crops; this is not ground truth.
- **not measured**: no evidence exists.

No human-reviewed annotations exist yet. Nothing here is a measured HOTA, IDF1, precision/recall or metre error.

## 1. Executive summary
- **The full 112.7 s video was processed end to end.** It has 3 camera segments and 11 flash frames. Both videos decode completely (6756/6756 frames). The original video is unchanged (SHA-256 identical).
- **Calibration safety: the main technical result of the night.**
  - The old quality rule (line-alignment inliers ≥ 0.55) accepted **27.6% of 980 deliberately wrong poses** on clips 1–5, plus a real wrong pose (court drawn on the stands, inliers 0.81).
  - The new grading needs line inliers ≥ 0.70 AND floor agreement (IoU ≥ 0.45 or court coverage ≥ 0.80) AND plausible camera motion. It accepted **2 of 1060** wrong poses (both in a synthetic arena) and rejected 1 of 98 visually verified real poses.
  - Clip 6 under the new grading: **99.2% / 99.85% of frames accepted** in the two main-camera segments, with no drift over 97 s (AI visual check on 36 overlay frames).
- **Ball:** found and fixed a bug that **deleted balls in flight during shots**. An old clutter rule dropped any ball whose floor projection lands more than 12 m off court.
  - All 48 sampled restored frames on clips 2/4/5/6 are the real ball (AI visual). No frame was lost.
  - Full video, segment 0: missing ball 35.7% → **29.2%**; longest gap 2.39 s → **1.30 s**.
- **Ground-contact check (new):** box height vs the expected height of a standing person at that court position.
  - Front-row spectators whose feet are hidden were being placed on the court and counted as players; this check catches them.
  - On the diagnostic interval: frames with more than 10 "players" 22% → 3%, and coordinate jumps 87 → 44 (proxies; every sampled case was verified AI visual).
- **Not improved:**
  - Long-term identity. Segment 0 has 48 player tracks in 97 s; about 10 is ideal.
  - Swaps at close crossings, and swaps between players and front-row spectators.
  - Offline stitching still finds no safe links, and a longer lost-track window merged players of different teams, so it was rejected.
  - These need either a re-ID model (needs your approval) or annotations.

(Sections 2–10 follow below.)
