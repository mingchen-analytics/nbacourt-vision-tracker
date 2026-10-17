# Video Indexing Engine: design and MVP (v0.1)

Status: MVP working on clip 6 (branch `feature/video-index`). This is a **timeline and indexing tool, not a tactics recognizer.**

## 1. Purpose and boundaries
```
full broadcast video ─► game timeline ─► candidate possessions ─► searchable, editable video intervals
```
- **Runs independently of the player-tracking pipeline.** Clip 6: 49 s of decoding plus 2 s of indexing for a 112.7 s video, vs ~94 min for tracking.
- **Joins the tracking engine later only through time.** Every event stores the source video's presentation timestamps (the same `timestamp_sec` as the tracking tables, and the same clock as HTML5 `currentTime`) plus frame indices. A tactical label added later is only a new event or label on an interval: nothing is re-processed.
- **Never stores track IDs.** `tracking_data_reference` names a tracking run and a frame range; local track IDs are not global player identities.

## 2. Evidence and what it can (not) say

| Signal | How (MVP) | What it means | What it does NOT mean |
|---|---|---|---|
| Camera cuts / flash frames | HSV-histogram distance; flash = f differs from both neighbours while f−1 ≈ f+1 (`index/shots.py`) | A view boundary | **Not** a possession boundary, not a stoppage |
| View type | Floor-colour fraction per segment (proxy), or a reviewed segments file | Main camera vs other | Not "replay" by itself |
| Scoreboard presence | Correlation of a fixed scoreboard patch with a reference | Scoreboard hidden (≥ 1 s) → `non_live` candidate, needs review | Replays that keep the scoreboard are not caught |
| Game clock | Glyph-template OCR (`index/scoreboard.py`) → stable 1-second runs | Running clock = live play; a value shown ≥ 2 s = clock stopped (start known to ±1 s) | Not who has the ball |
| Shot clock | Same reader; resets = value increases | Reset to 24: medium evidence of a change of possession. Reset to 14: usually the same team keeps the ball | Not every reset is a change of possession |
| Score | Same reader; stable values; changes of +1…+4 | The scoring team had the ball; the graphic lags the basket by 1–3 s | Free throws and and-ones are not separate possessions |
| Play-by-play | **Not implemented** | Would identify plays and players once the game is identified | — |

### Boundary rules (`index/events.py`)

| Evidence | Boundary placed at | Confidence |
|---|---|---|
| Reset to 24 + a score change ≤ 4 s later (made basket) | The reset | 0.9 |
| Score change without a reset (e.g. free throw) | The score change | 0.75 |
| Reset to 24 alone | The reset | 0.6 |
| Reset to 14 | Never a boundary | — |
| Cuts, flashes, replays | Never a boundary | — |

- Candidate possessions are the intervals between boundaries.
- **Offense team** is inferred only from score changes (the scorer had the ball; the other team has it next) and from alternation across boundaries of confidence ≥ 0.6. Conflicts stay unknown.
- An interval is marked `needs_review` when confidence < 0.7, the offense is unknown, or the video truncates it.

### Glyph-template OCR (no model download)
1. One decode pass stores the character glyphs of every scoreboard field in every 6th frame (10 Hz).
2. The glyphs are clustered per field. A person labels each cluster once per broadcast graphics package, from a contact sheet (`glyph_clusters_sheet.png`; ~45 labels for ESPN).
3. Reading accepts a glyph only if its best match is ≥ 0.80 NCC and beats the best other label by ≥ 0.05. Otherwise the whole field is `None`; nothing is guessed.
4. Values count only after 3 identical consecutive readings, which removes score-graphic animations.

## 3. Data model (`index/schema.py`, `nbatrack.index.events/0.1`)
- **Files:** `events_auto.json` / `.parquet` (automatic, never hand-edited) and `events_reviewed.json` (written by the UI; every save keeps the previous version in `history/`).
- **Header:** `game_id, video_id, video_path, video_sha256, fps, n_frames, duration_s, teams, flash_frames, provenance`.
- **Event fields:**
  - Identity and interval: `event_id, game_id, video_id, event_type, labels[], video_start_seconds, video_end_seconds, frame_start, frame_end`.
  - Game state: `period, game_clock_start/end, shot_clock_start, score_{away,home}_{start,end}`.
  - Teams: `offense_team_id, defense_team_id`.
  - Provenance: `period_source, game_clock_source, shot_clock_source, score_source, team_source` (`ocr_observed | interpolated | inferred | manual | null`).
  - Quality and evidence: `event_confidence, boundary_source[], evidence[], review_status` (`auto | needs_review | confirmed | edited | rejected`), `label_source` (`auto | manual`).
  - Links and notes: `camera_segment_id, tracking_data_reference, notes`.
- **Event types:**
  - Automatic: `camera_segment, non_live, score_change, shot_clock_reset, candidate_possession`.
  - Manual: `possession, shot_attempt, made_shot, turnover, foul, free_throw, rebound, timeout, replay, other`.
- **Auxiliary tables:**
  - `readings.parquet`: per sample, the parsed values; `None` = unreadable.
  - `game_clock_map.parquet`: video t0/t1 ↔ game clock; this is the key for future play-by-play alignment.
  - `camera_segments.parquet`.

## 4. Review UI (`tools/index_server.py` + `tools/index_ui/index.html`)
- **Implementation:** Python standard-library HTTP server on 127.0.0.1 plus one HTML page, with no new dependencies. It serves the **original** video with HTTP Range, so the browser can seek; the video is not copied.
- **What it does:**
  - Timeline rows: views, non-live, candidate possessions (coloured by offense), manual events, score / reset markers.
  - Event table with quick searches (candidate possessions, score changes, needs review, manual shot attempts) plus type, status and text filters.
  - Click to seek. Edit start/end (set to the playhead, ±1 frame), type, offense team, labels, status and notes. Add manual events at the playhead. Confirm or reject (automatic events cannot be deleted, only rejected).
  - Play an interval; optional clip export (ffmpeg) to `<out>/clips/`.

## 5. Results on clip 6 (`outputs/index/clip6/eval_report.json`)

| Measurement | Result | Kind |
|---|---|---|
| Scoreboard visible | 100% of samples | measured |
| Readable: game clock / shot clock / period | 100% / 100% / 100% | measured |
| Readable: away / home score | 98.5% / 97.9% (animations → `None`) | measured |
| Game-clock consistency | 110/110 tick steps are exactly −1 s; slope −0.999 game s per video s | measured self-consistency |
| Reading accuracy, 30 random frames | 119/119 field values agree with the assistant's reading. Hidden home score at f2652 → `None` (correct) | **assistant visual reference, not independent GT** |
| Score monotonicity | 0 readings below the running maximum | measured |
| Seek accuracy | 39/39 keyframe seeks pixel-identical to sequential decoding (PyAV) | measured; browser seeking itself not measured |
| Events | 29: 3 camera segments, 1 non-live (free-throw stoppage), 7 score changes, 8 resets to 24, 10 candidate possessions (5 needs_review) | output |
| Boundaries near cuts / inside non-main views | 0 / 0 (the 97–101 s close-up is mid-possession: the clock keeps running) | proxy |
| Boundaries within 1 s of a flash | 3 (photographer flashes at shots coincide with made baskets) | proxy, explained |
| Boundary precision / recall | **Not measured**: needs reviewed possessions | — |
| Runtime / memory | Extract 49.4 s (7.3 ms/frame, 0.44× real time), index 2.3 s; peak 660 MB | measured |

All 10 candidate possessions alternate LAL/DEN consistently with the 7 observed score changes (inferred, not reviewed).

## 6. Known limitations
- **Glyph templates are per graphics package.** A new broadcaster or season needs ~10 minutes of cluster labelling, and different resolutions need new ROIs. Game clocks under one minute ("SS.t") are not parsed yet.
- **Made-basket rule can mis-pair.** A reset to 24 followed within 4 s by the *other* team scoring (a very fast break) would be read as one made basket.
- **Replay detection is minimal.** Only a hidden scoreboard or a non-main view is flagged; replays that keep the bug or show the main angle are not detected.
- **No turnover, shot or rebound recognition.** These are manual event types for now.
- **Clip 6 is easy:** continuous live play with only one dead ball. Full games add timeouts, quarter breaks, commercials, free-throw sequences and replays.

## 7. Scaling from 2-minute clips to full games (recommendation)
1. **Decode cost.** Already ~0.44× real time on CPU for one pass: a 2.5 h broadcast ≈ 66 min. Sample the scoreboard at 5 Hz and compute cut statistics at 320×180. Use a background job with resumable chunks (as in tracking stage A).
2. **Storage.** One events file per game plus Parquet readings (a few MB). Keep the original video; serve it with Range requests. Optional clips on demand.
3. **Graphics packages.** Keep a small library of `scoreboard.json` + `glyph_labels.json` per broadcaster/season, and auto-detect which applies by template correlation. Add a tenths-of-a-second clock parser and a "no scoreboard" state for commercials.
4. **Game identification and play-by-play.** Once the game is known (date + teams from the bug), align the public play-by-play by `(period, game clock)` through `game_clock_map`. This gives shot / turnover / foul labels with player names, without vision. It is the highest-value next step after reviewed ground truth.
5. **Ground truth.** Review 2–3 full quarters in the UI (confirm or edit candidate possessions). Then measure boundary precision/recall with `tools/index_eval.py`. Only after that, consider learned replay / dead-ball classifiers.
6. **Integration.** The tracking engine is run only on intervals the index marks as live main-camera play (large savings on full games). Tactical modules read `(event interval) × (tracking tables)` by time.
