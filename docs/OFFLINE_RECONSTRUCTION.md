# Offline bidirectional trajectory reconstruction

Recorded video lets the system use both earlier and later observations. The work splits into three problems with different evidence and different risk. Implementation: `src/nbatrack/track/offline.py`, wired into stage B (`TrackConfig.stitch`, `TrackConfig.interpolate_max_gap_s`).

## Position states (schema 1.1.0)
| `position_status` | Meaning | Raw columns (`court_*_raw`, bbox, image_x/y) |
|---|---|---|
| `observed_projected` | A detection in this frame, projected through a good or fair calibration | filled |
| `interpolated` | No detection in this frame; position estimated between two reliable observations of the same track | **NULL**; these rows are new rows |
| `extrapolated` | Reserved for future off-screen inference; never produced yet | NULL |
| `unknown` | No usable position | NULL court values |

**Raw detection coordinates are never overwritten.**
- Stitching only changes `track_id`; the original fragment ID is kept in `source_track_id`.
- Interpolation only adds rows, each flagged `interpolated`.

## 1. Tracklet stitching (implemented; OFF by default)
A fragment *a* ending at t₀ and a fragment *b* starting at t₁ > t₀ may be linked only if all of these hold:
- the gap is at most 2 s;
- the court distance is at most v_max·gap + margin (7 m/s, 1 m);
- both have a known and identical class label (A / B / REF), if labels are available;
- appearance descriptors exist for both ends and are within the limit.

Links are chosen one-to-one by Hungarian assignment on a combined physics + appearance cost, and chains are followed transitively.

What happened on the clips (2026-10-15):
1. **First version (permissive):** it produced 3 links on clip 5. Visual check: at least one was wrong (a black-jersey Portland player was linked to white-jersey GSW #10). The cause was that the appearance descriptors were never computed, because fragments end in occlusions.
2. **Corrected version:** it requires appearance and known labels. It produced 0 links on all 4 clips. Remaining same-team candidates had appearance distances of 0.47–0.75, above the 0.45 limit: jersey color cannot tell teammates apart.
3. **Conclusion:** stitching needs an identity-level appearance cue (re-ID or jersey number) before it can help. It stays off until a reviewed-GT ablation shows a gain (`tools/tracker_ablation.py`, variants `S1_offline_stitch` and `V1+S1`).

## 2. Short-gap interpolation (implemented; ON, max 1.0 s)
For gaps of at most 1 s between two observed rows of the same track, missing frames are filled as follows:
- **Position:** cubic Hermite interpolation of the smoothed endpoint positions, with finite-difference velocities.
- **Sigma:** the larger endpoint sigma + 2 m/s × (time to the nearer observation). This is an uncalibrated heuristic.
- **Role and team:** copied only if both endpoints agree; otherwise `unknown`.

Interpolated rows on the five clips: 169, 205, 251, 155 and 161.

These rows count toward court-position coverage. They have no image box, so they are excluded from box-based tracking metrics.

## 3. Off-screen backward inference (design only; not implemented)
A person first observed at time t₁ was somewhere before t₁. Future observations alone cannot pin down where. What can honestly be stated is a **feasible region** for each earlier time t:

R(t) = { x on court : |x − x₁| ≤ v_max·(t₁ − t) + σ₁ } ∩ (court ∪ bench margin) ∖ V(t)

where V(t) is the floor area visible in the camera at time t, from the calibrated footprint. The person was **not** detected inside V(t), unless a detector miss is plausible there, for example in a crowded area.

Planned representation and rules:
- `position_status = "extrapolated"`, with the region stored as a polygon or a disc plus exclusion mask.
- No point estimate unless the region is small (e.g. the person entered from a frame edge one or two frames earlier).
- An identity-consistent earlier fragment (via stitching) takes priority, and turns the case into interpolation.
- If the region covers most of the court, leave the position `unknown`.

Validation will need annotated entries of people from off-screen.
