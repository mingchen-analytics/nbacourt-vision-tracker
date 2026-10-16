# Identity and classification improvement plan (evidence-based, 2026-10-15)

Status: no reviewed ground truth exists yet, so every number here is a **proxy or a visual spot check**. The order of work follows the failures actually observed, not assumed ones.

## Observed failures and root causes
| # | Failure | Evidence | Root cause |
|---|---|---|---|
| F1 | Identity switches at close crossings between opponents (e.g. clip 2: t003 POR #5 → GSW defender at ≈ frame 203) | Crop strips (`outputs/real/*/diagnostics_identity/strips`) and color-class change flags | IoU and the 1 m court gate cannot separate two people whose boxes overlap; bbox-bottom positions jump during the overlap |
| F2 | Fragmentation vs swaps trade-off | Court-gate sigma floor 0.05 m → 32 IDs on clip 2; 1.0 m → 13 IDs but swaps allowed | bbox-bottom ground point is noisy under occlusion (median 21% box-height change at speed-violation events) |
| F3 | Same-team fragments cannot be re-linked | Stitching diagnostics: same-label candidates have appearance distances of 0.47–0.75 | Hand-crafted color descriptors encode the jersey, not the person |
| F4 | Team/referee confusion under automatic k-means | Clips 4/5 visibly wrong; referee gray ≈ dark jerseys in L* (20–35 vs 13–32) | Color clustering is underdetermined; fixed by manual exemplars + kNN + votes + per-segment labels (holdout spot checks: clip 2 frame 420 9/9 plausible; clip 5 frame 300 5 correct, 2 unknown, 2 wrong) |
| F5 | Implausible speeds | 68–105 speed-violation rows per clip; heuristic causes: crossing overlap 15–36, ground-point jump 8–26, calibration jump ≤ 2 | Ground-contact localization under occlusion, plus residual switches. Calibration is rarely the cause. |

## Variants benchmarked so far (proxies only; all retained in `tools/tracker_ablation.py`)
| Variant | Proxy effect (4 clips) | Verdict pending GT |
|---|---|---|
| V0 baseline (CMC + court gate 1 m) | reference | — |
| V1 occlusion-aware gate (0.3 m clean / 1 m occluded) | speed-violation rows −14% to −44%; IDs ±0 to +5 | promising for coordinates; identity effect unknown |
| V2 team-aware association (color prototype) | more fragmentation in all clips | probably harmful in the current form |
| V3 longer recovery (2 s) | fewer IDs and births, but clip 2 has 11 IDs for 13 people (merges likely) | needs GT |
| S1 offline stitching | 0 links with safe rules | blocked by F3 |

## Prioritized next steps
1. **Reviewed GT for clips 2 and 5.** Required by every item below. Run `tools/tracker_ablation.py --gt ...` to replace the proxies with HOTA, AssA, IDF1 and IDSW.
2. **Identity-level appearance cue, to address F1 and F3.** Candidates, in order of cost:
   - (a) a lightweight person re-ID model (e.g. OSNet-x0.25, ~1–3 MB weights, MIT code; dataset licenses are research-only). This needs approval for the download.
   - (b) jersey-number OCR on unoccluded crops, as a sparse but strong identity anchor.

   Evaluate each **separately** on AssA / IDSW. Adopt it only if it beats V0 or V1.
3. **Occlusion-aware ground point, to address F2 and F5.** Adopt V1 if GT confirms it does not hurt identity. During heavy overlap, use the predicted court position from the track filter instead of the bbox bottom, and inflate sigma. Measure with foot-contact annotations (image-space px).
4. **Pose-based foot points.** Only if the bbox-bottom error measured on annotated foot points is large enough to matter. RTMPose Halpe26 heel/toe availability must be verified first.
5. **Stitching (S1).** Re-enable once step 2 gives discriminative descriptors; keep the strict physics and label rules.
6. **Team initialization UX.** Verify or replace the Claude-picked exemplars in `data/annotations/*/teams.json` using `tools/team_init_helper.py`. Add 2–3 referee exemplars per clip; clip 5 has only one.
