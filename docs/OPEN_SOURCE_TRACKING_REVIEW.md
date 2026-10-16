# Open-source tracking review and first integration (2026-10-15)

Purpose: stop re-inventing components that already exist. Seven public repositories were inspected for parts that could make NBA player and ball tracking more reliable on our broadcast clips. The highest-priority compatible part, a dedicated basketball ball detector (WASB), was integrated and benchmarked against the current system.

**Evidence levels used below**
- **measured**: computed by our tools on our clips.
- **visual spot check**: judged by the AI assistant from zoomed crops. This is *not* ground truth.
- **claimed**: stated by the repository or paper, not reproduced here.
- **not measured**: no evidence yet.

No ball or player ground truth has been reviewed yet (CVAT packages for clips 2 and 5 are still pending). So no accuracy number in this document is a validated accuracy.

## 1. How the repositories were inspected
- READMEs, source files, dependency lists, license files and model-download links were read on GitHub.
- Nothing was cloned into the project.
- One repository (WASB-SBDT, MIT) was cloned to `D:\external\WASB-SBDT`, outside the project. Only its HRNet model definition is imported, by `tools/export_wasb_onnx.py`, in a separate torch-only process. Its training and evaluation code was not run.
- One checkpoint was downloaded: WASB basketball, 5.8 MB, to `D:\external\weights`.

## 2. Technical comparison

| Repo | What it implements | Checkpoints | Failure modes it addresses | Evidence | License | Hardware / stack | Integrability with our pipeline |
|---|---|---|---|---|---|---|---|
| **nttcom/WASB-SBDT** (BMVC 2023) | Heatmap ball detector (HRNet, 3 frames in → 3 heatmaps out) + simple online tracker; models for soccer, tennis, badminton, volleyball, **basketball** | Basketball checkpoint public (Google Drive); weight license not stated; trained on the NBA broadcast dataset of Yan et al. (SAM) | Small, blurred, partly occluded ball; false positives on ball-like objects (temporal context) | Paper reports per-sport F1/AP (claimed); MODEL_ZOO has no numbers; **our benchmark below (measured + visual spot check)** | Code MIT; weights/dataset not stated | Original code asserts CUDA; we exported to ONNX → CPU 67 ms/frame (step 3), DirectML 25–27 ms/frame (step 3) / 62–74 ms/frame (step 1) at 512×288 | **High. Integrated** (`detect/wasb_ball.py`, ONNX Runtime only). Feeds our existing Viterbi ball tracker. |
| **cyrushadavi1/nba_video_analysis_v2** | Full stack: YOLO11 players + ByteTrack + offline stitching; SigLIP team clustering; jersey legibility (ResNet34) + PARSeq OCR; **WASB ball + Viterbi + interpolation with observed/interpolated/missing states**; court keypoints (YOLO11m-pose) | Fine-tuned YOLO11s (AGPL base), court pose model on HF, PARSeq/legibility (CC BY-NC) | ID switches (stitching, jersey numbers); ball (WASB) | Claims AP50 0.97 on SportsMOT and "21/24 players identified" (vague, not reproducible from repo) | **No license file** | Python ≥ 3.12, ~3 GB models, tuned for macOS MPS | Design reference only. The ball design (WASB → Viterbi → explicit states) is what we adopted. Code cannot be reused (no license, AGPL deps). |
| **hsiangwei0903/Deep-EIoU** | Online tracker for sports: expanding IoU (handles fast, irregular motion) + OSNet appearance; iterative scale matching | YOLOX detector + OSNet `sports_model.pth.tar-60` (Google Drive) | **Close-player swaps, fast direction changes** | Paper: 77.2 HOTA on SportsMOT (claimed) | No root license; `reid/` = torchreid (MIT); OSNet weights license not stated | Python 3.7, torch 1.13, CUDA 11.6, cython_bbox | Medium. Expanding-IoU cost is easy to re-implement in our tracker. The OSNet sports re-ID model can be exported to ONNX (~10–20 MB; **download needs approval**). |
| **DeepSportradar/player-reidentification-challenge** | Re-ID benchmark and baseline (open-reid ResNet50) on basketball crops (players, referees, coaches) | **No pretrained weights**; dataset 436 IDs / 8,569 crops | Re-identification across occlusions | Baseline ~72.7 mAP on challenge set (claimed) | License not stated (data terms on challenge page) | torch | Low now. Useful only as training data for a re-ID model, which needs training approval. |
| **jigggle/CourtVision** | YOLO26n detector with classes player / referee / basketball. **Only active players are labeled "player"**; bench and spectators are hard negatives, refined in a CVAT loop | No weights published | **Bench and sideline people mistaken for players**; ball | One validation clip: precision 0.904, mAP50 0.835, ball recall 0.50 (claimed); no IDF1/HOTA | No license; Ultralytics (AGPL) | Ultralytics | Labeling protocol is reusable. The model needs our own training data and a non-AGPL detector (e.g. RF-DETR fine-tune), so **training approval is needed**. |
| **hasan-bakr/Basketball-Game-Tracker** | RF-DETR via Roboflow Inference API + MCByte tracker + SAM masks + Cutie video segmentation; PARSeq jersey OCR; court keypoints | Roboflow-hosted (API key) | Occlusion (mask propagation), jersey identity | No metrics | MIT | GPU-heavy (SAM + Cutie), Roboflow account | Low. Hosted models (API key, account) conflict with "no paid services". Mask propagation is too heavy for this machine. |
| **Nolan-Lo/basketball-computer-vision** | YOLOv5l6u / YOLOv8x-pose, ByteTrack, fashion-CLIP teams, proximity possession | Google Drive | Team assignment | No metrics | No license; Ultralytics (AGPL) | GPU | Reference only. |

## 3. Mapping to the reported problems

| User-reported problem | Root cause (our diagnostics) | Most relevant external component | Status |
|---|---|---|---|
| "The ball location seems unknown" | COCO `sports ball` from RF-DETR: low recall, and many false positives on floor paint, shoes and jersey logos (§5) | **WASB** basketball heatmap model | **Integrated and benchmarked (§5)** |
| Two close players are confused | Bbox-bottom ground points jump during crossings; teammates are indistinguishable by color | Deep-EIoU (expanding IoU + OSNet re-ID); jersey OCR (nba_video_analysis_v2 / Basketball-Game-Tracker) | Next (§7, rank 2). Needs a ~15 MB re-ID download (approval). |
| Players near the top bench are lost or confused with bench people | Sideline people are near court edge; birth threshold and off-court rule interact | CourtVision active-player labeling | Rank 4; needs training data and approval |
| Players entering later are lost | New tracks need 3 hits; fragments are not re-linked (stitching is off: no identity cue) | Re-ID gallery (Deep-EIoU OSNet) enables offline stitching | Rank 2–3 |

## 4. Module reuse recommendation

| Module | Recommendation | Reason |
|---|---|---|
| Ball candidate detector | **Use WASB (done)** behind `TrackConfig.ball_detector`, RF-DETR kept as an option | Large precision gain in the spot check, cheap (5.8 MB, ~25–74 ms/frame) |
| Ball tracker | Keep ours (Viterbi + static-clutter filter + ≤ 8-frame interpolation + possession proxy) | Same design as nba_video_analysis_v2. Already emits detected / interpolated / missing states without inventing 3D. |
| Player tracker association | Re-implement Deep-EIoU's expanding-IoU cost inside our tracker (no code copy needed) | Small change; directly targets fast crossings |
| Appearance re-ID | Evaluate OSNet sports checkpoint via ONNX, used offline as a **track-level gallery** for stitching (mean of high-quality crops per tracklet), never as the only cue | Teammates have the same jersey color; re-ID can separate them. Must not merge aggressively: a link requires appearance + motion feasibility + no temporal overlap. |
| Jersey number OCR | Optional evidence only, later | PARSeq / legibility weights are CC BY-NC; numbers are legible in few frames at 720p |
| Active-player classifier | Adopt CourtVision's labeling protocol when training is approved; until then keep `role=unknown` explicit | No published weights; AGPL stack |
| Court keypoints | Keep our semi-automatic PTZ / homography calibration | Validated path. External court models use approximate court geometry. |

## 5. Experiment: ball detector comparison (reproducible)

Command (writes only to `outputs/experiments/ball_detector_comparison/`):
```
python tools/ball_detector_comparison.py --clips clip1 clip2 clip3 clip4 clip5 --provider dml
```

**Variants.** All five use the same tracker (`track_ball`: static-clutter filter, Viterbi, ≤ 8-frame interpolation, possession proxy), the same calibration and the same player tables. Only the ball candidates differ.
- `rfdetr`: current baseline. COCO `sports ball` from cached RF-DETR Small detections, score ≥ 0.1.
- `wasb_s1`: WASB with sliding 3-frame windows (each frame averaged over up to 3 predictions); heatmap threshold 0.3.
- `wasb_s1_p50`: as `wasb_s1`, keeping only blobs with peak ≥ 0.5.
- `wasb_s3`: WASB with non-overlapping windows (3× cheaper).
- `fused`: union of RF-DETR and WASB candidates.

### 5.1 Proxy metrics (measured, no GT)
Share of frames. `detected` = Viterbi chose a candidate; `missing` = no row.

| Clip | Variant | detected | interpolated | missing | held proxy | longest missing (s) | static unheld frames* |
|---|---|---|---|---|---|---|---|
| 1 (60 fps) | rfdetr | 0.76 | 0.09 | 0.15 | 0.36 | 0.52 | 0 |
| | wasb_s1 | 0.53 | 0.05 | 0.43 | 0.32 | 1.17 | 0 |
| 2 (30 fps) | rfdetr | 0.55 | 0.10 | 0.35 | 0.47 | 1.90 | 0 |
| | wasb_s1 | **0.80** | 0.06 | **0.14** | 0.64 | **0.63** | 0 |
| 3 (60 fps, wide) | rfdetr | 0.60 | 0.09 | 0.31 | 0.12 | 0.62 | **115** |
| | wasb_s1 | 0.13 | 0.01 | 0.86 | 0.05 | 9.70 | 0 |
| 4 (60 fps) | rfdetr | 0.56 | 0.12 | 0.32 | 0.29 | 1.80 | 0 |
| | wasb_s1 | **0.68** | 0.07 | **0.24** | 0.49 | **1.20** | 0 |
| 5 (60 fps) | rfdetr | 0.42 | 0.08 | 0.51 | 0.30 | 1.50 | 0 |
| | wasb_s1 | **0.52** | 0.06 | **0.43** | 0.40 | **1.03** | 0 |

\* Unheld selected detections whose floor projection coincides (< 0.3 m) with ≥ 0.5 s of other unheld selections anywhere in the clip. This catches a single static distractor but **misses** floor-paint false positives that hop between several dashes (clip 1). It is a weak proxy.

Full table, including `wasb_s3`, `wasb_s1_p50`, `fused` and pairwise agreement: `outputs/experiments/ball_detector_comparison/summary.csv`.
- Where both rfdetr and wasb_s1 detect, 69–100% of frames agree within 15 px (median 2.7–3.2 px).
- `wasb_s3` is within a few points of `wasb_s1` at 1/3 of the cost.
- `fused` has the highest coverage, but it inherits RF-DETR's false positives (see 5.2).

**Coverage is not accuracy.** The higher RF-DETR coverage on clips 1 and 3 turned out to be mostly false positives.

### 5.2 Random-frame visual spot check (assistant-judged, NOT ground truth)
- 15 frames per clip were sampled uniformly (seed 0) among frames where either variant has a detected ball.
- Both variants were judged on 100×100 px zoomed crops. Sheets: `<clip>/audit_random_rfdetr_vs_wasb_s1.jpg`. Per-frame labels: `visual_audit_assistant.csv`.

| Clip | RF-DETR: on ball / false positive | WASB: on ball / false positive | WASB has no detection |
|---|---|---|---|
| 1 | 10 / 5 | 11 / 0 | 4 |
| 2 | 8 / 4 | 15 / 0 | 0 |
| 3 | 4 / 7 (3 unsure) | 4 / 0 | 11 |
| 4 | 7 / 3 (2 unsure) | 12 / 0 (1 unsure) | 2 |
| 5 | 7 / 6 | 8 / 1 | 6 |
| **All** | **36 / 25 → precision ≈ 0.59** | **50 / 1 → precision ≈ 0.98** | 23 |

- **RF-DETR false-positive types:** orange or black floor paint dashes inside the paint, neon green and pink shoes, a jersey logo (Portland chest logo), feet in the bench area, empty floor.
- **WASB weakness:** recall on the wide camera of clip 3. The ball is ~6–8 px there, ~2.5 px after downscaling to 512×288. Of the 75 sampled frames, WASB found the true ball in 50 and RF-DETR in 36.
- **Limitations:**
  - The sample is small: 75 frames, ±~0.1 on precision.
  - The judge is a model.
  - The sample is conditioned on frames where at least one variant detected something, so frames that both variants miss are not counted.
  - Validated recall and precision need the reviewed CVAT ball points (§6).

### 5.3 Speed (measured, this laptop: AMD 780M iGPU via DirectML, otherwise CPU)

| | ms/frame |
|---|---|
| WASB step 1, DirectML | 62–74 |
| WASB step 3, DirectML | 25–27 |
| WASB step 3, CPU (90 frames, clip 2) | 67.5 |
| WASB step 1, CPU (90 frames, clip 2) | 189 |
| RF-DETR Small, DirectML, for reference (persons + ball, already paid by stage A) | ~318 |

ONNX vs original torch model: max abs heatmap difference 1.4e-7.

### 5.4 Decision
`ball_detector="wasb"` (step 1) is the new default. RF-DETR candidates remain available (`retrack --ball-detector rfdetr`). Fusion is **not** enabled: it adds RF-DETR's false positives back.

For the user's complaint ("the ball location is unknown"), this trades a little coverage on clips 1 and 3 for far fewer wrong ball positions:
- clips 2, 4, 5: more coverage, and the true ball;
- clips 1, 3: less coverage, but the extra RF-DETR "coverage" was mostly wrong.

**Known gap:** wide shots (clip 3). Candidate fixes are tiled or higher-resolution inference, or a lower threshold restricted to motion-consistent tracks. These must be evaluated on reviewed GT, not tuned on proxies.

## 6. Benchmark plan tied to the annotated videos
The CVAT packages for clips 2 and 5 (`outputs/annotation/clip{2,5}/`) are the evaluation set. Nothing below counts until the annotations are reviewed (`review=reviewed`; the eval tool enforces this).

1. **Ball**
   - Add a `ball` point label to the CVAT task: one point per frame where the ball is visible, every frame for 2 × ~10 s. Also mark `ball_not_visible` frames.
   - Metrics: precision, recall and F1 at a 10 px radius (about one ball diameter at 720p), and the false-positive rate on `ball_not_visible` frames. Report them separately for detected and interpolated rows.
   - Compare rfdetr / wasb_s1 / wasb_s3 / fused with `tools/ball_detector_comparison.py` (same command, plus `--gt`; to be added once GT exists).
2. **Players**
   - HOTA, IDF1 and ID switches (TrackEval) on the reviewed boxes, for: current tracker; plus expanding-IoU cost; plus OSNet appearance in association; plus offline gallery stitching.
   - Separately report swaps at close crossings: GT pairs < 1.5 m apart.
3. **Active-player role**
   - Precision and recall of `role=player` vs GT `active_player` on frames with bench people in view.
   - Count of unknown roles, kept explicit.
4. **Regression guard**
   - All five clips' proxies (`tracker_ablation.py`, `ball_detector_comparison.py`) are re-run for every change. A change that improves one clip and breaks another is reported, not hidden.

## 7. Ranked implementation plan

| Rank | Component | Expected effect | Needs | Status |
|---|---|---|---|---|
| 1 | WASB ball candidates + existing Viterbi | Ball found far more reliably, with far fewer false balls | — | **Done** (default; benchmarked §5) |
| 2 | OSNet sports re-ID (Deep-EIoU checkpoint) → ONNX, track-level gallery, used for (a) association cost at crossings and (b) conservative offline stitching | Fewer swaps at close crossings; re-linking players who leave and re-enter | **Approval for ~15–30 MB download** (weights license not stated: research use) | Proposed |
| 3 | Expanding-IoU association term (Deep-EIoU idea, re-implemented) | Fewer lost tracks on fast cuts | — | Proposed |
| 4 | Active-player detector following CourtVision's labeling protocol (fine-tune RF-DETR, non-AGPL) | Bench and sideline people no longer confused with players | Labeled data (CVAT) + GPU training (**approval**) | Later |
| 5 | Jersey-number OCR as optional evidence | Identity confirmation when legible | NC-licensed weights (research only) | Later |
| 6 | Ball on wide shots: tiled WASB inference | Recall on clip-3-type cameras | Evaluation on GT first | Later |

## 8. Files
- `src/nbatrack/detect/wasb_ball.py`: ONNX detector, streaming video runner, candidate → detection-row adapter.
- `tools/export_wasb_onnx.py`: one-off export in a torch-only process. Torch in the same process as OpenCV/PyAV crashed with an access violation on this machine.
- `tools/ball_detector_comparison.py`: the experiment above.
- `src/nbatrack/pipeline/video_pipeline.py`: `TrackConfig.ball_detector` / `ball_provider`. WASB candidates are cached in `cache/wasb_ball_candidates.parquet`.
- `tests/unit/test_wasb_ball.py`: post-processing and window bookkeeping, plus an ONNX shape test.
- `docs/LICENSES.md`: WASB code (MIT), weights and dataset (license not stated, research use), reviewed repositories.
