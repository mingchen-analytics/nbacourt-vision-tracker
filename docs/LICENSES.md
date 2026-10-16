# Third-party license register

Rule: the main implementation stays independent of AGPL components. Every dependency and every model weight is recorded here **before** it is integrated.

## Integrated (installed in `D:\conda\envs\nbatrack`)
| Component | Version | License | Use |
|---|---|---|---|
| NumPy | 2.4.6 | BSD-3 | arrays |
| SciPy | 1.17.1 | BSD-3 | least-squares optimization |
| OpenCV (opencv-python) | 5.0.0.93 | Apache-2.0 (OpenCV ≥ 4.5) | geometry, RANSAC, KLT, drawing |
| PyAV | 18.1.0 | BSD-3 (bundled FFmpeg: LGPL/GPL build, see note) | decoding with exact PTS, H.264 encoding |
| pyarrow | 26.0.0 | Apache-2.0 | Parquet/CSV |
| pandas | 3.0.6 | BSD-3 | analysis |
| PyYAML | 6.0.3 | MIT | config |
| pytest / pytest-cov | 9.1.1 / 7.1.0 | MIT | tests |
| psutil | 7.2.2 | BSD-3 | memory diagnostics |
| **RF-DETR** (rfdetr) | 1.11.2 | Apache-2.0 (N/S/M/Base/L; XL/2XL PML-1.0 not used) | player/ball detector; only used to download weights and export ONNX |
| RF-DETR Small COCO weights `rf-detr-small.pth` | n/a | Apache-2.0 (Roboflow states the N–L checkpoints are Apache-2.0) | stored in `models/rfdetr/` (D:), 368 MB; ONNX export 110 MB in `models/onnx/` |
| PyTorch / torchvision (CPU build) | 2.14.1 / 0.29.1 | BSD-3 | RF-DETR export only (not used at inference) |
| ONNX Runtime DirectML | 1.24.4 | MIT | inference (DirectML EP + CPU EP) |
| onnx / onnxsim | 1.23.2 / 0.8.1 | Apache-2.0 | export |
| TrackEval (git, JonathonLuiten/TrackEval) | 1.0.dev1 | MIT | reference HOTA/CLEAR/Identity metrics; used through `eval/trackeval_bridge.py` (numpy-2 aliases shimmed, package unmodified) |
| **WASB-SBDT** code (nttcom/WASB-SBDT, git clone in `D:\external\WASB-SBDT`, outside the project) | commit as cloned 2026-10-15 | MIT | Only `src/models/hrnet.py` (+ its config) is imported, by `tools/export_wasb_onnx.py`, in a torch-only process to export ONNX. Not imported at runtime. |
| WASB basketball checkpoint `wasb_basketball_best.pth.tar` (5.8 MB, `D:\external\weights\`) → `models/onnx/wasb_basketball_288x512.onnx` | n/a | **Not stated** by the authors. Trained on the NBA broadcast dataset of Yan et al. (SAM, ruiyan1995.github.io/SAM.html), whose license is also not stated | **Research / private use only** until the authors clarify. Default ball candidate source since 2026-10-15 (`TrackConfig.ball_detector="wasb"`); `--ball-detector rfdetr` avoids it. |
| gdown | 5.x | MIT | one-off download of the WASB checkpoint from the authors' Google Drive link |
| transformers, huggingface_hub, supervision, matplotlib (rfdetr dependencies) | 5.20.0, 2.2.0, 0.30.9, 3.11.2 | Apache-2.0, Apache-2.0, MIT, matplotlib (PSF-style) | pulled in by rfdetr; matplotlib also used for diagnostic plots |

Note on PyAV wheels: they bundle FFmpeg, including libx264 (GPL), which is used only to *write* diagnostic MP4s. That is fine for private research. Before any redistribution, check the bundled FFmpeg build's license, or switch the writer to a non-GPL encoder.

## Evaluated, NOT integrated
| Component | License | Status / reason |
|---|---|---|
| Ultralytics YOLO (v8/11) | AGPL-3.0 | Allowed only as a separate research baseline, in a separate environment |
| abdullahtarek/basketball_analysis (fork: chriztopherton/nba_tracking) code | MIT | Reference only |
| its `court_keypoint_detector.pt` | **No license stated.** Fine-tuned from Ultralytics `yolov8x-pose.pt` (AGPL-3.0); dataset Roboflow `fyp-3bwmg/reloc2-den7l` v1, license unknown | 18 keypoints; its geometry uses an approximate 28 × 15 m court. Not downloaded yet. If used, it must be a separate AGPL-side baseline, with its keypoints remapped to our exact landmark coordinates. |
| Roboflow `basketball-court-detection-2` dataset (33 keypoints) | Not verified (download page requires an account) | Candidate training data if training is approved later |
| RF-DETR (N–L) | Apache-2.0 (XL/2XL: PML-1.0, excluded) | **Integrated 2026-10-15** (Small), see above |
| rtmlib + RTMPose Halpe26 ONNX | rtmlib license not stated in the README; MMPose is Apache-2.0 | Planned pose-based ground-contact experiment. Heel/toe keypoints must be verified on the actual checkpoint before use. |
| boxmot (BoT-SORT etc.) | AGPL-3.0 | Not used; ideas re-implemented |
| supervision (ByteTrack) | MIT | Candidate tracker baseline |
| DeepSportRadar datasets | CC BY-NC(-ND/-SA), sources disagree | Research-only external validation candidate |
| DeepSportradar/player-reidentification-challenge (code, Synergy re-ID data) | No license file found; data terms on the challenge page | Reviewed 2026-10-15 (docs/OPEN_SOURCE_TRACKING_REVIEW.md). No pretrained weights. Candidate data for fine-tuning a re-ID model, only if training is approved. |
| hsiangwei0903/Deep-EIoU | No root license; `reid/` is torchreid (MIT) | Reviewed. Algorithm (expanding IoU + OSNet appearance) can be re-implemented; the sports OSNet checkpoint (Google Drive, ~ tens of MB) has no stated license; download needs approval. |
| cyrushadavi1/nba_video_analysis_v2 | No license file | Reviewed, reference only (ball pipeline design: WASB + Viterbi + interpolation states). Uses Ultralytics (AGPL) and PARSeq/legibility weights (CC BY-NC). |
| jigggle/CourtVision | No license file; uses Ultralytics YOLO26 (AGPL) | Reviewed, reference only (active-player labeling protocol). No weights published. |
| hasan-bakr/Basketball-Game-Tracker | MIT | Reviewed. Depends on Roboflow Inference API keys, SAM, Cutie; reference only. |
| Nolan-Lo/basketball-computer-vision | No license file; Ultralytics (AGPL) | Reviewed, reference only. |
