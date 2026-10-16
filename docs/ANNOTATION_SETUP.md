# Local CVAT annotation setup: state and steps

## Why localhost:8080 did not open (checked 2026-10-16 13:40)
CVAT was never installed. Nothing was listening on port 8080, and neither Docker nor WSL existed on the machine. It was not a configuration error.

## Done
| Step | Result |
|---|---|
| Hardware | Windows 11 Home build 26300, firmware virtualization enabled, 31 GB RAM. Free space: C: 17.7 GB, D: 315 GB |
| WSL2 platform | `wsl --install --no-distribution` (elevated, user approved UAC): exit 0. **Needs a reboot** |
| Docker Desktop 4.95.0 | Installer signature valid (Docker Inc). Installed with `--installation-dir=D:\DockerDesktop --wsl-default-data-root=D:\DockerData --backend=wsl-2`: exit 0. C: free 17.7 → 14.8 GB (system components); images and containers go to D: |
| Annotation package | `outputs/annotation/clip6_A_identity/`: 600-frame cut video, 23 person tracks plus 1 ball track, all `unreviewed` |

## After the reboot (to do)
1. Start Docker Desktop (`D:\DockerDesktop\Docker Desktop.exe`). Check `docker version` and that the data root is on D:.
2. Get CVAT: clone `cvat-ai/cvat` (MIT) at a release tag into `D:\external\cvat`, outside the project. Then run `docker compose up -d`; the images are about 6–8 GB, on D:.
3. Create the CVAT admin user (`docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`).
4. Create the task through the REST API:
   - labels from `01_labels_for_cvat.json`;
   - video `00_video_clip6_A_identity.mp4`;
   - pre-annotations `02_preannotations_CVAT_video_1.1.zip` imported as "CVAT 1.1".
5. Verify:
   - the UI at http://localhost:8080 responds;
   - the task has 600 frames;
   - the imported annotations contain 23 + 1 tracks;
   - then the user checks visually.
6. User guide (Traditional Chinese): `docs/CVAT_USER_GUIDE.md`.

## Rules
- Original videos, the git repository and the pipeline outputs are not modified.
- AI pre-annotations stay `review = unreviewed` until a human checks them. `tools/evaluate_against_gt.py` refuses unreviewed tracks.
