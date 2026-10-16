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

## After the reboot (done 2026-10-16 14:00–14:40)

| Check | Result |
|---|---|
| WSL2 | `docker-desktop` distribution running, version 2 |
| Docker | 29.8.2; VM 16 GB / 16 CPUs; disk images in `D:\DockerData` |
| CVAT | v2.79.0 cloned to `D:\external\cvat`; `docker compose up -d` started 18 containers, restart policy `always` |
| Endpoints | API `/api/server/about` 200, UI http://localhost:8080 200 |
| Admin user | `admin`; password in `D:\external\cvat_admin.txt` (outside git) |
| Task 1 `clip6_A_identity` | Created via `tools/cvat_setup_task.py`: 600 frames, tracks player 17 / referee 5 / other_person 1 / ball 1 = package |
| Round trip | CVAT export ("CVAT for video 1.1") parsed by `nbatrack.annot.cvat.parse_cvat_xml`: 6501 boxes identical to the package (< 0.05 px), 520 ball frames, 24 tracks unreviewed → `fully_reviewed = False` |
| Frame alignment | CVAT frames 0 / 300 / 599 match our decoded frames 0 / 300 / 599 (mean abs diff 0.8–1.3; neighbouring frames ≥ 10.7) |
| Bug found and fixed | The first interval packages had team A/B swapped: lightness alignment ran after the frame shift. Fixed in `tools/make_cvat_package.py` (manual-exemplar runs keep their labels); packages rebuilt and task 1 annotations re-imported. Overlay re-checked on 3 frames |
| Memory | CVAT containers about 3.2 GB; WSL VM (`vmmemWSL`) about 7.6 GB |
| Not verifiable by me | The browser UI rendering itself: the user confirms it visually |

User guide: `docs/CVAT_USER_GUIDE.md`.

## Original plan for after the reboot
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
