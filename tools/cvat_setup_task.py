"""Create a CVAT task from one of our annotation packages via the CVAT REST API (v2.x), and
verify it: labels, video, pre-annotations (imported as "CVAT 1.1"), frame count and tracks.

The package (tools/make_cvat_package.py) provides 01_labels_for_cvat.json, the video
(00_video_<clip>.mp4 for interval packages) and 02_preannotations_CVAT_video_1.1.zip. All
pre-annotated tracks are review=unreviewed AI predictions to be corrected.

Credentials: --user / --password, or env CVAT_USER / CVAT_PASSWORD, or a two-line file
(--cred-file: user on line 1, password on line 2; keep it out of git).

  python tools/cvat_setup_task.py --package outputs/annotation/clip6_A_identity --cred-file <path outside the repo>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


def wait_request(s: requests.Session, base: str, rq_id: str, what: str, timeout_s: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = s.get(f"{base}/api/requests/{rq_id}")
        r.raise_for_status()
        j = r.json()
        st = j.get("status")
        if st == "finished":
            return j
        if st == "failed":
            raise RuntimeError(f"{what} failed: {j.get('message')}")
        time.sleep(2)
    raise TimeoutError(f"{what} did not finish in {timeout_s} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--user", default=os.environ.get("CVAT_USER"))
    ap.add_argument("--password", default=os.environ.get("CVAT_PASSWORD"))
    ap.add_argument("--cred-file")
    ap.add_argument("--name", help="task name (default: package folder name)")
    ap.add_argument("--reimport", action="store_true",
                    help="existing task: REPLACE its annotations with the package pre-annotations "
                         "(only before any manual work has been saved; it overwrites corrections)")
    a = ap.parse_args()
    if a.cred_file:
        lines = Path(a.cred_file).read_text(encoding="utf-8").splitlines()
        a.user, a.password = lines[0].strip(), lines[1].strip()
    if not a.user or not a.password:
        sys.exit("credentials missing")
    pkg = Path(a.package)
    name = a.name or pkg.name
    labels = json.loads((pkg / "01_labels_for_cvat.json").read_text(encoding="utf-8"))
    videos = sorted(pkg.glob("00_video_*.mp4"))
    if not videos:
        sys.exit("package has no 00_video_*.mp4 (full-video packages: pass the original video instead)")
    video = videos[0]
    pre = pkg / "02_preannotations_CVAT_video_1.1.zip"
    interval = json.loads((pkg / "interval.json").read_text()) if (pkg / "interval.json").exists() else {}

    s = requests.Session()
    r = s.post(f"{a.url}/api/auth/login", json={"username": a.user, "password": a.password})
    r.raise_for_status()
    s.headers["Authorization"] = f"Token {r.json()['key']}"
    # reuse a task of the same name instead of creating duplicates
    r = s.get(f"{a.url}/api/tasks", params={"name": name})
    r.raise_for_status()
    same = [t for t in r.json()["results"] if t["name"] == name]
    if same:
        tid = same[0]["id"]
        if a.reimport:
            with open(pre, "rb") as fh:
                r = s.post(f"{a.url}/api/tasks/{tid}/annotations", params={"format": "CVAT 1.1"},
                           files={"annotation_file": (pre.name, fh, "application/zip")})
            r.raise_for_status()
            wait_request(s, a.url, r.json()["rq_id"], "annotation re-import")
            print(f"task '{name}' (id {tid}): annotations REPLACED by {pre.name}")
        else:
            print(f"task '{name}' already exists: id {tid} (not modified)")
    else:
        r = s.post(f"{a.url}/api/tasks", json={"name": name, "labels": labels})
        r.raise_for_status()
        tid = r.json()["id"]
        print("created task", tid)
        with open(video, "rb") as fh:
            r = s.post(f"{a.url}/api/tasks/{tid}/data",
                       data={"image_quality": 95, "use_zip_chunks": "true", "use_cache": "true",
                             "sorting_method": "predefined"},
                       files={"client_files[0]": (video.name, fh, "video/mp4")})
        r.raise_for_status()
        wait_request(s, a.url, r.json()["rq_id"], "video upload")
        print("video uploaded:", video.name)
        with open(pre, "rb") as fh:
            r = s.post(f"{a.url}/api/tasks/{tid}/annotations", params={"format": "CVAT 1.1"},
                       files={"annotation_file": (pre.name, fh, "application/zip")})
        r.raise_for_status()
        wait_request(s, a.url, r.json()["rq_id"], "annotation import")
        print("pre-annotations imported:", pre.name)

    # ---- verify
    meta = s.get(f"{a.url}/api/tasks/{tid}/data/meta").json()
    n_frames = meta.get("size")
    ann = s.get(f"{a.url}/api/tasks/{tid}/annotations").json()
    tracks = ann.get("tracks", [])
    lab = {l["id"]: l["name"] for l in s.get(f"{a.url}/api/labels", params={"task_id": tid, "page_size": 100}).json()["results"]}
    by_label: dict = {}
    for t in tracks:
        by_label[lab.get(t["label_id"], t["label_id"])] = by_label.get(lab.get(t["label_id"], t["label_id"]), 0) + 1
    with zipfile.ZipFile(pre) as z:
        root = ET.fromstring(z.read("annotations.xml"))
    exp = {}
    for t in root.iter("track"):
        exp[t.attrib["label"]] = exp.get(t.attrib["label"], 0) + 1
    exp_frames = interval.get("n_frames", int(root.find("./meta/task/size").text))
    ok = n_frames == exp_frames and by_label == exp
    print(json.dumps(dict(task_id=tid, url=f"{a.url}/tasks/{tid}", frames=n_frames, expected_frames=exp_frames,
                          tracks_by_label=by_label, expected_tracks_by_label=exp, verified=ok), indent=1))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
