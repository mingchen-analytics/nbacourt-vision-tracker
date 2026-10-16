"""Build a CVAT annotation package for one processed clip.

Output folder (outputs/annotation/<clip>/):
  01_labels_for_cvat.json          paste into CVAT label constructor ("Raw")
  02_preannotations_CVAT_video_1.1.zip   model predictions to CORRECT (all tracks unreviewed)
  03_frames_to_annotate.txt        frames for foot-contact points and court landmarks
  04_reference_frames/             JPGs of those frames with frame numbers (no predictions drawn)
  05_team_definition.txt           which real team is "A" / "B" for this clip
  README.txt                       pointer to docs/ANNOTATION_GUIDE.md

Foot-contact points and landmarks are deliberately NOT pre-filled: pre-filling them with
model output would bias the very quantities they are meant to evaluate.

Usage:
  python tools/make_cvat_package.py --clip clip2 --run outputs/real/clip2_ptz --video "data/raw/..." \
      --team-a "Portland Trail Blazers (black jerseys)" --team-b "Golden State Warriors (white jerseys)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nbatrack.annot.cvat import labels_json_text, predictions_to_cvat_xml, write_cvat_zip
from nbatrack.io.video import iter_frames


def align_team_labels(pl: pd.DataFrame, det: pd.DataFrame, dark_is_a: bool) -> pd.DataFrame:
    """Rename predicted team ids so that A = darker jerseys if `dark_is_a` (by median torso
    lightness of each predicted team), matching the clip's team definition."""
    m = pl.merge(det[["frame_index", "x1", "y1", "torso_L"]],
                 left_on=["frame_index", "bbox_x1", "bbox_y1"], right_on=["frame_index", "x1", "y1"], how="left")
    L = m.groupby("team_id").torso_L.median()
    if {"A", "B"} <= set(L.index):
        a_darker = L["A"] < L["B"]
        if a_darker != dark_is_a:
            pl = pl.copy()
            pl["team_id"] = pl.team_id.map({"A": "B", "B": "A"}).where(pl.team_id.notna(), None)
    return pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--team-a", required=True, help="description of team A (should be the darker jerseys)")
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--foot-every-s", type=float, default=1.0)
    ap.add_argument("--landmark-frames", type=int, default=4)
    ap.add_argument("--keyframe-step", type=int, default=5)
    ap.add_argument("--start-frame", type=int, help="annotate only an interval of a long video: first frame")
    ap.add_argument("--end-frame", type=int, help="last frame (inclusive)")
    ap.add_argument("--note", default="", help="why this interval was chosen (written to README)")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path("outputs/annotation") / a.clip
    (out / "04_reference_frames").mkdir(parents=True, exist_ok=True)
    ra = json.loads((run / "run_analyze.json").read_text(encoding="utf-8"))
    n, fps = ra["n_frames"], ra["fps"]
    pl = pd.read_parquet(run / "tables" / "player_observations.parquet")
    pl = pl[pl.bbox_x1.notna()]  # observed rows only (interpolated rows have no image box)
    bp = run / "tables" / "ball_observations.parquet"
    ball = pd.read_parquet(bp) if bp.exists() else None  # detected + interpolated ball points
    # Team ids: with manual exemplars (run_track.json team_method) A/B already mean the teams of
    # teams.json and are kept. Otherwise (k-means) A = darker jerseys by torso lightness. This
    # must use SOURCE frame numbers (it matches rows to detections), i.e. run before any shift:
    # matching after the interval shift swapped A/B in the first clip-6 interval packages.
    rt = run / "run_track.json"
    team_method = json.loads(rt.read_text(encoding="utf-8")).get("team_method") if rt.exists() else None
    if team_method != "manual_exemplars":
        pl = align_team_labels(pl, pd.read_parquet(run / "cache" / "detections.parquet"), dark_is_a=True)
    offset = 0
    if a.start_frame is not None:
        # interval of a long video: CVAT frame k = source frame start + k. The interval is cut
        # frame-exactly into its own video (re-encoded from decoded frames), which the annotator
        # uploads instead of the full source.
        from nbatrack.viz.side_by_side import VideoSink
        offset, end = a.start_frame, a.end_frame if a.end_frame is not None else n - 1
        pl = pl[pl.frame_index.between(offset, end)].copy()
        pl["frame_index"] -= offset
        if ball is not None:
            ball = ball[ball.frame_index.between(offset, end)].copy()
            ball["frame_index"] -= offset
        n = end - offset + 1
        cut = out / f"00_video_{a.clip}.mp4"
        prev = json.loads((out / "interval.json").read_text()) if (out / "interval.json").exists() else {}
        if not (cut.exists() and prev.get("start_frame") == offset and prev.get("end_frame") == end):
            sink = VideoSink(cut, fps)
            for fr in iter_frames(a.video):
                if fr.frame_index > end:
                    break
                if fr.frame_index >= offset:
                    sink.write(fr.image, fr.timestamp_sec)
            sink.close()
        (out / "interval.json").write_text(json.dumps(dict(source=str(a.video), run=str(run), start_frame=offset,
                                                           end_frame=end, n_frames=n, fps=fps, note=a.note), indent=1))
    # Split pre-annotated tracks where the pipeline's role/team label changes (a
    # suspected identity switch), so the annotator sees the suspected switch point.
    pl = pl.sort_values(["track_id", "frame_index"]).copy()
    key = pl.role.fillna("-").astype(str) + "|" + pl.team_id.where(pl.team_id.isin(["A", "B"]), "-").astype(str)
    run = (key != key.groupby(pl.track_id).shift()).groupby(pl.track_id).cumsum()
    pl["track_id"] = pl.track_id + "_" + run.astype(str)

    (out / "01_labels_for_cvat.json").write_text(labels_json_text(), encoding="utf-8")
    xml = predictions_to_cvat_xml(pl, n, (ra["width"], ra["height"]), a.clip, a.keyframe_step, ball=ball)
    write_cvat_zip(xml, out / "02_preannotations_CVAT_video_1.1.zip")

    step = max(1, int(round(a.foot_every_s * fps)))
    foot = list(range(step // 2, n, step))
    lmk = [int(round(x)) for x in np.linspace(0, n - 1, a.landmark_frames)]
    lines = ["# Frame numbers are CVAT frame numbers (0-based), same as our frame_index.",
             f"# Video: {Path(a.video).name}  ({n} frames, {fps:g} fps)",
             "",
             "FOOT-CONTACT frames (label foot_contact; one point per visible player / referee):",
             ", ".join(map(str, foot)),
             "",
             "COURT-LANDMARK frames (label court_landmark; as many clearly visible landmarks as you can, ideally >= 5):",
             ", ".join(map(str, lmk))]
    (out / "03_frames_to_annotate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    want = set(foot) | set(lmk)
    for fr in iter_frames(a.video):
        k = fr.frame_index - offset  # CVAT frame number
        if k > n - 1:
            break
        if k in want:
            img = fr.image.copy()
            tag = ("FOOT " if k in foot else "") + ("LANDMARK" if k in lmk else "")
            txt = f"frame {k}  t={fr.timestamp_sec:.3f}s  {tag}" + (f"  (source frame {fr.frame_index})" if offset else "")
            cv2.putText(img, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(img, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(out / "04_reference_frames" / f"frame_{k:04d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "05_team_definition.txt").write_text(
        f"team A = {a.team_a}\nteam B = {a.team_b}\n"
        "Referees use the label `referee` (no team). Bench players, coaches, staff, ball kids and\n"
        "spectators that stand on the floor / apron use `other_person` (optional, only if they\n"
        "are fully visible on the floor area).\n", encoding="utf-8")
    upload = f"00_video_{a.clip}.mp4 (this folder; frames {offset}-{offset + n - 1} of {Path(a.video).name})" \
        if offset or a.start_frame is not None else Path(a.video).name
    (out / "README.txt").write_text(
        "Follow docs/ANNOTATION_GUIDE.md (step-by-step). Upload this video file:\n"
        f"  {upload}\n"
        + (f"Why this interval: {a.note}\n" if a.note else "")
        + "Pre-annotations are UNREVIEWED model output: correct them; they are not ground truth.\n"
        "When finished, export 'CVAT for video 1.1' and save the zip as\n"
        f"  data/annotations/{a.clip}/gt_cvat_export.zip\n", encoding="utf-8")
    print(f"package written to {out} ({len(pl.track_id.unique())} pre-annotated tracks, "
          f"{len(foot)} foot frames, {len(lmk)} landmark frames)")


if __name__ == "__main__":
    main()
