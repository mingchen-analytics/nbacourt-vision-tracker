"""Evaluation of the Video Indexing Engine on one indexed video.

Measured (no ground truth needed):
  * scoreboard readability per field, game-clock tick consistency, video<->game-clock slope
  * score monotonicity (scores never go down)
  * seek accuracy: for event start times + random times, a keyframe seek (PyAV) must return the
    same frame (pixel-identical) as sequential decoding at that timestamp
  * boundaries near camera cuts / flash frames / inside non-main views (false-boundary proxy)
  * runtime and memory (from extract/index reports)
Reference checks:
  * --visual-ref CSV (frame_index, away, home, clock, shot) read by a person / the assistant from
    the reference sheet written by --make-ref-sheet: reading accuracy on those frames
  * boundary precision / recall vs events_reviewed.json, ONLY over events with
    review_status confirmed/edited (otherwise reported as "not measured")

  python tools/index_eval.py --out outputs/index/clip6 [--make-ref-sheet 30] [--visual-ref ref.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.index.schema import read_events  # noqa: E402
from nbatrack.io.video import iter_frames  # noqa: E402


def seek_frame(video: str, t: float) -> tuple[float, np.ndarray]:
    """Keyframe seek + decode forward to the first frame with timestamp >= t (what a player does)."""
    with av.open(video) as c:
        s = c.streams.video[0]
        tb = s.time_base
        t0 = float(s.start_time * tb) if s.start_time is not None else 0.0
        c.seek(int((t + t0) / tb), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            ts = float(Fraction(fr.pts) * tb) - t0
            if ts >= t - 1e-6:
                return ts, fr.to_ndarray(format="bgr24")
    return float("nan"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random-seeks", type=int, default=20)
    ap.add_argument("--make-ref-sheet", type=int, default=0, help="write a sheet of N random scoreboard crops to read by eye")
    ap.add_argument("--visual-ref", help="CSV frame_index,away,home,clock,shot read from the reference sheet")
    ap.add_argument("--scoreboard", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    header, events = read_events(out / "events_auto.json")
    video = header["video_path"] if Path(header["video_path"]).is_absolute() else str(ROOT / header["video_path"])
    rd = pd.read_parquet(out / "readings.parquet")
    rep = json.loads((out / "index_report.json").read_text())
    res: dict = {"note": "measured on the system's own output unless stated; no reviewed ground truth unless stated"}

    # ---- readings
    vis = rd.scoreboard_visible
    res["readability"] = {f: round(float(rd.loc[vis, f].notna().mean()), 4) for f in
                          ("away_score", "home_score", "period", "game_clock_s", "shot_clock")}
    res["game_clock"] = rep["clock_checks"]
    cm = pd.read_parquet(out / "game_clock_map.parquet")
    if len(cm) > 2:
        sl = np.polyfit(cm.video_t0.to_numpy(), cm.game_clock_s.to_numpy(), 1)[0]
        res["game_clock"]["fit_slope_game_s_per_video_s"] = round(float(sl), 4)
    mono = {}
    for f in ("away_score", "home_score"):
        x = rd[f].dropna().to_numpy()
        # count sustained decreases (3 consecutive readings below the running max)
        runmax = np.maximum.accumulate(x) if len(x) else x
        mono[f] = int(((x < runmax).astype(int)).sum())
    res["score_readings_below_running_max"] = mono

    # ---- seek accuracy
    rng = np.random.default_rng(0)
    starts = sorted({round(e["video_start_seconds"], 3) for e in events})
    rand = sorted(rng.uniform(0, header["duration_s"] - 0.5, a.n_random_seeks).round(3).tolist())
    targets = sorted(set(starts) | set(rand))
    seq = {}
    want = set()
    st = pd.read_parquet(out / "frame_stats.parquet")
    tsv = st.timestamp_sec.to_numpy()
    for t in targets:
        k = int(np.searchsorted(tsv, t - 1e-6))
        if k < len(tsv):
            want.add(int(st.frame_index.iloc[k]))
    for fr in iter_frames(video):
        if fr.frame_index in want:
            seq[round(fr.timestamp_sec, 6)] = fr.image
        if fr.frame_index > max(want):
            break
    ok, rows = 0, []
    for t in targets:
        ts, img = seek_frame(video, t)
        ref = seq.get(round(ts, 6))
        same = ref is not None and img is not None and np.array_equal(ref, img)
        ok += same
        rows.append(dict(target=t, got_ts=round(ts, 4), identical=bool(same)))
    res["seek"] = dict(n=len(targets), pixel_identical=ok, rate=round(ok / max(1, len(targets)), 4),
                       method="PyAV keyframe seek + forward decode vs sequential decode; browser seeking not measured")

    # ---- boundaries near cuts / flashes / non-main views (false-boundary proxy)
    fps = header["fps"]
    flashes = np.array(header.get("flash_frames", [])) / fps
    cuts = np.array([e["video_start_seconds"] for e in events if e["event_type"] == "camera_segment"][1:])
    nonmain = [(e["video_start_seconds"], e["video_end_seconds"]) for e in events
               if e["event_type"] == "camera_segment" and "main_broadcast" not in e["labels"]]
    bt = [b["t"] for b in rep["boundaries"]]
    near = lambda arr, t, w: bool(len(arr)) and bool((np.abs(arr - t) <= w).any())
    res["boundaries_proxy"] = dict(
        n=len(bt), within_1s_of_flash=sum(near(flashes, t, 1.0) for t in bt),
        within_1s_of_cut=sum(near(cuts, t, 1.0) for t in bt),
        inside_non_main_view=sum(any(a0 <= t <= a1 for a0, a1 in nonmain) for t in bt),
        cuts=len(cuts), flashes=len(flashes),
        note="boundaries come only from scoreboard evidence; this counts coincidences, it is not an error rate")

    # ---- reviewed ground truth (if any)
    rv = out / "events_reviewed.json"
    if rv.exists():
        _, rev = read_events(rv)
        gt = [e for e in rev if e["event_type"] in ("candidate_possession", "possession")
              and e["review_status"] in ("confirmed", "edited")]
        if gt:
            gtb = sorted({round(e["video_start_seconds"], 3) for e in gt} | {round(e["video_end_seconds"], 3) for e in gt})
            tp = sum(any(abs(t - g) <= 1.0 for g in gtb) for t in bt)
            rec = sum(any(abs(t - g) <= 1.0 for t in bt) for g in gtb)
            res["boundary_vs_reviewed"] = dict(n_pred=len(bt), n_gt=len(gtb), precision=tp / max(1, len(bt)),
                                               recall=rec / max(1, len(gtb)), tolerance_s=1.0,
                                               note="ground truth = reviewed (confirmed/edited) possessions only")
        else:
            res["boundary_vs_reviewed"] = "not measured (no confirmed/edited possessions yet)"
    else:
        res["boundary_vs_reviewed"] = "not measured (no events_reviewed.json yet)"

    # ---- visual reference
    if a.make_ref_sheet:
        sc = json.loads(Path(a.scoreboard or header["provenance"]["scoreboard"]).read_text())
        x1 = min(v[0] for v in sc["fields"].values()) - 4
        x2 = max(v[2] for v in sc["fields"].values()) + 4
        y1 = min(v[1] for v in sc["fields"].values()) - 4
        y2 = max(v[3] for v in sc["fields"].values()) + 4
        pick = sorted(rng.choice(rd.frame_index.to_numpy(), a.make_ref_sheet, replace=False).tolist())
        tiles, want = [], set(pick)
        for fr in iter_frames(video):
            if fr.frame_index in want:
                c = cv2.resize(fr.image[y1:y2, x1:x2], None, fx=1.5, fy=1.5)
                lab = np.zeros((c.shape[0], 90, 3), np.uint8)
                cv2.putText(lab, f"f{fr.frame_index}", (4, c.shape[0] // 2 + 5), 0, 0.5, (255, 255, 255), 1)
                tiles.append(np.hstack([lab, c]))
            if fr.frame_index > max(want):
                break
        cv2.imwrite(str(out / "visual_ref_sheet.png"), np.vstack(tiles))
        pd.DataFrame(dict(frame_index=pick)).to_csv(out / "visual_ref_frames.csv", index=False)
        res["visual_ref_sheet"] = str(out / "visual_ref_sheet.png")
    if a.visual_ref:
        ref = pd.read_csv(a.visual_ref, comment="#")
        m = ref.merge(rd, on="frame_index", how="left")
        acc = {}
        for col, f in (("away", "away_score"), ("home", "home_score"), ("clock", "game_clock_s"), ("shot", "shot_clock")):
            truth = m[col].astype(float)
            pred = m[f].astype(float)
            acc[f] = dict(n=int(truth.notna().sum()), correct=int(((pred == truth) & truth.notna()).sum()),
                          unreadable=int((pred.isna() & truth.notna()).sum()),
                          wrong=int(((pred != truth) & pred.notna() & truth.notna()).sum()))
        res["visual_reference_check"] = dict(fields=acc, source=a.visual_ref,
                                             note="reference values read by eye from visual_ref_sheet.png; not independent GT if read by the assistant")
    res["runtime"] = dict(extract_seconds=rep["extract"]["seconds"], extract_ms_per_frame=rep["extract"]["ms_per_frame"],
                          extract_peak_private_mb=rep["extract"]["peak_private_mb"], index_seconds=rep["index_seconds"],
                          video_seconds=header["duration_s"])
    (out / "eval_report.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
