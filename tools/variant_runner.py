"""Isolated stage-B variants of an existing run (one change per variant).

Copies the stage-A inputs of --base (run_analyze.json, overlay, calibration diagnostics,
cache/, stage-A tables) into <out>/<variant>/, runs track_and_export with the variant's
TrackConfig, then the issue detector and the proxy metrics. The base run is never modified.

Variants are defined in VARIANTS below; select with --variants.

  python tools/variant_runner.py --base outputs/real/clip6_long_115s_diag \
      --out outputs/experiments/clip6_diag_variants --team-init data/annotations/clip6/teams.json \
      --variants kmeans exemplars exemplars_stitch exemplars_lost3s
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.pipeline.video_pipeline import DEFAULT_TRACKER, TrackConfig, track_and_export  # noqa: E402

STAGE_A_TABLES = ("calibration", "frames", "videos", "camera_segments", "court_landmarks", "landmark_observations")


def variants(team_init: str | None) -> dict[str, TrackConfig]:
    ex = dict(team_init=team_init)
    return {
        "kmeans": TrackConfig(),  # automatic team clustering (no exemplars)
        "exemplars": TrackConfig(**ex),
        "exemplars_stitch": TrackConfig(**ex, stitch=True),
        "exemplars_lost3s": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, max_time_lost_s=3.0)),
        "exemplars_noint": TrackConfig(**ex, interpolate_max_gap_s=None),
        "exemplars_occsigma": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, court_min_sigma_occluded_m=2.0)),
        "exemplars_birth05": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, new_track_thresh=0.5)),
        "exemplars_feet06_trk": TrackConfig(**ex, min_height_ratio=0.6, height_ratio_gates_tracking=True),
        "exemplars_feet06": TrackConfig(**ex, min_height_ratio=0.6),  # output only
        "exemplars_trunc06": TrackConfig(**ex, min_track_height_ratio=0.6),
        "exemplars_feet06_trunc06": TrackConfig(**ex, min_height_ratio=0.6, min_track_height_ratio=0.6),
        "exemplars_trunc06_lost3s": TrackConfig(**ex, min_track_height_ratio=0.6,
                                                tracker=replace(DEFAULT_TRACKER, max_time_lost_s=3.0)),
        # current defaults (height rules on) = baseline for the height-consistency gate experiment
        "current": TrackConfig(**ex),
        "hgate035": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, height_ratio_gate=0.35)),
        "hgate035_unocc": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, height_ratio_gate=0.35,
                                                            height_gate_max_occ=0.1)),
        "hgate025": TrackConfig(**ex, tracker=replace(DEFAULT_TRACKER, height_ratio_gate=0.25)),
    }


def prepare(base: Path, dst: Path) -> None:
    (dst / "tables").mkdir(parents=True, exist_ok=True)
    (dst / "cache").mkdir(exist_ok=True)
    for f in ("run_analyze.json", "overlay_mask.png", "calibration_diagnostics.csv"):
        if (base / f).exists():
            shutil.copy(base / f, dst / f)
    for t in STAGE_A_TABLES:
        if (base / "tables" / f"{t}.parquet").exists():
            shutil.copy(base / "tables" / f"{t}.parquet", dst / "tables" / f"{t}.parquet")
    for f in (base / "cache").iterdir():
        if not (dst / "cache" / f.name).exists():
            shutil.copy(f, dst / "cache" / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--team-init")
    ap.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    V = variants(a.team_init)
    summary = {}
    for name in a.variants:
        dst = out / name
        prepare(base, dst)
        res = track_and_export(dst, V[name])
        subprocess.run([sys.executable, str(ROOT / "tools" / "tracking_issues.py"), str(dst), "--clips", "0"],
                       check=True, capture_output=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "long_video_metrics.py"), str(dst),
                        "--out", str(dst / "metrics.json")], check=True, capture_output=True)
        m = json.loads((dst / "metrics.json").read_text())[name]
        iss = json.loads((dst / "tracking_issues_summary.json").read_text())
        segs = [s for s in m["segments"] if s["supported"]]
        summary[name] = dict(stitch_links=len(res.get("stitch_links", [])), n_tracks=res["n_tracks"], roles=res["roles"],
                             issues=iss["by_category"],
                             per_segment=[{k: s.get(k) for k in ("camera_segment_id", "tracks_by_role", "player_tracks_per_minute",
                                                                 "player_track_duration_s_p50_p90_max", "players_per_frame",
                                                                 "referees_per_frame_p50_max", "rows_speed_violation_per_1000",
                                                                 "tracks_with_suspected_id_switch", "interpolated_player_rows")}
                                          for s in segs])
        print(name, json.dumps(summary[name], default=float), flush=True)
    (out / "variant_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
