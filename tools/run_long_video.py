"""Run the full pipeline on a long (multi-shot) video with per-stage timing, memory and
output-integrity checks.

Stages: A analyze (RF-DETR + per-segment calibration) -> WASB ball candidates ->
appearance descriptors (only with --team-init) -> B track/export -> render -> verify.
Every stage is skipped when its output already exists (use --redo STAGE to force).

  python tools/run_long_video.py data/raw/clip6_long_115s.mp4 --video-id clip6_LAL_at_DEN_long \
      --segments data/annotations/clip6/segments.json --out outputs/real/clip6_long_115s [--max-frames 1020]

Writes <out>/run_long.json (timings, memory, integrity) and top-level aliases
tracking_data.parquet / ball_tracking.parquet (byte copies of tables/player_observations
and tables/ball_observations, same schema).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.pipeline.video_pipeline import (  # noqa: E402
    AnalyzeConfig,
    TrackConfig,
    _ball_candidates,
    analyze_video,
    track_and_export,
)


class PeakMem(threading.Thread):
    """Samples this process's private (commit) memory every 0.25 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.p = psutil.Process()
        self.peak = 0.0
        self.stage_peak: dict[str, float] = {}
        self.stage = "init"
        self.stop = False

    def run(self):
        while not self.stop:
            m = self.p.memory_info().private / 2**20
            self.peak = max(self.peak, m)
            self.stage_peak[self.stage] = max(self.stage_peak.get(self.stage, 0.0), m)
            time.sleep(0.25)


def keep_awake(on: bool) -> None:
    """Ask Windows not to idle-sleep while this process works (SetThreadExecutionState with
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED; released on exit). This is a per-process request, not a
    change of power settings; a closed lid or manual sleep still suspends the run (which then
    simply continues on resume)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def verify_video(path: Path) -> dict:
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    r = subprocess.run([ff, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    n = subprocess.run([fp, "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return dict(path=str(path), decode_error_lines=len([x for x in r.stderr.splitlines() if x.strip()]),
                packets=int(n.stdout.strip() or 0), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--team-init")
    ap.add_argument("--team-frames", type=int, nargs="*", default=[])
    ap.add_argument("--stitch", action="store_true")
    ap.add_argument("--provider", default="dml")
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    mem = PeakMem()
    mem.start()
    rec = json.loads((out / "run_long.json").read_text()) if (out / "run_long.json").exists() else {}
    rec.setdefault("stages_s", {})
    t_all = time.time()

    def stage(name):
        mem.stage = name
        return time.time()

    # A: detection + calibration
    if not (out / "run_analyze.json").exists() or "analyze" in a.redo:
        t0 = stage("analyze")
        analyze_video(a.video, out, None, a.video_id, AnalyzeConfig(provider=a.provider, max_frames=a.max_frames),
                      segments_json=a.segments)
        rec["stages_s"]["analyze"] = time.time() - t0
    run_a = json.loads((out / "run_analyze.json").read_text(encoding="utf-8"))
    # WASB ball candidates (cached; track_and_export reuses the cache)
    cache = out / "cache" / "wasb_ball_candidates.parquet"
    if "ball" in a.redo and cache.exists():
        cache.unlink()
    if not cache.exists():
        import pandas as pd
        t0 = stage("wasb_ball")
        _ball_candidates(TrackConfig(ball_provider=a.provider), out, run_a, pd.read_parquet(out / "cache" / "detections.parquet"))
        rec["stages_s"]["wasb_ball"] = time.time() - t0
    # appearance descriptors + numbered pictures for manual team exemplars
    if a.team_frames and (not (out / "cache" / "appearance.npy").exists() or "appearance" in a.redo):
        t0 = stage("appearance")
        subprocess.run([sys.executable, str(ROOT / "tools" / "team_init_helper.py"), "render", "--run", str(out),
                        "--video", a.video, "--frames", *map(str, a.team_frames)], check=True)
        rec["stages_s"]["appearance"] = time.time() - t0
    # B: tracking / export
    t0 = stage("track")
    res = track_and_export(out, TrackConfig(team_init=a.team_init, stitch=a.stitch, ball_provider=a.provider))
    rec["stages_s"]["track"] = time.time() - t0
    rec["track_summary"] = {k: res[k] for k in ("n_tracks", "roles", "rows", "interpolated_rows", "ball_detector",
                                                "ball_frames", "ball_detected_frames", "ball_possession_proxy_frames",
                                                "stitch_enabled") if k in res}
    rec["track_summary"]["stitch_links"] = len(res.get("stitch_links", []))
    for src, dst in (("player_observations", "tracking_data"), ("ball_observations", "ball_tracking")):
        shutil.copyfile(out / "tables" / f"{src}.parquet", out / f"{dst}.parquet")
    # render + verify
    if not a.no_render:
        from nbatrack.viz.render import render_videos
        t0 = stage("render")
        paths = render_videos(a.video, out, max_frames=a.max_frames)
        rec["stages_s"]["render"] = time.time() - t0
        rec["videos"] = [verify_video(Path(p)) for p in paths]
    mem.stop = True
    rec.update(video=a.video, n_frames=run_a["n_frames"], fps=run_a["fps"],
               analyze_ms_per_frame=run_a["ms_per_frame"], analyze_ms_per_frame_p95=run_a.get("ms_per_frame_p95"),
               analyze_peak_private_mb=run_a["peak_private_memory_mb"],
               peak_private_mb_this_run=mem.peak, peak_private_mb_by_stage=mem.stage_peak,
               wall_s_this_run=time.time() - t_all)
    (out / "run_long.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    keep_awake(False)
    print(json.dumps(rec, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
