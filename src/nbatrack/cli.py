"""Command-line entry point: `nbatrack <command> ...`"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_probe(a) -> int:
    from nbatrack.io.video import probe
    print(json.dumps(probe(a.video, with_hash=a.hash).to_dict(), indent=2))
    return 0


def _cmd_synth(a) -> int:
    from nbatrack.pipeline.synthetic_demo import run_synthetic_demo
    m = run_synthetic_demo(a.out, n_frames=a.frames, calib=a.calib, seed=a.seed,
                           render_video=not a.no_video)
    print(json.dumps(m, indent=2))
    return 0


def _cmd_process(a) -> int:
    from nbatrack.pipeline.video_pipeline import AnalyzeConfig, analyze_video, track_and_export
    from nbatrack.viz.render import render_videos
    acfg = AnalyzeConfig(provider=a.provider, calib_mode=a.calib_mode, max_frames=a.max_frames)
    if not a.keyframe and not a.segments:
        print("--keyframe or --segments is required", file=sys.stderr)
        return 2
    run = analyze_video(a.video, a.out, a.keyframe, a.video_id, acfg, segments_json=a.segments)
    print(json.dumps({k: run[k] for k in ("n_frames", "ms_per_frame", "processing_fps",
                                          "peak_private_memory_mb")}, indent=2, default=float))
    print(json.dumps(track_and_export(a.out), indent=2, default=float)[:2000])
    if not a.no_render:
        print(render_videos(a.video, a.out, max_frames=a.max_frames))
    return 0


def _cmd_retrack(a) -> int:
    from nbatrack.pipeline.video_pipeline import TrackConfig, track_and_export
    from nbatrack.track.tracker import TrackerConfig
    from nbatrack.viz.render import render_videos
    from dataclasses import replace
    from nbatrack.pipeline.video_pipeline import DEFAULT_TRACKER
    tcfg = TrackConfig(tracker=replace(DEFAULT_TRACKER, use_court=not a.no_court,
                                       court_min_sigma_m=a.court_min_sigma),
                       team_init=a.team_init, stitch=a.stitch,
                       interpolate_max_gap_s=None if a.no_interp else 1.0,
                       ball_detector=a.ball_detector)
    res = track_and_export(a.out, tcfg)
    print(json.dumps({k: res[k] for k in ("n_tracks", "roles", "rows")}, indent=2))
    if a.render:
        run = json.loads((__import__("pathlib").Path(a.out) / "run_analyze.json").read_text())
        print(render_videos(run["source"], a.out))
    return 0


def _cmd_calibrate(a) -> int:  # pragma: no cover - interactive
    from nbatrack.calib.manual import ManualCalibrationSession, run_gui
    from nbatrack.io.video import iter_frames, probe
    info = probe(a.video)
    for fr in iter_frames(a.video):
        if fr.frame_index == a.frame:
            break
    else:
        print(f"frame {a.frame} not found", file=sys.stderr)
        return 1
    s = ManualCalibrationSession(a.video_id or info.path, fr.frame_index, fr.timestamp_sec,
                                 (info.width, info.height), annotator=a.annotator)
    ok = run_gui(fr.image, s, a.out)
    print("saved" if ok else "cancelled", a.out)
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nbatrack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="print video metadata")
    p.add_argument("video")
    p.add_argument("--hash", action="store_true")
    p.set_defaults(fn=_cmd_probe)
    p = sub.add_parser("synth-demo", help="end-to-end run on a synthetic broadcast")
    p.add_argument("--out", default="outputs/synthetic_demo")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--calib", choices=["ptz", "camera", "homography"], default="ptz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(fn=_cmd_synth)
    p = sub.add_parser("process", help="run the real-video pipeline + render diagnostics")
    p.add_argument("video")
    p.add_argument("--keyframe", help="landmark JSON of the keyframe (single-shot video)")
    p.add_argument("--segments", help="camera segments JSON (multi-shot video; one keyframe per supported segment)")
    p.add_argument("--video-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", choices=["cpu", "dml"], default="dml")
    p.add_argument("--calib-mode", choices=["ptz", "free"], default="ptz")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-render", action="store_true")
    p.set_defaults(fn=_cmd_process)
    p = sub.add_parser("retrack", help="re-run tracking/export from cached detections")
    p.add_argument("out")
    p.add_argument("--court-min-sigma", type=float, default=1.0)
    p.add_argument("--no-court", action="store_true")
    p.add_argument("--team-init", help="teams.json with manual exemplars")
    p.add_argument("--stitch", action="store_true", help="offline tracklet stitching (experimental)")
    p.add_argument("--no-interp", action="store_true", help="disable short-gap interpolation")
    p.add_argument("--ball-detector", choices=["wasb", "rfdetr"], default="wasb",
                   help="ball candidate source (wasb = dedicated basketball heatmap model)")
    p.add_argument("--render", action="store_true")
    p.set_defaults(fn=_cmd_retrack)
    p = sub.add_parser("calibrate", help="manual landmark calibration of one frame (GUI)")
    p.add_argument("video")
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--video-id")
    p.add_argument("--annotator", default="")
    p.set_defaults(fn=_cmd_calibrate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
