"""Verify decoded timestamps against ffprobe on synthetic CFR and VFR files."""

import json
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from nbatrack.io.video import iter_frames, probe
from nbatrack.synthetic.video import numbered_frames, write_video

FFPROBE = shutil.which("ffprobe")


def _ffprobe_pts(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=pts,pts_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return [int(f["pts"]) for f in json.loads(out.stdout)["frames"]]


@pytest.mark.parametrize("vfr", [False, True])
def test_timestamps_match_written_and_ffprobe(tmp_path, vfr):
    n = 45
    if vfr:  # irregular frame spacing (e.g. dropped / duplicated frames in a broadcast)
        rng = np.random.default_rng(0)
        dt = rng.choice([1 / 30, 1 / 60, 2 / 30], size=n)
        t = np.concatenate([[0.0], np.cumsum(dt[:-1])]) + 0.5  # non-zero stream start
    else:
        t = np.arange(n) / 30.0
    imgs = numbered_frames(n)
    path = tmp_path / ("vfr.mp4" if vfr else "cfr.mp4")
    pts_written = write_video(path, imgs, list(t))
    frames = list(iter_frames(path))
    assert [f.frame_index for f in frames] == list(range(n))
    got_pts = [f.pts for f in frames]
    # The muxer may rescale the time base; compare in seconds.
    info = probe(path)
    tb = Fraction(info.time_base)
    got_sec = np.array([float(p * tb) for p in got_pts])
    shift = got_sec[0] - t[0]  # container may add a constant offset (edit list)
    assert np.allclose(got_sec - shift, t, atol=1e-4)
    assert frames[0].timestamp_sec == pytest.approx(0.0, abs=1e-9)
    assert np.allclose([f.timestamp_sec for f in frames], t - t[0], atol=1e-4)
    # brightness encodes the index -> frames are in the right order (yuv420p range
    # conversion shifts gray levels by a few units; levels are >= 30 apart)
    assert [int(round(f.image.mean())) for f in frames] == pytest.approx(
        [(i * 37) % 200 + 30 for i in range(n)], abs=8)
    if FFPROBE:
        assert got_pts == _ffprobe_pts(path)
    assert len(pts_written) == n


def test_stride_and_window_keep_global_indices(tmp_path):
    t = np.arange(30) / 30.0
    path = tmp_path / "c.mp4"
    write_video(path, numbered_frames(30), list(t))
    sel = list(iter_frames(path, start_sec=0.3, end_sec=0.7, stride=2))
    assert [f.frame_index for f in sel] == [10, 12, 14, 16, 18, 20]
    assert sel[0].timestamp_sec == pytest.approx(10 / 30, abs=1e-4)


def test_probe_fields(tmp_path):
    path = tmp_path / "p.mp4"
    write_video(path, numbered_frames(10, (640, 360)), list(np.arange(10) / 30))
    info = probe(path, with_hash=True)
    assert (info.width, info.height) == (640, 360)
    assert info.codec == "h264"
    assert len(info.sha256) == 64
