"""Local review UI for the Video Indexing Engine (Python standard library only).

  python tools/index_server.py --out outputs/index/clip6 [--port 8765]
  -> open http://127.0.0.1:8765

Serves the ORIGINAL video file (HTTP Range requests, so the browser can seek), the automatic
events (events_auto.json) and the reviewed events (events_reviewed.json, created on first save;
every save keeps the previous version under history/). Optional clip export uses ffmpeg into
<out>/clips/. Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.index.schema import EVENT_FIELDS, read_events, validate_event, write_events  # noqa: E402

UI = Path(__file__).resolve().parent / "index_ui" / "index.html"


def make_handler(out: Path):
    auto = out / "events_auto.json"
    reviewed = out / "events_reviewed.json"
    header, _ = read_events(auto)
    video = Path(header["video_path"])
    if not video.is_absolute():
        video = ROOT / video

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/doc":
                src = reviewed if reviewed.exists() else auto
                h, ev = read_events(src)
                cm = []
                p = out / "game_clock_map.parquet"
                if p.exists():
                    import pandas as pd
                    cm = pd.read_parquet(p).round(3).values.tolist()
                return self._json(dict(header=h, events=ev, source=src.name, clock_map=cm,
                                       event_fields=EVENT_FIELDS))
            if self.path.startswith("/video"):
                return self._video()
            if self.path.startswith("/clips/"):
                f = out / "clips" / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "video/mp4")
            self._send(404, b"not found", "text/plain")

        def _video(self):
            size = video.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:  # suffix range
                        start = size - int(m.group(2))
            end = min(end, size - 1, start + 8 * 1024 * 1024 - 1)  # at most 8 MB per response
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(video, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 16, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass

        def do_POST(self):
            try:
                self._post()
            except Exception as e:  # report instead of dropping the connection
                self._json(dict(ok=False, error=f"{type(e).__name__}: {e}"), 500)

        def _post(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/save":
                ev = body.get("events", [])
                errs = {e.get("event_id"): validate_event(e) for e in ev}
                errs = {k: v for k, v in errs.items() if v}
                if errs:
                    return self._json(dict(ok=False, errors=errs), 400)
                if reviewed.exists():
                    hist = out / "history"
                    hist.mkdir(exist_ok=True)
                    shutil.copy(reviewed, hist / f"events_reviewed_{time.strftime('%Y%m%d_%H%M%S')}.json")
                h = dict(header)
                h["reviewed_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                write_events(reviewed, h, ev)
                return self._json(dict(ok=True, saved=str(reviewed), n=len(ev)))
            if self.path == "/api/export_clip":
                ff = shutil.which("ffmpeg")
                if not ff:
                    return self._json(dict(ok=False, error="ffmpeg not found"), 500)
                s, t = float(body["start"]), float(body["end"])
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", body.get("name", "clip"))[:80] + ".mp4"
                (out / "clips").mkdir(exist_ok=True)
                dst = out / "clips" / name
                r = subprocess.run([ff, "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", str(video), "-t", f"{max(0.05, t - s):.3f}",
                                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(dst)],
                                   capture_output=True, text=True)
                return self._json(dict(ok=r.returncode == 0, path=str(dst), url=f"/clips/{name}", error=r.stderr[-500:]))
            self._send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(out))
    print(f"Video index review UI: http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
