"""Tiny local web UI: pick waypoints on a map, build, download the track as a zip.

Standard library only. Binds to localhost by default - it holds your API key.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..track import TrackBuilder, TrackConfig

INDEX = Path(__file__).with_name("index.html")


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.state = "running"
        self.log: list[str] = []
        self.zip_path: Path | None = None

    def write(self, msg):
        self.log.append(str(msg))


def run_job(job: Job, cfg: TrackConfig, offline: bool, out: Path, api_key: str):
    from ..cli import make_source, zip_track

    try:
        source = make_source(offline, api_key, out / "_cache", int(os.environ.get("SV2AC_MAX_IMAGES", 1000)), log=job.write)
        track_dir = TrackBuilder(cfg, source, out / job.id, log=job.write).build()
        job.zip_path = zip_track(track_dir)
        job.state = "done"
    except Exception as e:  # report any failure back to the page
        job.write(f"ERROR: {e}")
        traceback.print_exc()
        job.state = "error"


def make_handler(out: Path, api_key: str):
    jobs: dict[str, Job] = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            if parts == ["api", "config"]:
                return self._json({"has_key": bool(api_key)})
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"] and parts[2] in jobs:
                job = jobs[parts[2]]
                if len(parts) == 3:
                    return self._json({"state": job.state, "log": job.log})
                if parts[3:] == ["download"] and job.zip_path:
                    data = job.zip_path.read_bytes()
                    return self._send(200, data, "application/zip", {"Content-Disposition": f'attachment; filename="{job.zip_path.name}"'})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/build":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                offline = bool(body.get("offline")) or not api_key
                cfg = TrackConfig(
                    name=str(body.get("name") or "Street View Track")[:60],
                    waypoints=[(float(p[0]), float(p[1])) for p in body["waypoints"]],
                    closed=bool(body.get("closed", True)),
                    follow_roads=bool(body.get("follow_roads", True)),
                    road_width=float(body.get("road_width", 9)),
                    sv_spacing=max(10.0, float(body.get("sv_spacing", 25))),
                    pitboxes=max(1, min(30, int(body.get("pitboxes", 10)))),
                    sky_mask=bool(body.get("sky_mask", True)),
                )
            except (KeyError, ValueError, TypeError) as e:
                return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            job = Job()
            jobs[job.id] = job
            if offline:
                job.write("Offline mode: synthetic elevation and imagery (no API key in use).")
            threading.Thread(target=run_job, args=(job, cfg, offline, out, api_key), daemon=True).start()
            self._json({"job": job.id})

    return Handler


def serve(host: str, port: int, out: Path, api_key: str | None = None):
    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")
    out.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(out, api_key))
    print(f"sv2ac UI on http://{host}:{port}  ({'Google APIs enabled' if api_key else 'no API key - offline mode only'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
