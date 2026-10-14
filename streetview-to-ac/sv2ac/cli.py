"""Command line interface: `python -m sv2ac build ...` / `python -m sv2ac serve`."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .inputs import load_file, parse_waypoints
from .sources import GoogleSource, OfflineSource
from .track import TrackBuilder, TrackConfig


def make_source(offline: bool, api_key: str | None, cache_dir: Path, max_images: int, log=print):
    if offline:
        return OfflineSource(log=log)
    return GoogleSource(api_key or os.environ.get("GOOGLE_MAPS_API_KEY", ""), cache_dir, max_images, log=log)


def zip_track(track_dir: Path) -> Path:
    return Path(shutil.make_archive(str(track_dir), "zip", track_dir.parent, track_dir.name))


def build_cmd(a):
    if a.waypoints:
        pts = parse_waypoints(a.waypoints)
    elif a.file:
        pts = load_file(a.file)
    else:
        sys.exit("Give --waypoints or --file")
    follow = a.follow_roads
    if follow is None:
        follow = len(pts) <= 100  # dense GPS traces are already on the road
    cfg = TrackConfig(
        name=a.name, waypoints=pts, closed=not a.point_to_point, follow_roads=follow,
        road_width=a.road_width, shoulder_width=a.shoulder, sv_spacing=a.sv_spacing,
        sample_spacing=a.sample_spacing, smoothing_m=a.smoothing, pitboxes=a.pitboxes,
        sky_mask=not a.no_sky_mask, flip_start=a.flip_start, author=a.author,
        country=a.country, city=a.city, track_id=a.track_id or "",
    )
    out = Path(a.out)
    source = make_source(a.offline, a.api_key, out / "_cache", a.max_images)
    track_dir = TrackBuilder(cfg, source, out).build()
    if a.zip:
        print(f"Zipped -> {zip_track(track_dir)}")


def serve_cmd(a):
    from .web.server import serve

    serve(a.host, a.port, Path(a.out), a.api_key)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sv2ac", description="Generate Assetto Corsa tracks from Google Street View.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a track")
    b.add_argument("--name", required=True)
    b.add_argument("--track-id", help="folder name (default: derived from --name)")
    b.add_argument("--waypoints", help='"lat,lng; lat,lng; ..."')
    b.add_argument("--file", help="GPX, KML or text file of waypoints")
    b.add_argument("--point-to-point", action="store_true", help="A-to-B stage instead of a closed loop")
    roads = b.add_mutually_exclusive_group()
    roads.add_argument("--follow-roads", dest="follow_roads", action="store_true", default=None, help="snap waypoints to roads with the Routes API")
    roads.add_argument("--no-follow-roads", dest="follow_roads", action="store_false", help="use waypoints as the exact centreline")
    b.add_argument("--road-width", type=float, default=9.0)
    b.add_argument("--shoulder", type=float, default=2.5, help="grass verge width before the barrier (m)")
    b.add_argument("--sample-spacing", type=float, default=2.0, help="mesh resolution along the road (m)")
    b.add_argument("--smoothing", type=float, default=8.0, help="centreline smoothing radius (m)")
    b.add_argument("--sv-spacing", type=float, default=25.0, help="distance between Street View samples (m)")
    b.add_argument("--max-images", type=int, default=1000, help="cap on billed Street View image downloads")
    b.add_argument("--pitboxes", type=int, default=10)
    b.add_argument("--no-sky-mask", action="store_true", help="keep the sky in roadside photo panels")
    b.add_argument("--flip-start", action="store_true", help="rotate spawn points 180 deg if cars spawn backwards")
    b.add_argument("--author", default="sv2ac")
    b.add_argument("--country", default="")
    b.add_argument("--city", default="")
    b.add_argument("--offline", action="store_true", help="no API calls: synthetic elevation and imagery (for testing)")
    b.add_argument("--api-key", help="Google Maps Platform key (default: $GOOGLE_MAPS_API_KEY)")
    b.add_argument("--out", default="tracks")
    b.add_argument("--zip", action="store_true")
    b.set_defaults(func=build_cmd)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--out", default="tracks")
    s.add_argument("--api-key")
    s.set_defaults(func=serve_cmd)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
