"""Parse waypoints from text, GPX or KML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    """'lat,lng; lat,lng; ...' (semicolons or newlines between points)."""
    pts = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lat, lng = (float(v) for v in chunk.split(","))
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"Invalid coordinate: {chunk}")
        pts.append((lat, lng))
    return pts


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def load_gpx(path: Path) -> list[tuple[float, float]]:
    root = ET.parse(path).getroot()
    for kind in ("trkpt", "rtept", "wpt"):
        pts = [(float(e.get("lat")), float(e.get("lon"))) for e in root.iter() if _local(e.tag) == kind]
        if pts:
            return pts
    raise ValueError(f"No track/route points in {path}")


def load_kml(path: Path) -> list[tuple[float, float]]:
    root = ET.parse(path).getroot()
    pts = []
    for e in root.iter():
        if _local(e.tag) == "coordinates" and e.text:
            for tup in e.text.split():
                lng, lat = tup.split(",")[:2]
                pts.append((float(lat), float(lng)))
    if not pts:
        raise ValueError(f"No coordinates in {path}")
    return pts


def load_file(path: str | Path) -> list[tuple[float, float]]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".gpx":
        return load_gpx(path)
    if ext == ".kml":
        return load_kml(path)
    return parse_waypoints(path.read_text())
