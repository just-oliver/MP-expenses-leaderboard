"""Data sources: Google Maps Platform (routes, elevation, Street View) and an offline stand-in."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .geo import decode_polyline, encode_polyline

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
SV_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


class SourceError(RuntimeError):
    pass


class GoogleSource:
    """Thin client for the Google Maps Platform APIs we need, with an on-disk cache.

    Cached responses mean re-running a build (e.g. after tweaking road width)
    does not re-bill you for the same Street View images.
    """

    def __init__(self, api_key: str, cache_dir: Path, max_images: int = 1000, log=print):
        if not api_key:
            raise SourceError("A Google Maps Platform API key is required (set GOOGLE_MAPS_API_KEY or use --offline).")
        self.key = api_key
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.images_fetched = 0
        self.log = log

    # -- plumbing -----------------------------------------------------------------
    def _cache_path(self, kind: str, ident: str, ext: str) -> Path:
        h = hashlib.sha1(ident.encode()).hexdigest()[:20]
        return self.cache / f"{kind}_{h}.{ext}"

    def _request(self, url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(url, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise SourceError(f"HTTP {e.code} from {url.split('?')[0]}: {body}") from None

    def _get_json(self, base: str, params: dict, kind: str) -> dict:
        ident = base + "?" + urllib.parse.urlencode(sorted(params.items()))
        path = self._cache_path(kind, ident, "json")
        if path.exists():
            return json.loads(path.read_text())
        url = base + "?" + urllib.parse.urlencode({**params, "key": self.key})
        result = json.loads(self._request(url))
        status = result.get("status", "OK")
        if status not in ("OK", "ZERO_RESULTS"):
            raise SourceError(f"{kind} request failed: {status} {result.get('error_message', '')}")
        path.write_text(json.dumps(result))
        return result

    # -- APIs ---------------------------------------------------------------------
    def route(self, waypoints, closed: bool) -> list[tuple[float, float]]:
        """Snap waypoints to drivable roads with the Routes API; returns a dense lat/lng path.

        The API allows 25 intermediate waypoints per request, so long lists are
        routed in consecutive legs and stitched together.
        """
        pts = list(waypoints)
        if closed:
            pts = pts + [pts[0]]
        path: list[tuple[float, float]] = []
        step = 26
        for i in range(0, len(pts) - 1, step):
            leg = self._route_leg(pts[i : i + step + 1])
            path.extend(leg if not path else leg[1:])
        return path

    def _route_leg(self, pts):
        def ll(p):
            return {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}

        body = {
            "origin": ll(pts[0]),
            "destination": ll(pts[-1]),
            "intermediates": [{**ll(p), "via": True} for p in pts[1:-1]],
            "travelMode": "DRIVE",
            "polylineQuality": "HIGH_QUALITY",
        }
        payload = json.dumps(body, sort_keys=True)
        path = self._cache_path("route", payload, "json")
        if path.exists():
            result = json.loads(path.read_text())
        else:
            raw = self._request(
                ROUTES_URL,
                data=payload.encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": self.key,
                    "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.distanceMeters",
                },
            )
            result = json.loads(raw)
            path.write_text(json.dumps(result))
        if not result.get("routes"):
            raise SourceError("Routes API found no drivable route between the waypoints.")
        return decode_polyline(result["routes"][0]["polyline"]["encodedPolyline"])

    def elevations(self, latlngs) -> list[float]:
        out = []
        batch = 256  # keeps the URL comfortably under the 16k char limit
        for i in range(0, len(latlngs), batch):
            chunk = latlngs[i : i + batch]
            res = self._get_json(ELEVATION_URL, {"locations": "enc:" + encode_polyline(chunk)}, "elev")
            out.extend(r["elevation"] for r in res["results"])
        return out

    def streetview_meta(self, lat: float, lng: float, radius: int = 40) -> dict | None:
        """Free metadata call; returns pano info or None if there is no coverage nearby."""
        res = self._get_json(
            SV_META_URL,
            {"location": f"{lat:.6f},{lng:.6f}", "radius": radius, "source": "outdoor"},
            "svmeta",
        )
        return res if res.get("status") == "OK" else None

    def streetview_image(self, pano_id: str, heading: float, fov: float, pitch: float = 0.0, size: int = 640) -> Image.Image | None:
        params = {
            "pano": pano_id,
            "heading": f"{heading:.1f}",
            "fov": f"{fov:.1f}",
            "pitch": f"{pitch:.1f}",
            "size": f"{size}x{size}",
            "return_error_code": "true",
        }
        ident = SV_IMAGE_URL + "?" + urllib.parse.urlencode(sorted(params.items()))
        path = self._cache_path("sv", ident, "jpg")
        if path.exists():
            return Image.open(path).convert("RGB")
        if self.images_fetched >= self.max_images:
            return None
        url = SV_IMAGE_URL + "?" + urllib.parse.urlencode({**params, "key": self.key})
        try:
            raw = self._request(url)
        except SourceError as e:
            self.log(f"  street view image failed: {e}")
            return None
        self.images_fetched += 1
        path.write_bytes(raw)
        return Image.open(io.BytesIO(raw)).convert("RGB")


class OfflineSource:
    """Synthetic stand-in so the whole pipeline can run without an API key.

    Routes are the raw waypoints, elevation is a gentle procedural terrain, and
    "Street View" images are procedurally drawn streetscapes.
    """

    def __init__(self, seed: int = 1, log=print):
        self.seed = seed
        self.images_fetched = 0
        self.log = log

    def route(self, waypoints, closed: bool):
        pts = list(waypoints)
        if closed:
            pts.append(pts[0])
        return pts

    def elevations(self, latlngs):
        return [
            40 + 6 * math.sin(lat * 900 + 1.3) * math.cos(lng * 700) + 2 * math.sin(lng * 2100)
            for lat, lng in latlngs
        ]

    def streetview_meta(self, lat, lng, radius=40):
        return {"status": "OK", "pano_id": f"offline_{lat:.5f}_{lng:.5f}", "location": {"lat": lat, "lng": lng}, "date": "offline"}

    def streetview_image(self, pano_id, heading, fov, pitch=0.0, size=640):
        rnd = random.Random(f"{self.seed}:{pano_id}:{round(heading)}")
        self.images_fetched += 1
        img = Image.new("RGB", (size, size))
        d = ImageDraw.Draw(img)
        horizon = size // 2
        for y in range(horizon):  # sky gradient
            t = y / horizon
            d.line([(0, y), (size, y)], fill=(int(120 + 80 * t), int(170 + 60 * t), 235))
        d.rectangle([0, horizon, size, size], fill=(95, 110, 80))
        x = rnd.randint(-60, 0)
        while x < size:
            w = rnd.randint(60, 160)
            h = rnd.randint(60, 260)
            if rnd.random() < 0.25:  # a tree instead of a building
                cx = x + w // 2
                d.rectangle([cx - 5, horizon - 40, cx + 5, horizon + 10], fill=(80, 60, 40))
                d.ellipse([cx - w // 2, horizon - 40 - h // 2, cx + w // 2, horizon - 20], fill=(40 + rnd.randint(0, 40), 110, 50))
            else:
                col = tuple(rnd.randint(90, 220) for _ in range(3))
                d.rectangle([x, horizon - h, x + w, horizon + 15], fill=col)
                for wy in range(horizon - h + 12, horizon - 10, 28):
                    for wx in range(x + 10, x + w - 18, 26):
                        d.rectangle([wx, wy, wx + 12, wy + 16], fill=(60, 80, 110))
            x += w + rnd.randint(0, 30)
        return img.filter(ImageFilter.GaussianBlur(0.6))
