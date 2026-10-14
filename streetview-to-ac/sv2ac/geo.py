"""Geographic helpers: polyline codec, local projection, resampling and smoothing."""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS = 6378137.0


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into (lat, lng) pairs."""
    points, index, lat, lng = [], 0, 0, 0
    while index < len(encoded):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lng += delta
        points.append((lat / 1e5, lng / 1e5))
    return points


def encode_polyline(points) -> str:
    """Encode (lat, lng) pairs as a Google encoded polyline."""
    out, prev_lat, prev_lng = [], 0, 0
    for lat, lng in points:
        ilat, ilng = round(lat * 1e5), round(lng * 1e5)
        for delta in (ilat - prev_lat, ilng - prev_lng):
            v = ~(delta << 1) if delta < 0 else delta << 1
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        prev_lat, prev_lng = ilat, ilng
    return "".join(out)


class LocalProjection:
    """Equirectangular projection around an origin.

    Output axes follow Assetto Corsa / FBX Y-up convention:
    x = east, y = up, z = south (so north is -z).
    Accurate to well under 0.1% over the few kilometres a track spans.
    """

    def __init__(self, lat0: float, lng0: float):
        self.lat0, self.lng0 = lat0, lng0
        self.kx = math.radians(1) * EARTH_RADIUS * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_RADIUS

    def to_xz(self, lat, lng):
        lat, lng = np.asarray(lat, float), np.asarray(lng, float)
        return (lng - self.lng0) * self.kx, -(lat - self.lat0) * self.ky

    def to_latlng(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        return self.lat0 - z / self.ky, self.lng0 + x / self.kx


def cumulative_length(xz: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def dedupe(xz: np.ndarray, min_dist: float = 0.05) -> np.ndarray:
    keep = [0]
    for i in range(1, len(xz)):
        if np.linalg.norm(xz[i] - xz[keep[-1]]) > min_dist:
            keep.append(i)
    return xz[keep]


def resample(xz: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Resample a polyline at (approximately) uniform spacing.

    For closed loops the returned array does not repeat the first point.
    """
    pts = dedupe(xz)
    if closed:
        if np.linalg.norm(pts[0] - pts[-1]) < 0.5:
            pts = pts[:-1]
        pts = np.vstack([pts, pts[:1]])
    s = cumulative_length(pts)
    total = s[-1]
    n = max(2, int(round(total / spacing)))
    if closed:
        targets = np.linspace(0, total, n, endpoint=False)
    else:
        targets = np.linspace(0, total, n + 1)
    return np.column_stack([np.interp(targets, s, pts[:, k]) for k in range(pts.shape[1])])


def gaussian_smooth(values: np.ndarray, sigma_samples: float, closed: bool) -> np.ndarray:
    """Gaussian smoothing along axis 0. Open curves keep their end points fixed."""
    if sigma_samples <= 0:
        return values.copy()
    radius = int(math.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    k /= k.sum()
    v = values.reshape(len(values), -1)
    n = len(v)
    if closed:
        idx = np.arange(-radius, n + radius) % n
        padded = v[idx]
    else:
        # reflect-pad around the end points (odd reflection keeps tangents sane)
        left = 2 * v[0] - v[1 : radius + 1][::-1]
        right = 2 * v[-1] - v[-radius - 1 : -1][::-1]
        if len(left) < radius:  # very short curves
            return values.copy()
        padded = np.vstack([left, v, right])
    out = np.column_stack([np.convolve(padded[:, c], k, mode="valid") for c in range(v.shape[1])])
    if not closed:
        out[0], out[-1] = v[0], v[-1]
    return out.reshape(values.shape)


def tangents(xz: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        t = np.roll(xz, -1, axis=0) - np.roll(xz, 1, axis=0)
    else:
        t = np.gradient(xz, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def heading_deg(tangent_xz: np.ndarray) -> np.ndarray:
    """Compass heading (degrees clockwise from north) for x=east, z=south tangents."""
    return np.degrees(np.arctan2(tangent_xz[..., 0], -tangent_xz[..., 1])) % 360
