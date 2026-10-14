"""Build an Assetto Corsa track project from a route + Street View imagery."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import acfiles, textures
from .geo import LocalProjection, cumulative_length, gaussian_smooth, heading_deg, resample, tangents
from .mesh import MAX_VERTS, Material, MeshObject, Scene

UP = np.array([0.0, 1.0, 0.0])
CAMERA_HEIGHT = 2.5  # approx. height of the Street View camera above the road


@dataclass
class TrackConfig:
    name: str
    waypoints: list  # [(lat, lng), ...]
    closed: bool = True
    follow_roads: bool = True
    road_width: float = 9.0
    shoulder_width: float = 2.5
    wall_height: float = 1.2
    sample_spacing: float = 2.0
    smoothing_m: float = 8.0
    elevation_spacing: float = 10.0
    elevation_smoothing_m: float = 25.0
    sv_spacing: float = 25.0
    panel_distance: float | None = None  # default: max(sv_spacing / 2, edge of verge + 2 m)
    sky_mask: bool = True
    sv_size: int = 640
    atlas_tile: int = 512
    pitboxes: int = 10
    flip_start: bool = False
    author: str = "sv2ac"
    country: str = ""
    city: str = ""
    description: str = ""
    track_id: str = field(default="")

    def __post_init__(self):
        if not self.track_id:
            self.track_id = re.sub(r"[^a-z0-9_]+", "_", self.name.lower()).strip("_") or "sv_track"
        self.waypoints = [tuple(map(float, p)) for p in self.waypoints]
        if len(self.waypoints) < 2:
            raise ValueError("Need at least two waypoints.")


def _sky_alpha(img: Image.Image) -> Image.Image:
    """Make sky transparent: per column, walk down from the top while pixels look like sky."""
    a = np.asarray(img.convert("RGB"), float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    bright = (r + g + b) / 3
    sat = a.max(axis=2) - a.min(axis=2)
    skyish = ((b >= r - 5) & (bright > 140) & ((sat < 40) | (b > g))) | (bright > 225)
    # stop at the first non-sky pixel in each column
    run = np.cumprod(skyish, axis=0).astype(bool)
    alpha = np.where(run, 0, 255).astype(np.uint8)
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


class TrackBuilder:
    def __init__(self, cfg: TrackConfig, source, out_root: Path, log=print):
        self.cfg = cfg
        self.src = source
        self.out = Path(out_root) / cfg.track_id
        self.log = log

    # ------------------------------------------------------------------ centreline
    def _centreline(self):
        c = self.cfg
        if c.follow_roads:
            self.log(f"Routing {len(c.waypoints)} waypoints along roads...")
            path = self.src.route(c.waypoints, c.closed)
        else:
            path = list(c.waypoints) + ([c.waypoints[0]] if c.closed else [])
        self.path_ll = path
        self.proj = LocalProjection(*path[0])
        x, z = self.proj.to_xz([p[0] for p in path], [p[1] for p in path])
        xz = np.column_stack([x, z])
        dense = resample(xz, 1.0, c.closed)
        dense = gaussian_smooth(dense, c.smoothing_m, c.closed)
        self.xz = resample(dense, c.sample_spacing, c.closed)
        n = len(self.xz)
        if c.closed:
            self.length = cumulative_length(np.vstack([self.xz, self.xz[:1]]))[-1]
            self.s = np.arange(n) * self.length / n
        else:
            self.s = cumulative_length(self.xz)
            self.length = self.s[-1]
        self.ds = self.length / (n if c.closed else n - 1)
        self.log(f"Centreline: {self.length / 1000:.2f} km, {n} samples")
        if self.length < 200:
            raise ValueError("Track is shorter than 200 m - add more waypoints.")

    def _elevation(self):
        c = self.cfg
        step = max(1, int(round(c.elevation_spacing / self.ds)))
        idx = np.arange(0, len(self.xz), step)
        if not c.closed and idx[-1] != len(self.xz) - 1:
            idx = np.append(idx, len(self.xz) - 1)
        lat, lng = self.proj.to_latlng(self.xz[idx, 0], self.xz[idx, 1])
        self.log(f"Fetching elevation for {len(idx)} points...")
        elev = np.asarray(self.src.elevations(list(zip(lat.tolist(), lng.tolist()))), float)
        if c.closed:
            sp = np.append(self.s[idx], self.length)
            el = np.append(elev, elev[0])
        else:
            sp, el = self.s[idx], elev
        y = np.interp(self.s, sp, el)
        y = gaussian_smooth(y, c.elevation_smoothing_m / self.ds, c.closed)
        self.y = y - y[0]
        grade = np.abs(np.diff(self.y)) / self.ds
        self.log(f"Elevation range {self.y.min():.1f}..{self.y.max():.1f} m, max grade {grade.max() * 100:.1f}%")

    def _frames(self):
        c = self.cfg
        t = tangents(self.xz, c.closed)
        self.t2 = t
        self.P = np.column_stack([self.xz[:, 0], self.y, self.xz[:, 1]])
        self.right = np.column_stack([-t[:, 1], np.zeros(len(t)), t[:, 0]])
        self.heading = heading_deg(t)
        # signed curvature: positive = turning right (so the right-hand side is the inside)
        th = np.unwrap(np.radians(self.heading))
        if c.closed:
            dth = ((np.roll(th, -1) - np.roll(th, 1) + np.pi) % (2 * np.pi) - np.pi) / 2
        else:
            dth = np.gradient(th)
        kappa = dth / self.ds
        self.kappa = gaussian_smooth(kappa, 3, c.closed)

    def _offset(self, o: float) -> np.ndarray:
        """Per-sample lateral offset, pulled in on the inside of tight corners so meshes don't fold."""
        out = np.full(len(self.P), o, float)
        k = self.kappa * np.sign(o)  # >0 when this offset is on the inside of the bend
        with np.errstate(divide="ignore"):
            limit = np.where(k > 1e-6, 0.85 / np.maximum(k, 1e-6), np.inf)
        return np.sign(o) * np.minimum(abs(o), np.maximum(limit, self.cfg.road_width / 2 + 0.5))

    # ------------------------------------------------------------------ meshes
    def _row_indices(self):
        n = len(self.P)
        if self.cfg.closed:
            return np.append(np.arange(n), 0), np.append(self.s, self.length)
        return np.arange(n), self.s

    def _strip(self, scene, prefix, material, profile, along_scale, along="v"):
        """profile: list of (offset, height, cross_coord). Rows follow the track."""
        idx, sv = self._row_indices()
        offsets = [self._offset(o) if abs(o) > self.cfg.road_width / 2 + 0.01 else np.full(len(self.P), o) for o, _, _ in profile]
        chunk = max(2, min(400, MAX_VERTS // len(profile) - 1))
        part = 0
        for start in range(0, len(idx) - 1, chunk):
            rows = idx[start : start + chunk + 1]
            srow = sv[start : start + chunk + 1]
            pos = np.zeros((len(rows), len(profile), 3))
            uv = np.zeros((len(rows), len(profile), 2))
            for ci, (_, h, cross) in enumerate(profile):
                pos[:, ci] = self.P[rows] + self.right[rows] * offsets[ci][rows, None] + UP * h
                a = srow * along_scale
                uv[:, ci] = np.column_stack([np.full(len(rows), cross), a]) if along == "v" else np.column_stack([a, np.full(len(rows), cross)])
            obj = MeshObject(f"{prefix}_{part:03d}", material)
            obj.add_grid(pos, uv)
            scene.add(obj)
            part += 1

    def _build_base(self, scene, tex_dir):
        c = self.cfg
        w2, sh = c.road_width / 2, c.shoulder_width
        tile = 16.0
        tile = self.length / max(1, round(self.length / tile)) if c.closed else tile
        textures.save(textures.asphalt(c.road_width, tile), tex_dir / "asphalt.png")
        textures.save(textures.grass(), tex_dir / "grass.png")
        textures.save(textures.ground(), tex_dir / "ground.png")
        textures.save(textures.barrier(), tex_dir / "barrier.png")
        for name in ("asphalt", "grass", "ground", "barrier"):
            scene.add_material(Material(name, f"textures/{name}.png"))

        self._strip(scene, "1ROAD", "asphalt", [(-w2, 0, 0), (0, 0.03, 0.5), (w2, 0, 1)], 1 / tile)
        self._strip(scene, "1GRASS_L", "grass", [(-w2 - sh, -0.08, 0), (-w2, 0, sh / 4)], 1 / 4)
        self._strip(scene, "1GRASS_R", "grass", [(w2, 0, 0), (w2 + sh, -0.08, sh / 4)], 1 / 4)
        wo = w2 + sh
        self._strip(scene, "1WALL_L", "barrier", [(-wo, c.wall_height, 1), (-wo, -0.5, 0)], 1 / 4, along="u")
        self._strip(scene, "1WALL_R", "barrier", [(wo, -0.5, 0), (wo, c.wall_height, 1)], 1 / 4, along="u")
        g = self.panel_distance + 40
        self._strip(scene, "GROUND_L", "ground", [(-g, -1.0, 0), (-wo - 0.05, -0.1, (g - wo) / 8)], 1 / 8)
        self._strip(scene, "GROUND_R", "ground", [(wo + 0.05, -0.1, 0), (g, -1.0, (g - wo) / 8)], 1 / 8)

    # ------------------------------------------------------------------ street view
    def _streetview(self, scene, tex_dir, ui_dir):
        c = self.cfg
        S, D = c.sv_spacing, self.panel_distance
        fov = math.degrees(2 * math.atan((S / 2) / D))
        half = c.sv_size / 2
        focal = half / math.tan(math.radians(fov / 2))
        H = CAMERA_HEIGHT + S / 2  # tallest wall the image can cover at distance D
        bottom = -1.0
        row_top = max(0, half - focal * (H - CAMERA_HEIGHT) / D)
        row_bot = min(c.sv_size, half + focal * (CAMERA_HEIGHT - bottom) / D)
        tile_h = int(c.atlas_tile * (H - bottom) / S / 2) * 2
        tile_h = max(64, min(c.atlas_tile * 2, tile_h))
        atlas = textures.Atlas("sv_atlas", c.atlas_tile, tile_h, alpha=c.sky_mask)
        centres = np.arange(S / 2, self.length - (0 if c.closed else S / 2) + 1e-6, S)
        if c.closed:
            centres = centres[centres < self.length]
        self.log(f"Street View: {len(centres)} sample points every {S:.0f} m, fov {fov:.0f} deg, panels {H:.1f} m tall at {D:.1f} m")

        panels = []  # (page, uvrect, centre_s, side)
        preview = None
        missing = 0
        for k, sc in enumerate(centres):
            i = int(round(sc / self.ds)) % len(self.P)
            lat, lng = self.proj.to_latlng(self.P[i, 0], self.P[i, 2])
            meta = self.src.streetview_meta(float(lat), float(lng))
            if not meta:
                missing += 1
                continue
            pano = meta["pano_id"]
            if preview is None:
                preview = self.src.streetview_image(pano, self.heading[i], 80, 0, c.sv_size)
            for side, dh in (("R", 90), ("L", -90)):
                img = self.src.streetview_image(pano, (self.heading[i] + dh) % 360, fov, 0, c.sv_size)
                if img is None:
                    continue
                img = img.resize((c.sv_size, c.sv_size)).crop((0, int(row_top), c.sv_size, int(row_bot)))
                if c.sky_mask:
                    img = _sky_alpha(img)
                page, rect = atlas.add(img)
                panels.append((page, rect, sc, side))
            if (k + 1) % 20 == 0:
                self.log(f"  {k + 1}/{len(centres)} points processed")
        self.sv_stats = {"points": int(len(centres)), "no_coverage": missing, "panels": len(panels), "images_downloaded": self.src.images_fetched}
        self.log(f"Street View: {len(panels)} panels, {missing} points without coverage, {self.src.images_fetched} images downloaded")

        for name in atlas.save(tex_dir):
            scene.add_material(Material(name, f"textures/{name}.png", "ksPerPixelAT" if c.sky_mask else "ksPerPixel", alpha=c.sky_mask))
        objs: dict[int, MeshObject] = {}
        n = len(self.P)
        for page, (u0, v0, u1, v1), sc, side in panels:
            i0 = int(math.floor((sc - S / 2) / self.ds))
            i1 = int(math.ceil((sc + S / 2) / self.ds))
            rows = np.arange(i0, i1 + 1)
            rows = rows % n if c.closed else np.clip(rows, 0, n - 1)
            if len(np.unique(rows)) < 2:
                continue
            t = np.linspace(0, 1, len(rows))
            sign = 1 if side == "R" else -1
            off = self._offset(sign * D)[rows]
            # right side: image left = forward end; left side: image left = rear end
            u = u0 + (1 - t if side == "R" else t) * (u1 - u0)
            heights = [(bottom, v0), (H, v1)] if side == "R" else [(H, v1), (bottom, v0)]
            pos = np.zeros((len(rows), 2, 3))
            uv = np.zeros((len(rows), 2, 2))
            for ci, (h, v) in enumerate(heights):
                pos[:, ci] = self.P[rows] + self.right[rows] * off[:, None] + UP * h
                uv[:, ci] = np.column_stack([u, np.full(len(rows), v)])
            obj = objs.setdefault(page, MeshObject(f"SV_PANELS_{page:02d}", atlas.page_name(page)))
            obj.add_grid(pos, uv)
        for obj in objs.values():
            scene.add(obj)

        if preview is None:
            preview = Image.new("RGB", (640, 640), (70, 80, 90))
        textures.save(ImageOps.fit(preview, (640, 360)), ui_dir / "preview.png")

    # ------------------------------------------------------------------ empties
    def _empty(self, name, s, lateral=0.0, lift=0.3):
        n = len(self.P)
        i = int(round(s / self.ds)) % n if self.cfg.closed else int(np.clip(round(s / self.ds), 0, n - 1))
        p = self.P[i] + self.right[i] * lateral + UP * lift
        f = np.array([self.t2[i, 0], 0.0, self.t2[i, 1]])
        if self.cfg.flip_start:
            f = -f
        return {"name": name, "position": [round(v, 4) for v in p.tolist()], "forward": [round(v, 5) for v in f.tolist()], "up": [0, 1, 0]}

    def _start_line(self) -> float:
        """Put a loop's start/finish where the road is straightest across the grid and run to turn 1."""
        before = 10 + 8.0 * self.cfg.pitboxes
        after = 60.0
        k = np.abs(self.kappa) * self.ds
        n = len(k)
        nb, na = int(before / self.ds), int(after / self.ds)
        csum = np.concatenate([[0], np.cumsum(np.concatenate([k, k, k]))])
        idx = np.arange(n) + n  # centre copy of the tripled array
        cost = csum[idx + na + 1] - csum[idx - nb]
        return float(self.s[int(np.argmin(cost))])

    def _empties(self):
        c = self.cfg
        L, w = self.length, c.road_width
        spacing, lane = 8.0, w / 4
        e = []
        if c.closed:
            line = self._start_line()
            for k in range(c.pitboxes):
                s = (line - 6 - spacing * k) % L
                lat_off = lane * (1 if k % 2 == 0 else -1)
                e.append(self._empty(f"AC_START_{k}", s, lat_off))
                e.append(self._empty(f"AC_PIT_{k}", s, lat_off))
            e.append(self._empty("AC_HOTLAP_START_0", (line - 80) % L))
            e.append(self._empty("AC_TIME_0_L", line, -(w / 2 + 0.5), 0))
            e.append(self._empty("AC_TIME_0_R", line, w / 2 + 0.5, 0))
        else:
            line = 6 + spacing * c.pitboxes
            if line > L * 0.5:
                raise ValueError("Point-to-point route is too short for the grid; reduce --pitboxes.")
            finish = L - 20
            for k in range(c.pitboxes):
                s = line - 6 - spacing * k
                lat_off = lane * (1 if k % 2 == 0 else -1)
                e.append(self._empty(f"AC_START_{k}", s, lat_off))
                e.append(self._empty(f"AC_PIT_{k}", s, lat_off))
            e.append(self._empty("AC_HOTLAP_START_0", line - 6))
            e.append(self._empty("AC_AB_START_L", line, -(w / 2 + 0.5), 0))
            e.append(self._empty("AC_AB_START_R", line, w / 2 + 0.5, 0))
            e.append(self._empty("AC_AB_FINISH_L", finish, -(w / 2 + 0.5), 0))
            e.append(self._empty("AC_AB_FINISH_R", finish, w / 2 + 0.5, 0))
        return e

    # ------------------------------------------------------------------ main
    def build(self) -> Path:
        c = self.cfg
        if self.out.exists():
            shutil.rmtree(self.out)
        src_dir = self.out / "source"
        tex_dir = src_dir / "textures"
        ui_dir = self.out / "ui"
        for d in (tex_dir, ui_dir, self.out / "data"):
            d.mkdir(parents=True, exist_ok=True)

        self._centreline()
        self._elevation()
        self._frames()
        self.panel_distance = c.panel_distance or max(c.sv_spacing / 2, c.road_width / 2 + c.shoulder_width + 2)

        scene = Scene()
        self._build_base(scene, tex_dir)
        self._streetview(scene, tex_dir, ui_dir)
        scene.write_obj(src_dir / f"{c.track_id}.obj")
        empties = self._empties()

        stats = scene.stats()
        self.log(f"Mesh: {stats['objects']} objects, {stats['vertices']} vertices, {stats['triangles']} triangles")
        lat0, lng0 = self.path_ll[0]
        (src_dir / "scene.json").write_text(json.dumps({
            "track_id": c.track_id,
            "obj": f"{c.track_id}.obj",
            "fbx": f"{c.track_id}.fbx",
            "empties": empties,
            "materials": {m.name: {"texture": m.texture, "shader": m.shader_hint, "alpha": m.alpha} for m in scene.materials.values()},
            "origin_latlng": [lat0, lng0],
        }, indent=1))
        shutil.copy(Path(__file__).with_name("blender_build_fbx.py"), src_dir / "blender_build_fbx.py")

        centre3 = self.P
        acfiles.write_all(self.out, c, centre3, self.length, (lat0, lng0), stats, getattr(self, "sv_stats", {}))
        (self.out / "sv2ac_config.json").write_text(json.dumps(asdict(c), indent=1))
        self.log(f"Done -> {self.out}")
        return self.out
