import io
import json
import re

import numpy as np
import pytest

from sv2ac import geo, inputs
from sv2ac.mesh import MAX_VERTS
from sv2ac.sources import GoogleSource, OfflineSource
from sv2ac.track import TrackBuilder, TrackConfig

LOOP = [(51.5007, -0.1246), (51.5035, -0.1195), (51.5065, -0.1270), (51.5030, -0.1320)]


def quiet(*_):
    pass


def test_polyline_roundtrip_matches_google_example():
    enc = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    pts = geo.decode_polyline(enc)
    assert pts == [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]
    assert geo.encode_polyline(pts) == enc


def test_projection_roundtrip_and_axes():
    p = geo.LocalProjection(51.5, -0.12)
    x, z = p.to_xz(51.501, -0.119)
    assert x > 0 and z < 0  # east is +x, north is -z
    lat, lng = p.to_latlng(x, z)
    assert lat == pytest.approx(51.501) and lng == pytest.approx(-0.119)


def test_resample_closed_uniform():
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], float)
    r = geo.resample(sq, 5.0, closed=True)
    seg = np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1)
    assert len(r) == 80 and np.allclose(seg, 5.0)


def test_heading():
    assert geo.heading_deg(np.array([0.0, -1.0])) == pytest.approx(0)  # north
    assert geo.heading_deg(np.array([1.0, 0.0])) == pytest.approx(90)  # east


def test_parse_inputs(tmp_path):
    assert inputs.parse_waypoints("1,2; 3,4\n5,6") == [(1, 2), (3, 4), (5, 6)]
    gpx = tmp_path / "a.gpx"
    gpx.write_text('<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
                   '<trkpt lat="1" lon="2"/><trkpt lat="3" lon="4"/></trkseg></trk></gpx>')
    assert inputs.load_file(gpx) == [(1, 2), (3, 4)]
    kml = tmp_path / "a.kml"
    kml.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><LineString>'
                   '<coordinates>2,1,0 4,3,0</coordinates></LineString></Placemark></kml>')
    assert inputs.load_file(kml) == [(1, 2), (3, 4)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("tracks")
    cfg = TrackConfig(name="Test Loop", waypoints=LOOP, pitboxes=6)
    b = TrackBuilder(cfg, OfflineSource(log=quiet), out, log=quiet)
    return b, b.build()


def test_track_layout(built):
    _, d = built
    for f in ["data/surfaces.ini", "data/map.ini", "map.png", "ui/ui_track.json", "ui/outline.png",
              "ui/preview.png", "source/test_loop.obj", "source/test_loop.mtl", "source/scene.json",
              "source/blender_build_fbx.py", "source/textures/asphalt.png", "source/textures/sv_atlas_00.png"]:
        assert (d / f).exists(), f
    ui = json.loads((d / "ui/ui_track.json").read_text())
    assert ui["pitboxes"] == "6" and ui["run"] in ("clockwise", "anticlockwise")


def test_obj_objects_fit_kn5_limits_and_reference_materials(built):
    _, d = built
    text = (d / "source/test_loop.obj").read_text()
    mtl = (d / "source/test_loop.mtl").read_text()
    counts, cur = {}, None
    for line in text.splitlines():
        if line.startswith("o "):
            cur = line[2:]
            counts[cur] = 0
        elif line.startswith("v "):
            counts[cur] += 1
        elif line.startswith("usemtl "):
            assert f"newmtl {line[7:]}" in mtl
    assert max(counts.values()) <= MAX_VERTS
    for prefix in ("1ROAD", "1GRASS", "1WALL", "SV_PANELS", "GROUND"):
        assert any(n.startswith(prefix) for n in counts), prefix


def test_road_faces_point_up_and_walls_face_road(built):
    b, d = built
    text = (d / "source/test_loop.obj").read_text()

    def normals(obj):
        block = text.split(f"o {obj}\n")[1].split("\no ")[0]
        return np.array([[float(v) for v in ln.split()[1:]] for ln in block.splitlines() if ln.startswith("vn ")])

    def verts(obj):
        block = text.split(f"o {obj}\n")[1].split("\no ")[0]
        return np.array([[float(v) for v in ln.split()[1:]] for ln in block.splitlines() if ln.startswith("v ")])

    assert (normals("1ROAD_000")[:, 1] > 0.9).all()
    for side in ("1WALL_L_000", "1WALL_R_000", "SV_PANELS_00"):
        v, n = verts(side), normals(side)
        nearest = np.argmin(np.linalg.norm(v[:, None, [0, 2]] - b.P[None, :, [0, 2]], axis=2), axis=1)
        to_road = b.P[nearest] - v
        assert (np.einsum("ij,ij->i", n[:, [0, 2]], to_road[:, [0, 2]]) > 0).mean() > 0.9


def test_spawns_on_road_facing_along_track(built):
    b, d = built
    scene = json.loads((d / "source/scene.json").read_text())
    names = [e["name"] for e in scene["empties"]]
    assert {f"AC_START_{i}" for i in range(6)} <= set(names)
    assert {f"AC_PIT_{i}" for i in range(6)} <= set(names)
    assert {"AC_TIME_0_L", "AC_TIME_0_R", "AC_HOTLAP_START_0"} <= set(names)
    for e in scene["empties"]:
        if not e["name"].startswith(("AC_START", "AC_PIT")):
            continue
        p = np.array(e["position"])
        dist = np.linalg.norm(b.P[:, [0, 2]] - p[[0, 2]], axis=1)
        i = int(np.argmin(dist))
        assert dist[i] < b.cfg.road_width / 2
        t = np.array([b.t2[i, 0], 0, b.t2[i, 1]])
        assert np.dot(t, e["forward"]) > 0.95


def test_start_line_is_on_a_straight(built):
    b, _ = built
    s = b._start_line()
    i = int(round(s / b.ds))
    # the grid area behind the line should be nearly straight (< ~2 deg per 10 m)
    window = np.arange(i - int(40 / b.ds), i + 1) % len(b.P)
    assert np.abs(b.kappa[window]).max() < np.radians(2) / 10


def test_map_ini_maps_world_to_pixels(built):
    b, d = built
    ini = dict(ln.split("=") for ln in (d / "data/map.ini").read_text().splitlines() if "=" in ln)
    sf, xo, zo = float(ini["SCALE_FACTOR"]), float(ini["X_OFFSET"]), float(ini["Z_OFFSET"])
    px = (b.P[:, 0] + xo) / sf
    pz = (b.P[:, 2] + zo) / sf
    assert px.min() >= 0 and px.max() <= int(ini["WIDTH"])
    assert pz.min() >= 0 and pz.max() <= int(ini["HEIGHT"])


def test_point_to_point(tmp_path):
    cfg = TrackConfig(name="Stage", waypoints=[(46.0, 11.0), (46.005, 11.004), (46.01, 11.0)], closed=False, pitboxes=4)
    d = TrackBuilder(cfg, OfflineSource(log=quiet), tmp_path, log=quiet).build()
    names = [e["name"] for e in json.loads((d / "source/scene.json").read_text())["empties"]]
    assert {"AC_AB_START_L", "AC_AB_START_R", "AC_AB_FINISH_L", "AC_AB_FINISH_R"} <= set(names)


class FakeHTTP:
    """Stands in for urllib.request.urlopen and returns canned Google responses."""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=None):
        from urllib.parse import parse_qs, urlparse

        url = req.full_url
        self.calls.append(url)
        if "computeRoutes" in url:
            body = json.loads(req.data)
            pts = [body["origin"]] + body["intermediates"] + [body["destination"]]
            ll = [(p["location"]["latLng"]["latitude"], p["location"]["latLng"]["longitude"]) for p in pts]
            data = {"routes": [{"polyline": {"encodedPolyline": geo.encode_polyline(ll)}}]}
        elif "elevation" in url:
            locs = geo.decode_polyline(parse_qs(urlparse(url).query)["locations"][0][4:])
            data = {"status": "OK", "results": [{"elevation": 10.0} for _ in locs]}
        elif "metadata" in url:
            data = {"status": "OK", "pano_id": "abc"}
        else:
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, "JPEG")
            return io.BytesIO(buf.getvalue())
        return io.BytesIO(json.dumps(data).encode())


def test_google_source_batching_caching_and_cap(tmp_path, monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr("urllib.request.urlopen", fake)
    src = GoogleSource("KEY", tmp_path, max_images=2, log=quiet)

    wps = [(51.5 + i * 1e-3, -0.1) for i in range(40)]
    path = src.route(wps, closed=True)
    assert len([c for c in fake.calls if "computeRoutes" in c]) == 2  # 41 points -> two legs
    assert path[0] == path[-1]

    pts = [(51.5 + i * 1e-4, -0.1) for i in range(300)]
    fake.calls.clear()
    elev = src.elevations(pts)
    assert len(elev) == 300 and len(fake.calls) == 2
    src.elevations(pts)
    assert len(fake.calls) == 2  # cached

    assert src.streetview_meta(51.5, -0.1)["pano_id"] == "abc"
    imgs = [src.streetview_image("abc", h, 90) for h in (0, 90, 180)]
    assert imgs[0] is not None and imgs[1] is not None and imgs[2] is None  # capped at 2
    assert src.streetview_image("abc", 0, 90) is not None  # cache hit is free
    assert all("key=KEY" in c for c in fake.calls if "maps.googleapis" in c)
    assert not any("KEY" in p.read_bytes().decode(errors="ignore") for p in tmp_path.iterdir())


def test_full_build_with_mocked_google(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", FakeHTTP())
    src = GoogleSource("KEY", tmp_path / "cache", log=quiet)
    d = TrackBuilder(TrackConfig(name="Mocked", waypoints=LOOP), src, tmp_path, log=quiet).build()
    assert (d / "source/textures/sv_atlas_00.png").exists()
