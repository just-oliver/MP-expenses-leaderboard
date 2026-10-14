"""Blender script: OBJ + scene.json -> FBX ready for Assetto Corsa's ksEditor.

Usage:  blender -b --python blender_build_fbx.py -- <source_dir>

Runs inside Blender's own Python, so it must not import sv2ac.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def ac_to_blender(v):
    """AC/FBX space (x, y-up, z) -> Blender space (x, y, z-up)."""
    x, y, z = v
    return Vector((x, -z, y))


def main(src: Path):
    scene = json.loads((src / "scene.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    obj_path = str(src / scene["obj"])
    if hasattr(bpy.ops.wm, "obj_import"):  # Blender 3.3+
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="NEGATIVE_Z", up_axis="Y")
    else:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="-Z", axis_up="Y")

    # AC expects spawn/timing empties with local +Z pointing along the track and +Y up
    # (in FBX/AC space). The FBX exporter converts Blender space back with the inverse
    # of ac_to_blender, so build each empty's basis directly in converted coordinates.
    for e in scene["empties"]:
        fwd = Vector(e["forward"]).normalized()
        up = Vector(e["up"]).normalized()
        x_axis = up.cross(fwd).normalized()
        cols = [ac_to_blender(x_axis), ac_to_blender(up), ac_to_blender(fwd)]
        m = Matrix((
            (cols[0].x, cols[1].x, cols[2].x, 0),
            (cols[0].y, cols[1].y, cols[2].y, 0),
            (cols[0].z, cols[1].z, cols[2].z, 0),
            (0, 0, 0, 1),
        ))
        m.translation = ac_to_blender(e["position"])
        empty = bpy.data.objects.new(e["name"], None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 2.0
        empty.matrix_world = m
        bpy.context.scene.collection.objects.link(empty)

    fbx_path = str(src / scene["fbx"])
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        object_types={"MESH", "EMPTY"},
        mesh_smooth_type="FACE",
        path_mode="RELATIVE",
        bake_space_transform=False,
        add_leaf_bones=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(src / (Path(scene["fbx"]).stem + ".blend")))
    print(f"sv2ac: wrote {fbx_path} with {len(scene['empties'])} empties")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main(Path(argv[0] if argv else Path(__file__).parent).resolve())
