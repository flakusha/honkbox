"""honkbox Blender asset recipe (FEAT-blender-asset-preflight).

Reads a JSON job description, applies the requested stages headlessly, writes
the produced artifacts and a result JSON. The recipe never opens a UI and
never touches the user's Blender profile: the runner passes
``--factory-startup``.

Progress is reported as ``HONKBOX_PROGRESS={json}`` lines on stdout; the
result JSON (produced artifacts + warnings) is written to the path the job
names.

"""

import json
import math
import os
import sys

import bpy

PROGRESS_PREFIX = "HONKBOX_PROGRESS="

# Automatic retopology yields a BASE mesh. The job never promises a
# deformation rig for a hero character.
BASE_MESH_WARNING = (
    "the retopologized mesh is a base mesh; no hero-character deformation "
    "rig is implied"
)

# Bake types the recipe knows, keyed by the job's map name.
BAKE_TYPES = {
    "color": "DIFFUSE",
    "normal": "NORMAL",
    "roughness": "ROUGHNESS",
}


class RecipeError(RuntimeError):
    """A named, reportable failure."""


def progress(stage, event="start", **fields):
    """Report one progress event."""
    payload = {"stage": stage, "event": event}
    payload.update(fields)
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def parse_paths(argv):
    """The job JSON path and the optional result path.

    Both come after the ``--`` separator: the job description first, the
    result path second.
    """
    args = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    if not args:
        raise RecipeError("no job description argument")
    result = args[1] if len(args) > 1 else None
    return args[0], result


def active_object():
    """The active object, or the first mesh in the scene."""
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        for candidate in bpy.data.objects:
            if candidate.type == "MESH":
                return candidate
    return obj


def clear_scene():
    """Drop the startup scene (Blender opens with a cube, a camera and a light)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def build_source(source):
    """Put the source mesh into the scene; return the mesh object."""
    kind = source.get("kind", "primitive")
    if kind == "primitive":
        primitive = source.get("primitive", "cube")
        size = float(source.get("size", 2.0))
        if primitive == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=size / 2.0, segments=16, ring_count=8
            )
        elif primitive == "ico_sphere":
            bpy.ops.mesh.primitive_ico_sphere_add(radius=size / 2.0, subdivisions=2)
        else:
            bpy.ops.mesh.primitive_cube_add(size=size)
        obj = active_object()
        if obj is None:
            raise RecipeError("the primitive produced no object")
        obj.name = "honkbox_asset"
        return obj

    path = source.get("path", "")
    fmt = source.get("format", "glb")
    if not path or not os.path.isfile(path):
        raise RecipeError("source mesh not found: %s" % path)
    if fmt == "glb":
        bpy.ops.import_scene.gltf(filepath=path)
    elif fmt == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif fmt == "ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RecipeError("unsupported source format: %s" % fmt)
    obj = active_object()
    if obj is None:
        raise RecipeError("the imported mesh has no active object")
    return obj


def unwrap(obj, margin):
    """Smart UV project, in edit mode."""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66.0), island_margin=float(margin)
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def ensure_uv(obj):
    """A UV map on the mesh (unwrapped when it has none)."""
    if not obj.data.uv_layers:
        unwrap(obj, 0.0)
    return obj.data.uv_layers[0]


def material_of(obj):
    """The object's material, created when it has none."""
    if not obj.data.materials:
        obj.data.materials.append(bpy.data.materials.new(name="honkbox"))
    material = obj.data.materials[0]
    material.use_nodes = True
    return material


def principled_of(tree):
    """The Principled BSDF node of a material tree."""
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def attach_texture(obj, image_path, node_name):
    """Attach an image through a ``ShaderNodeTexImage``."""
    if not os.path.isfile(image_path):
        raise RecipeError("texture not found: %s" % image_path)
    tree = material_of(obj).node_tree
    node = tree.nodes.get(node_name) or tree.nodes.new("ShaderNodeTexImage")
    node.name = node_name
    node.image = bpy.data.images.load(image_path, check_existing=True)
    node.location = (-400.0, 300.0)
    principled = principled_of(tree)
    if principled is not None:
        tree.links.new(node.outputs["Color"], principled.inputs["Base Color"])
    return node


def bake_maps(obj, maps, resolution, output_dir, prefix):
    """Bake the requested maps to PNG files; return their paths."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    ensure_uv(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    tree = material_of(obj).node_tree
    written = []
    for map_name in maps:
        if map_name not in BAKE_TYPES:
            raise RecipeError("unsupported bake map: %s" % map_name)
        image = bpy.data.images.new(
            name="honkbox_%s" % map_name, width=int(resolution), height=int(resolution)
        )
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.location = (-400.0, -200.0)
        # The bake target stays UNLINKED while it is written: linking it to
        # the input it bakes is a circular dependency, and the source texture
        # (the one honkbox attached) must survive the bake.
        tree.nodes.active = node
        bpy.ops.object.bake(type=BAKE_TYPES[map_name])
        path = os.path.join(output_dir, "%s_%s.png" % (prefix, map_name))
        image.filepath_raw = path
        image.file_format = "PNG"
        image.save()
        written.append(path)
    return written


def voxel_remesh(obj, voxel_size):
    """Voxel remesh through a REMESH modifier (deterministic)."""
    modifier = obj.modifiers.new(name="honkbox_remesh", type="REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = float(voxel_size)
    modifier.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def export_glb(obj, path):
    """Export the scene as a binary glTF."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    return path


def stage_uv_unpack(obj, op):
    """Smart UV project."""
    unwrap(obj, op.get("margin", 0.0))
    return []


def stage_uv_pack(warnings):
    """Pack the file-backed images into the blend."""
    packed = 0
    for image in bpy.data.images:
        if image.source == "FILE":
            image.pack()
            packed += 1
    warnings.append("packed %d image(s) into the blend" % packed)
    return []


def stage_voxel_remesh(obj, op, warnings):
    """Voxel remesh (deterministic: no random seed)."""
    voxel_remesh(obj, op.get("voxel_size", 0.1))
    warnings.append(BASE_MESH_WARNING)
    return []


def stage_quadriflow(obj, op, warnings):
    """Quadriflow retopology, where the Blender build still has it."""
    try:
        bpy.ops.object.quadriflow(target_faces=int(op.get("target_faces", 1000)))
    except AttributeError:
        # Blender 4.5 dropped the operator; the stage is skipped, not failed.
        warnings.append("quadriflow is not in this Blender build; the stage was skipped")
        return []
    warnings.append(BASE_MESH_WARNING)
    return []


def stage_attach_texture(obj, op):
    """Attach a texture through a ShaderNodeTexImage."""
    attach_texture(obj, op["image"], op.get("node", "honkbox_texture"))
    return []


def stage_bake(obj, op, job):
    """Bake the requested maps to PNG files."""
    paths = bake_maps(
        obj,
        op.get("maps", ["color"]),
        int(op.get("resolution", 256)),
        job["output_dir"],
        job.get("prefix", "honkbox_asset"),
    )
    return [{"path": path, "kind": "map"} for path in paths]


def stage_export_glb(obj, op, job):
    """Export the scene as a binary glTF."""
    path = os.path.join(job["output_dir"], "%s.glb" % job.get("prefix", "honkbox_asset"))
    export_glb(obj, path)
    return [{"path": path, "kind": "model"}]


def apply_stage(obj, op, job, artifacts, warnings):
    """Run one stage; return the artifacts it produced."""
    stage = op.get("op", "")
    if stage == "uv_unpack":
        return stage_uv_unpack(obj, op)
    if stage == "uv_pack":
        return stage_uv_pack(warnings)
    if stage == "voxel_remesh":
        return stage_voxel_remesh(obj, op, warnings)
    if stage == "quadriflow":
        return stage_quadriflow(obj, op, warnings)
    if stage == "attach_texture":
        return stage_attach_texture(obj, op)
    if stage == "bake":
        return stage_bake(obj, op, job)
    if stage == "export_glb":
        return stage_export_glb(obj, op, job)
    raise RecipeError("unknown stage: %s" % stage)


def cancelled(job):
    """Whether the runner asked the job to stop.

    The runner drops a marker file; the recipe only looks for it BETWEEN
    stages, so a bake in flight is never interrupted mid-write.
    """
    path = job.get("cancel_path")
    return bool(path) and os.path.isfile(path)


def write_result(result_path, artifacts, warnings, cancelled_now=False):
    """Write the result JSON the runner reads."""
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifacts": artifacts,
                "warnings": warnings,
                "cancelled": cancelled_now,
                "blender_version": bpy.app.version_string,
            },
            handle,
        )


def main():
    """Run the recipe described by the job JSON."""
    job_path, result_arg = parse_paths(sys.argv)
    with open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    result_path = (
        result_arg
        or job.get("result_path")
        or os.path.join(output_dir, "result.json")
    )
    warnings = []
    artifacts = []

    progress("source")
    clear_scene()
    obj = build_source(job["source"])
    progress("source", event="done")

    for op in job.get("stages", []):
        stage = op.get("op", "")
        if cancelled(job):
            write_result(result_path, artifacts, warnings, cancelled_now=True)
            progress(stage, event="cancelled")
            return
        progress(stage)
        artifacts.extend(apply_stage(obj, op, job, artifacts, warnings))
        progress(stage, event="done")

    write_result(result_path, artifacts, warnings)
    progress("done", event="done")


if __name__ == "__main__":
    main()
