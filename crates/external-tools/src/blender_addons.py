"""Blender add-on and operator probe (honkbox).

Executed as ``blender -b -noaudio --python-expr <this file>`` and prints one
JSON payload after the ``HONKBOX_ADDONS=`` marker. The probe reports FACTS
only - which add-ons are installed, what they declare, and whether a candidate
operator exists and passes its poll with a prepared scene. honkbox decides
the tiers and the headless verdict in Rust, where both are unit-tested.

The probe never executes an operator: execution can write files, and a probe
must not. ``poll()`` has no side effects and is the honest signal.
"""

import addon_utils
import json
import os
import sys

import bpy

try:
    import tomllib
except ImportError:  # Blender built on Python < 3.11 has no tomllib
    tomllib = None

# Candidate operators, injected by the Rust side (see BASELINE_OPERATORS).
CANDIDATES = globals().get("HONKBOX_CANDIDATES", [])


def _manifest(path):
    """The extension manifest above ``path``, when the add-on ships one."""
    if tomllib is None or not path:
        return None
    directory = os.path.dirname(os.path.abspath(path))
    while True:
        candidate = os.path.join(directory, "blender_manifest.toml")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "rb") as handle:
                    return tomllib.load(handle)
            except Exception:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def _enabled(name):
    """Whether the add-on is enabled in the probed profile."""
    try:
        return bool(addon_utils.check(name)[0])
    except Exception:
        return name in bpy.context.preferences.addons


def _addons():
    """Every installed add-on with its declared version and license."""
    modules = {}
    try:
        for module in addon_utils.modules():
            modules[module.__name__] = module
    except Exception:
        pass
    for name in list(bpy.context.preferences.addons.keys()):
        modules.setdefault(name, sys.modules.get(name))

    records = []
    for name, module in sorted(modules.items()):
        try:
            info = getattr(module, "bl_info", None) or {}
            path = getattr(module, "__file__", "") or ""
            manifest = _manifest(path) or {}
            version = manifest.get("version")
            if not version:
                version = ".".join(str(part) for part in (info.get("version") or ()))
            licenses = manifest.get("license") or []
            license_id = info.get("license") or (", ".join(str(each) for each in licenses) or None)
            records.append(
                {
                    "id": str(manifest.get("id") or name),
                    "file": path,
                    "version": str(version) if version else None,
                    "license": str(license_id) if license_id else None,
                    "enabled": _enabled(name),
                }
            )
        except Exception:
            continue
    records.sort(key=lambda record: record["id"])
    return records


def _prepare_scene():
    """A cube with a UV map, so data polls can pass headless."""
    try:
        bpy.ops.mesh.primitive_cube_add()
        active = bpy.context.active_object
        if active is not None:
            active.data.uv_layers.new(name="UVMap")
    except Exception:
        pass


def _mode(mode):
    """Switch the edit mode; ``False`` when the switch is impossible."""
    try:
        bpy.ops.object.mode_set(mode=mode)
        return True
    except Exception:
        return False


def _resolve(candidate):
    """The operator object for ``namespace.name``, or ``None``."""
    namespace, _, name = candidate.partition(".")
    module = getattr(bpy.ops, namespace, None)
    if module is None or name not in dir(module):
        return None
    return getattr(module, name)


def _poll(operator):
    """Whether the operator's poll passes in the current context."""
    try:
        return bool(operator.poll())
    except Exception:
        return False


def _operators(candidates):
    """Candidate operators: present in this build and polling headless."""
    _prepare_scene()
    records = {
        candidate: {
            "id": candidate,
            "available": False,
            "object_poll": False,
            "edit_poll": False,
        }
        for candidate in candidates
    }
    for candidate in candidates:
        operator = _resolve(candidate)
        if operator is None:
            continue
        records[candidate]["available"] = True
        records[candidate]["object_poll"] = _poll(operator)
    if _mode("EDIT"):
        for candidate in candidates:
            operator = _resolve(candidate)
            if operator is not None:
                records[candidate]["edit_poll"] = _poll(operator)
        _mode("OBJECT")
    return [records[candidate] for candidate in sorted(records)]


PAYLOAD = {
    "scripts": bpy.utils.system_resource("SCRIPTS") or "",
    "addons": _addons(),
    "operators": _operators(CANDIDATES),
}
print("HONKBOX_ADDONS=" + json.dumps(PAYLOAD))
