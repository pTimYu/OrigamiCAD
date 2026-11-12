"""Type definitions shared by hexagon layout and kinematic operations."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from ...core.topology import panel_rigidity_plan, surface_topology

Coordinate: TypeAlias = tuple[float, float]
PointID: TypeAlias = str
SurfaceID: TypeAlias = str
CreaseKind: TypeAlias = Literal["mountain", "valley"]
CreaseSide: TypeAlias = Literal["previous_quad", "current_quad"]


class TriangleKind(TypedDict):
    surface: SurfaceID
    kind: CreaseKind
    local_index: int
    unit: int


class LocalCrease(TypedDict):
    """Resolved local view, also the format used by older metadata files."""
    unit: int
    local_index: int
    edge: list[PointID]
    triangle: SurfaceID
    quad: SurfaceID
    kind: CreaseKind
    side: CreaseSide


class PhysicalCrease(TypedDict):
    edge: list[PointID]
    triangle: SurfaceID
    quad: SurfaceID
    kind: CreaseKind


class CreaseReference(TypedDict):
    crease: str
    local_index: int
    side: CreaseSide
    reversed: bool


class HexUnit(TypedDict):
    count: int
    mid: list[PointID]
    side: list[PointID]
    triangles: list[SurfaceID]
    parallelograms: list[SurfaceID]
    surfaces: list[SurfaceID]
    triangle_kinds: list[TriangleKind]
    local_creases: list[CreaseReference]


def crease_key(crease):
    return tuple(sorted(crease["edge"])), crease["triangle"], crease["quad"]


@dataclass(frozen=True)
class HexConstraintPlan:
    """Validated metadata and ordered physical constraints for one topology."""

    triangle_kinds: dict[str, CreaseKind]
    triangles: tuple[str, ...]
    crease_entries: tuple
    skipped: tuple
    shared: tuple


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _line_records(owner):
    for lid, line in owner.lines.items():
        if isinstance(line, dict):
            yield lid, line["start"], line["end"], line["kind"]
        else:
            yield lid, line.start, line.end, line.kind


def _plan_signature(owner):
    # Public dictionaries and nested vertex/reference lists can be edited
    # directly. Compare values, not object IDs or lengths. Exclude XYZ so
    # continuation and coordinate updates reuse the compiled plan.
    return (tuple(owner.points), surface_topology(owner), tuple(_line_records(owner)),
            _freeze(getattr(owner, "hex_units", [])),
            _freeze(getattr(owner, "hex_creases", {})))


def _compile_constraint_plan(owner, units, registry):
    surfaces = dict(surface_topology(owner))
    triangle_kinds = {}
    for unit in units:
        for record in unit.get("triangle_kinds", []):
            sid, kind = record.get("surface"), record.get("kind")
            if sid not in surfaces or len(surfaces[sid]) != 3:
                raise ValueError(f"Triangle metadata references missing or non-triangle surface '{sid}'.")
            if kind not in {"mountain", "valley"}:
                raise ValueError(f"Invalid mountain/valley metadata for triangle '{sid}'.")
            if record.get("unit", unit["count"]) != unit["count"]:
                raise ValueError(f"Inconsistent unit in triangle metadata for '{sid}'.")
            if sid in triangle_kinds and triangle_kinds[sid] != kind:
                raise ValueError(f"Conflicting mountain/valley metadata for triangle '{sid}'.")
            triangle_kinds[sid] = kind
    required = {sid for unit in units for sid in unit.get("triangles", [])}
    missing = required - triangle_kinds.keys()
    if missing:
        raise ValueError(f"Missing triangle_kinds metadata for {sorted(missing)}. Rebuild the 2D metadata.")

    owners = {}
    for unit in units:
        for ref in unit.get("local_creases", []):
            owners.setdefault(ref["crease"], []).append(
                (unit["count"], ref["local_index"], ref["side"]))
    line_kinds = {}
    for _, a, b, kind in _line_records(owner):
        line_kinds.setdefault(tuple(sorted((a, b))), kind)
    for sid, expected_kind in triangle_kinds.items():
        vertices = surfaces[sid]
        for i, a in enumerate(vertices):
            kind = line_kinds.get(tuple(sorted((a, vertices[(i + 1) % 3]))))
            if kind in {"mountain", "valley"} and kind != expected_kind:
                raise ValueError(
                    f"Conflicting mountain/valley metadata: triangle '{sid}' has mixed crease kinds "
                    "or disagrees with triangle_kinds."
                )
    entries, skipped, shared = [], [], []
    edge_panels = {}
    for cid, labels in owners.items():
        crease = registry[cid]
        a, b = crease["edge"]
        tri, quad = crease["triangle"], crease["quad"]
        missing_geometry = None
        if tri not in surfaces:
            missing_geometry = "missing triangle", tri
        elif quad not in surfaces:
            missing_geometry = "missing quad", quad
        elif a not in owner.points or b not in owner.points:
            missing_geometry = ("missing edge points",)
        if missing_geometry:
            skipped.extend((*label, *missing_geometry) for label in labels)
            continue
        if triangle_kinds.get(tri) != crease["kind"]:
            raise ValueError(f"Conflicting mountain/valley or missing triangle metadata for crease '{cid}'.")
        edge = tuple(sorted((a, b)))
        for sid, size in ((tri, 3), (quad, 4)):
            vertices = surfaces[sid]
            boundary = {tuple(sorted((p, vertices[(i + 1) % len(vertices)])))
                        for i, p in enumerate(vertices)}
            if len(vertices) != size or edge not in boundary:
                raise ValueError(f"Crease '{cid}' is not a boundary edge of panel '{sid}'.")
        if edge in edge_panels and edge_panels[edge] != (tri, quad):
            raise ValueError(f"Conflicting adjacent panels for crease edge {edge}.")
        edge_panels[edge] = tri, quad
        # Validate line/metadata consistency; never infer a classification
        # from geometry or from the digits in an element's name.
        if line_kinds.get(edge) in {"mountain", "valley"} and line_kinds[edge] != crease["kind"]:
            raise ValueError(f"Conflicting mountain/valley assignment for crease '{cid}'.")
        entries.append((_physical_definition(crease), labels[0]))
        shared.extend((*label, tri, quad) for label in labels[1:])
    panel_rigidity_plan(owner)
    return HexConstraintPlan(
        triangle_kinds, tuple(sid for sid, vertices in surfaces.items() if len(vertices) == 3),
        tuple(entries), tuple(skipped), tuple(shared),
    )


def hex_constraint_plan(owner):
    """Reuse the build/import plan, or refresh it after a topology edit.

    Existing triangle_kinds and physical crease metadata remain the source
    of classifications. Older inline crease records are normalized once;
    absent triangle labels are an error, not an invitation to scan lines.
    """
    cached = getattr(owner, "_hex_constraint_cache", None)
    if cached is not None and cached[0] == _plan_signature(owner):
        return cached[1]
    normalize_hex_creases(owner)
    return owner._hex_constraint_cache[1]


def _validate_physical(crease):
    try:
        edge = crease["edge"]
        if (not isinstance(edge, (list, tuple)) or len(edge) != 2 or edge[0] == edge[1]
                or not all(isinstance(pid, str) for pid in edge)
                or not isinstance(crease["triangle"], str)
                or not isinstance(crease["quad"], str)):
            raise ValueError("Invalid physical crease geometry.")
        if crease["kind"] not in {"mountain", "valley"}:
            raise ValueError("A physical crease must be mountain or valley.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete physical crease metadata.") from exc


def _physical_definition(crease):
    return {"edge": list(crease["edge"]), "triangle": crease["triangle"],
            "quad": crease["quad"], "kind": crease["kind"]}


@contextmanager
def indexed_creases(pattern):
    """Index one append-only layout operation, including nested unit builders."""
    if getattr(pattern, "_hex_crease_index", None) is not None:
        yield
        return
    registry = getattr(pattern, "hex_creases", {})
    pattern.hex_creases = registry
    index = {}
    for cid, crease in registry.items():
        _validate_physical(crease)
        key = crease_key(crease)
        if key in index:
            raise ValueError(f"Duplicate physical crease definitions: {index[key]}, {cid}.")
        index[key] = cid
    pattern._hex_crease_index = index
    try:
        yield
    finally:
        pattern._hex_crease_index = None


def register_crease(pattern, local):
    """Store a hinge once; return its unit-local reference and orientation."""
    _validate_physical(local)
    key = crease_key(local)
    index = pattern._hex_crease_index
    cid = index.get(key)
    if cid is None:
        cid = f"crease_u{local['unit']}_i{local['local_index']}_{local['side']}"
        if cid in pattern.hex_creases:
            raise ValueError(f"Crease ID '{cid}' describes different geometry.")
        pattern.hex_creases[cid] = _physical_definition(local)
        index[key] = cid
    physical = pattern.hex_creases[cid]
    if physical["kind"] != local["kind"]:
        raise ValueError(f"Conflicting mountain/valley definitions for crease '{cid}'.")
    return {"crease": cid, "local_index": local["local_index"], "side": local["side"],
            "reversed": list(local["edge"]) != physical["edge"]}


def iter_local_creases(owner, unit):
    """Resolve unit references without changing their local edge direction.

    Older inline crease dictionaries are also accepted. Resolved dictionaries
    are snapshots for reading; edit physical definitions through ``hex_creases``.
    """
    registry = getattr(owner, "hex_creases", {})
    for ref in unit.get("local_creases", []):
        if "crease" not in ref:
            yield ref
            continue
        cid = ref["crease"]
        if not isinstance(cid, str) or cid not in registry:
            raise ValueError(f"Unknown physical crease reference '{cid}'.")
        if any(key in ref for key in ("edge", "triangle", "quad", "kind")):
            raise ValueError(f"Crease reference '{cid}' also contains an inline definition.")
        if not isinstance(ref.get("reversed", False), bool):
            raise ValueError(f"Invalid local direction for crease '{cid}'.")
        if "local_index" not in ref or "side" not in ref:
            raise ValueError(f"Incomplete local reference to crease '{cid}'.")
        physical = registry[cid]
        _validate_physical(physical)
        edge = list(physical["edge"])
        if ref.get("reversed", False):
            edge.reverse()
        yield {**physical, "edge": edge, "unit": unit["count"],
               "local_index": ref["local_index"], "side": ref["side"]}


def normalize_hex_creases(owner):
    """Upgrade legacy inline records and prune unreferenced physical creases.

    Validate before replacing metadata. Existing registry IDs survive normal
    loads and cavity removal; consistent legacy descriptions are interned.
    """
    registry = getattr(owner, "hex_creases", {})
    units = getattr(owner, "hex_units", [])
    normalized_registry, index, normalized_units = {}, {}, []
    counts = set()
    for unit in units:
        count = unit["count"]
        if count in counts:
            raise ValueError(f"Duplicate hex-unit count '{count}'.")
        counts.add(count)
        refs, local_labels = [], set()
        for source, local in zip(unit.get("local_creases", []), iter_local_creases(owner, unit)):
            _validate_physical(local)
            if (type(local.get("local_index")) is not int
                    or not 0 <= local["local_index"] < 6
                    or local.get("side") not in ("previous_quad", "current_quad")):
                raise ValueError(f"Invalid local crease label in unit {count}.")
            label = local["local_index"], local["side"]
            if local.get("unit", count) != count or label in local_labels:
                raise ValueError(f"Duplicate or inconsistent local crease in unit {count}: {label}.")
            local_labels.add(label)
            key = crease_key(local)
            cid = index.get(key)
            if cid is None:
                cid = source.get("crease", f"crease_u{count}_i{label[0]}_{label[1]}")
                if cid in normalized_registry:
                    raise ValueError(f"Crease ID '{cid}' describes different geometry.")
                physical = registry[cid] if "crease" in source else local
                normalized_registry[cid] = _physical_definition(physical)
                index[key] = cid
            physical = normalized_registry[cid]
            if physical["kind"] != local["kind"]:
                raise ValueError(f"Conflicting mountain/valley definitions for crease '{cid}'.")
            refs.append({"crease": cid, "local_index": label[0], "side": label[1],
                         "reversed": list(local["edge"]) != physical["edge"]})
        normalized_units.append({**unit, "local_creases": refs})
    # Compile after layout merging/cavity removal, or while loading metadata.
    # Validate before publishing either the normalized records or the plan.
    plan = _compile_constraint_plan(owner, normalized_units, normalized_registry)
    owner.hex_units = normalized_units
    owner.hex_creases = normalized_registry
    owner._hex_constraint_cache = _plan_signature(owner), plan


def load_hex_metadata(owner, metadata):
    owner.hex_units = deepcopy(metadata.get("hex_units", []))
    owner.hex_creases = deepcopy(metadata.get("hex_creases", {}))
    normalize_hex_creases(owner)
