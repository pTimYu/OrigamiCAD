"""Type definitions shared by hexagon layout and kinematic operations."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Literal, TypeAlias, TypedDict

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
    owner.hex_units = normalized_units
    owner.hex_creases = normalized_registry


def load_hex_metadata(owner, metadata):
    owner.hex_units = deepcopy(metadata.get("hex_units", []))
    owner.hex_creases = deepcopy(metadata.get("hex_creases", {}))
    normalize_hex_creases(owner)
