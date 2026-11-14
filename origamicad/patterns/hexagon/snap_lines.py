"""Snap lines of the hexagon pattern and the exact folds they allow.

With rigid panels and horizontal triangles, triangles can only translate, and
a parallelogram moves its two triangles apart by its flat width times the
cosine of its fold angle. A parallelogram that has snapped through 90 degrees
has the supplementary dihedral angle, so it contributes the same distance
with the opposite sign.

Going once around a unit chain these six contributions must cancel. Their
directions are 60 degrees apart, so, leaving aside the alternating state of
an isolated unit, they cancel only if opposite parallelograms of the unit are
in the same state. Opposite parallelograms are parallel, and each one is
shared with the next unit chain in that direction. The state therefore
carries on from unit to unit: all parallelograms that cross one row of unit
chains snap together. Such a set is a *snap line*.

Snapping a line moves the two parts of the pattern on either side of the row
towards each other, without straining any panel. The folded shape for any
choice of snapped lines is known in closed form.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np

from .metadata import hex_constraint_plan, iter_local_creases


def _unit_quads(owner, unit) -> dict[int, str]:
    """Map a unit's local parallelogram positions (0-5) to panel IDs."""
    quads: dict[int, str] = {}
    for crease in iter_local_creases(owner, unit):
        index = crease["local_index"]
        position = (index - 1) % 6 if crease["side"] == "previous_quad" else index
        quads[position] = crease["quad"]
    return quads


def snap_lines(owner) -> list[tuple[str, ...]]:
    """Return the snap lines of a pattern or model.

    Each line is a tuple of parallelogram IDs that can only snap together:
    opposite parallelograms of every unit chain are joined, and the joined
    sets are returned in a stable order. Every parallelogram is in exactly
    one line.
    """
    parent: dict[str, str] = {}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for unit in owner.hex_units:
        quads = _unit_quads(owner, unit)
        for quad in quads.values():
            parent.setdefault(quad, quad)
        for position in range(3):
            first, second = quads.get(position), quads.get(position + 3)
            if first is not None and second is not None:
                parent[find(first)] = find(second)
    lines: dict[str, list[str]] = defaultdict(list)
    for quad in parent:
        lines[find(quad)].append(quad)
    return [tuple(line) for line in lines.values()]


def _quad_links(owner, xy):
    """For every parallelogram: its two triangles and its flat width vector."""
    creases = defaultdict(list)
    for unit in owner.hex_units:
        for crease in iter_local_creases(owner, unit):
            entry = (crease["triangle"], tuple(crease["edge"]))
            if entry[0] not in [triangle for triangle, _ in creases[crease["quad"]]]:
                creases[crease["quad"]].append(entry)
    links = {}
    for quad, entries in creases.items():
        if len(entries) != 2:
            raise ValueError(f"Parallelogram '{quad}' must join exactly two triangles.")
        (first, edge), (second, other_edge) = entries
        surface = owner.surfaces[quad]
        cycle = list(surface["vertices"] if isinstance(surface, dict) else surface.vertices)
        start = edge[0]
        position = cycle.index(start)
        across = next(
            cycle[(position + offset) % 4] for offset in (1, -1)
            if cycle[(position + offset) % 4] in other_edge
        )
        axis = xy[edge[1]] - xy[edge[0]]
        axis /= np.linalg.norm(axis)
        side = xy[across] - xy[start]
        # Only the part of the side at right angles to the crease folds.
        links[quad] = (first, second, side - (side @ axis) * axis)
    return links


def fold_with_snapped_panels(
    model,
    dihedral_deg: float,
    snapped: Iterable[str] = (),
    *,
    valley_z: float = 0.0,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Return the exact rigid fold in which the given parallelograms are snapped.

    ``model`` is the flat ``Cadder`` of a hexagon pattern and is not changed.
    Every parallelogram is folded to ``dihedral_deg``, except those in
    ``snapped``, which take the supplementary angle. The result is the
    model's coordinate vector for that fold, with one valley triangle left
    where it is in the flat pattern.

    The states must be compatible: ``snapped`` should be a union of
    :func:`snap_lines`. Otherwise the triangles cannot close around some unit
    chain and a ``ValueError`` names the triangle where that shows.
    """
    fold = np.deg2rad(180.0 - float(dihedral_deg))
    if not 0.0 < fold < np.pi:
        raise ValueError("dihedral_deg must be between 0 and 180 degrees.")
    snapped = set(snapped)
    point_ids = model.point_ids()
    flat = model.get_coordinate_vector().reshape(-1, 3)
    xy = {pid: flat[i, :2].copy() for i, pid in enumerate(point_ids)}
    links = _quad_links(model, xy)
    unknown = snapped - links.keys()
    if unknown:
        raise ValueError(f"Unknown parallelograms: {sorted(unknown)}.")

    graph = defaultdict(list)
    for quad, (first, second, width) in links.items():
        shift = ((-1.0 if quad in snapped else 1.0) * np.cos(fold) - 1.0) * width
        graph[first].append((second, shift))
        graph[second].append((first, -shift))

    plan = hex_constraint_plan(model)
    kinds = plan.triangle_kinds
    scale = max(1.0, float(np.ptp(flat[:, :2], axis=0).max()))
    moves: dict[str, np.ndarray] = {}
    for root in sorted(graph, key=lambda sid: (kinds[sid] != "valley", sid)):
        if root in moves:
            continue
        moves[root] = np.zeros(2)
        queue = deque([root])
        while queue:
            triangle = queue.popleft()
            for neighbor, shift in graph[triangle]:
                proposed = moves[triangle] + shift
                known = moves.get(neighbor)
                if known is None:
                    moves[neighbor] = proposed
                    queue.append(neighbor)
                elif float(np.linalg.norm(known - proposed)) > tolerance * scale:
                    raise ValueError(
                        "These panel states are not a rigid fold: the triangles "
                        f"do not close at '{neighbor}'."
                    )

    width = float(np.mean([np.linalg.norm(link[2]) for link in links.values()]))
    height = width * float(np.sin(fold))
    folded = flat.copy()
    index = {pid: i for i, pid in enumerate(point_ids)}
    for triangle, move in moves.items():
        z = valley_z + (height if kinds[triangle] == "mountain" else 0.0)
        for pid in model._surface_vertices(triangle):
            row = index[pid]
            folded[row, :2] = flat[row, :2] + move
            folded[row, 2] = z
    return folded.ravel()
