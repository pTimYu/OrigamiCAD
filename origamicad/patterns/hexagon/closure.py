"""Close a pattern that is open on one side with its half-turn continuation.

A wall that surrounds a cavity on three sides cannot carry a pressure applied
"from all sides": its outer boundary has no inside. The missing half is taken
to be the same lattice continued over the 180-degree rotation of the pattern
about a point on its open side. Unlike a mirror image, a rotated copy has the
handedness of the lattice, so the two halves join panel to panel.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from ...core.two_d_drawer import TwoDDrawer
from .layout import hex_unit_chain
from .metadata import normalize_hex_creases

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


class ClosedPattern(NamedTuple):
    """A closed pattern and how it relates to the open one it came from.

    ``pattern`` is drawn in the lattice's own frame. A point of it maps to the
    open pattern's frame by ``rotation @ (x, y) + translation``. The units,
    panels, and points of the open pattern keep their IDs; ``added_units``
    are the unit counts of the continuation.
    """

    pattern: TwoDDrawer
    rotation: np.ndarray
    translation: np.ndarray
    added_units: tuple[int, ...]


def _lattice(pattern):
    """Return the lattice frame and each complete unit's integer cell."""
    units = [unit for unit in pattern.hex_units if len(unit["mid"]) == 6]
    if len(units) != len(pattern.hex_units) or not units:
        raise ValueError("Half-turn closure needs a pattern of complete unit chains.")

    def xy(point_id):
        point = pattern.points[point_id]
        return np.array([point.x, point.y], dtype=float)

    first, second = xy(units[0]["mid"][0]), xy(units[0]["mid"][1])
    side = float(np.linalg.norm(second - first))
    angle = math.atan2(second[1] - first[1], second[0] - first[0])
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)],
    ])
    height = side * math.sqrt(3.0) / 2.0
    # Two lattice vectors between neighboring void centers, 60 degrees apart.
    basis = np.array([[2.5 * side, 2.0 * side], [-height, 2.0 * height]])
    origins = [rotation.T @ xy(unit["mid"][0]) for unit in units]
    cells = []
    for origin in origins:
        coordinates = np.linalg.solve(basis, origin - origins[0])
        cell = np.rint(coordinates)
        if float(np.max(np.abs(coordinates - cell))) > 1e-6:
            raise ValueError("The unit chains do not lie on one hexagon lattice.")
        cells.append((int(cell[0]), int(cell[1])))
    reverse = units[0]["triangle_kinds"][0]["kind"] == "valley"
    return units, side, rotation, basis, origins[0], cells, reverse


def _enclosed_components(cells: set) -> list[set]:
    """Return the groups of empty cells that the given cells surround."""
    low_i = min(i for i, _ in cells) - 1
    high_i = max(i for i, _ in cells) + 1
    low_j = min(j for _, j in cells) - 1
    high_j = max(j for _, j in cells) + 1
    empty = {
        (i, j) for i in range(low_i, high_i + 1) for j in range(low_j, high_j + 1)
    } - cells

    def flood(start):
        component, stack = {start}, [start]
        while stack:
            i, j = stack.pop()
            for di, dj in _NEIGHBORS:
                cell = (i + di, j + dj)
                if cell in empty and cell not in component:
                    component.add(cell)
                    stack.append(cell)
        return component

    outside = flood((low_i, low_j))
    remaining = empty - outside
    components = []
    while remaining:
        component = flood(next(iter(remaining)))
        components.append(component)
        remaining -= component
    return components


def find_half_turn_center(cells) -> Optional[tuple[int, int]]:
    """Find the half-turn that closes a set of lattice cells around one cavity.

    The result ``(s, t)`` maps cell ``(i, j)`` to ``(s - i, t - j)``. The
    rotated copy must not overlap the original, and together they must
    surround exactly one group of at least two empty cells. Among such
    half-turns the one with the largest cavity is chosen, which is the copy
    that faces the original across the open side; the longest seam decides
    ties. Return None if there is none, as for a pattern with no open cavity.
    """
    cells = set(cells)
    # The copy must touch the original, so some cell maps onto an empty cell
    # next to the pattern.
    shell = {(i + di, j + dj) for i, j in cells for di, dj in _NEIGHBORS} - cells
    best = None
    for s, t in sorted({(a + c, b + d) for a, b in cells for c, d in shell}):
        image = {(s - i, t - j) for i, j in cells}
        if image & cells:
            continue
        seam = sum(
            (i + di, j + dj) in image for i, j in cells for di, dj in _NEIGHBORS
        )
        if seam < 2:
            continue
        cavities = _enclosed_components(cells | image)
        if len(cavities) == 1 and len(cavities[0]) >= 2:
            score = (len(cavities[0]), seam)
            if best is None or score > best[0]:
                best = (score, (s, t))
    return None if best is None else best[1]


def close_open_pattern(pattern) -> Optional[ClosedPattern]:
    """Return the pattern closed by its half-turn continuation, or None.

    None means that no half-turn of the pattern encloses a cavity: the
    pattern is already closed, or it has no cavity open on one side.
    """
    units, side, rotation, basis, origin, cells, reverse = _lattice(pattern)
    center = find_half_turn_center(cells)
    if center is None:
        return None
    closed = TwoDDrawer(unit=pattern.unit, point_tol=getattr(pattern, "point_tol", 1e-6))
    drawn = []
    for unit, cell in zip(units, cells):
        start = origin + basis @ np.array(cell, dtype=float)
        drawn.append(hex_unit_chain(
            closed, start_point=(float(start[0]), float(start[1])), l=side,
            count=int(unit["count"]), reverse=reverse,
        ))
    next_count = max(int(unit["count"]) for unit in units) + 1
    added = []
    for cell in sorted({(center[0] - i, center[1] - j) for i, j in cells}):
        start = origin + basis @ np.array(cell, dtype=float)
        drawn.append(hex_unit_chain(
            closed, start_point=(float(start[0]), float(start[1])), l=side,
            count=next_count, reverse=reverse,
        ))
        added.append(next_count)
        next_count += 1
    closed.hex_units = drawn
    normalize_hex_creases(closed)

    for surface_id, surface in pattern.surfaces.items():
        if surface_id not in closed.surfaces or (
            list(closed.surfaces[surface_id].vertices) != list(surface.vertices)
        ):
            raise ValueError(
                "The pattern could not be rebuilt cell by cell for closure; "
                f"panel '{surface_id}' differs."
            )
    reference = units[0]["mid"][0]
    canonical = np.array([closed.points[reference].x, closed.points[reference].y])
    actual = np.array([pattern.points[reference].x, pattern.points[reference].y])
    return ClosedPattern(closed, rotation, actual - rotation @ canonical, tuple(added))
